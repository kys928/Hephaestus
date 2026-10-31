"""Actual immutable training -> frozen generation -> evaluator receipt handoff.

Only the generation backend and training provider are CPU test doubles. The
production wiring, checkpoint hash validation, generation service, semantic
scoring, persisted immutable sources and restart reconstruction are real.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from hephaestus.control.staged_state import (
    StagedAutonomousServices, StagedOperationRequest, WORKFLOW_RECORD_COLLECTION,
)
from hephaestus.generation.backends import DeterministicFakeGenerationBackend
from hephaestus.schemas.experiment_contract import ExperimentProposal, TrainingRunHandle
from hephaestus.storage.verified import canonical_bytes, read_verified_json
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest

from test_evaluation_generation_bridge import _handle as baseline_handle, _output
from test_staged_real_training_handoff import _runtime, _checkpoint


class CompletedCpuTrainer:
    def __init__(self, run):
        self.run = run
        self.calls = 0

    def status(self, run_id):
        self.calls += 1
        assert run_id == self.run.run_id
        return TrainingRunHandle.from_dict(self.run.to_dict())

    def launch(self, proposal):
        raise AssertionError("evaluation must not relaunch a completed training job")


def _candidate_checkpoint(root: Path) -> tuple[str, str]:
    path = Path(_checkpoint(root))
    handoff = path / "loading_instructions.json"
    handoff.write_text(json.dumps({
        "backend": "transformers_causal_lm",
        "model_artifact_ref": str(path / "model"),
        "tokenizer_artifact_ref": str(path / "tokenizer"),
        "trust_remote_code": False,
    }), encoding="utf-8")
    components = {
        p.relative_to(path).as_posix(): "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*") if p.is_file() and p.name != "checkpoint_manifest.json"
    }
    digest = "sha256:" + hashlib.sha256(canonical_bytes(components)).hexdigest()
    (path / "checkpoint_manifest.json").write_text(json.dumps({
        "components": components, "manifest_hash": digest, "partial_write": False,
    }), encoding="utf-8")
    assert validate_checkpoint_manifest(path) == (True, digest)
    return str(path), str(handoff)


def _store_stage(runtime, substep, kind, payload, *, ordinal):
    row = {
        "record_id": f"immutable-stage-{ordinal}",
        "workflow_id": "workflow-candidate", "run_id": "candidate",
        "lineage_id": "lineage-1", "stage_name": "smoke_test",
        "phase": "evaluator", "substep": substep,
        "operation_id": "op-" + substep,
        "sequence": ordinal,
        "kind": kind, "payload": payload,
    }
    ref = runtime.artifact_store.put_bytes(canonical_bytes(row), media_type="application/json")
    runtime.state_repository.append(WORKFLOW_RECORD_COLLECTION, {
        **row, "artifact_ref": ref.artifact_ref,
    })


def _request(substep, previous):
    return StagedOperationRequest(
        workflow_id="workflow-candidate", run_id="candidate",
        lineage_id="lineage-1", stage_name="smoke_test",
        phase="evaluator", substep=substep,
        operation_id="op-" + substep, attempt=1,
        input_refs=(), prior_outputs=previous,
    )


def _components(tmp_path, *, enabled=True):
    runtime = _runtime(tmp_path)
    baseline, _ = baseline_handle(tmp_path, "baseline")
    candidate_checkpoint, handoff = _candidate_checkpoint(tmp_path)
    candidate = TrainingRunHandle(
        run_id="candidate", experiment_id="exp-generation",
        backend_id="cpu-training-test-double", status="completed",
        checkpoint_refs=[candidate_checkpoint],
        metadata={"generation_handoff_ref": handoff},
    )
    trainer = CompletedCpuTrainer(candidate)
    runtime.training_lifecycle.delegate = trainer
    proposal = ExperimentProposal(
        experiment_id="exp-generation", run_id="candidate",
        lineage_id="lineage-1", stage_name="smoke_test",
        diagnosis_report_id="verified-diagnosis",
        intervention_id="policy-admitted-intervention",
        primary_variable="lora_rank", baseline_ref="baseline", status="ready",
    )
    _store_stage(runtime, "training_input_binding", "experiment_training_binding",
                 proposal.to_dict(), ordinal=1)
    pinned = runtime.artifact_store.put_bytes(canonical_bytes({
        "schema": "verified-baseline-run.v1", "handle": baseline.to_dict(),
    }), media_type="application/json")
    backend = DeterministicFakeGenerationBackend(fallback=_output)

    class NotUsed:
        def execute(self, request):
            raise AssertionError("unexpected evaluator fallback")

    def wire(runtime):
        services = runtime.with_training_services(
            StagedAutonomousServices(evaluator=NotUsed()), execute_enabled=False,
        )
        return runtime.with_evaluation_services(
            services, baseline_run_artifact_ref=pinned.artifact_ref,
            generation_backend=backend, inference_enabled=enabled,
        )
    return runtime, trainer, wire, candidate_checkpoint, backend


def test_real_generation_and_comparison_from_verified_immutable_training_artifacts(tmp_path):
    runtime, trainer, wire, checkpoint, backend = _components(tmp_path)
    service = wire(runtime)
    initial = {
        "runtime_monitor.training_status_poll": {
            "training_status": "completed", "checkpoint_ref": checkpoint,
            "experiment_id": "exp-generation",
        },
    }
    checkpoint_step = service.evaluator.execute(_request("checkpoint_resolution", initial))
    assert checkpoint_step.status == "completed"
    assert checkpoint_step.metadata["checkpoint_ref"] == checkpoint
    prior = {**initial, "evaluator.checkpoint_resolution": checkpoint_step.metadata}
    prompt = service.generation.execute(_request("generation_prompt_materialization", prior))
    assert prompt.status == "completed"
    assert prompt.metadata["eval_pack_id"] == "semantic_behavior"
    for step in ("baseline_generation", "candidate_generation"):
        response = service.generation.execute(_request(step, prior))
        assert response.status == "completed"
        assert response.metadata["generation_complete"]
        prior["evaluator." + step] = response.metadata
    assert backend.calls == 2

    # Recreate the entire production runtime/adapter objects from filesystem
    # artifacts and durable SQL state. No in-memory generation handle reused.
    restarted = _runtime(tmp_path)
    restarted.training_lifecycle.delegate = trainer
    evaluation = wire(restarted).evaluator
    comparison = evaluation.execute(_request("semantic_comparison", prior))
    assert comparison.status == "completed"
    assert comparison.records[0].kind == "experiment_comparison"
    assert comparison.metadata["comparison_receipt_ref"].startswith("sha256:")
    assert backend.calls == 2
    _store_stage(restarted, "semantic_comparison", "experiment_comparison",
                 comparison.records[0].payload, ordinal=2)
    prior["evaluator.semantic_comparison"] = comparison.metadata

    # Every downstream evaluator stage reconstructs the exact comparison
    # from its immutable receipt and re-scores persisted generated samples.
    again = _runtime(tmp_path)
    again.training_lifecycle.delegate = trainer
    completed = wire(again).evaluator
    for step in ("deterministic_regression_evidence",
                 "repeatability_variance_evidence", "human_review_references"):
        result = completed.execute(_request(step, prior))
        assert result.status == "completed"
        assert result.records
    assert backend.calls == 2


def test_inference_is_disabled_by_default_even_after_training_completed(tmp_path):
    runtime, trainer, wire, checkpoint, backend = _components(tmp_path, enabled=False)
    result = wire(runtime).generation.execute(_request("baseline_generation", {
        "runtime_monitor.training_status_poll": {
            "training_status": "completed", "checkpoint_ref": checkpoint,
            "experiment_id": "exp-generation",
        },
    }))
    assert result.status == "approval_pending"
    assert "real_generation_disabled" in result.blocking_issues
    assert backend.calls == 0


def test_wrong_baseline_reference_or_tampered_checkpoint_blocks_before_generation(tmp_path):
    runtime, trainer, wire, checkpoint, backend = _components(tmp_path)
    initial = {
        "runtime_monitor.training_status_poll": {
            "training_status": "completed", "checkpoint_ref": checkpoint,
            "experiment_id": "exp-generation",
        },
    }
    (Path(checkpoint) / "model" / "adapter_model.safetensors").write_bytes(b"tampered")
    result = wire(runtime).generation.execute(_request("baseline_generation", initial))
    assert result.status == "blocked"
    assert "verified_evaluation_handoff_invalid:ValueError" in result.blocking_issues
    assert backend.calls == 0
