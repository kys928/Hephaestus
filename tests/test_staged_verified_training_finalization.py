"""Planner discovery -> approved data+model -> immutable PEFT launch proposal.

Real local acquisition, actual preprocessing and persisted SHA artifacts are
used. Model directories and acceptance decisions are declared CPU fixtures;
no optimization or artificial evaluation result is claimed here.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import json

import pytest

from hephaestus.backends.hf_causal_lm import directory_content_identity
from hephaestus.control.staged_state import StagedOperationRequest
from hephaestus.production.staged_finalization import StagedExperimentFinalizer
from hephaestus.production.staged_training import StagedTrainingLifecycleAdapter
from hephaestus.schemas.discovery_contract import ModelCandidate, ModelSelectionDecision
from hephaestus.schemas.experiment_contract import ExperimentProposal
from hephaestus.storage.verified import canonical_bytes
from test_staged_approved_local_data_factory import (
    _setup, _completed_source, _factory, _write_row,
)


class CheckedTokenizer:
    def __init__(self, ref):
        self.tokenizer_ref = ref
        self.checker_id = "cpu-tokenizer-compatibility-probe-v1"
    def check(self, texts):
        return bool(texts and all(isinstance(x, str) for x in texts)), {
            "checked_rows": len(texts), "tokenizer_ref": self.tokenizer_ref,
        }


class RealCheckedTokenizer:
    def __init__(self, ref):
        self.tokenizer_ref = ref
        self.checker_id = "local-transformers-tokenizer-encode-v1"
    def check(self, texts):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_ref, local_files_only=True)
        lengths = [len(tokenizer(str(text), add_special_tokens=True)["input_ids"])
                   for text in texts]
        return bool(lengths and all(0 < n <= 32 for n in lengths)), {
            "verified_tokens": sum(lengths), "checked_rows": len(lengths),
            "model_tokenizer_path": self.tokenizer_ref,
        }


def _req():
    return StagedOperationRequest(
        workflow_id="wf-local-data", run_id="data-run",
        lineage_id="lineage-1", stage_name="smoke_test",
        phase="training_engineer", substep="training_input_binding",
        operation_id="op-training_input_binding", attempt=1,
        input_refs=(), prior_outputs={},
    )


def _setup_finalized(tmp_path, *, checked_tokenizer=True,
                     approved_recipe=True, wrong_model=False, approved_data=True,
                     real_model=False):
    runtime, source = _setup(tmp_path, approved=approved_data)
    if real_model:
        from test_real_hf_training import _tiny_transformers_fixture
        (tmp_path / "model-input").mkdir()
        model_dir, model_revision, count = _tiny_transformers_fixture(
            tmp_path / "model-input")
        tokenizer_dir = model_dir
        tokenizer_revision = model_revision
    else:
        model_dir = tmp_path / "pinned-model"
        tokenizer_dir = tmp_path / "pinned-tokenizer"
        model_dir.mkdir()
        tokenizer_dir.mkdir()
        (model_dir / "config.json").write_text('{"model_type":"tiny_cpu_fixture"}')
        (tokenizer_dir / "tokenizer.json").write_text('{"version":"fixture"}')
        model_revision = directory_content_identity(model_dir)
        tokenizer_revision = directory_content_identity(tokenizer_dir)
        count = 300
    if checked_tokenizer:
        runtime.data_preprocessor.tokenizer_checker = (
            RealCheckedTokenizer(str(tokenizer_dir)) if real_model else
            CheckedTokenizer(str(tokenizer_dir)))
    if approved_data:
        _completed_source(runtime, _factory(runtime))
    candidate = ModelCandidate(
        candidate_id="verified-local-model", provider_id="local_fixture",
        model_id=str(model_dir), revision=model_revision,
        architecture_family="gpt2" if real_model else "decoder_only",
        tokenizer_ref=str(tokenizer_dir), license="apache-2.0",
        capabilities=["causal_lm"], missing_metadata=[], risk_signals=[],
        evidence_refs=["model://verified-local-fixture"],
    )
    model_selection = ModelSelectionDecision(
        decision_id="selected-model-1", request_id="model-search-1",
        status="selected", selected_candidate_id="verified-local-model",
        ranked_candidate_ids=["verified-local-model"],
        required_approvals=[],
    )
    _write_row(runtime, "model_discovery", "model_discovery_result", {
        "request_id": "model-search-1", "status": "discovered",
        "kind": "model", "candidates": [candidate.to_dict()], "issues": [],
    }, ordinal=20)
    _write_row(runtime, "model_selection", "model_selection_decision",
               model_selection.to_dict(), ordinal=21)
    peft_config = {
        "rank": 4, "alpha": 8.0, "dropout": 0.0,
        "target_modules": ["c_attn"] if real_model else ["q_proj", "v_proj"],
        "continuation_kind": "fresh", "bias": "none",
    }
    approved = {
        "schema": "approved-peft-recipe.v1",
        "training_mode": "lora_finetune", "peft_config": peft_config,
        "max_steps": 2, "model_selection_id": model_selection.decision_id,
    }
    recipe = runtime.artifact_store.put_bytes(canonical_bytes(approved), media_type="application/json")
    recipe_ref = recipe.artifact_ref if approved_recipe else "recipe://unapproved"
    proposal = ExperimentProposal(
        experiment_id="real-scientific-experiment", run_id="data-run",
        lineage_id="lineage-1", stage_name="smoke_test",
        diagnosis_report_id="diag-1", intervention_id="intervention-1",
        primary_variable="lora_rank", baseline_ref="frozen-parent",
        status="pending", dataset_selection_id="selected-1",
        model_selection_id=model_selection.decision_id,
        required_approvals=["dataset_acquisition"],
        metadata={"unresolved_discovery":["dataset", "model"]},
        training_constraints={
            "model_id": str(model_dir) if not wrong_model else str(tmp_path / "foreign"),
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "training_mode": "lora_finetune", "peft_config": peft_config,
            "training_recipe_ref": recipe_ref, "seed": 19,
            "max_steps": 2, "optimizer": "adamw",
            "scheduler": "constant", "context_length": 16 if real_model else 128,
            "vocabulary_size": 9 if real_model else 256,
            "parameter_count": count, "hidden_size": 16,
            "device": "cpu", "dtype": "float32",
            "learning_rate": 0.01, "batch_size": 1, "max_total_tokens": 256,
            "local_files_only": True,
            "special_token_ids": ({"eos_token_id": 1, "pad_token_id": 0, "unk_token_id": 2}
                                  if real_model else {"eos_token_id": 2, "pad_token_id": 0}),
        },
    )
    _write_row(runtime, "experiment_proposal", "experiment_proposal",
               proposal.to_dict(), ordinal=22)
    service = StagedExperimentFinalizer(
        runtime.state_repository, runtime.artifact_store,
        StagedTrainingLifecycleAdapter(runtime.state_repository, runtime.artifact_store,
                                       runtime.training_lifecycle, execute_enabled=False),
        approved_recipe_refs=frozenset((recipe.artifact_ref,)),
        approved_baseline_refs=frozenset(("frozen-parent",)),
    )
    return runtime, service, _req(), source


def test_real_data_model_and_operator_evidence_finalizes_peft_training_inputs(tmp_path):
    runtime, finalizer, request, _ = _setup_finalized(tmp_path)
    proposal = finalizer.finalize(request)
    assert proposal.status == "approved"
    assert proposal.required_approvals == []
    assert not proposal.metadata.get("unresolved_discovery")
    assert proposal.model_selection_id == "selected-model-1"
    assert proposal.dataset_selection_id == "selected-1"
    config = proposal.training_constraints
    assert config["training_mode"] == "lora_finetune"
    assert config["model_revision"].startswith("sha256:")
    assert config["trainable_data_contract_ref"].endswith("trainable_data_contract.json")
    assert config["trainable_data_contract_hash"].startswith("sha256:")
    assert config["processed_dataset_hash"].startswith("sha256:")
    assert config["processing_evidence_hash"].startswith("sha256:")
    assert json.loads(Path(config["trainable_data_contract_ref"]).read_text())["run_id"] == "data-run"
    assert json.loads(Path(config["processing_evidence_ref"]).read_text())["tokenizer_compatibility"]["compatible"]
    result = finalizer.execute(request)
    assert result.status == "completed"
    assert result.records[0].kind == "experiment_training_binding"
    assert result.records[0].payload["training_constraints"] == config


@pytest.mark.parametrize("variant", [
    "no_approval", "not_checked", "unapproved_recipe", "wrong_model",
    "tampered_processed", "tampered_contract", "tampered_model",
])
def test_unverified_training_inputs_refuse_finalization(tmp_path, variant):
    runtime, finalizer, request, source = _setup_finalized(
        tmp_path, checked_tokenizer=variant != "not_checked",
        approved_recipe=variant != "unapproved_recipe",
        wrong_model=variant == "wrong_model",
        approved_data=variant != "no_approval",
    )
    if variant == "tampered_processed":
        data = finalizer._load(request, "manifest_production", "data_factory_result")
        Path(data["processed_dataset_ref"]).write_bytes(b"tampered")
    elif variant == "tampered_contract":
        data = finalizer._load(request, "manifest_production", "data_factory_result")
        source = Path(data["processing_evidence_ref"]).parent / "trainable_data_contract.json"
        source.write_text('{"run_id":"other"}')
    elif variant == "tampered_model":
        data = finalizer._load(request, "model_discovery", "model_discovery_result")
        Path(data["candidates"][0]["model_id"], "config.json").write_text('{"changed":true}')
    result = finalizer.execute(request)
    assert result.status == "blocked"
    assert any("scientific_training_finalization_failed" in x for x in result.blocking_issues)


def test_cannot_consider_untrusted_writable_recipe_to_be_policy_approved(tmp_path):
    runtime, finalizer, request, _ = _setup_finalized(tmp_path)
    finalizer.approved_recipe_refs = frozenset()
    result = finalizer.execute(request)
    assert result.status == "blocked"
    assert "scientific_training_finalization_failed:PermissionError" in result.blocking_issues


@pytest.mark.real_cpu
def test_admitted_verified_proposal_runs_real_tiny_lora_and_generates_on_cpu(tmp_path):
    """Actual optimizer/PEFT/model weights, NOT a simulated scientific success."""
    import importlib.util
    if any(importlib.util.find_spec(name) is None
           for name in ("torch", "transformers", "tokenizers", "peft")):
        pytest.skip("requires opt-in real CPU PEFT dependencies")
    from hephaestus.training.hf_lifecycle import (
        TransformersTrainingLifecycleService, validate_checkpoint_manifest,
    )
    from hephaestus.generation.backends import TransformersCausalLMGenerationBackend
    from hephaestus.generation.models import GenerationTask
    from test_real_hf_training import _wait

    runtime, finalizer, request, _ = _setup_finalized(tmp_path, real_model=True)
    proposal = finalizer.finalize(request)
    assert proposal.training_constraints["training_mode"] == "lora_finetune"
    worker = TransformersTrainingLifecycleService(tmp_path / "real-cpu-lora-runs")
    started = worker.launch(proposal)
    assert started.status == "running", started.to_dict()
    result = _wait(worker, proposal.run_id, {"completed", "failed"}, timeout=100)
    assert result.status == "completed", (result.to_dict(),
        Path(str(result.metadata.get("log_ref", ""))).read_text(encoding="utf-8")[-5000:]
        if result.metadata.get("log_ref") else "no worker log")
    assert result.checkpoint_refs
    checkpoint = Path(result.checkpoint_refs[-1])
    assert validate_checkpoint_manifest(checkpoint)[0]
    assert (checkpoint / "model" / "adapter_model.safetensors").is_file()
    assert not (checkpoint / "model" / "model.safetensors").exists()

    instructions = json.loads((checkpoint / "loading_instructions.json").read_text())
    generated = TransformersCausalLMGenerationBackend().generate_batch(
        [GenerationTask("probe", "generation_probes", "hello world", 17)],
        run_id="post-lora-generation",
        loading_instructions=instructions,
        decoding_config={"max_new_tokens": 8, "temperature": 1.0},
    )
    assert len(generated) == 1 and generated[0].output
    # An independent new service must resolve existing completed work and
    # must not change trainable data nor launch another optimization.
    restored = TransformersTrainingLifecycleService(
        tmp_path / "real-cpu-lora-runs").status(proposal.run_id)
    assert restored.status == "completed"
    assert restored.checkpoint_refs == result.checkpoint_refs

    # Use an entirely local *test-only*, SHA-verified frozen evaluation pack:
    # tiny positional embeddings cannot safely generate 96 tokens from the
    # protected production semantic pack. Do not modify that frozen pack.
    from hephaestus.generation.service import EvaluationGenerationService
    from hephaestus.evaluation.experiment_service import ExperimentEvaluationService
    from hephaestus.control.semantic_judge import SemanticComparisonJudgeAdapter
    from hephaestus.schemas.experiment_contract import TrainingRunHandle
    import hashlib
    config_dir = tmp_path / "cpu-eval-configs"
    pack_dir = config_dir / "eval_packs"
    pack_dir.mkdir(parents=True)
    original_pack = json.loads(
        (Path("configs") / "eval_packs" / "semantic_behavior_v1.yaml").read_text())
    pack = dict(original_pack)
    pack["pack_name"] = "tiny_cpu_repair_v1"
    pack["eval_pack_id"] = "tiny_cpu_repair"
    pack["source_ref"] = "configs/eval_packs/tiny_cpu_repair_v1.yaml"
    pack["decoding_config"] = {
        "temperature": 0.0, "top_p": 1.0, "max_new_tokens": 8, "seeds": [11, 29],
    }
    for bundle in ("generation_probes", "continuation_prompts", "ranking_sets",
                   "regression_prompts", "structure_tests", "repetition_checks",
                   "length_termination_checks"):
        pack[bundle] = []
    pack["generation_probes"] = [{
        "task_id": "tiny_short_probe", "prompt": "hello world",
        "checks": [{"name": "nonempty_bounded_answer", "type": "max_words",
                    "value": 12, "dimension": "excessive_length", "hard": False}],
    }]
    pack["required_evidence"] = {"comparison_runs": 2, "seeds_per_task": 2}
    pack.pop("content_hash")
    pack["content_hash"] = hashlib.sha256(
        json.dumps(pack, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode()).hexdigest()
    (pack_dir / "tiny_cpu_repair_v1.yaml").write_text(
        json.dumps(pack, sort_keys=True), encoding="utf-8")

    model_dir = Path(proposal.training_constraints["model_id"])
    # The production generator correctly requires a finalized checkpoint
    # manifest even for a frozen, never-trained parent. Package the genuine
    # pretrained tiny weights into a separately verifiable baseline snapshot.
    import shutil
    frozen_checkpoint = tmp_path / "frozen-parent-finalized"
    shutil.copytree(model_dir, frozen_checkpoint / "model")
    shutil.copytree(model_dir, frozen_checkpoint / "tokenizer")
    component_hashes = {
        p.relative_to(frozen_checkpoint).as_posix(): "sha256:" + hashlib.sha256(
            p.read_bytes()).hexdigest()
        for p in frozen_checkpoint.rglob("*") if p.is_file()
    }
    (frozen_checkpoint / "checkpoint_manifest.json").write_text(json.dumps({
        "components": component_hashes,
        "manifest_hash": "sha256:" + hashlib.sha256(canonical_bytes(component_hashes)).hexdigest(),
        "partial_write": False,
    }), encoding="utf-8")
    assert validate_checkpoint_manifest(frozen_checkpoint)[0]
    baseline_instructions = tmp_path / "frozen-parent-loading.json"
    baseline_instructions.write_text(json.dumps({
        "backend": "transformers_causal_lm",
        "model_artifact_ref": str(frozen_checkpoint / "model"),
        "tokenizer_artifact_ref": str(frozen_checkpoint / "tokenizer"),
        "trust_remote_code": False,
    }), encoding="utf-8")
    baseline = TrainingRunHandle(
        run_id=proposal.baseline_ref, experiment_id=proposal.experiment_id,
        backend_id="transformers_causal_lm", status="completed",
        checkpoint_refs=[str(frozen_checkpoint)],
        metadata={"generation_handoff_ref": str(baseline_instructions)},
    )
    generation = EvaluationGenerationService(
        tmp_path / "actual-generated-evidence",
        TransformersCausalLMGenerationBackend(),
        pack_name="tiny_cpu_repair_v1", config_dir=config_dir,
        artifact_store=runtime.artifact_store,
    )
    before = generation.generate(baseline)
    after = generation.generate(restored)
    assert before.report.completed and after.report.completed, (
        before.report.to_dict(), after.report.to_dict())
    assert len(before.report.samples) == len(after.report.samples) == 2
    compared = ExperimentEvaluationService(
        pack_name="tiny_cpu_repair_v1", config_dir=config_dir,
    ).compare(proposal, [before.run_handle, after.run_handle])
    assert compared.primary_outcome != "invalid_comparison", compared.to_dict()
    assert compared.evaluation_report_refs or compared.evidence_refs
    verdict = SemanticComparisonJudgeAdapter().decide(
        compared, run_id=proposal.run_id, lineage_id=proposal.lineage_id,
        candidate_checkpoint_ref=str(checkpoint), monitor_outcome="healthy",
        recent_failure_count=0, has_stable_checkpoint=True,
    )
    assert verdict.run_id == proposal.run_id
    assert verdict.lineage_id == proposal.lineage_id
    assert any("certification_state=" in reason for reason in verdict.reasons)
    # The tiny training pack is deliberately not a sealed scientific holdout:
    # generation+Judge execution is not a scientifically certified improvement.


def test_wrong_frozen_parent_ref_stops_training_before_validation(tmp_path):
    runtime, finalizer, request, _ = _setup_finalized(tmp_path)
    finalizer.approved_baseline_refs = frozenset(("another-parent",))
    outcome = finalizer.execute(request)
    assert outcome.status == "blocked"
    assert "scientific_training_finalization_failed:PermissionError" in outcome.blocking_issues
