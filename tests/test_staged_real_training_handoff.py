"""Real staged training handoff with explicit CPU lifecycle test doubles.

The production adapter, orchestrator, workflow state, immutable receipts,
checkpoint integrity validation and resume are real. The CPU lifecycle here
is NOT actual PEFT optimization; existing tiny-model tests prove that part.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from hephaestus.control.staged_state import StagedAutonomousDependencies, StagedOutputRecord
from hephaestus.production.composition import ProductionCompositionRoot, ProductionCompositionSettings
from hephaestus.production.loop import ProductionLoopRunner
from hephaestus.production.staged_driver import StagedProductionCycleDriver
from hephaestus.production.staged_training import StagedTrainingLifecycleAdapter
from hephaestus.schemas.contract_common import ContractIssue
from hephaestus.schemas.experiment_contract import ExperimentProposal, TrainingRunHandle
from hephaestus.storage.verified import canonical_bytes
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest

from test_staged_autonomous_orchestration import FakeOperationRouter, _services


def _runtime(root):
    return ProductionCompositionRoot(ProductionCompositionSettings(
        state_root=root / "state", artifact_root=root / "artifacts",
    )).build()


def _checkpoint(root: Path) -> str:
    folder = root / "synthetic-finalized-checkpoint"
    (folder / "model").mkdir(parents=True)
    (folder / "model" / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
    (folder / "model" / "adapter_model.safetensors").write_bytes(b"synthetic-model-not-real-training")
    entries = {
        p.relative_to(folder).as_posix(): "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (folder / "model").rglob("*") if p.is_file()
    }
    digest = "sha256:" + hashlib.sha256(canonical_bytes(entries)).hexdigest()
    (folder / "checkpoint_manifest.json").write_text(json.dumps({
        "components": entries, "manifest_hash": digest, "partial_write": False,
    }), encoding="utf-8")
    assert validate_checkpoint_manifest(folder) == (True, digest)
    return str(folder)


class ControlledLifecycle:
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint
        self.active = {}
        self.launches = 0
        self.done = False

    def status(self, run_id):
        proposal = self.active.get(run_id)
        if proposal is None:
            return TrainingRunHandle(
                run_id=run_id, experiment_id="unknown", backend_id="cpu-test-double",
                status="failed",
                issues=[ContractIssue(
                    code="run_not_found", category="missing_evidence",
                    message="The test training worker has not launched this run.", blocking=True,
                )],
            )
        return self._handle(proposal)

    def launch(self, proposal):
        self.launches += 1
        if proposal.run_id in self.active:
            raise AssertionError("double launch detected")
        self.active[proposal.run_id] = proposal
        return self._handle(proposal)

    def _handle(self, proposal):
        return TrainingRunHandle(
            run_id=proposal.run_id, experiment_id=proposal.experiment_id,
            backend_id="cpu-test-double", status="completed" if self.done else "running",
            checkpoint_refs=[self.checkpoint] if self.done else [],
        )


class PlannedWorker(FakeOperationRouter):
    def __init__(self, *, invalid_data=False, unresolved=False, approvals=False):
        super().__init__(action="continue_lineage_best")
        self.invalid_data = invalid_data
        self.unresolved = unresolved
        self.approvals = approvals

    def execute(self, request):
        result = super().execute(request)
        if request.substep == "experiment_proposal":
            proposal = ExperimentProposal(
                experiment_id="scientific-experiment-" + request.run_id,
                run_id=request.run_id, lineage_id=request.lineage_id,
                stage_name=request.stage_name, diagnosis_report_id="diagnosis-verified",
                intervention_id="model-approved-intervention",
                primary_variable="lora_rank",
                baseline_ref="checkpoint://frozen-baseline",
                training_constraints={
                    "model_id": "tiny-pinned-test-model",
                    "model_revision": "f" * 40,
                    "data_contract_ref": "" if self.invalid_data else "sha256:" + "b" * 64,
                    "training_recipe_ref": "recipe://controlled-test",
                    "max_steps": 3,
                },
                required_approvals=["operator_approval"] if self.approvals else [],
                status="ready", metadata={"unresolved_discovery": (
                    ["dataset_selection"] if self.unresolved else []
                )},
            )
            return replace(result, records=(
                StagedOutputRecord("experiment_proposal", proposal.to_dict()),
            ), metadata={"experiment_id": proposal.experiment_id})
        if request.substep == "semantic_comparison":
            return replace(result, metadata={**result.metadata,
                "comparison_id": "comparison-" + request.run_id})
        if request.substep == "governed_verdict":
            return replace(result, metadata={**result.metadata, "confidence": 0.8})
        return result


def _driver(worker, *, execute_enabled):
    def provide(runtime, state, cycle):
        services = _services(worker)
        services.action_executor = None
        # Retain the real policy-bound and guarded lifecycle in production
        # but inject an explicitly synthetic CPU worker for the tests.
        services = runtime.with_training_services(services, execute_enabled=execute_enabled)
        return StagedAutonomousDependencies(
            services=services, state_repository=runtime.state_repository,
            artifact_store=runtime.artifact_store,
        )
    return StagedProductionCycleDriver(provide)


def test_training_disabled_blocks_before_launch_and_preserves_campaign(tmp_path):
    runtime = _runtime(tmp_path)
    worker = PlannedWorker()
    lifecycle = ControlledLifecycle(_checkpoint(tmp_path))
    runtime.training_lifecycle.delegate = lifecycle
    state = ProductionLoopRunner(runtime, _driver(worker, execute_enabled=False),
        maximum_cycles=1).run(program_id="disabled-campaign",
                              lineage_id="L", stage_name="smoke_test")
    assert state.status == "waiting" and state.cycle_index == 0
    assert "training_execution_disabled" in state.metadata["pending_cycle"]["evidence"]["blocking_issues"]
    assert lifecycle.launches == 0


def test_live_training_service_is_launched_once_and_recovered_after_restart(tmp_path):
    runtime = _runtime(tmp_path)
    worker = PlannedWorker()
    lifecycle = ControlledLifecycle(_checkpoint(tmp_path))
    runtime.training_lifecycle.delegate = lifecycle
    state = ProductionLoopRunner(runtime, _driver(worker, execute_enabled=True),
        maximum_cycles=1).run(program_id="real-handoff-campaign",
                              lineage_id="L", stage_name="smoke_test")
    assert state.status == "waiting" and state.cycle_index == 0
    assert lifecycle.launches == 1

    resumed = _runtime(tmp_path)
    resumed.training_lifecycle.delegate = lifecycle
    lifecycle.done = True
    end = ProductionLoopRunner(resumed, _driver(PlannedWorker(), execute_enabled=True),
        maximum_cycles=1).run(program_id="real-handoff-campaign",
                              lineage_id="L", stage_name="smoke_test")
    assert end.cycle_index == 1 and end.status == "stopped"
    assert lifecycle.launches == 1
    assert end.metadata["latest_cycle"]["checkpoint_ref"] == lifecycle.checkpoint

    from hephaestus.control.staged_state import WORKFLOW_RECORD_COLLECTION
    kinds = [r["kind"] for r in resumed.state_repository.all(WORKFLOW_RECORD_COLLECTION)]
    assert "experiment_training_binding" in kinds
    assert "training_launch_configuration" in kinds
    assert "checkpoint_resolution" in kinds
    assert "training_run_handle" in kinds


@pytest.mark.parametrize("invalid,unresolved,approvals", [
    (True, False, False), (False, True, False), (False, False, True),
])
def test_missing_data_discovery_or_operator_approval_prevents_launch(
    tmp_path, invalid, unresolved, approvals,
):
    runtime = _runtime(tmp_path)
    lifecycle = ControlledLifecycle(_checkpoint(tmp_path))
    runtime.training_lifecycle.delegate = lifecycle
    worker = PlannedWorker(invalid_data=invalid, unresolved=unresolved, approvals=approvals)
    state = ProductionLoopRunner(runtime, _driver(worker, execute_enabled=True),
        maximum_cycles=1).run(program_id="rejected-input",
                              lineage_id="L", stage_name="smoke_test")
    assert state.status == "blocked" and state.cycle_index == 0
    assert lifecycle.launches == 0


def test_corrupted_finalized_checkpoint_does_not_reach_evaluation(tmp_path):
    runtime = _runtime(tmp_path)
    lifecycle = ControlledLifecycle(_checkpoint(tmp_path))
    runtime.training_lifecycle.delegate = lifecycle
    worker = PlannedWorker()
    wait = ProductionLoopRunner(runtime, _driver(worker, execute_enabled=True),
        maximum_cycles=1).run(program_id="tamper-campaign",
                              lineage_id="L", stage_name="smoke_test")
    assert wait.status == "waiting"
    lifecycle.done = True
    (Path(lifecycle.checkpoint) / "model" / "adapter_model.safetensors").write_bytes(b"tampered")
    result = ProductionLoopRunner(runtime, _driver(worker, execute_enabled=True),
        maximum_cycles=1).run(program_id="tamper-campaign",
                              lineage_id="L", stage_name="smoke_test")
    assert result.status == "blocked" and result.cycle_index == 0
    assert lifecycle.launches == 1
