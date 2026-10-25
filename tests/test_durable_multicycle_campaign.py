"""Real production loop + 45-step staged orchestration across 3 simulated cycles.

This is a CPU integration of *control flow*, immutable evidence/history,
durable action ownership, and process reconstruction. Explicitly synthetic
stage workers are injected; no claim of real five-model inference/training.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import pytest

from hephaestus.control.staged_state import StagedAutonomousDependencies, StagedOperationRequest, StagedOutputRecord
from hephaestus.production.composition import ProductionCompositionRoot, ProductionCompositionSettings
from hephaestus.production.loop import ProductionLoopRunner
from hephaestus.production.staged_driver import StagedProductionCycleDriver
from hephaestus.production.state import ACTION_EXECUTIONS
from hephaestus.storage.verified import read_verified_json
from test_staged_autonomous_orchestration import FakeOperationRouter, _services


@dataclass
class ScientificSimulationWorker(FakeOperationRouter):
    """Simulates external workers but uses genuine staging, storage and memory."""
    runtime: object = None
    wait_cycle_two: bool = False
    invocations: list[tuple[str, str]] = field(default_factory=list)
    memory_seen: dict[str, list[dict]] = field(default_factory=dict)

    def execute(self, request: StagedOperationRequest):
        cycle = int(request.run_id.rsplit("-", 1)[-1])
        self.invocations.append((request.run_id, request.substep))
        if request.substep == "intervention_ranking":
            # The next simulated intervention is selected from *actual*
            # immutable experiment memory, not a variable in the loop.
            memories = self.runtime.planner.memory_query.dead_ends_for_lineage(request.lineage_id)
            self.memory_seen[request.run_id] = list(memories)
            chosen = "curriculum_reweight" if memories else "wrapper_diagnostic"
            self._chosen = chosen
        if request.substep == "governed_verdict":
            self.action = "reject_candidate" if cycle == 1 else (
                "request_recheck" if cycle == 2 else "abort_run")
        if request.substep == "training_status_poll":
            self.poll_statuses = ["running"] if self.wait_cycle_two and cycle == 2 else ["completed"]
        result = super().execute(request)
        md = dict(result.metadata)
        records = list(result.records)
        if request.substep == "intervention_ranking":
            md["intervention_id"] = "intervention-" + request.run_id
            md["intervention_kind"] = self._chosen
            records = [StagedOutputRecord("intervention_proposal", {
                "run_id": request.run_id, "intervention_id": md["intervention_id"],
                "intervention_kind": self._chosen, "primary_variable": self._chosen,
                "required_inputs": [],
            })]
        elif request.substep == "experiment_proposal":
            md["experiment_id"] = "experiment-" + request.run_id
            records = [StagedOutputRecord("experiment_proposal", {
                "experiment_id": md["experiment_id"],
                "intervention_id": "intervention-" + request.run_id,
                "primary_variable": self._chosen,
                "run_id": request.run_id, "lineage_id": request.lineage_id,
                "stage_name": request.stage_name, "required_evidence": [],
                "controlled_variables": {"frozen_parent": "V3.1"},
                "baseline_ref": "checkpoint://V3.1",
            })]
        elif request.substep == "semantic_comparison":
            md["comparison_id"] = "comparison-" + request.run_id
            outcome = "regressed" if cycle == 1 else "improved"
            records = [StagedOutputRecord("experiment_comparison", {
                "comparison_id": md["comparison_id"],
                "experiment_id": "experiment-" + request.run_id,
                "run_id": request.run_id, "primary_outcome": outcome,
                "deterministic_gate_status": "failed" if cycle == 1 else "passed",
                "evidence_level": "explicit_synthetic_cpu_test_worker",
            })]
        elif request.substep == "governed_verdict":
            md["confidence"] = 0.78
        elif request.substep == "promotion_gate":
            md["certification_state"] = "not_certified"
            md["promotion_allowed"] = False
            md["action_allowed"] = False
        return replace(result, metadata=md, records=tuple(records))


def _runtime(root):
    return ProductionCompositionRoot(ProductionCompositionSettings(
        state_root=root / "state", artifact_root=root / "artifacts")).build()


def _factory(worker):
    def provide(runtime, state, cycle):
        worker.runtime = runtime
        services = _services(worker)
        services.action_executor = None  # production loop alone applies actions
        return StagedAutonomousDependencies(services, runtime.state_repository)
    return provide


def test_three_complete_spine_cycles_with_failure_memory_and_restart(tmp_path):
    runtime = _runtime(tmp_path)
    first = ScientificSimulationWorker(wait_cycle_two=True)
    initial = ProductionLoopRunner(
        runtime, StagedProductionCycleDriver(_factory(first)), maximum_cycles=3,
    ).run(program_id="recovery-campaign", lineage_id="evaluator-lineage",
          stage_name="smoke_test")
    assert initial.status == "waiting" and initial.cycle_index == 1
    assert initial.completed_cycles == ["recovery-campaign:cycle:1"]
    assert first.memory_seen["recovery-campaign-cycle-1"] == []
    second_memory = first.memory_seen["recovery-campaign-cycle-2"]
    assert len(second_memory) == 1
    assert second_memory[0]["memory_type"] == "known_dead_end"
    assert second_memory[0]["metadata"]["intervention_kind"] == "wrapper_diagnostic"
    assert second_memory[0]["metadata"]["causal_failure_claim"] is False
    assert sum(sub == "lifecycle_launch" for _, sub in first.invocations) == 2

    # Recreate the production composition and worker, not merely the runner.
    restarted = _runtime(tmp_path)
    second = ScientificSimulationWorker()
    completed = ProductionLoopRunner(
        restarted, StagedProductionCycleDriver(_factory(second)), maximum_cycles=3,
    ).run(program_id="recovery-campaign", lineage_id="evaluator-lineage",
          stage_name="smoke_test")
    assert completed.status == "stopped"
    assert completed.stop_reason == "judge_action:abort_run"
    assert completed.cycle_index == 3
    assert len(set(completed.completed_cycles)) == 3
    assert not any(run.endswith("-1") for run, _ in second.invocations)
    assert not any(run.endswith("-2") and step == "lifecycle_launch" for run, step in second.invocations)
    assert second.memory_seen["recovery-campaign-cycle-3"]
    assert second.memory_seen["recovery-campaign-cycle-3"][0]["metadata"]["intervention_kind"] == "wrapper_diagnostic"
    actions = [r for r in restarted.state_repository.all(ACTION_EXECUTIONS)
               if r.get("status") == "applied"]
    assert [r["effective_action"] for r in actions] == [
        "reject_candidate", "request_recheck", "abort_run",
    ]
    assert len({r["execution_id"] for r in actions}) == 3
    events = [r["kind"] for r in restarted.state_repository.all("production_loop_events")
              if r["program_id"] == "recovery-campaign"]
    assert events.count("cycle_completed") == 3
    assert "cycle_waiting" in events
    from hephaestus.control.staged_state import WORKFLOW_RECORD_COLLECTION
    comparisons = [r for r in restarted.state_repository.all(WORKFLOW_RECORD_COLLECTION)
                   if r["kind"] == "experiment_comparison"]
    assert len(comparisons) == 3
    for row in comparisons:
        assert row.get("artifact_ref", "").startswith("sha256:")
        stored = read_verified_json(restarted.artifact_store, row["artifact_ref"])
        assert stored == {k: v for k, v in row.items() if k != "artifact_ref"}

    reloaded = _runtime(tmp_path)
    same = ProductionLoopRunner(
        reloaded, StagedProductionCycleDriver(_factory(ScientificSimulationWorker())),
        maximum_cycles=3,
    ).run(program_id="recovery-campaign", lineage_id="evaluator-lineage",
          stage_name="smoke_test")
    assert same.cycle_index == 3 and len(same.completed_cycles) == 3
    assert len([r for r in reloaded.state_repository.all(ACTION_EXECUTIONS)
                if r.get("status") == "applied"]) == 3


def test_campaign_identity_is_immutable_and_cannot_reset_existing_state(tmp_path):
    runtime = _runtime(tmp_path)
    first = ScientificSimulationWorker(wait_cycle_two=True)
    runner = ProductionLoopRunner(runtime, StagedProductionCycleDriver(_factory(first)), maximum_cycles=3)
    assert runner.run(program_id="identity-campaign", lineage_id="original",
                      stage_name="smoke_test").status == "waiting"
    with pytest.raises(ValueError, match="identity changed"):
        runner.run(program_id="identity-campaign", lineage_id="different",
                   stage_name="smoke_test")
    with pytest.raises(ValueError, match="identity changed"):
        runner.run(program_id="identity-campaign", lineage_id="original",
                   stage_name="changed-stage")
    with pytest.raises(ValueError, match="cannot overwrite"):
        runner.run(program_id="identity-campaign", lineage_id="original",
                   stage_name="smoke_test", resume=False)
    with pytest.raises(ValueError, match="positive cycle limit"):
        ProductionLoopRunner(runtime, StagedProductionCycleDriver(_factory(first)),
                             maximum_cycles=0).run(program_id="identity-campaign",
                             lineage_id="original", stage_name="smoke_test")
    with pytest.raises(ValueError, match="cycle budget changed"):
        ProductionLoopRunner(runtime, StagedProductionCycleDriver(_factory(first)),
                             maximum_cycles=2).run(program_id="identity-campaign",
                             lineage_id="original", stage_name="smoke_test")


def test_role_model_proposal_flows_into_real_production_loop(tmp_path):
    """One real model-backed control cycle; generation remains a declared CPU stub."""
    from hephaestus.control.staged_state import WORKFLOW_RECORD_COLLECTION
    from hephaestus.llm.role_contract import RoleMasteryContract
    from hephaestus.llm.certified_runtime import CertifiedRoleReasoner, VerifiedRoleAdapters
    from hephaestus.schemas.scientific_evidence import ScientificEvidenceBinding
    from hephaestus.storage.verified import canonical_bytes
    from hephaestus.schemas.diagnosis_contract import DiagnosticHypothesis
    from hephaestus.control.staged_state import StagedOutputRecord
    from test_model_backed_staged_roles import ModelProposals, ROOT
    from test_closed_loop_planner import _diagnosis
    from test_certified_role_inference import _archive, _roles, FakeS3

    class SimulatedModelScientificWorker(ScientificSimulationWorker):
        def execute(self, request):
            result = super().execute(request)
            if request.substep == "governed_verdict":
                # smoke_test permits continue_lineage_best, not reject_candidate.
                # Preserve the stage policy rather than bending it for the test.
                metadata = {**result.metadata, "proposed_action": "continue_lineage_best"}
                return replace(result, metadata=metadata, records=(
                    StagedOutputRecord("judge_exit", {
                        "run_id": request.run_id,
                        "proposed_action": "continue_lineage_best",
                        "confidence": 0.78,
                    }),
                ))
            if request.substep == "evidence_based_diagnosis":
                report = _diagnosis(hypotheses=[DiagnosticHypothesis(
                    "hyp-eval", "evaluation_integrity",
                    "Diagnose the verified, frozen Evaluation D failure.",
                    supporting_evidence_refs=[initial.artifact_ref],
                    recommended_intervention_kinds=["repair_evaluation", "collect_more_evidence"],
                    confidence=0.8)])
                report.run_id = request.run_id
                report.lineage_id = request.lineage_id
                report.stage_name = request.stage_name
                return replace(result, records=(StagedOutputRecord("diagnosis_report", report.to_dict()),))
            return result

    runtime = _runtime(tmp_path)
    program = "model-campaign"
    run_id = program + "-cycle-1"
    lineage = "model-evaluator-lineage"
    initial = runtime.artifact_store.put_bytes(canonical_bytes({
        "run_id": run_id, "lineage_id": lineage, "stage_name": "smoke_test",
        "kind": "eval_report",
        "payload": {"status": "failed", "eval_integrity_verified": True},
    }))
    runtime.model_evidence.store.register(ScientificEvidenceBinding(
        "initial-failure", run_id, run_id, lineage, "smoke_test",
        "eval_report", initial.artifact_ref,
    ))
    blobs = {role: _archive(role=role) for role in (
        "controller", "diagnosis", "planner", "evaluator", "judge")}
    s3 = FakeS3(blobs)
    model = ModelProposals(judge_action="continue_lineage_best")
    reasoner = CertifiedRoleReasoner(
        _roles(blobs),
        RoleMasteryContract.load(ROOT / "configs/models/hephaestus_role_mastery_runtime_v1.json"),
        VerifiedRoleAdapters(s3, "read-only", tmp_path / "model-cache"),
        model, runtime.artifact_store,
    )
    worker = SimulatedModelScientificWorker()
    def provide(rt, state, cycle):
        worker.runtime = rt
        services = _services(worker)
        services.action_executor = None
        return StagedAutonomousDependencies(
            services=rt.with_role_models(services, reasoner),
            state_repository=rt.state_repository, artifact_store=rt.artifact_store,
        )

    state = ProductionLoopRunner(
        runtime, StagedProductionCycleDriver(provide), maximum_cycles=1,
    ).run(program_id=program, lineage_id=lineage, stage_name="smoke_test")
    assert state.cycle_index == 1 and state.status == "stopped"
    assert model.calls == ["controller", "diagnosis", "planner", "evaluator", "judge"]
    records = runtime.state_repository.all(WORKFLOW_RECORD_COLLECTION)
    proposals = [r["payload"] for r in records if r["kind"] == "experiment_proposal"]
    assert len(proposals) == 1 and proposals[0]["metadata"]["model_originated"] is True
    assert proposals[0]["primary_variable"] == "diagnostic_measurement"
    assert state.latest_experiment_id == proposals[0]["experiment_id"]
    actions = [r for r in runtime.state_repository.all(ACTION_EXECUTIONS)
               if r.get("status") == "applied"]
    assert len(actions) == 1 and actions[0]["effective_action"] == "continue_lineage_best"
    assert len(s3.gets) == 5
