"""Orchestrator that preserves explicit stage ordering and role boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from hephaestus.backends.base import ExecutionBackend
from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.control.spine import SPINE_ORDER, PhaseResult, SpineCoordinator, SpinePhase
from hephaestus.policy.judge_policy import JudgePolicy
from hephaestus.policy.runtime_policy import RuntimePolicy
from hephaestus.policy.stage_policy import StagePolicy
from hephaestus.roles.data_acquisition_audit import DataAcquisitionAuditRole
from hephaestus.roles.data_preprocessor import DataPreprocessorRole
from hephaestus.roles.evaluator import EvaluatorRole
from hephaestus.roles.judge_entry import JudgeEntryRole
from hephaestus.roles.judge_exit import JudgeExitRole
from hephaestus.roles.planner import PlannerRole
from hephaestus.roles.reporter import ReporterRole
from hephaestus.roles.runtime_monitor import RuntimeMonitorRole
from hephaestus.roles.training_engineer import TrainingEngineerRole
from hephaestus.schemas.decision_record import DecisionRecord
from hephaestus.schemas.eval_report import EvalReport
from hephaestus.schemas.lineage_state import LineageState
from hephaestus.schemas.run_record import RunRecord
from hephaestus.state.artifact_index import ArtifactIndex
from hephaestus.state.decision_store import DecisionStore
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.manifest_store import ManifestStore
from hephaestus.state.report_store import ReportStore
from hephaestus.state.run_store import RunStore


@dataclass(slots=True)
class ControlContext:
    run_id: str
    lineage_id: str
    stage_name: str
    artifact_root: str
    outputs: dict[str, object] = field(default_factory=dict)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_action(value: object) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    return str(value)


@dataclass(slots=True)
class DefaultSpineCoordinator(SpineCoordinator):
    context: ControlContext
    lineage_store: LineageStore
    run_store: RunStore
    decision_store: DecisionStore
    manifest_store: ManifestStore
    report_store: ReportStore
    artifact_index: ArtifactIndex
    backend: ExecutionBackend
    runtime_policy: RuntimePolicy
    stage_policy: StagePolicy
    judge_policy: JudgePolicy

    def run_phase(self, phase: SpinePhase, run_id: str) -> PhaseResult:
        if phase is SpinePhase.JUDGE_ENTRY:
            entry, decision = JudgeEntryRole().run(
                run_id=run_id,
                lineage_id=self.context.lineage_id,
                stage_name=self.context.stage_name,
                created_at=_now(),
            )
            output = entry.to_dict()
            self.decision_store.append(decision.to_dict())
            self.context.outputs[phase.value] = output
            return PhaseResult(phase, "ok", [], output)

        if phase is SpinePhase.PLANNER:
            output = PlannerRole().run(run_id, self.context.stage_name).to_dict()
            self.context.outputs[phase.value] = output
            self.report_store.append({"kind": "experiment_plan", **output})
            return PhaseResult(phase, "ok", [], output)

        if phase is SpinePhase.DATA_ACQUISITION_AUDIT:
            profile, manifest = DataAcquisitionAuditRole(self.backend).run(run_id, self.context.lineage_id)
            output = {"dataset_profile": profile.to_dict(), "dataset_manifest": manifest.to_dict()}
            self.context.outputs[phase.value] = output
            self.manifest_store.append(manifest.to_dict())
            self.report_store.append({"kind": "dataset_profile", **profile.to_dict()})
            self.artifact_index.append({"run_id": run_id, "kind": "dataset_manifest", "ref": manifest.artifact_ref})
            return PhaseResult(phase, "ok", [manifest.artifact_ref], output)

        if phase is SpinePhase.DATA_PREPROCESSOR:
            manifest_id = str(self.context.outputs[SpinePhase.DATA_ACQUISITION_AUDIT.value]["dataset_manifest"]["manifest_id"])
            report, contract = DataPreprocessorRole(self.backend).run(run_id, manifest_id)
            output = {"preprocessing_report": report.to_dict(), "trainable_data_contract": contract.to_dict()}
            self.context.outputs[phase.value] = output
            self.report_store.append({"kind": "preprocessing_report", **report.to_dict()})
            self.report_store.append({"kind": "trainable_data_contract", **contract.to_dict()})
            self.artifact_index.append({"run_id": run_id, "kind": "processed_dataset", "ref": report.processed_dataset_ref})
            return PhaseResult(phase, "ok", [report.processed_dataset_ref], output)

        if phase is SpinePhase.TRAINING_ENGINEER:
            data_contract = self.context.outputs[SpinePhase.DATA_PREPROCESSOR.value]["trainable_data_contract"]
            plan, launch = TrainingEngineerRole().run(
                run_id,
                self.context.stage_name,
                self.context.artifact_root,
                data_contract,
                backend_name=getattr(self.backend, "name", "dry_run"),
                dry_run=getattr(self.backend, "name", "dry_run") == "dry_run",
            )
            output = {"training_plan": plan.to_dict(), "launch_config": launch.to_dict()}
            self.context.outputs[phase.value] = output
            self.report_store.append({"kind": "training_plan", **plan.to_dict()})
            self.report_store.append({"kind": "launch_config", **launch.to_dict()})
            return PhaseResult(phase, "ok", [], output)

        if phase is SpinePhase.RUNTIME_MONITOR:
            planner = self.context.outputs[SpinePhase.PLANNER.value]
            training = self.context.outputs[SpinePhase.TRAINING_ENGINEER.value]
            data_contract = self.context.outputs[SpinePhase.DATA_PREPROCESSOR.value]["trainable_data_contract"]
            monitor = RuntimeMonitorRole(self.backend, self.runtime_policy).run(
                run_id=run_id,
                experiment_plan=planner,
                training_plan=training["training_plan"],
                launch_config=training["launch_config"],
                data_contract=data_contract,
            )
            output = {
                "outcome": monitor.outcome,
                "recommendation": monitor.recommendation,
                "events": [event.to_dict() for event in monitor.events],
                "incidents": [incident.to_dict() for incident in monitor.incidents],
                "training_outputs": monitor.training_outputs,
            }
            self.context.outputs[phase.value] = output
            for event in monitor.events:
                if event.payload_ref:
                    self.artifact_index.append({"run_id": run_id, "kind": event.category.value, "ref": event.payload_ref})
            for incident in monitor.incidents:
                self.report_store.append({"kind": "incident", **incident.to_dict()})
            return PhaseResult(phase, "ok", [event.payload_ref for event in monitor.events if event.payload_ref], output)

        if phase is SpinePhase.EVALUATOR:
            stage_profile = self.stage_policy.resolve(self.context.stage_name)
            runtime_output = self.context.outputs[SpinePhase.RUNTIME_MONITOR.value]
            report = EvaluatorRole().run(run_id, stage_profile, training_outputs=runtime_output["training_outputs"])
            output = report.to_dict()
            self.context.outputs[phase.value] = output
            self.report_store.append({"kind": "eval_report", **output})
            for ref in report.intermediate_artifact_refs:
                self.artifact_index.append({"run_id": run_id, "kind": "intermediate_eval", "ref": ref})
            self.artifact_index.append({"run_id": run_id, "kind": "checkpoint", "ref": report.checkpoint_resolution["selected_checkpoint_ref"]})
            return PhaseResult(phase, "ok", report.intermediate_artifact_refs, output)

        if phase is SpinePhase.JUDGE_EXIT:
            eval_report = self.context.outputs[SpinePhase.EVALUATOR.value]
            monitor_outcome = str(self.context.outputs[SpinePhase.RUNTIME_MONITOR.value]["outcome"])
            judge = JudgeExitRole(self.judge_policy).run(
                run_id,
                self.context.lineage_id,
                eval_report=EvalReport.from_dict(eval_report),
                monitor_outcome=monitor_outcome,
            )
            output = judge.to_dict()
            self.context.outputs[phase.value] = output
            decision = DecisionRecord(f"dec-{run_id}-exit", run_id, self.context.lineage_id, "judge_exit", judge.next_action.value, "; ".join(judge.reasons), judge.confidence, created_at=_now())
            self.decision_store.append(decision.to_dict())
            self.report_store.append({"kind": "judge_exit", **output})
            return PhaseResult(phase, "ok", [], output)

        raise ValueError(f"Unhandled phase {phase}")


@dataclass(slots=True)
class Orchestrator:
    coordinator: DefaultSpineCoordinator

    def run(self, run_id: str) -> list[PhaseResult]:
        results: list[PhaseResult] = []
        started = _now()
        for phase in SPINE_ORDER:
            results.append(self.coordinator.run_phase(phase=phase, run_id=run_id))

        eval_id = str(self.coordinator.context.outputs[SpinePhase.EVALUATOR.value]["eval_id"])
        action = _canonical_action(self.coordinator.context.outputs[SpinePhase.JUDGE_EXIT.value]["next_action"])
        monitor_outcome = str(self.coordinator.context.outputs[SpinePhase.RUNTIME_MONITOR.value]["outcome"])
        runtime_status = str(self.coordinator.context.outputs[SpinePhase.RUNTIME_MONITOR.value]["training_outputs"].get("status", "failed"))
        run_status = "completed" if monitor_outcome == "healthy" and runtime_status == "completed" else "failed"
        run_record = RunRecord(
            run_id=run_id,
            lineage_id=self.coordinator.context.lineage_id,
            stage_name=self.coordinator.context.stage_name,
            status=run_status,
            artifact_root=self.coordinator.context.artifact_root,
            started_at=started,
            completed_at=_now(),
            phase_order=[phase.value for phase in SPINE_ORDER],
            monitor_outcome=monitor_outcome,
            eval_report_id=eval_id,
            judge_action=action,
        )
        self.coordinator.run_store.append(run_record.to_dict())

        lineage = self.coordinator.lineage_store.get_current()
        best_checkpoint_ref = self.coordinator.context.outputs[SpinePhase.EVALUATOR.value]["checkpoint_resolution"]["selected_checkpoint_ref"]
        updated = LineageState(
            lineage_id=self.coordinator.context.lineage_id,
            parent_lineage_id=(lineage or {}).get("parent_lineage_id"),
            status="active",
            stage_name=self.coordinator.context.stage_name,
            latest_run_id=run_id,
            best_checkpoint_ref=best_checkpoint_ref,
            last_decision_id=f"dec-{run_id}-exit",
            run_count=int((lineage or {}).get("run_count", 0)) + 1,
            artifact_refs=[best_checkpoint_ref],
            updated_at=_now(),
        )
        self.coordinator.lineage_store.set_current(updated.to_dict())

        ReporterRole().run(run_id, action, monitor_outcome)
        return results


def build_orchestrator(
    state_root: Path,
    run_id: str,
    lineage_id: str = "lineage-main",
    stage_name: str = "early_pretraining",
    backend: ExecutionBackend | None = None,
    runtime_policy: RuntimePolicy | None = None,
    stage_policy: StagePolicy | None = None,
    judge_policy: JudgePolicy | None = None,
) -> Orchestrator:
    context = ControlContext(run_id=run_id, lineage_id=lineage_id, stage_name=stage_name, artifact_root=f"artifacts/{run_id}")
    return Orchestrator(
        coordinator=DefaultSpineCoordinator(
            context=context,
            lineage_store=LineageStore(state_root),
            run_store=RunStore(state_root),
            decision_store=DecisionStore(state_root),
            manifest_store=ManifestStore(state_root),
            report_store=ReportStore(state_root),
            artifact_index=ArtifactIndex(state_root),
            backend=backend or DryRunBackend(),
            runtime_policy=runtime_policy or RuntimePolicy(),
            stage_policy=stage_policy or StagePolicy(),
            judge_policy=judge_policy or JudgePolicy(),
        )
    )
