"""Orchestrator that preserves explicit stage ordering and role boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from hephaestus.backends.base import ExecutionBackend
from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.control.branching import create_branch_state
from hephaestus.control.lineage_transition import compute_lineage_signals
from hephaestus.control.restart import create_restart_state
from hephaestus.control.rollback import apply_rollback
from hephaestus.control.spine import SPINE_ORDER, PhaseResult, SpineCoordinator, SpinePhase
from hephaestus.policy.judge_policy import JudgePolicy
from hephaestus.policy.promotion_policy import PromotionPolicy
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
from hephaestus.state.query import Query
from hephaestus.state.report_store import ReportStore
from hephaestus.state.run_store import RunStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_action(value: object) -> str:
    if hasattr(value, "value"):
        return str(getattr(value, "value"))
    return str(value)


@dataclass(slots=True)
class ControlContext:
    run_id: str
    lineage_id: str
    stage_name: str
    artifact_root: str
    outputs: dict[str, object] = field(default_factory=dict)


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
    promotion_policy: PromotionPolicy
    query: Query

    def run_phase(self, phase: SpinePhase, run_id: str) -> PhaseResult:
        lineage_state = self.lineage_store.get_current(self.context.lineage_id)
        recent_failures = self.query.recent_failures(self.context.lineage_id)

        if phase is SpinePhase.JUDGE_ENTRY:
            entry, decision = JudgeEntryRole(self.judge_policy).run(
                run_id=run_id,
                lineage_id=self.context.lineage_id,
                stage_name=self.context.stage_name,
                created_at=_now(),
                lineage_state=lineage_state,
                recent_failures=recent_failures,
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
            stage_profile = self.stage_policy.resolve(self.context.stage_name)
            monitor = RuntimeMonitorRole(self.backend, self.runtime_policy).run(
                run_id=run_id,
                experiment_plan=planner,
                training_plan=training["training_plan"],
                launch_config=training["launch_config"],
                data_contract=data_contract,
                stage_profile=stage_profile,
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
            judge = JudgeExitRole(self.judge_policy, self.promotion_policy).run(
                run_id,
                self.context.lineage_id,
                eval_report=EvalReport.from_dict(eval_report),
                monitor_outcome=monitor_outcome,
                recent_failure_count=len(recent_failures),
                has_stable_checkpoint=bool((lineage_state or {}).get("last_stable_checkpoint_ref")),
                stage_profile=self.stage_policy.resolve(self.context.stage_name),
            )
            output = judge.to_dict()
            self.context.outputs[phase.value] = output
            decision = DecisionRecord(
                f"dec-{run_id}-exit",
                run_id,
                self.context.lineage_id,
                "judge_exit",
                judge.next_action.value,
                "; ".join(judge.reasons),
                judge.confidence,
                created_at=_now(),
                metadata={
                    "monitor_outcome": monitor_outcome,
                    "recent_failure_count": len(recent_failures),
                },
            )
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

        eval_output = self.coordinator.context.outputs[SpinePhase.EVALUATOR.value]
        eval_id = str(eval_output["eval_id"])
        checkpoint_ref = str(eval_output["checkpoint_resolution"].get("selected_checkpoint_ref", "")) or None
        action = _canonical_action(self.coordinator.context.outputs[SpinePhase.JUDGE_EXIT.value]["next_action"])
        monitor_outcome = str(self.coordinator.context.outputs[SpinePhase.RUNTIME_MONITOR.value]["outcome"])
        runtime_status = str(self.coordinator.context.outputs[SpinePhase.RUNTIME_MONITOR.value]["training_outputs"].get("status", "failed"))
        run_status = "completed" if monitor_outcome == "healthy" and runtime_status == "completed" else "failed"

        prior = self.coordinator.lineage_store.get_current(self.coordinator.context.lineage_id) or {}
        loop_index = int(prior.get("loop_index", 0)) + 1

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
            loop_index=loop_index,
            checkpoint_ref=checkpoint_ref,
        )
        self.coordinator.run_store.append(run_record.to_dict())

        self._apply_lineage_transition(
            run_id=run_id,
            action=action,
            run_status=run_status,
            checkpoint_ref=checkpoint_ref,
            eval_output=eval_output,
            prior_state=prior,
            loop_index=loop_index,
        )

        ReporterRole().run(run_id, action, monitor_outcome)
        return results

    def _apply_lineage_transition(
        self,
        run_id: str,
        action: str,
        run_status: str,
        checkpoint_ref: str | None,
        eval_output: dict[str, object],
        prior_state: dict[str, object],
        loop_index: int,
    ) -> None:
        deterministic_passed = bool(eval_output["regression_summary"].get("deterministic_passed", False))
        confidence = float(eval_output.get("confidence", 0.0))
        signal_update = compute_lineage_signals(
            prior_state=prior_state,
            run_id=run_id,
            run_status=run_status,
            action=action,
            checkpoint_ref=checkpoint_ref,
            deterministic_passed=deterministic_passed,
            confidence=confidence,
            promotion_policy=self.coordinator.promotion_policy,
        )

        state = LineageState(
            lineage_id=self.coordinator.context.lineage_id,
            parent_lineage_id=prior_state.get("parent_lineage_id") if prior_state else None,
            stage_name=self.coordinator.context.stage_name,
            status=signal_update.promotion.status,
            trust_level=signal_update.trust_level,
            loop_index=loop_index,
            latest_run_id=run_id,
            best_checkpoint_ref=signal_update.promotion.best_checkpoint_ref,
            last_stable_checkpoint_ref=signal_update.promotion.last_stable_checkpoint_ref,
            recent_failures=signal_update.failures,
            known_pathologies=signal_update.known_pathologies,
            last_decision=action,
            last_decision_id=f"dec-{run_id}-exit",
            branch_origin_checkpoint_ref=prior_state.get("branch_origin_checkpoint_ref") if prior_state else None,
            child_lineage_ids=list(prior_state.get("child_lineage_ids", [])),
            run_count=int(prior_state.get("run_count", 0)) + 1,
            updated_at=_now(),
        )

        if action == "rollback_to_checkpoint":
            rollback = apply_rollback(state.to_dict())
            if rollback.succeeded:
                state.best_checkpoint_ref = rollback.target_checkpoint_ref
            else:
                state.status = "blocked"
                state.known_pathologies = [*state.known_pathologies, *rollback.notes][-5:]

        if action == "branch_new_experiment":
            child_id = f"{state.lineage_id}-branch-{state.loop_index}"
            branch = create_branch_state(
                parent_state=state.to_dict(),
                child_lineage_id=child_id,
                stage_name=self.coordinator.context.stage_name,
                origin_checkpoint_ref=checkpoint_ref or state.best_checkpoint_ref,
                updated_at=_now(),
            )
            self.coordinator.lineage_store.set_current(branch.child_state)
            self.coordinator.lineage_store.add_child(state.lineage_id, child_id)
            if child_id not in state.child_lineage_ids:
                state.child_lineage_ids = [*state.child_lineage_ids, child_id]

        if action == "restart_lineage":
            restart = create_restart_state(
                prior_state=state.to_dict(),
                lineage_id=state.lineage_id,
                stage_name=self.coordinator.context.stage_name,
                updated_at=_now(),
                reason="explicit_restart_action",
            )
            self.coordinator.lineage_store.set_current(restart.reset_state)
            return

        self.coordinator.lineage_store.set_current(state.to_dict())


def build_orchestrator(
    state_root: Path,
    run_id: str,
    lineage_id: str = "lineage-main",
    stage_name: str = "early_pretraining",
    backend: ExecutionBackend | None = None,
    runtime_policy: RuntimePolicy | None = None,
    stage_policy: StagePolicy | None = None,
    judge_policy: JudgePolicy | None = None,
    promotion_policy: PromotionPolicy | None = None,
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
            promotion_policy=promotion_policy or PromotionPolicy(),
            query=Query(state_root),
        )
    )
