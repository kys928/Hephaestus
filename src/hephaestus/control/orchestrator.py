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
from hephaestus.policy.action_registry import evaluate_action_boundary
from hephaestus.policy.approval_policy import ApprovalPolicy
from hephaestus.policy.judge_policy import JudgePolicy
from hephaestus.policy.promotion_gates import evaluate_promotion_gates
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
from hephaestus.safety.checkpoint_guard import check_checkpoint_candidates, check_selected_checkpoint
from hephaestus.safety.dataset_guard import check_dataset_manifest, check_trainable_data_contract
from hephaestus.safety.eval_guard import check_eval_report
from hephaestus.safety.launch_guard import check_launch_request
from hephaestus.safety.policy import decide_boundary
from hephaestus.safety.role_boundary_guard import check_phase_order
from hephaestus.safety.state_guard import check_lineage_state
from hephaestus.schemas.safety_guard import SafetyGuardInput
from hephaestus.schemas.approval_decision import ApprovalDecision
from hephaestus.schemas.approval_request import ApprovalRequest
from hephaestus.schemas.decision_record import DecisionRecord
from hephaestus.schemas.eval_report import EvalReport
from hephaestus.schemas.lineage_state import LineageState
from hephaestus.schemas.replay_metadata import build_replay_metadata
from hephaestus.schemas.run_record import RunRecord
from hephaestus.state.artifact_index import ArtifactIndex
from hephaestus.state.decision_store import DecisionStore
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.manifest_store import ManifestStore
from hephaestus.state.memory_builder import build_memory_records_from_run
from hephaestus.state.memory_store import MemoryStore
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
    memory_store: MemoryStore
    artifact_index: ArtifactIndex
    backend: ExecutionBackend
    runtime_policy: RuntimePolicy
    stage_policy: StagePolicy
    judge_policy: JudgePolicy
    promotion_policy: PromotionPolicy
    approval_policy: ApprovalPolicy
    query: Query
    operator_responses: dict[str, dict[str, str]] = field(default_factory=dict)

    def run_phase(self, phase: SpinePhase, run_id: str) -> PhaseResult:
        lineage_state = self.lineage_store.get_current(self.context.lineage_id)
        recent_failures = self.query.recent_failures(self.context.lineage_id)
        recent_repeatability = self.query.checkpoint_repeatability_summary(self.context.lineage_id)
        relevant_memories = [
            *self.query.dead_ends_for_lineage(self.context.lineage_id, limit=5),
            *self.query.prior_promotion_blocks(self.context.lineage_id, limit=5),
            *self.query.data_issues_for_lineage(self.context.lineage_id, limit=5),
            *self.query.eval_issues_for_lineage(self.context.lineage_id, limit=5),
        ]
        relevant_memories = relevant_memories[-10:]

        if phase is SpinePhase.JUDGE_ENTRY:
            entry, decision = JudgeEntryRole(self.judge_policy).run(
                run_id=run_id,
                lineage_id=self.context.lineage_id,
                stage_name=self.context.stage_name,
                created_at=_now(),
                lineage_state=lineage_state,
                recent_failures=recent_failures,
                recent_repeatability=recent_repeatability,
                relevant_memories=relevant_memories,
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
            profile, manifest = DataAcquisitionAuditRole(self.backend).run(
                run_id,
                self.context.lineage_id,
                stage_name=self.context.stage_name,
            )
            output = {"dataset_profile": profile.to_dict(), "dataset_manifest": manifest.to_dict()}
            guard_decision = decide_boundary(
                run_id=run_id,
                lineage_id=self.context.lineage_id,
                boundary="data_acquisition_audit",
                guard_results=[
                    check_dataset_manifest(
                        SafetyGuardInput(
                            guard_id="dataset_manifest_guard",
                            run_id=run_id,
                            lineage_id=self.context.lineage_id,
                            boundary="data_acquisition_audit",
                            payload=output["dataset_manifest"],
                        )
                    )
                ],
            )
            output["safety"] = guard_decision.to_dict()
            if not guard_decision.allowed:
                self.report_store.append({"kind": "safety_guard", **guard_decision.to_dict()})
                return PhaseResult(phase, "blocked", [], output)
            self.context.outputs[phase.value] = output
            self.manifest_store.append(manifest.to_dict())
            self.report_store.append({"kind": "dataset_profile", **profile.to_dict()})
            manifest_ref = str(manifest.artifact_ref or "")
            if manifest_ref:
                self.artifact_index.append({"run_id": run_id, "kind": "dataset_manifest", "ref": manifest_ref})
            return PhaseResult(phase, "ok", [manifest_ref] if manifest_ref else [], output)

        if phase is SpinePhase.DATA_PREPROCESSOR:
            manifest_id = str(self.context.outputs[SpinePhase.DATA_ACQUISITION_AUDIT.value]["dataset_manifest"]["manifest_id"])
            report, contract = DataPreprocessorRole(self.backend).run(run_id, manifest_id)
            output = {"preprocessing_report": report.to_dict(), "trainable_data_contract": contract.to_dict()}
            guard_decision = decide_boundary(
                run_id=run_id,
                lineage_id=self.context.lineage_id,
                boundary="data_preprocessor",
                guard_results=[
                    check_trainable_data_contract(
                        SafetyGuardInput(
                            guard_id="trainable_data_contract_guard",
                            run_id=run_id,
                            lineage_id=self.context.lineage_id,
                            boundary="data_preprocessor",
                            payload=output["trainable_data_contract"],
                        )
                    )
                ],
            )
            output["safety"] = guard_decision.to_dict()
            if not guard_decision.allowed:
                self.report_store.append({"kind": "safety_guard", **guard_decision.to_dict()})
                return PhaseResult(phase, "blocked", [], output)
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
            guard_decision = decide_boundary(
                run_id=run_id,
                lineage_id=self.context.lineage_id,
                boundary="training_engineer_launch",
                guard_results=[
                    check_launch_request(
                        SafetyGuardInput(
                            guard_id="launch_guard",
                            run_id=run_id,
                            lineage_id=self.context.lineage_id,
                            boundary="training_engineer_launch",
                            payload={**output, "data_contract": data_contract},
                        )
                    )
                ],
            )
            output["safety"] = guard_decision.to_dict()
            if not guard_decision.allowed:
                self.report_store.append({"kind": "safety_guard", **guard_decision.to_dict()})
                return PhaseResult(phase, "blocked", [], output)
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
            guard_decision = decide_boundary(
                run_id=run_id,
                lineage_id=self.context.lineage_id,
                boundary="runtime_monitor_checkpoint_candidates",
                guard_results=[
                    check_checkpoint_candidates(
                        SafetyGuardInput(
                            guard_id="checkpoint_candidate_guard",
                            run_id=run_id,
                            lineage_id=self.context.lineage_id,
                            boundary="runtime_monitor_checkpoint_candidates",
                            payload=monitor.training_outputs,
                        )
                    )
                ],
            )
            output["safety"] = guard_decision.to_dict()
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
            guard_decision = decide_boundary(
                run_id=run_id,
                lineage_id=self.context.lineage_id,
                boundary="evaluation_promotion_boundary",
                guard_results=[
                    check_eval_report(
                        SafetyGuardInput(
                            guard_id="eval_guard",
                            run_id=run_id,
                            lineage_id=self.context.lineage_id,
                            boundary="evaluation_promotion_boundary",
                            payload=output,
                        )
                    ),
                    check_selected_checkpoint(
                        SafetyGuardInput(
                            guard_id="selected_checkpoint_guard",
                            run_id=run_id,
                            lineage_id=self.context.lineage_id,
                            boundary="evaluation_promotion_boundary",
                            payload=output,
                        )
                    ),
                ],
            )
            output["safety"] = guard_decision.to_dict()
            self.context.outputs[phase.value] = output
            self.report_store.append({"kind": "eval_report", **output})
            for ref in report.intermediate_artifact_refs:
                self.artifact_index.append({"run_id": run_id, "kind": "intermediate_eval", "ref": ref})
            self.artifact_index.append({"run_id": run_id, "kind": "checkpoint", "ref": report.checkpoint_resolution["selected_checkpoint_ref"]})
            return PhaseResult(phase, "ok", report.intermediate_artifact_refs, output)

        if phase is SpinePhase.JUDGE_EXIT:
            eval_report = self.context.outputs[SpinePhase.EVALUATOR.value]
            monitor_outcome = str(self.context.outputs[SpinePhase.RUNTIME_MONITOR.value]["outcome"])
            candidate_ref = str(eval_report.get("checkpoint_resolution", {}).get("selected_checkpoint_ref", ""))
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
            requested_action = judge.next_action.value
            effective_action = requested_action
            prior_trust = str((lineage_state or {}).get("trust_level", "unknown"))
            boundary_context: dict[str, object] = {}
            boundary = evaluate_action_boundary(requested_action, context=boundary_context)

            gate = self.approval_policy.decide(
                action=requested_action,
                stage_name=self.context.stage_name,
                trust_level=prior_trust,
            )
            if bool(boundary.get("requires_approval", False)) and gate.outcome == "auto_allowed":
                gate = self.approval_policy.decide(
                    action=requested_action,
                    stage_name=self.context.stage_name,
                    trust_level=prior_trust,
                )
                gate.outcome = "approval_required_high_risk" if bool(boundary.get("high_risk", False)) else "approval_required"
                gate.risk_level = "high" if bool(boundary.get("high_risk", False)) else "moderate"
                gate.required_approval_type = "operator_high_risk_approval" if bool(boundary.get("high_risk", False)) else "operator_approval"
                gate.reason = f"action_registry:{requested_action}:{gate.outcome}"

            governance: dict[str, object] = {
                "approval_outcome": gate.outcome,
                "approval_risk_level": gate.risk_level,
                "required_approval_type": gate.required_approval_type,
                "requested_action": requested_action,
                "effective_action": effective_action,
                "approval_request_id": None,
                "approval_status": "not_required" if gate.outcome == "auto_allowed" else "pending",
                "operator_override": False,
                "action_boundary": boundary,
                "action_category": boundary.get("category", "unknown"),
                "action_forbidden": bool(boundary.get("forbidden", False)),
                "action_requires_approval": bool(boundary.get("requires_approval", False)),
                "action_high_risk": bool(boundary.get("high_risk", False)),
            }

            if not bool(boundary.get("known_action", False)) or bool(boundary.get("forbidden", False)):
                effective_action = "continue_lineage_best"
                governance["approval_status"] = "rejected"
                governance["approval_outcome"] = "blocked_unknown_or_forbidden_action"
                governance["required_approval_type"] = "none"
                governance["approval_risk_level"] = "high" if bool(boundary.get("forbidden", False)) else "moderate"
                governance["effective_action"] = effective_action

            if gate.outcome in {"approval_required", "approval_required_high_risk", "override_not_allowed"} and governance["approval_outcome"] != "blocked_unknown_or_forbidden_action":
                request_id = f"apr-{run_id}-{requested_action}"
                request = ApprovalRequest(
                    request_id=request_id,
                    decision_id=f"dec-{run_id}-exit",
                    lineage_id=self.context.lineage_id,
                    run_id=run_id,
                    proposed_action=requested_action,
                    reason=gate.reason,
                    risk_level=gate.risk_level,
                    required_approval_type=gate.required_approval_type,
                    status="pending",
                    created_at=_now(),
                    metadata={"checkpoint_ref": candidate_ref},
                )
                self.decision_store.append_approval_request(request.to_dict())
                governance["approval_request_id"] = request_id

                response = self.operator_responses.get(request_id) or self.operator_responses.get(requested_action)
                if response:
                    note = str(response.get("note", ""))
                    operator_id = str(response.get("operator_id", "operator.unknown"))
                    resolution = self.approval_policy.resolve_operator_outcome(
                        str(response.get("outcome", "rejected")),
                        override_allowed=gate.outcome != "override_not_allowed",
                    )
                    decision = ApprovalDecision(
                        decision_event_id=f"apd-{run_id}-{requested_action}",
                        request_id=request_id,
                        lineage_id=self.context.lineage_id,
                        run_id=run_id,
                        operator_id=operator_id,
                        outcome=resolution.outcome,
                        status=resolution.status,
                        note=note,
                        effect_on_action=resolution.effect_on_action,
                        created_at=_now(),
                        metadata={
                            "proposed_action": requested_action,
                            "checkpoint_ref": candidate_ref,
                            "resolution_reason": resolution.reason,
                            "override_blocked": resolution.override_blocked,
                        },
                    )
                    self.decision_store.append_approval_decision(decision.to_dict())
                    governance["approval_status"] = resolution.status
                    governance["operator_override"] = resolution.is_override and not resolution.override_blocked
                    governance["override_blocked"] = resolution.override_blocked
                    if resolution.status != "approved":
                        effective_action = "continue_lineage_best"
                else:
                    effective_action = "continue_lineage_best"
                governance["effective_action"] = effective_action

            data_manifest_rows = self.manifest_store.list_for_run(run_id)
            data_manifest = data_manifest_rows[-1] if data_manifest_rows else None
            gate_report = evaluate_promotion_gates(
                run_id=run_id,
                lineage_id=self.context.lineage_id,
                requested_action=requested_action,
                eval_report=EvalReport.from_dict(eval_report),
                lineage_state=lineage_state,
                data_manifest=data_manifest,
                approval_metadata=governance,
            )
            if effective_action == requested_action:
                effective_action = gate_report.recommended_effective_action
            governance["effective_action"] = effective_action
            governance["promotion_gate_report"] = gate_report.to_dict()
            governance["blocking_failures"] = list(gate_report.blocking_failures)
            governance["gate_warnings"] = list(gate_report.warnings)
            governance["promotion_allowed"] = gate_report.promotion_allowed
            governance["rollback_allowed"] = gate_report.rollback_allowed
            governance["branch_allowed"] = gate_report.branch_allowed
            governance["restart_allowed"] = gate_report.restart_allowed
            governance["reject_allowed"] = gate_report.reject_allowed
            governance["confidence_ceiling"] = gate_report.confidence_ceiling
            governance["recommended_effective_action"] = gate_report.recommended_effective_action

            self.context.outputs[phase.value] = output
            decision = DecisionRecord(
                f"dec-{run_id}-exit",
                run_id,
                self.context.lineage_id,
                "judge_exit",
                requested_action,
                "; ".join(judge.reasons),
                judge.confidence,
                created_at=_now(),
                metadata={
                    "monitor_outcome": monitor_outcome,
                    "recent_failure_count": len(recent_failures),
                    "checkpoint_ref": candidate_ref,
                    "certification_state": next((item.split("=", 1)[1] for item in judge.reasons if item.startswith("certification_state=")), "certification_not_eligible"),
                    "repeatability_sufficient": bool(eval_report.get("repeatability_sufficient", False)),
                    "variance_risk": str(eval_report.get("variance_risk", "unknown")),
                    "repeated_eval_count": int(eval_report.get("repeated_eval_count", 0)),
                    "consistency_score": float(eval_report.get("consistency_score", 0.0)),
                    **governance,
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
            if results[-1].status == "blocked":
                return results

        spine_guard = decide_boundary(
            run_id=run_id,
            lineage_id=self.coordinator.context.lineage_id,
            boundary="orchestrator_phase_order",
            guard_results=[
                check_phase_order(
                    SafetyGuardInput(
                        guard_id="role_boundary_guard",
                        run_id=run_id,
                        lineage_id=self.coordinator.context.lineage_id,
                        boundary="orchestrator_phase_order",
                        payload={"phase_order": [item.phase.value for item in results]},
                    )
                )
            ],
        )
        if not spine_guard.allowed:
            self.coordinator.report_store.append({"kind": "safety_guard", **spine_guard.to_dict()})
            return results

        eval_output = self.coordinator.context.outputs[SpinePhase.EVALUATOR.value]
        eval_id = str(eval_output["eval_id"])
        checkpoint_ref = str(eval_output["checkpoint_resolution"].get("selected_checkpoint_ref", "")) or None
        replay_metadata = build_replay_metadata(
            checkpoint_ref=checkpoint_ref,
            checkpoint_evidence=dict(eval_output.get("checkpoint_resolution", {})),
        )
        judge_output = self.coordinator.context.outputs[SpinePhase.JUDGE_EXIT.value]
        action = _canonical_action(judge_output["next_action"])
        judge_decision = self.coordinator.decision_store.get(f"dec-{run_id}-exit") or {}
        governance = dict(judge_decision.get("metadata", {}))
        effective_action = str(governance.get("effective_action", action))
        monitor_outcome = str(self.coordinator.context.outputs[SpinePhase.RUNTIME_MONITOR.value]["outcome"])
        runtime_status = str(self.coordinator.context.outputs[SpinePhase.RUNTIME_MONITOR.value]["training_outputs"].get("status", "failed"))
        run_status = "completed" if monitor_outcome == "healthy" and runtime_status == "completed" else "failed"

        prior = self.coordinator.lineage_store.get_current(self.coordinator.context.lineage_id) or {}
        loop_index = int(prior.get("loop_index", 0)) + 1

        data_manifest = self.coordinator.manifest_store.list_for_run(run_id)
        latest_data_manifest = data_manifest[-1] if data_manifest else {}
        data_manifest_id = str(latest_data_manifest.get("manifest_id", "")) or None

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
            data_manifest_id=data_manifest_id,
            replay_metadata=replay_metadata,
        )
        self.coordinator.run_store.append(run_record.to_dict())

        self._apply_lineage_transition(
            run_id=run_id,
            requested_action=action,
            effective_action=effective_action,
            run_status=run_status,
            checkpoint_ref=checkpoint_ref,
            eval_output=eval_output,
            prior_state=prior,
            loop_index=loop_index,
            governance=governance,
        )
        self._build_and_persist_memory(run_id=run_id, run_record=run_record.to_dict())

        ReporterRole().run(run_id, action, monitor_outcome)
        return results

    def _build_and_persist_memory(self, run_id: str, run_record: dict[str, object]) -> None:
        try:
            lineage = self.coordinator.lineage_store.get_current(self.coordinator.context.lineage_id)
            decisions = [row for row in self.coordinator.decision_store.all() if str(row.get("run_id")) == run_id]
            reports = [row for row in self.coordinator.report_store.all() if str(row.get("run_id")) == run_id]
            manifests = self.coordinator.manifest_store.list_for_run(run_id)
            records = build_memory_records_from_run(
                run_id=run_id,
                run_record=run_record,
                lineage_state=lineage,
                decisions=decisions,
                reports=reports,
                manifests=manifests,
            )
            for record in records:
                self.coordinator.memory_store.append(record)
        except Exception as exc:  # pragma: no cover - protective non-fatal guard
            self.coordinator.report_store.append(
                {
                    "kind": "warning",
                    "run_id": run_id,
                    "warning_type": "memory_builder_failed",
                    "warning": str(exc),
                }
            )

    def _apply_lineage_transition(
        self,
        run_id: str,
        requested_action: str,
        effective_action: str,
        run_status: str,
        checkpoint_ref: str | None,
        eval_output: dict[str, object],
        prior_state: dict[str, object],
        loop_index: int,
        governance: dict[str, object],
    ) -> None:
        action = effective_action
        deterministic_passed = bool(eval_output["regression_summary"].get("deterministic_passed", False))
        confidence = float(eval_output.get("confidence", 0.0))
        blocked = bool(governance.get("approval_status") in {"pending", "rejected", "expired", "superseded"})
        promotion_allowed = bool(governance.get("promotion_allowed", True))
        signal_update = compute_lineage_signals(
            prior_state=prior_state,
            run_id=run_id,
            run_status=run_status,
            action=action,
            checkpoint_ref=checkpoint_ref,
            deterministic_passed=deterministic_passed,
            confidence=confidence,
            promotion_bundle_passed=bool(eval_output.get("promotion_bundle_passed", False)),
            evidence_completeness=float(eval_output.get("evidence_completeness", 0.0)),
            certification_readiness=str(eval_output.get("certification_readiness", "certification_not_eligible")),
            recheck_recommended=bool(eval_output.get("recheck_recommended", False)),
            stage_certification_eligibility=str(eval_output.get("evaluation_bundle_summary", {}).get("stage_certification_profile", {}).get("eligibility", "standard")),
            stage_require_recheck=bool(eval_output.get("evaluation_bundle_summary", {}).get("stage_certification_profile", {}).get("require_recheck", False)),
            stage_min_consistent_runs=int(eval_output.get("evaluation_bundle_summary", {}).get("effective_min_consistent_runs", 1)),
            observed_consistent_runs=int(eval_output.get("observed_consistent_runs", 0)),
            min_promotion_evidence=int(eval_output.get("evaluation_bundle_summary", {}).get("required_evidence", {}).get("promotion_runs", 1)),
            observed_evidence_runs=int(eval_output.get("evaluation_bundle_summary", {}).get("observed_evidence_runs", 1)),
            min_stable_evidence=int(eval_output.get("evaluation_bundle_summary", {}).get("required_evidence", {}).get("stable_runs", 1)),
            min_certification_evidence=int(eval_output.get("evaluation_bundle_summary", {}).get("required_evidence", {}).get("certification_runs", 2)),
            stability_confidence=float(eval_output.get("stability_confidence", 0.0)),
            min_stability_confidence=float(eval_output.get("evaluation_bundle_summary", {}).get("min_stability_confidence", 0.0)),
            stage_thresholds=dict(eval_output.get("evaluation_bundle_summary", {}).get("stage_thresholds", {})),
            promotion_policy=self.coordinator.promotion_policy,
            repeatability_sufficient=bool(eval_output.get("repeatability_sufficient", False)),
            variance_risk=str(eval_output.get("variance_risk", "unknown")),
        )
        if blocked:
            signal_update.promotion.best_checkpoint_ref = prior_state.get("best_checkpoint_ref")
            signal_update.promotion.last_stable_checkpoint_ref = prior_state.get("last_stable_checkpoint_ref")
            signal_update.promotion.certified_stable_checkpoint_ref = prior_state.get("certified_stable_checkpoint_ref")
            signal_update.promotion.last_certification_result = str(
                prior_state.get("last_certification_result", signal_update.promotion.last_certification_result)
            )
            signal_update.promotion.status = str(prior_state.get("status", signal_update.promotion.status))
            signal_update.trust_level = str(prior_state.get("trust_level", signal_update.trust_level))
            signal_update.failures = list(prior_state.get("recent_failures", []))
            signal_update.known_pathologies = list(prior_state.get("known_pathologies", []))
        if not promotion_allowed:
            signal_update.promotion.best_checkpoint_ref = prior_state.get("best_checkpoint_ref")
            signal_update.promotion.last_stable_checkpoint_ref = prior_state.get("last_stable_checkpoint_ref")
            signal_update.promotion.certified_stable_checkpoint_ref = prior_state.get("certified_stable_checkpoint_ref")

        state = LineageState(
            lineage_id=self.coordinator.context.lineage_id,
            parent_lineage_id=prior_state.get("parent_lineage_id") if prior_state else None,
            child_lineage_ids=list(prior_state.get("child_lineage_ids", [])),
            stage_name=self.coordinator.context.stage_name,
            status=signal_update.promotion.status,
            trust_level=signal_update.trust_level,
            origin_run_id=str(prior_state.get("origin_run_id") or run_id),
            origin_checkpoint_ref=prior_state.get("origin_checkpoint_ref") or checkpoint_ref,
            branch_origin_checkpoint_ref=prior_state.get("branch_origin_checkpoint_ref") if prior_state else None,
            created_at=str(prior_state.get("created_at") or _now()),
            updated_at=_now(),
            architecture_contract_ref=prior_state.get("architecture_contract_ref"),
            tokenizer_contract_ref=prior_state.get("tokenizer_contract_ref"),
            data_policy_ref=(
                str(
                    self.coordinator.context.outputs.get(SpinePhase.DATA_ACQUISITION_AUDIT.value, {})
                    .get("dataset_manifest", {})
                    .get("stage_data_policy_ref", "")
                )
                or prior_state.get("data_policy_ref")
            ),
            training_recipe_ref=prior_state.get("training_recipe_ref"),
            eval_policy_ref=prior_state.get("eval_policy_ref"),
            loop_index=loop_index,
            run_count=int(prior_state.get("run_count", 0)) + 1,
            latest_run_id=run_id,
            best_checkpoint_ref=signal_update.promotion.best_checkpoint_ref,
            last_stable_checkpoint_ref=signal_update.promotion.last_stable_checkpoint_ref,
            certified_stable_checkpoint_ref=signal_update.promotion.certified_stable_checkpoint_ref,
            last_certification_result=signal_update.promotion.last_certification_result,
            last_repeated_eval_count=int(eval_output.get("repeated_eval_count", 0)),
            last_consistency_score=float(eval_output.get("consistency_score", 0.0)),
            last_variance_risk=str(eval_output.get("variance_risk", "unknown")),
            certification_recheck_count=int(eval_output.get("certification_recheck_count", 0)),
            repeatability_sufficient=bool(eval_output.get("repeatability_sufficient", False)),
            recent_failures=signal_update.failures,
            known_pathologies=signal_update.known_pathologies,
            last_decision=action,
            last_decision_id=f"dec-{run_id}-exit",
            last_requested_action=requested_action,
            last_effective_action=action,
            last_approval_status=str(governance.get("approval_status", "not_required")),
            pending_approval=bool(governance.get("approval_status") == "pending"),
            last_high_impact_request_id=str(governance.get("approval_request_id") or "") or None,
            major_interventions=[dict(item) for item in prior_state.get("major_interventions", []) if isinstance(item, dict)],
            metadata=dict(prior_state.get("metadata", {})),
        )

        if action == "rollback_to_checkpoint":
            rollback = apply_rollback(state.to_dict())
            state.major_interventions = [
                *state.major_interventions,
                {
                    "type": "rollback",
                    "run_id": run_id,
                    "target_checkpoint_ref": rollback.target_checkpoint_ref,
                    "created_at": _now(),
                },
            ][-20:]
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

        state_payload = state.to_dict()
        state_guard = decide_boundary(
            run_id=run_id,
            lineage_id=self.coordinator.context.lineage_id,
            boundary="lineage_state_persistence",
            guard_results=[
                check_lineage_state(
                    SafetyGuardInput(
                        guard_id="state_guard",
                        run_id=run_id,
                        lineage_id=self.coordinator.context.lineage_id,
                        boundary="lineage_state_persistence",
                        payload=state_payload,
                    )
                )
            ],
        )
        state_payload["metadata"] = {**dict(state_payload.get("metadata", {})), "last_safety_guard": state_guard.to_dict()}
        if not state_guard.allowed:
            state_payload["status"] = "blocked"
        self.coordinator.lineage_store.set_current(state_payload)


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
    approval_policy: ApprovalPolicy | None = None,
    operator_responses: dict[str, dict[str, str]] | None = None,
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
            memory_store=MemoryStore(state_root),
            artifact_index=ArtifactIndex(state_root),
            backend=backend or DryRunBackend(),
            runtime_policy=runtime_policy or RuntimePolicy(),
            stage_policy=stage_policy or StagePolicy(),
            judge_policy=judge_policy or JudgePolicy(),
            promotion_policy=promotion_policy or PromotionPolicy(),
            approval_policy=approval_policy or ApprovalPolicy(),
            query=Query(state_root),
            operator_responses=operator_responses or {},
        )
    )
