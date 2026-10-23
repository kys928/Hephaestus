"""Production-loop bridge to the real governed staged orchestrator.

No fabricated phase records, no direct model decisions and NO side effects at
the Judge's action_application stage. The production loop owns that commit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from hephaestus.control.spine import SPINE_ORDER
from hephaestus.control.staged_autonomous import build_staged_autonomous_orchestrator
from hephaestus.control.staged_state import (
    StagedAutonomousDependencies,
    StagedOperationRequest,
    StagedOperationResult,
    StagedOutputRecord,
)
from hephaestus.policy.action_registry import evaluate_action_boundary

from .composition import ProductionRuntime
from .loop import ProductionCycleResult
from .state import ProductionLoopState


class StagedActionIntentService:
    """Complete staged Judge validation; defer *all* lineage mutation to loop."""

    def execute(self, request: StagedOperationRequest) -> StagedOperationResult:
        if request.substep != "action_application":
            raise ValueError("action intent service only handles action_application")
        intent = request.prior_outputs.get("judge_exit.action_boundary", {})
        approval = request.prior_outputs.get("judge_exit.action_approval_gate", {})
        promotion = request.prior_outputs.get("judge_exit.promotion_gate", {})
        action = str(intent.get("proposed_action", ""))
        approval_status = str(approval.get("approval_status", "not_required"))
        boundary = evaluate_action_boundary(action, context={"approval_status": approval_status})
        if not boundary["allowed"]:
            return StagedOperationResult(
                status="blocked",
                blocking_issues=("action_intent_not_authorized",),
                resumable=False,
            )
        if action == "promote_checkpoint" and not bool(
            promotion.get("action_allowed", promotion.get("promotion_allowed", False))
        ):
            return StagedOperationResult(
                status="blocked",
                blocking_issues=("promotion_gate_not_authorized",),
                resumable=False,
            )
        return StagedOperationResult(
            records=(
                StagedOutputRecord(
                    "action_decision",
                    {
                        "operation_id": request.operation_id,
                        "action": action,
                        "status": "validated_intent_not_applied",
                        "approval_status": approval_status,
                    },
                ),
            ),
            metadata={"proposed_action": action, "action_applied": False},
        )


CycleDependenciesFactory = Callable[
    [ProductionRuntime, ProductionLoopState, int],
    StagedAutonomousDependencies,
]


@dataclass(slots=True)
class StagedProductionCycleDriver:
    """One real eight-phase staged cycle, restartable using persisted state.

    The caller MUST supply actual operation services. Missing services block in
    the existing orchestrator, not with artificial success. A production cycle
    cannot complete until all 45 staged substeps and replay checks complete.
    """

    dependencies_for_cycle: CycleDependenciesFactory

    def execute_cycle(
        self,
        *,
        runtime: ProductionRuntime,
        state: ProductionLoopState,
        cycle_index: int,
    ) -> ProductionCycleResult:
        if cycle_index < 1:
            raise ValueError("cycle_index must be positive")
        run_id = f"{state.program_id}-cycle-{cycle_index}"
        deps = self.dependencies_for_cycle(runtime, state, cycle_index)
        if deps.state_repository is not runtime.state_repository:
            raise ValueError("cycle state must use the production runtime's durable repository")
        if deps.services.action_executor is not None and not isinstance(
            deps.services.action_executor, StagedActionIntentService
        ):
            raise ValueError("staged action executor would duplicate the production loop action")
        deps.services.action_executor = StagedActionIntentService()

        workflow = build_staged_autonomous_orchestrator(
            workflow_id=f"workflow-{run_id}",
            run_id=run_id,
            lineage_id=state.lineage_id,
            stage_name=state.stage_name,
            dependencies=deps,
        )
        progress = workflow.run(run_id)
        evidence = {
            "workflow_id": progress.workflow_id,
            "workflow_status": progress.status,
            "current_phase": progress.current_phase,
            "current_substep": progress.current_substep,
            "blocking_issues": list(progress.blocking_issues),
        }
        if not progress.completion_marker:
            outcome = "pending" if progress.resumable else "blocked"
            return ProductionCycleResult(
                cycle_id=f"{state.program_id}:cycle:{cycle_index}",
                run_id=run_id,
                experiment_id="",
                status=outcome,
                judge_action="",
                evidence=evidence,
            )

        def output(phase: str, substep: str) -> dict[str, object]:
            key = f"{phase}.{substep}"
            step = progress.steps.get(key)
            if step is None or not step.completion_marker:
                raise ValueError(f"completed workflow missing completed step: {key}")
            return step.output

        verdict = output("judge_exit", "governed_verdict")
        boundary = output("judge_exit", "action_boundary")
        intent = output("judge_exit", "action_application")
        approval = output("judge_exit", "action_approval_gate")
        gate = output("judge_exit", "promotion_gate")
        replay = output("judge_exit", "replay_evidence")
        comparison = output("evaluator", "semantic_comparison")
        checkpoint = output("evaluator", "checkpoint_resolution")
        proposal = output("planner", "experiment_proposal")
        action = str(boundary.get("proposed_action", "")).strip()
        if not action or action != str(verdict.get("proposed_action", "")).strip():
            raise ValueError("Judge action differs from governed boundary")
        if intent.get("proposed_action") != action or intent.get("action_applied") is not False:
            raise ValueError("staged Judge attempted side effects or lost validated intent")
        if replay.get("replay_complete") is not True:
            raise ValueError("completed workflow lacks verified replay metadata")
        experiment_id = str(proposal.get("experiment_id", "")).strip()
        comparison_ref = str(comparison.get("comparison_id", "")).strip()
        if not experiment_id or not comparison_ref:
            raise ValueError("completed cycle lacks experiment/comparison identities")
        if not isinstance(verdict.get("confidence"), (int, float)):
            raise ValueError("Judge confidence is missing or malformed")
        confidence = float(verdict["confidence"])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Judge confidence outside [0, 1]")

        phase_evidence: dict[str, list[str]] = {}
        for phase in SPINE_ORDER:
            completed = [
                step.operation_id
                for step in progress.steps.values()
                if step.phase == phase.value and step.completion_marker
            ]
            if not completed:
                raise ValueError(f"completed cycle has no phase operations: {phase.value}")
            phase_evidence[phase.value] = completed
        return ProductionCycleResult(
            cycle_id=f"{state.program_id}:cycle:{cycle_index}",
            run_id=run_id,
            experiment_id=experiment_id,
            status="completed",
            judge_action=action,
            checkpoint_ref=str(checkpoint.get("checkpoint_ref") or "") or None,
            confidence=confidence,
            promotion_allowed=bool(gate.get("promotion_allowed", False)),
            certification_state=str(gate.get("certification_state") or "not_certified"),
            approval_status=str(approval.get("approval_status") or "not_required"),
            approval_ref=str(approval.get("approval_ref") or "") or None,
            comparison_ref=comparison_ref,
            phase_evidence=phase_evidence,
            evidence=evidence,
        )
