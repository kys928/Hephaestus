"""Persistent multi-experiment production loop.

The loop owns continuation, recovery and final action application; scientific
phase implementations remain behind a cycle-driver boundary.  A cycle is not
accepted unless evidence exists for every mandatory control-spine phase.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

from hephaestus.control.spine import SPINE_ORDER

from .composition import ProductionRuntime
from .recovery import RecoverableInfrastructureError
from .state import ProductionLoopState


@dataclass(slots=True)
class ProductionCycleResult:
    cycle_id: str
    run_id: str
    experiment_id: str
    status: str
    judge_action: str
    checkpoint_ref: str | None = None
    confidence: float = 0.0
    promotion_allowed: bool = False
    certification_state: str | None = None
    approval_status: str = "not_required"
    approval_ref: str | None = None
    comparison_ref: str | None = None
    phase_evidence: dict[str, list[str]] = field(default_factory=dict)
    evidence: dict[str, object] = field(default_factory=dict)
    next_cycle: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ProductionCycleDriver(Protocol):
    def execute_cycle(
        self,
        *,
        runtime: ProductionRuntime,
        state: ProductionLoopState,
        cycle_index: int,
    ) -> ProductionCycleResult: ...


_STOP_ACTIONS = {
    "promote_checkpoint",
    "abort_run",
    "quarantine_lineage",
    "archive_lineage",
    "mark_lineage_poisoned",
}
_CONTINUE_ACTIONS = {
    "reject_checkpoint",
    "reject_candidate",
    "continue_lineage_best",
    "continue_from_checkpoint",
    "rerun_same_config",
    "request_recheck",
    "rollback_to_checkpoint",
    "branch_new_experiment",
    "restart_lineage",
}


@dataclass(slots=True)
class ProductionLoopRunner:
    runtime: ProductionRuntime
    driver: ProductionCycleDriver
    maximum_cycles: int = 8
    stop_on_promotion: bool = True

    def run(
        self,
        *,
        program_id: str,
        lineage_id: str,
        stage_name: str,
        resume: bool = True,
    ) -> ProductionLoopState:
        store = self.runtime.loop_state
        state = store.get(program_id) if resume else None
        if state is None:
            state = ProductionLoopState(
                program_id=program_id,
                lineage_id=lineage_id,
                stage_name=stage_name,
                status="running",
            )
            store.save(state)
        elif state.status in {"completed", "stopped", "blocked"}:
            return state
        else:
            state.status = "running"
            store.save(state)

        while state.cycle_index < self.maximum_cycles:
            cycle_index = state.cycle_index + 1
            operation_id = f"{program_id}:cycle:{cycle_index}"
            store.event(program_id, "cycle_started", {"cycle_index": cycle_index, "operation_id": operation_id})

            try:
                result = self.runtime.infrastructure_recovery.run(
                    operation_id,
                    lambda _attempt: self.driver.execute_cycle(
                        runtime=self.runtime,
                        state=state,
                        cycle_index=cycle_index,
                    ),
                )
            except RecoverableInfrastructureError as exc:
                state.status = "blocked"
                state.stop_reason = f"infrastructure_recovery_exhausted:{exc.code}"
                state.recovery_attempts += 1
                store.event(program_id, "cycle_blocked", {"cycle_index": cycle_index, "failure_code": exc.code})
                store.save(state)
                return state

            self._validate_cycle(result)
            state.cycle_index = cycle_index
            state.latest_run_id = result.run_id
            state.latest_experiment_id = result.experiment_id
            state.latest_comparison_ref = result.comparison_ref
            state.latest_judge_action = result.judge_action
            state.completed_cycles.append(result.cycle_id)
            state.metadata["latest_cycle"] = result.to_dict()

            action_record = self.runtime.action_executor.apply(
                lineage_id=state.lineage_id,
                run_id=result.run_id,
                requested_action=result.judge_action,
                checkpoint_ref=result.checkpoint_ref,
                approval_status=result.approval_status,
                approval_ref=result.approval_ref,
                promotion_allowed=result.promotion_allowed,
                certification_state=result.certification_state,
                confidence=result.confidence,
            )
            state.metadata["latest_action_execution"] = action_record
            store.event(program_id, "cycle_completed", result.to_dict())

            effective = str(action_record["effective_action"])
            if effective == "promote_checkpoint" and self.stop_on_promotion:
                state.status = "completed"
                state.stop_reason = "candidate_promoted"
                store.save(state)
                return state
            if effective in _STOP_ACTIONS:
                state.status = "stopped"
                state.stop_reason = f"judge_action:{effective}"
                store.save(state)
                return state
            if effective not in _CONTINUE_ACTIONS:
                state.status = "blocked"
                state.stop_reason = f"unsupported_continuation_action:{effective}"
                store.save(state)
                return state

            if result.next_cycle is not None:
                state.metadata["next_cycle"] = dict(result.next_cycle)
            else:
                state.metadata["next_cycle"] = {
                    "source_cycle_id": result.cycle_id,
                    "prior_judge_action": effective,
                    "prior_comparison_ref": result.comparison_ref,
                    "prior_evidence": result.evidence,
                    "continuation_reason": "judge_action_requires_another_control_spine_pass",
                }
            state.status = "running"
            store.event(
                program_id,
                "continuation_planned",
                {"cycle_index": cycle_index, "effective_action": effective, "next_cycle": state.metadata["next_cycle"]},
            )
            store.save(state)

        state.status = "stopped"
        state.stop_reason = "maximum_cycles_reached"
        store.save(state)
        return state

    @staticmethod
    def _validate_cycle(result: ProductionCycleResult) -> None:
        if result.status != "completed":
            raise ValueError(f"cycle did not complete: {result.status}")
        expected = [phase.value for phase in SPINE_ORDER]
        missing = [phase for phase in expected if not result.phase_evidence.get(phase)]
        if missing:
            raise ValueError(f"cycle lacks mandatory phase evidence: {missing}")
        if not result.judge_action:
            raise ValueError("completed cycle lacks Judge exit action")
        if result.judge_action == "promote_checkpoint" and not result.promotion_allowed:
            raise ValueError("cycle requested promotion without promotion-gate evidence")
        if result.judge_action == "promote_checkpoint" and not result.checkpoint_ref:
            raise ValueError("cycle requested promotion without checkpoint reference")
