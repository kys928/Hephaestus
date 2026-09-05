"""Concrete governed action execution for production-loop Judge decisions."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from hephaestus.control.staged_state import (
    StagedOperationRequest,
    StagedOperationResult,
    StagedOutputRecord,
)
from hephaestus.policy.action_registry import canonical_action_name, evaluate_action_boundary
from hephaestus.schemas.lineage_state import LineageState
from hephaestus.state.lineage_store import LineageStore
from hephaestus.storage.base import StateRepository

from .state import ACTION_EXECUTIONS


def _stable_id(*values: object) -> str:
    raw = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "action-" + hashlib.sha256(raw).hexdigest()[:20]


@dataclass(slots=True)
class GovernedActionExecutor:
    """Apply finite Judge actions only after their own governance gate authorizes them.

    The executor mutates compact lineage truth, never checkpoint bytes. Every
    attempt is append-only and keyed by a stable execution ID so process retries
    cannot apply the same transition twice.
    """

    state_root: Path
    repository: StateRepository

    @property
    def lineage_store(self) -> LineageStore:
        return LineageStore(self.state_root)

    def apply(
        self,
        *,
        lineage_id: str,
        run_id: str,
        requested_action: str,
        checkpoint_ref: str | None = None,
        approval_status: str = "not_required",
        approval_ref: str | None = None,
        promotion_allowed: bool = False,
        rollback_allowed: bool = False,
        branch_allowed: bool = False,
        restart_allowed: bool = False,
        certification_state: str | None = None,
        confidence: float = 0.0,
        child_lineage_id: str | None = None,
    ) -> dict[str, object]:
        canonical = canonical_action_name(requested_action)
        boundary = evaluate_action_boundary(canonical, context={"approval_status": approval_status})
        if not boundary["allowed"]:
            raise PermissionError(f"action is not authorized: {canonical}: {boundary['reasons']}")

        transition_gate = {
            "promote_checkpoint": promotion_allowed,
            "rollback_to_checkpoint": rollback_allowed,
            "branch_new_experiment": branch_allowed,
            "restart_lineage": restart_allowed,
        }.get(canonical, True)
        if not transition_gate:
            raise PermissionError(f"governed transition gate did not authorize {canonical}")

        execution_id = _stable_id(lineage_id, run_id, canonical, checkpoint_ref, approval_ref)
        previous = self.repository.get_latest(ACTION_EXECUTIONS, "execution_id", execution_id)
        if previous is not None and previous.get("status") == "applied":
            return dict(previous)

        current_payload = self.lineage_store.get_current(lineage_id)
        if current_payload is None:
            current = LineageState(lineage_id=lineage_id, stage_name="smoke_test")
        else:
            current = LineageState.from_dict(current_payload)
        before = current.to_dict()
        now = datetime.now(timezone.utc).isoformat()

        current.latest_run_id = run_id
        current.last_requested_action = requested_action
        current.last_effective_action = canonical
        current.last_approval_status = approval_status
        current.pending_approval = False
        current.updated_at = now

        if canonical == "promote_checkpoint":
            if not checkpoint_ref:
                raise ValueError("promotion requires candidate checkpoint_ref")
            current.best_checkpoint_ref = checkpoint_ref
            current.status = "stable" if certification_state == "certification_passed" else "active"
            current.trust_level = "high" if confidence >= 0.85 else "medium"
            if confidence >= 0.85:
                current.last_stable_checkpoint_ref = checkpoint_ref
            if certification_state == "certification_passed":
                current.certified_stable_checkpoint_ref = checkpoint_ref
                current.last_certification_result = "certification_passed"
        elif canonical == "reject_candidate":
            current.status = "active"
            current.recent_failures = [*current.recent_failures, run_id][-5:]
            current.known_pathologies = [*current.known_pathologies, "candidate_rejected"][-5:]
        elif canonical == "rollback_to_checkpoint":
            target = checkpoint_ref or current.last_stable_checkpoint_ref or current.best_checkpoint_ref
            if not target:
                raise ValueError("rollback requires a known checkpoint")
            current.best_checkpoint_ref = target
            current.status = "active"
            current.known_pathologies = [*current.known_pathologies, "rollback_applied"][-5:]
        elif canonical == "branch_new_experiment":
            origin = checkpoint_ref or current.best_checkpoint_ref
            if not origin:
                raise ValueError("branch requires an origin checkpoint")
            child_id = child_lineage_id or f"{lineage_id}-branch-{hashlib.sha256((run_id + origin).encode()).hexdigest()[:8]}"
            if self.lineage_store.get_current(child_id) is None:
                child = LineageState(
                    lineage_id=child_id,
                    parent_lineage_id=lineage_id,
                    stage_name=current.stage_name,
                    status="exploratory",
                    trust_level="unknown",
                    origin_run_id=run_id,
                    origin_checkpoint_ref=origin,
                    branch_origin_checkpoint_ref=origin,
                    created_at=now,
                    updated_at=now,
                    architecture_contract_ref=current.architecture_contract_ref,
                    tokenizer_contract_ref=current.tokenizer_contract_ref,
                    data_policy_ref=current.data_policy_ref,
                    training_recipe_ref=current.training_recipe_ref,
                    eval_policy_ref=current.eval_policy_ref,
                )
                self.lineage_store.set_current(child.to_dict())
                self.lineage_store.add_child(lineage_id, child_id)
            current.metadata["latest_child_lineage_id"] = child_id
        elif canonical == "restart_lineage":
            current.status = "restarted"
            current.trust_level = "low"
            current.best_checkpoint_ref = checkpoint_ref
            current.last_stable_checkpoint_ref = checkpoint_ref
            current.certified_stable_checkpoint_ref = None
            current.last_certification_result = "certification_not_eligible"
        elif canonical in {"continue_lineage_best", "continue_from_checkpoint", "rerun_same_config"}:
            current.status = "active"
            if canonical == "continue_from_checkpoint" and checkpoint_ref:
                current.best_checkpoint_ref = checkpoint_ref
        elif canonical == "abort_run":
            current.status = "blocked"
        elif canonical == "quarantine_lineage":
            current.status = "poisoned"
            current.trust_level = "low"
        elif canonical == "archive_lineage":
            current.status = "archived"
        elif canonical == "mark_lineage_stable":
            current.status = "stable"
            current.trust_level = "high"
        elif canonical == "mark_lineage_poisoned":
            current.status = "poisoned"
            current.trust_level = "low"
        elif canonical in {"observe_state", "read_memory", "summarize_run", "summarize_lineage", "record_incident", "request_recheck"}:
            pass
        else:
            raise NotImplementedError(f"registered action has no production executor: {canonical}")

        self.lineage_store.set_current(current.to_dict())
        record = {
            "execution_id": execution_id,
            "status": "applied",
            "lineage_id": lineage_id,
            "run_id": run_id,
            "requested_action": requested_action,
            "effective_action": canonical,
            "checkpoint_ref": checkpoint_ref,
            "approval_status": approval_status,
            "approval_ref": approval_ref,
            "transition_gates": {
                "promotion_allowed": promotion_allowed,
                "rollback_allowed": rollback_allowed,
                "branch_allowed": branch_allowed,
                "restart_allowed": restart_allowed,
            },
            "certification_state": certification_state,
            "confidence": confidence,
            "applied_at": now,
            "before": before,
            "after": current.to_dict(),
        }
        self.repository.append(ACTION_EXECUTIONS, record)
        return record

    def execute(self, request: StagedOperationRequest) -> StagedOperationResult:
        outputs = request.prior_outputs
        boundary = outputs.get("judge_exit.action_boundary", {})
        approval = outputs.get("judge_exit.action_approval_gate", {})
        promotion = outputs.get("judge_exit.promotion_gate", {})
        verdict = outputs.get("judge_exit.governed_verdict", {})
        action = str(boundary.get("proposed_action") or verdict.get("proposed_action") or "")
        checkpoint_ref = str(verdict.get("checkpoint_ref") or promotion.get("checkpoint_ref") or "") or None
        try:
            record = self.apply(
                lineage_id=request.lineage_id,
                run_id=request.run_id,
                requested_action=action,
                checkpoint_ref=checkpoint_ref,
                approval_status=str(approval.get("approval_status", "not_required")),
                approval_ref=str(approval.get("approval_ref", "")) or None,
                promotion_allowed=bool(promotion.get("promotion_allowed", False)),
                rollback_allowed=bool(promotion.get("rollback_allowed", False)),
                branch_allowed=bool(promotion.get("branch_allowed", False)),
                restart_allowed=bool(promotion.get("restart_allowed", False)),
                certification_state=str(promotion.get("certification_state", "")) or None,
                confidence=float(verdict.get("confidence", 0.0) or 0.0),
            )
        except (PermissionError, ValueError, NotImplementedError) as exc:
            return StagedOperationResult(
                status="blocked",
                blocking_issues=(f"action_execution_blocked:{type(exc).__name__}",),
                records=(StagedOutputRecord("action_decision", {"status": "blocked", "action": action, "reason": str(exc)}),),
                resumable=False,
            )
        return StagedOperationResult(
            records=(StagedOutputRecord("action_decision", record),),
            output_refs=(record["execution_id"],),
            metadata={"action_applied": True, "effective_action": record["effective_action"]},
        )
