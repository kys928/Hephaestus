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
from hephaestus.policy.approval_policy import ApprovalPolicy
from hephaestus.policy.action_authorization import ActionAuthorizer
from hephaestus.state.single_writer import single_writer
from hephaestus.storage.base import ArtifactStore
from hephaestus.schemas.lineage_state import LineageState
from hephaestus.state.lineage_store import LineageStore
from hephaestus.state.run_store import RunStore
from hephaestus.state.checkpoint_identity_store import CheckpointIdentityStore
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest
from hephaestus.storage.base import StateRepository

from .state import ACTION_EXECUTIONS


def _stable_id(*values: object) -> str:
    raw = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "action-" + hashlib.sha256(raw).hexdigest()[:20]


def _approval_required_by_policy(outcome: str) -> bool:
    return outcome in {
        "approval_required",
        "approval_required_high_risk",
        "override_not_allowed",
    }


@dataclass(slots=True)
class GovernedActionExecutor:
    """Apply finite Judge actions only after their own governance gate authorizes them.

    Promotion eligibility is deliberately *not* a generic lineage-action gate.
    Only ``promote_checkpoint`` requires independently recomputed certification. Rollback, branch,
    and restart are authorized independently by the central action registry and
    ApprovalPolicy, plus their own lineage preconditions. Legacy transition flags
    remain accepted for compatibility but have no authorization effect.

    The executor mutates compact lineage truth, never checkpoint bytes. Every
    attempt is append-only and keyed by a stable execution ID so process retries
    cannot apply the same transition twice.
    """

    state_root: Path
    repository: StateRepository
    approval_policy: ApprovalPolicy | None = None
    artifact_store: ArtifactStore | None = None

    @property
    def lineage_store(self) -> LineageStore:
        return LineageStore(self.state_root)

    def apply(self, **kwargs) -> dict[str, object]:
        with single_writer(self.state_root):
            self.reconcile_prepared()
            return self._apply(**kwargs)

    def reconcile_prepared(self) -> list[dict[str, object]]:
        """Finish already-authorized intents before admitting another mutation."""
        with single_writer(self.state_root):
            latest = {str(r["execution_id"]): r for r in self.repository.all(ACTION_EXECUTIONS)}
            return [self._reconcile(r) for r in latest.values() if r.get("status") == "prepared"]

    def _reconcile(self, previous: dict) -> dict:
        before, after = previous.get("before"), previous.get("after")
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise RuntimeError("malformed prepared action journal")
        current = self.lineage_store.get_current(str(previous["lineage_id"]))
        if current is None:
            current = LineageState(lineage_id=str(previous["lineage_id"]), stage_name="smoke_test").to_dict()
        if current not in (before, after):
            raise RuntimeError("lineage diverged since prepared action; refusing overwrite")
        if current == before:
            action = str(previous["effective_action"])
            self._validate_preconditions(LineageState.from_dict(before), action,
                                         previous.get("checkpoint_ref"))
            policy = self.approval_policy or ApprovalPolicy()
            boundary = evaluate_action_boundary(action)
            outcome = policy.decide(action, str(before["stage_name"]), str(before["trust_level"]))
            if boundary["requires_approval"] or _approval_required_by_policy(outcome.outcome):
                # Prepared is not permission to mutate later after revocation,
                # expiry, policy drift or checkpoint corruption. Already-applied
                # after-states only need journal/mirror completion.
                scope = previous.get("authorization", {}).get("scope", {})
                ActionAuthorizer(self.state_root, policy, self.artifact_store).verify(
                    approval_ref=str(previous.get("approval_ref") or ""), action=action,
                    run_id=str(previous["run_id"]), current=before,
                    checkpoint_ref=previous.get("checkpoint_ref"),
                    child_lineage_id=scope.get("child_lineage_id"))
        child = previous.get("child_after")
        if child is not None:
            if not isinstance(child, dict) or not child.get("lineage_id"):
                raise RuntimeError("malformed prepared child lineage")
            existing = self.lineage_store.get_current(str(child["lineage_id"]))
            if existing is not None and existing != child:
                raise RuntimeError("prepared child lineage conflicts with current state")
            self.lineage_store.set_current(child)
        self.lineage_store.set_current(after)  # Also heals an interrupted legacy mirror write.
        applied = {**previous, "status": "applied"}
        self.repository.append(ACTION_EXECUTIONS, applied)
        return applied

    def _apply(
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

        current_payload = self.lineage_store.get_current(lineage_id)
        if current_payload is None:
            current = LineageState(lineage_id=lineage_id, stage_name="smoke_test")
        else:
            current = LineageState.from_dict(current_payload)
        before = current.to_dict()

        execution_id = _stable_id(lineage_id, run_id, canonical, checkpoint_ref, child_lineage_id)
        previous = self.repository.get_latest(ACTION_EXECUTIONS, "execution_id", execution_id)
        if previous is not None:
            if previous.get("approval_ref") != approval_ref:
                raise PermissionError("action retry changed authorization identity")
            if previous.get("status") == "applied":
                return dict(previous)
            raise RuntimeError("unreconciled action journal state")
        for row in self.repository.all(ACTION_EXECUTIONS):
            if row.get("lineage_id") == lineage_id and row.get("run_id") == run_id:
                raise PermissionError("a different action is already committed for this run")
            if approval_ref and row.get("approval_ref") == approval_ref:
                raise PermissionError("approval has already been consumed")

        policy = self.approval_policy or ApprovalPolicy()
        policy_decision = policy.decide(canonical, current.stage_name, current.trust_level)
        boundary = evaluate_action_boundary(canonical, context={"approval_status": approval_status})
        protected = bool(boundary["requires_approval"]) or _approval_required_by_policy(policy_decision.outcome)
        authorization: dict = {}
        if protected:
            if not approval_ref:
                if approval_status in {"approved", "override_approved"}:
                    raise PermissionError("approved action is missing durable approval reference")
                raise PermissionError("action registry did not authorize action without approval")
            authorization = ActionAuthorizer(self.state_root, policy, self.artifact_store).verify(
                approval_ref=approval_ref, action=canonical, run_id=run_id, current=before,
                checkpoint_ref=checkpoint_ref, child_lineage_id=child_lineage_id)
            approval_status = str(authorization["approval_status"])
        else:
            approval_status = "not_required"
        boundary = evaluate_action_boundary(canonical, context={"approval_status": approval_status})
        if not boundary["allowed"]:
            raise PermissionError(f"action registry did not authorize {canonical}: {boundary['reasons']}")
        checkpoint_identity = self._validate_preconditions(current, canonical, checkpoint_ref)

        if canonical == "promote_checkpoint":
            # Authoritative immutable evaluation, never the caller's boolean.
            certification_state = str(authorization["certification_state"])
            confidence = float(authorization["confidence"])
            promotion_allowed = True

        child_after: dict[str, object] | None = None
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
            existing_child = self.lineage_store.get_current(child_id)
            if existing_child is None:
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
                child_after = child.to_dict()
            elif existing_child.get("parent_lineage_id") != lineage_id:
                raise RuntimeError("branch lineage identity conflict")
            current.child_lineage_ids = list(dict.fromkeys([*current.child_lineage_ids, child_id]))
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

        record = {
            "execution_id": execution_id,
            "status": "applied",
            "lineage_id": lineage_id,
            "run_id": run_id,
            "requested_action": requested_action,
            "effective_action": canonical,
            "checkpoint_ref": checkpoint_ref,
            "checkpoint_identity": checkpoint_identity,
            "approval_status": approval_status,
            "approval_ref": approval_ref,
            "authorization": authorization,
            "governance": {
                "action_registry": boundary,
                "approval_policy": {
                    "outcome": policy_decision.outcome,
                    "risk_level": policy_decision.risk_level,
                    "required_approval_type": policy_decision.required_approval_type,
                    "reason": policy_decision.reason,
                },
                "promotion_gate_applied": canonical == "promote_checkpoint",
            },
            "transition_gates": {
                "promotion_allowed": promotion_allowed if canonical == "promote_checkpoint" else None,
                "legacy_transition_hints_ignored": {
                    "rollback_allowed": rollback_allowed,
                    "branch_allowed": branch_allowed,
                    "restart_allowed": restart_allowed,
                },
            },
            "certification_state": certification_state,
            "confidence": confidence,
            "applied_at": now,
            "before": before,
            "after": current.to_dict(),
            "child_after": child_after,
        }
        # Journal first, then cross-store mutation. On retry, a prepared record
        # is reconciled against before/after identities, never blindly re-applied.
        self.repository.append(ACTION_EXECUTIONS, {**record, "status": "prepared"})
        if child_after is not None:
            self.lineage_store.set_current(child_after)
        self.lineage_store.set_current(current.to_dict())
        self.repository.append(ACTION_EXECUTIONS, record)
        return record

    def _validate_preconditions(self, current: LineageState, canonical: str,
                                checkpoint_ref: str | None) -> str | None:
        """Shared admission before normal or prepared-but-unapplied mutation."""
        if current.status in {"poisoned", "archived", "deprecated", "blocked"} and canonical in {
            "continue_lineage_best", "continue_from_checkpoint",
            "rerun_same_config", "reject_candidate",
        }:
            raise PermissionError("terminal or poisoned lineage cannot autonomously continue")
        if canonical == "continue_from_checkpoint":
            if not checkpoint_ref:
                raise PermissionError("continuation requires an explicit checkpoint identity")
            known = {
                current.best_checkpoint_ref,
                current.last_stable_checkpoint_ref,
                current.certified_stable_checkpoint_ref,
                current.origin_checkpoint_ref,
                current.branch_origin_checkpoint_ref,
            }
            known.update(
                row.get("checkpoint_ref") for row in RunStore(self.state_root).all()
                if row.get("lineage_id") == current.lineage_id
            )
            known.update(row.checkpoint_ref for row in CheckpointIdentityStore(self.repository).for_lineage(current.lineage_id))
            if checkpoint_ref not in known:
                raise PermissionError("continuation checkpoint does not belong to lineage")
            verified, manifest_identity = validate_checkpoint_manifest(Path(checkpoint_ref))
            if not verified:
                raise PermissionError("continuation checkpoint manifest is invalid")
            return CheckpointIdentityStore(self.repository).verify(
                current.lineage_id, checkpoint_ref, manifest_identity)
        return None


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
