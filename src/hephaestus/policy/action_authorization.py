"""Authoritative action authorization. Caller status/boolean hints grant nothing."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from hephaestus.schemas.action_authorization import ActionAuthorizationScope
from hephaestus.schemas.approval_request import ApprovalRequest
from hephaestus.schemas.eval_report import EvalReport
from hephaestus.state.decision_store import DecisionStore
from hephaestus.state.run_store import RunStore
from hephaestus.storage.base import ArtifactStore
from .action_registry import canonical_action_name, evaluate_action_boundary
from .approval_policy import ApprovalPolicy
from .promotion_gates import evaluate_promotion_gates
from .promotion_policy import PromotionPolicy


def lineage_digest(state: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def required_approval_type(policy: ApprovalPolicy, action: str, current: dict) -> str:
    decision = policy.decide(action, str(current["stage_name"]), str(current["trust_level"]))
    if decision.required_approval_type != "none":
        return decision.required_approval_type
    boundary = evaluate_action_boundary(action)
    return "operator_high_risk_approval" if boundary["high_risk"] else "operator_approval"


def build_action_approval_request(*, request_id: str, current: dict, run_id: str,
                                  action: str, checkpoint_ref: str | None,
                                  expires_at: str, policy: ApprovalPolicy,
                                  child_lineage_id: str | None = None,
                                  eval_report_ref: str | None = None,
                                  data_manifest_ref: str | None = None) -> ApprovalRequest:
    """Prepare an exact, reviewable request; this never grants an approval."""
    action = canonical_action_name(action)
    boundary = evaluate_action_boundary(action)
    if not boundary["known_action"] or boundary["forbidden"]:
        raise ValueError("cannot request an unknown or forbidden action")
    if _timestamp(expires_at) <= datetime.now(timezone.utc):
        raise ValueError("approval request must expire in the future")
    scope = ActionAuthorizationScope(action, str(current["lineage_id"]), run_id,
        str(current["stage_name"]), checkpoint_ref, lineage_digest(current), expires_at,
        child_lineage_id, eval_report_ref, data_manifest_ref)
    return ApprovalRequest(request_id, f"decision-{run_id}", str(current["lineage_id"]),
        run_id, action, "Review exact action, target and immutable evidence",
        "high" if boundary["high_risk"] else "moderate",
        required_approval_type(policy, action, current),
        created_at=datetime.now(timezone.utc).isoformat(),
        metadata={"action_scope": scope.to_dict(), "checkpoint_ref": checkpoint_ref})


def _timestamp(value: str) -> datetime:
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("approval timestamps must include a timezone")
    return stamp


def _artifact(store: ArtifactStore | None, ref: str | None) -> dict:
    if store is None or not ref or not ref.startswith("sha256:"):
        raise PermissionError("immutable scientific evidence is required")
    with store.open(ref) as stream:
        raw = stream.read(16 * 1024 * 1024 + 1)
    if len(raw) > 16 * 1024 * 1024 or "sha256:" + hashlib.sha256(raw).hexdigest() != ref:
        raise PermissionError("scientific evidence hash mismatch")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise PermissionError("scientific evidence must be an object")
    return value


@dataclass(slots=True)
class ActionAuthorizer:
    root: Path
    policy: ApprovalPolicy
    artifacts: ArtifactStore | None = None

    def verify(self, *, approval_ref: str, action: str, run_id: str,
               current: dict, checkpoint_ref: str | None,
               child_lineage_id: str | None) -> dict:
        store = DecisionStore(self.root)
        # A reference identifies an actual decision event, never arbitrary text
        # or just a request that might have been rejected/superseded later.
        decision = next((r for r in reversed(store.all_approval_decisions())
                         if r.get("decision_event_id") == approval_ref), None)
        if decision is None:
            raise PermissionError("authoritative approval decision not found")
        request = store.get_approval_request(str(decision.get("request_id", "")))
        latest = store.get_latest_approval_decision(str(decision.get("request_id", "")))
        if request is None or latest != decision:
            raise PermissionError("approval was superseded or its request is missing")
        if (decision.get("status") != "approved" or decision.get("outcome") not in
                {"approved", "override_approved"} or not decision.get("operator_id")
                or decision.get("operator_id") == "operator.unknown"
                or decision.get("effect_on_action") != "execute_requested_action"
                or request.get("status") not in {"pending", "approved"}):
            raise PermissionError("approval is not an authorized operator decision")
        for record in (request, decision):
            if record.get("run_id") != run_id or record.get("lineage_id") != current["lineage_id"]:
                raise PermissionError("approval run or lineage mismatch")
        if canonical_action_name(str(request.get("proposed_action"))) != action:
            raise PermissionError("approval action mismatch")
        try:
            scope = ActionAuthorizationScope(**request["metadata"]["action_scope"])
            now = datetime.now(timezone.utc)
            if (_timestamp(scope.expires_at) <= now or
                    _timestamp(str(decision["created_at"])) > now or
                    _timestamp(str(request["created_at"])) > _timestamp(str(decision["created_at"]))):
                raise PermissionError("approval is expired or has invalid chronology")
        except (KeyError, TypeError, ValueError) as exc:
            raise PermissionError("approval is missing a valid bounded action scope") from exc
        expected = (action, current["lineage_id"], run_id, current["stage_name"],
                    checkpoint_ref, lineage_digest(current), child_lineage_id)
        if (scope.action, scope.lineage_id, scope.run_id, scope.stage_name,
                scope.checkpoint_ref, scope.lineage_sha256, scope.child_lineage_id) != expected:
            raise PermissionError("approval scope does not match current action/state")
        policy = self.policy.decide(action, str(current["stage_name"]), str(current["trust_level"]))
        if request.get("required_approval_type") != required_approval_type(self.policy, action, current):
            raise PermissionError("approval type no longer matches policy")
        if policy.outcome == "override_not_allowed" and decision["outcome"] == "override_approved":
            raise PermissionError("approval policy forbids override")
        result = {"approval_status": decision["outcome"], "approval_ref": approval_ref,
                  "request_id": request["request_id"], "scope": scope.to_dict()}
        if action in {"rollback_to_checkpoint", "branch_new_experiment", "continue_from_checkpoint"}:
            known = {current.get(key) for key in ("best_checkpoint_ref", "last_stable_checkpoint_ref",
                "certified_stable_checkpoint_ref", "origin_checkpoint_ref", "branch_origin_checkpoint_ref")}
            known.update(r.get("checkpoint_ref") for r in RunStore(self.root).all()
                         if r.get("lineage_id") == current["lineage_id"])
            if not checkpoint_ref or checkpoint_ref not in known:
                raise PermissionError("checkpoint does not belong to the authorized lineage")
        if action == "promote_checkpoint":
            report = EvalReport.from_dict(_artifact(self.artifacts, scope.eval_report_ref))
            manifest = _artifact(self.artifacts, scope.data_manifest_ref)
            run = RunStore(self.root).get(run_id)
            if (run is None or run.get("lineage_id") != current["lineage_id"] or
                    run.get("checkpoint_ref") != checkpoint_ref):
                raise PermissionError("candidate run/checkpoint lineage is not authoritative")
            if (report.run_id != run_id or report.stage_name != current["stage_name"] or
                    report.checkpoint_resolution.get("selected_checkpoint_ref") != checkpoint_ref or
                    report.eval_pack_integrity_level != "content_hash_verified" or
                    report.scorecard_integrity_level != "content_hash_verified" or
                    report.deterministic_passed is not True or report.failed_gates or
                    report.promotion_bundle_passed is not True or report.evidence_completeness != 1.0):
                raise PermissionError("scientific certification identity or hard gates failed")
            gate = evaluate_promotion_gates(run_id, str(current["lineage_id"]), action,
                                            report, current, manifest,
                                            {"approval_status": result["approval_status"]})
            cert = PromotionPolicy().decide(
                deterministic_passed=report.deterministic_passed, confidence=report.confidence,
                promotion_bundle_passed=report.promotion_bundle_passed,
                evidence_completeness=report.evidence_completeness,
                certification_readiness=report.certification_readiness,
                recheck_recommended=report.recheck_recommended or report.recheck_needed,
                observed_evidence_runs=report.repeated_eval_count,
                observed_consistent_runs=report.observed_consistent_runs,
                stability_confidence=report.stability_confidence,
                repeatability_sufficient=report.repeatability_sufficient,
                variance_risk=report.variance_risk,
            )
            if not gate.promotion_allowed or cert.certification_state != "certification_passed":
                raise PermissionError("independent promotion/certification gates refused action")
            result.update(certification_state=cert.certification_state, confidence=report.confidence,
                          promotion_gate=gate.to_dict())
        return result
