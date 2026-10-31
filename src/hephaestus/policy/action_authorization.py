"""Authoritative action authorization. Caller status/boolean hints grant nothing."""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from hephaestus.schemas.action_authorization import ActionAuthorizationScope
from hephaestus.schemas.approval_request import ApprovalRequest
from hephaestus.schemas.approval_decision import ApprovalDecision
from hephaestus.schemas.eval_report import EvalReport
from hephaestus.schemas.scorecard import Scorecard
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest
from hephaestus.config_loader import load_named_config
from hephaestus.evaluation.pack_loader import load_eval_pack
from hephaestus.evaluation.regression_checks import build_regression_summary
from hephaestus.state.decision_store import DecisionStore
from hephaestus.state.run_store import RunStore
from hephaestus.storage.base import ArtifactStore
from hephaestus.storage.verified import canonical_bytes, read_verified_json
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
    checkpoint_hash = None
    if action in {"promote_checkpoint", "activate_role_adapter"}:
        valid, checkpoint_hash = validate_checkpoint_manifest(str(checkpoint_ref or ""))
        if not valid:
            raise ValueError("checkpoint action requires verified materialized checkpoint bytes")
    scope = ActionAuthorizationScope(action, str(current["lineage_id"]), run_id,
        str(current["stage_name"]), checkpoint_ref, lineage_digest(current), expires_at,
        child_lineage_id, eval_report_ref, data_manifest_ref, checkpoint_hash)
    return ApprovalRequest(request_id, f"decision-{run_id}", str(current["lineage_id"]),
        run_id, action, "Review exact action, target and immutable evidence",
        "high" if boundary["high_risk"] else "moderate",
        required_approval_type(policy, action, current),
        created_at=datetime.now(timezone.utc).isoformat(),
        metadata={"action_scope": scope.to_dict(), "checkpoint_ref": checkpoint_ref})


def build_scoped_approval_decision(request: ApprovalRequest, *, decision_event_id: str,
                                   operator_id: str, outcome: str, note: str,
                                   policy: ApprovalPolicy) -> ApprovalDecision:
    """Record an explicit operator response to exactly the reviewed request bytes."""
    if not operator_id.strip() or operator_id == "operator.unknown":
        raise ValueError("an identified operator is required")
    resolution = policy.resolve_operator_outcome(outcome,
        override_allowed=request.required_approval_type != "operator_approval_no_override")
    digest = hashlib.sha256(canonical_bytes(request.to_dict())).hexdigest()
    return ApprovalDecision(decision_event_id, request.request_id, request.lineage_id,
        request.run_id, operator_id, resolution.outcome, resolution.status, note,
        resolution.effect_on_action, datetime.now(timezone.utc).isoformat(),
        {"approval_request_sha256":digest, "proposed_action":request.proposed_action,
         "checkpoint_ref":request.metadata.get("checkpoint_ref")})


def _timestamp(value: str) -> datetime:
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("approval timestamps must include a timezone")
    return stamp


def _artifact(store: ArtifactStore | None, ref: str | None) -> dict:
    if store is None or not ref:
        raise PermissionError("immutable scientific evidence is required")
    try:
        return read_verified_json(store, ref)
    except (ValueError, OSError) as exc:
        raise PermissionError("scientific evidence hash/identity mismatch") from exc



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
        if decision.get("metadata", {}).get("approval_request_sha256") != hashlib.sha256(canonical_bytes(request)).hexdigest():
            raise PermissionError("approval request bytes changed after the operator decision")
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
        if action == "activate_role_adapter":
            valid, checkpoint_hash = validate_checkpoint_manifest(str(checkpoint_ref or ""))
            if not valid or checkpoint_hash != scope.checkpoint_manifest_hash:
                raise PermissionError("activation checkpoint manifest differs from operator-reviewed bytes")
        if action == "promote_checkpoint":
            stage = load_named_config(self.policy.config_dir, "stage_profiles", str(current["stage_name"]))
            if action not in stage.get("allowed_next_actions", []):
                raise PermissionError("active stage does not permit promotion")
            pack = load_eval_pack(str(stage["eval_pack"]), config_dir=self.policy.config_dir)
            if pack.get("content_hash_verified") is not True:
                raise PermissionError("active certification pack has no independently verified frozen content hash")
            valid, checkpoint_hash = validate_checkpoint_manifest(str(checkpoint_ref or ""))
            if not valid or checkpoint_hash != scope.checkpoint_manifest_hash:
                raise PermissionError("candidate checkpoint bytes differ from the operator-reviewed manifest")
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
            if (report.pack_name != stage["eval_pack"] or report.eval_pack_id != pack["eval_pack_id"] or
                    report.eval_pack_version != pack["eval_pack_version"]):
                raise PermissionError("scientific report does not match the active frozen certification pack")
            for value in (report.confidence, report.evidence_completeness, report.stability_confidence, report.consistency_score):
                if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise PermissionError("invalid scientific confidence or completeness value")
            scorecard = Scorecard.from_dict(report.deterministic_scorecard)
            if (scorecard.run_id != run_id or scorecard.checkpoint_ref != checkpoint_ref or
                    scorecard.eval_pack_id != report.eval_pack_id or scorecard.eval_pack_version != report.eval_pack_version or
                    scorecard.thresholds != stage["deterministic_gates"] or
                    scorecard.deterministic_passed is not True or scorecard.failed_gates or
                    not set(pack["required_metrics"]) <= set(scorecard.metrics) or
                    any(type(x) not in (int, float) or not math.isfinite(x) for x in scorecard.metrics.values())):
                raise PermissionError("scorecard identity, metrics or policy thresholds failed verification")
            regression = build_regression_summary(run_id, scorecard.metrics,
                stage["deterministic_gates"], pack["regression_bundles"])
            if (not regression.deterministic_passed or
                    not all(bundle["passed"] for bundle in regression.bundle_results.values()) or
                    not scorecard.gate_results or
                    any(gate.get("passed") is not True for gate in scorecard.gate_results.values())):
                raise PermissionError("recomputed deterministic scientific gates failed")
            stage_cert = stage.get("certification_profile", {})
            required = pack["minimum_evidence"]
            required_rechecks = max(int(stage_cert.get("required_rechecks", 0)),
                                   int(pack["repeatability_requirements"].get("required_rechecks", 0)))
            if (report.repeated_eval_count < required_rechecks or
                    report.observed_consistent_runs < max(int(stage_cert.get("min_consistent_runs", 1)),
                        int(pack["recheck_requirements"]["min_consistent_runs"])) or
                    report.consistency_score < max(float(stage_cert.get("min_repeat_consistency", 0)),
                        float(pack["repeatability_requirements"].get("min_repeat_consistency", 0)))):
                raise PermissionError("frozen repeatability requirements are not satisfied")
            gate = evaluate_promotion_gates(run_id, str(current["lineage_id"]), action,
                                            report, current, manifest,
                                            {"approval_status": result["approval_status"]})
            cert = PromotionPolicy().decide(
                deterministic_passed=report.deterministic_passed, confidence=report.confidence,
                promotion_bundle_passed=report.promotion_bundle_passed,
                evidence_completeness=report.evidence_completeness,
                certification_readiness=report.certification_readiness,
                recheck_recommended=report.recheck_recommended or report.recheck_needed,
                observed_evidence_runs=1+report.repeated_eval_count,
                min_promotion_evidence=int(required["promotion_runs"]),
                min_stable_evidence=int(required["stable_runs"]),
                min_certification_evidence=int(required["certification_runs"]),
                stage_certification_eligibility=str(stage_cert.get("eligibility", "standard")),
                stage_require_recheck=bool(stage_cert.get("require_recheck", False)),
                stage_min_consistent_runs=int(stage_cert.get("min_consistent_runs", 1)),
                min_stability_confidence=float(pack["certification_bundle"]["min_stability_confidence"]),
                stage_thresholds=stage.get("stage_thresholds") or pack["stage_tolerances"].get(str(stage["strictness"]), {}),
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
