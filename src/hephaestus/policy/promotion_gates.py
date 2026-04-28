from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from hephaestus.schemas.eval_report import EvalReport
from hephaestus.schemas.gate_result import GateResult
from hephaestus.schemas.promotion_gate_report import PromotionGateReport

_PROMOTION_ACTIONS = {"promote_checkpoint", "continue_from_checkpoint", "continue_lineage_best"}
_SAFE_FALLBACK = "continue_lineage_best"
_REJECT_FALLBACK = "reject_checkpoint"
_LINEAGE_BLOCKED = {"poisoned", "deprecated", "archived", "suspect", "blocked"}
_INTEGRITY_BLOCKING = {"insufficient"}
_APPROVAL_BLOCKED = {"pending", "rejected", "expired", "superseded"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_gate(
    gate_results: list[GateResult],
    blocking_failures: list[str],
    warnings: list[str],
    gate_id: str,
    gate_name: str,
    passed: bool,
    severity: str,
    reason: str,
    blocking: bool,
    metadata: dict[str, Any] | None = None,
) -> None:
    gate = GateResult(
        gate_id=gate_id,
        gate_name=gate_name,
        passed=passed,
        severity=severity,
        reason=reason,
        blocking=blocking,
        evidence_refs=[],
        metadata=metadata or {},
    )
    gate_results.append(gate)
    if not passed:
        if blocking:
            blocking_failures.append(reason)
        else:
            warnings.append(reason)


def evaluate_promotion_gates(
    run_id: str,
    lineage_id: str,
    requested_action: str,
    eval_report: EvalReport,
    lineage_state: dict[str, object] | None,
    data_manifest: dict[str, object] | None = None,
    approval_metadata: dict[str, object] | None = None,
) -> PromotionGateReport:
    state = lineage_state or {}
    approval = approval_metadata or {}
    confidence_ceiling = 1.0
    gate_results: list[GateResult] = []
    blocking_failures: list[str] = []
    warnings: list[str] = []
    checkpoint_resolution = dict(eval_report.checkpoint_resolution or {})
    candidate_ref = str(checkpoint_resolution.get("selected_checkpoint_ref", "")) or None
    approval_status = str(approval.get("approval_status", "not_required"))
    requested = str(requested_action or "")

    deterministic_missing = not bool(eval_report.deterministic_scorecard)
    deterministic_passed = bool(eval_report.deterministic_passed) and not deterministic_missing
    deterministic_block = requested in _PROMOTION_ACTIONS and (deterministic_missing or not deterministic_passed)
    _add_gate(
        gate_results,
        blocking_failures,
        warnings,
        gate_id="deterministic_scorecard",
        gate_name="Deterministic scorecard gate",
        passed=not deterministic_block,
        severity="critical" if deterministic_block else "info",
        reason=(
            "Deterministic evidence missing: deterministic_scorecard absent."
            if deterministic_missing
            else "Deterministic gates failed; promotion-like actions are blocked."
        ),
        blocking=deterministic_block,
        metadata={
            "requested_action": requested,
            "deterministic_passed": bool(eval_report.deterministic_passed),
            "deterministic_scorecard_present": bool(eval_report.deterministic_scorecard),
            "failed_gates": list(eval_report.failed_gates),
        },
    )

    eval_pack_level = str(eval_report.eval_pack_integrity_level or "insufficient")
    eval_pack_block = requested in _PROMOTION_ACTIONS and eval_pack_level in _INTEGRITY_BLOCKING
    if eval_pack_level in {"reference_only", "inline_unhashed"}:
        confidence_ceiling = min(confidence_ceiling, 0.65)
        warnings.append(f"eval_pack_integrity_limited:{eval_pack_level}")
    _add_gate(
        gate_results,
        blocking_failures,
        warnings,
        gate_id="eval_pack_integrity",
        gate_name="Eval pack integrity gate",
        passed=not eval_pack_block,
        severity="major" if eval_pack_block else ("warning" if eval_pack_level in {"reference_only", "inline_unhashed"} else "info"),
        reason=(
            f"Eval pack integrity '{eval_pack_level}' is insufficient for promotion-like actions."
            if eval_pack_block
            else f"Eval pack integrity is '{eval_pack_level}'."
        ),
        blocking=eval_pack_block,
        metadata={"eval_pack_integrity_level": eval_pack_level, "eval_pack_id": eval_report.eval_pack_id},
    )

    scorecard_level = str(eval_report.scorecard_integrity_level or "insufficient")
    scorecard_block = requested in _PROMOTION_ACTIONS and scorecard_level in _INTEGRITY_BLOCKING
    if scorecard_level in {"reference_only", "inline_unhashed"}:
        confidence_ceiling = min(confidence_ceiling, 0.75)
        warnings.append(f"scorecard_integrity_limited:{scorecard_level}")
    _add_gate(
        gate_results,
        blocking_failures,
        warnings,
        gate_id="scorecard_integrity",
        gate_name="Scorecard integrity gate",
        passed=not scorecard_block,
        severity="major" if scorecard_block else ("warning" if scorecard_level in {"reference_only", "inline_unhashed"} else "info"),
        reason=(
            f"Scorecard integrity '{scorecard_level}' is insufficient for promotion-like actions."
            if scorecard_block
            else f"Scorecard integrity is '{scorecard_level}'."
        ),
        blocking=scorecard_block,
        metadata={
            "scorecard_integrity_level": scorecard_level,
            "failed_gates": list(eval_report.failed_gates),
            "passed_gates": list(eval_report.passed_gates),
        },
    )

    manifest = data_manifest or {}
    manifest_level = str(manifest.get("manifest_integrity_level", "missing" if not data_manifest else "insufficient"))
    manifest_score = float(manifest.get("completeness_score", 0.0)) if data_manifest else 0.0
    manifest_block = requested == "promote_checkpoint" and manifest_level == "insufficient"
    if not data_manifest:
        confidence_ceiling = min(confidence_ceiling, 0.55)
        warnings.append("data_manifest_missing")
    elif manifest_level in {"reference_only", "partial"}:
        confidence_ceiling = min(confidence_ceiling, 0.7)
        warnings.append(f"data_manifest_integrity_limited:{manifest_level}")
    _add_gate(
        gate_results,
        blocking_failures,
        warnings,
        gate_id="data_manifest",
        gate_name="Data manifest gate",
        passed=not manifest_block,
        severity="major" if manifest_block else ("warning" if manifest_level in {"missing", "reference_only", "partial"} else "info"),
        reason=(
            "Data manifest missing; evidence is incomplete."
            if not data_manifest
            else f"Data manifest integrity is '{manifest_level}' with completeness={manifest_score:.3f}."
        ),
        blocking=manifest_block,
        metadata={
            "manifest_present": bool(data_manifest),
            "manifest_integrity_level": manifest_level,
            "completeness_score": manifest_score,
            "manifest_id": manifest.get("manifest_id"),
        },
    )

    prior_best = str(state.get("best_checkpoint_ref", "") or "")
    prior_stable = str(state.get("last_stable_checkpoint_ref", "") or "")
    checkpoint_block = requested in _PROMOTION_ACTIONS and not candidate_ref
    rollback_allowed = bool(prior_stable or prior_best)
    rollback_block = requested == "rollback_to_checkpoint" and not rollback_allowed
    branch_origin = candidate_ref or prior_best or prior_stable
    _add_gate(
        gate_results,
        blocking_failures,
        warnings,
        gate_id="checkpoint_candidate",
        gate_name="Checkpoint candidate gate",
        passed=not (checkpoint_block or rollback_block),
        severity="critical" if (checkpoint_block or rollback_block) else ("warning" if not branch_origin else "info"),
        reason=(
            "Promotion-like action requires candidate_checkpoint_ref."
            if checkpoint_block
            else "Rollback requested but no stable/best checkpoint target exists."
            if rollback_block
            else "Branch origin checkpoint is missing; branch may proceed only with explicit warning."
            if not branch_origin
            else "Checkpoint evidence available for requested action."
        ),
        blocking=checkpoint_block or rollback_block,
        metadata={
            "candidate_checkpoint_ref": candidate_ref,
            "last_stable_checkpoint_ref": prior_stable or None,
            "best_checkpoint_ref": prior_best or None,
            "branch_origin_checkpoint_ref": branch_origin,
        },
    )

    lineage_status = str(state.get("status", "unknown"))
    lineage_block = requested in _PROMOTION_ACTIONS and lineage_status in _LINEAGE_BLOCKED
    _add_gate(
        gate_results,
        blocking_failures,
        warnings,
        gate_id="lineage_status",
        gate_name="Lineage trust/status gate",
        passed=not lineage_block,
        severity="critical" if lineage_block else "info",
        reason=(
            f"Lineage status '{lineage_status}' is unsafe for promotion-like actions."
            if lineage_block
            else f"Lineage status '{lineage_status}' allows requested action checks to continue."
        ),
        blocking=lineage_block,
        metadata={"lineage_status": lineage_status, "trust_level": state.get("trust_level", "unknown")},
    )

    approval_block = requested in {"promote_checkpoint", "rollback_to_checkpoint", "branch_new_experiment", "restart_lineage"} and approval_status in _APPROVAL_BLOCKED
    _add_gate(
        gate_results,
        blocking_failures,
        warnings,
        gate_id="approval_status",
        gate_name="Approval gate",
        passed=not approval_block,
        severity="major" if approval_block else "info",
        reason=(
            f"Approval status '{approval_status}' blocks execution of high-impact action '{requested}'."
            if approval_block
            else f"Approval status is '{approval_status}'."
        ),
        blocking=approval_block,
        metadata=approval,
    )

    repeatability_sufficient = bool(eval_report.repeatability_sufficient)
    variance_risk = str(eval_report.variance_risk or "unknown")
    stable_like = requested == "promote_checkpoint"
    repeatability_block = stable_like and (not repeatability_sufficient or variance_risk == "high")
    if not repeatability_sufficient or variance_risk in {"high", "medium"}:
        confidence_ceiling = min(confidence_ceiling, 0.8 if variance_risk == "medium" else 0.65)
    _add_gate(
        gate_results,
        blocking_failures,
        warnings,
        gate_id="repeatability_variance",
        gate_name="Repeatability/variance gate",
        passed=not repeatability_block,
        severity="major" if repeatability_block else ("warning" if variance_risk in {"high", "medium"} else "info"),
        reason=(
            "Repeatability or variance risk is unsafe for stable/certified promotion."
            if repeatability_block
            else f"Repeatability status={repeatability_sufficient}, variance_risk='{variance_risk}'."
        ),
        blocking=repeatability_block,
        metadata={
            "repeatability_sufficient": repeatability_sufficient,
            "variance_risk": variance_risk,
            "certification_readiness": eval_report.certification_readiness,
        },
    )

    promotion_allowed = not any(
        (not gate.passed and gate.blocking and requested in _PROMOTION_ACTIONS)
        for gate in gate_results
    )
    branch_allowed = requested == "branch_new_experiment" and not approval_block and (bool(branch_origin) or True)
    restart_allowed = requested == "restart_lineage" and not approval_block
    reject_allowed = True

    recommended = requested
    if requested in _PROMOTION_ACTIONS and not promotion_allowed:
        recommended = _REJECT_FALLBACK if deterministic_block else _SAFE_FALLBACK
    elif requested == "rollback_to_checkpoint" and rollback_block:
        recommended = "branch_new_experiment" if branch_origin else _SAFE_FALLBACK
    elif requested == "branch_new_experiment" and not branch_origin:
        warnings.append("branch_origin_missing")
        confidence_ceiling = min(confidence_ceiling, 0.7)
    elif requested == "restart_lineage" and approval_block:
        recommended = _SAFE_FALLBACK

    return PromotionGateReport(
        report_id=f"gate-{run_id}-{requested or 'none'}",
        run_id=run_id,
        lineage_id=lineage_id,
        candidate_checkpoint_ref=candidate_ref,
        requested_action=requested,
        recommended_effective_action=recommended,
        gate_results=[item.to_dict() for item in gate_results],
        blocking_failures=blocking_failures,
        warnings=warnings,
        promotion_allowed=promotion_allowed,
        rollback_allowed=rollback_allowed,
        branch_allowed=branch_allowed,
        restart_allowed=restart_allowed,
        reject_allowed=reject_allowed,
        confidence_ceiling=max(0.0, min(confidence_ceiling, 1.0)),
        created_at=_now(),
        metadata={
            "eval_pack_integrity_level": eval_pack_level,
            "scorecard_integrity_level": scorecard_level,
            "manifest_integrity_level": manifest_level,
            "approval_status": approval_status,
            "lineage_status": lineage_status,
            "branch_origin_checkpoint_ref": branch_origin,
        },
    )
