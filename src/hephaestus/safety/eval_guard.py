"""Evaluation report checks for promotion and judge-exit boundaries."""

from __future__ import annotations

from hephaestus.schemas.eval_report import EvalReport
from hephaestus.schemas.safety_guard import SafetyGuardInput, SafetyGuardResult
from hephaestus.safety._helpers import result


def check_eval_report(inp: SafetyGuardInput) -> SafetyGuardResult:
    report = EvalReport.from_dict(inp.payload)
    reasons: list[str] = []
    warnings: list[str] = []
    if report.run_id != inp.run_id:
        reasons.append("run_id_mismatch")
    if not report.checkpoint_resolution.get("selected_checkpoint_ref"):
        reasons.append("selected_checkpoint_ref_missing")
    if not report.deterministic_scorecard:
        reasons.append("deterministic_scorecard_missing")
    if not report.deterministic_passed:
        reasons.append("deterministic_gates_failed")
    if report.eval_pack_integrity_level in {"insufficient", "unknown"}:
        reasons.append(f"eval_pack_integrity_{report.eval_pack_integrity_level}")
    if report.scorecard_integrity_level in {"insufficient", "unknown"}:
        reasons.append(f"scorecard_integrity_{report.scorecard_integrity_level}")
    if report.variance_risk in {"high", "unknown"}:
        warnings.append(f"variance_risk_{report.variance_risk}")
    return result(inp, reasons, warnings, {"eval_id": report.eval_id, "confidence": report.confidence})
