"""Lineage state persistence checks."""

from __future__ import annotations

from hephaestus.schemas.lineage_state import LineageState
from hephaestus.schemas.safety_guard import SafetyGuardInput, SafetyGuardResult
from hephaestus.safety._helpers import result, text


def check_lineage_state(inp: SafetyGuardInput) -> SafetyGuardResult:
    state = LineageState.from_dict(inp.payload)
    reasons: list[str] = []
    warnings: list[str] = []
    if state.lineage_id != inp.lineage_id:
        reasons.append("lineage_id_mismatch")
    if state.latest_run_id and state.latest_run_id != inp.run_id:
        warnings.append("latest_run_id_differs_from_boundary_run")
    if state.pending_approval and state.last_effective_action in {"promote_checkpoint", "rollback_to_checkpoint", "branch_new_experiment", "restart_lineage"}:
        reasons.append("pending_approval_with_high_impact_effective_action")
    if state.last_effective_action == "promote_checkpoint" and not text(state.best_checkpoint_ref):
        reasons.append("promoted_without_best_checkpoint_ref")
    return result(inp, reasons, warnings, {"status": state.status, "trust_level": state.trust_level})
