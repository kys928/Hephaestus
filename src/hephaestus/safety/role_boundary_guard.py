"""Role-boundary checks that preserve the mandatory control spine."""

from __future__ import annotations

from hephaestus.control.spine import SPINE_ORDER
from hephaestus.schemas.safety_guard import SafetyGuardInput, SafetyGuardResult
from hephaestus.safety._helpers import result, text

_EXPECTED = [phase.value for phase in SPINE_ORDER]


def check_phase_order(inp: SafetyGuardInput) -> SafetyGuardResult:
    observed = [text(item) for item in inp.payload.get("phase_order", [])] if isinstance(inp.payload.get("phase_order"), list) else []
    reasons: list[str] = []
    if observed != _EXPECTED[: len(observed)]:
        reasons.append("phase_order_violation")
    if len(observed) > len(_EXPECTED):
        reasons.append("unknown_extra_phase")
    return result(inp, reasons, metadata={"observed_phase_order": observed, "expected_phase_order": _EXPECTED})


def check_boundary_payload(inp: SafetyGuardInput) -> SafetyGuardResult:
    reasons = [] if text(inp.boundary) and text(inp.guard_id) and text(inp.run_id) and text(inp.lineage_id) else ["boundary_identity_incomplete"]
    return result(inp, reasons)
