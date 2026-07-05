from __future__ import annotations

from typing import Iterable

from hephaestus.schemas.safety_guard import SafetyGuardInput, SafetyGuardResult


def text(value: object) -> str:
    return str(value or "").strip()


def mapping(value: object) -> dict[str, object]:
    return {str(k): v for k, v in value.items()} if isinstance(value, dict) else {}


def sequence_of_mappings(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [mapping(item) for item in value if isinstance(item, dict)]


def result(inp: SafetyGuardInput, reasons: Iterable[str], warnings: Iterable[str] = (), metadata: dict[str, object] | None = None) -> SafetyGuardResult:
    reason_list = sorted(set(reasons))
    warning_list = sorted(set(warnings))
    severity = "critical" if reason_list else ("warning" if warning_list else "info")
    return SafetyGuardResult(
        guard_id=inp.guard_id,
        run_id=inp.run_id,
        lineage_id=inp.lineage_id,
        boundary=inp.boundary,
        allowed=not reason_list,
        severity=severity,
        reasons=reason_list,
        warnings=warning_list,
        metadata=metadata or {},
    )
