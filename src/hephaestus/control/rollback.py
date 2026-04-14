"""Rollback transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RollbackTransition:
    target_checkpoint_ref: str | None
    succeeded: bool
    notes: list[str]


def apply_rollback(lineage_state: dict[str, Any] | None, explicit_target: str | None = None) -> RollbackTransition:
    state = lineage_state or {}
    target = explicit_target or state.get("last_stable_checkpoint_ref")
    if not target:
        return RollbackTransition(None, False, ["no_valid_rollback_target"])
    return RollbackTransition(str(target), True, ["rollback_target_selected"])
