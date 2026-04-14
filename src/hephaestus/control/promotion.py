"""Checkpoint promotion transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PromotionTransition:
    best_checkpoint_ref: str | None
    last_stable_checkpoint_ref: str | None
    status: str
    trust_level: str
    notes: list[str]


def apply_promotion(
    lineage_state: dict[str, Any] | None,
    candidate_checkpoint_ref: str | None,
    promotion_state: str,
    deterministic_passed: bool,
    confidence: float,
    stable_confidence_threshold: float,
) -> PromotionTransition:
    state = lineage_state or {}
    best = state.get("best_checkpoint_ref")
    stable = state.get("last_stable_checkpoint_ref")
    notes: list[str] = []

    if not candidate_checkpoint_ref:
        notes.append("missing_candidate_checkpoint")
        return PromotionTransition(best, stable, "unstable", "low", notes)

    if promotion_state in {"promoted_best", "stable"}:
        best = candidate_checkpoint_ref
        notes.append("best_checkpoint_updated")

    if promotion_state == "stable" and deterministic_passed and confidence >= stable_confidence_threshold:
        stable = candidate_checkpoint_ref
        notes.append("stable_checkpoint_updated")

    if promotion_state == "rejected":
        notes.append("checkpoint_rejected")

    status = "active" if deterministic_passed else "degraded"
    trust = "high" if promotion_state == "stable" else ("medium" if promotion_state in {"promoted_best", "candidate_best"} else "low")
    return PromotionTransition(best, stable, status, trust, notes)
