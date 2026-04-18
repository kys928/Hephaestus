"""Checkpoint promotion transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PromotionTransition:
    best_checkpoint_ref: str | None
    last_stable_checkpoint_ref: str | None
    certified_stable_checkpoint_ref: str | None
    last_certification_result: str
    status: str
    trust_level: str
    notes: list[str]


def apply_promotion(
    lineage_state: dict[str, Any] | None,
    candidate_checkpoint_ref: str | None,
    promotion_state: str,
    certification_state: str,
    deterministic_passed: bool,
    confidence: float,
    stable_confidence_threshold: float,
) -> PromotionTransition:
    state = lineage_state or {}
    best = state.get("best_checkpoint_ref")
    stable = state.get("last_stable_checkpoint_ref")
    certified = state.get("certified_stable_checkpoint_ref")
    notes: list[str] = []

    if not candidate_checkpoint_ref:
        notes.append("missing_candidate_checkpoint")
        return PromotionTransition(
            best,
            stable,
            certified,
            certification_state,
            "unstable",
            "low",
            notes,
        )

    if promotion_state in {"promoted_best", "stable", "certified_stable"}:
        best = candidate_checkpoint_ref
        notes.append("best_checkpoint_updated")

    if promotion_state in {"stable", "certified_stable"} and deterministic_passed and confidence >= stable_confidence_threshold:
        stable = candidate_checkpoint_ref
        notes.append("stable_checkpoint_updated")

    if promotion_state == "certified_stable" and certification_state == "certification_passed":
        certified = candidate_checkpoint_ref
        notes.append("certified_stable_checkpoint_updated")

    if promotion_state == "rejected":
        notes.append("checkpoint_rejected")

    status = "active" if deterministic_passed else "degraded"
    if promotion_state == "certified_stable":
        trust = "high"
    elif promotion_state == "stable":
        trust = "high"
    elif promotion_state in {"promoted_best", "candidate_best"}:
        trust = "medium"
    else:
        trust = "low"
    return PromotionTransition(best, stable, certified, certification_state, status, trust, notes)
