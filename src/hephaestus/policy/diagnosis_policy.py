"""Deterministic confidence and evidence policy for diagnosis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DiagnosisPolicy:
    """Conservative thresholds used by the rule-based diagnosis service."""

    minimum_hypothesis_confidence: float = 0.45
    close_hypothesis_margin: float = 0.05
    close_hypothesis_confidence_ceiling: float = 0.72
    unverified_eval_downstream_confidence_ceiling: float = 0.35
    contradiction_penalty: float = 0.30
    additional_independent_evidence_bonus: float = 0.05
    maximum_hypothesis_confidence: float = 0.95


DIAGNOSTIC_DOMAIN_ORDER = (
    "evaluation_integrity",
    "launch_or_reproducibility",
    "runtime_or_system",
    "data_quality",
    "data_coverage",
    "data_format_or_wrapper",
    "tokenizer",
    "checkpoint_integrity",
    "architecture",
    "optimizer_or_scheduler",
    "numerical_instability",
    "undertraining",
    "overfitting",
    "decoding",
    "model_family_limitation",
    "inconclusive",
)
