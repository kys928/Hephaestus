"""Authoritative deterministic diagnostic rules."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from hephaestus.diagnosis.normalization import NormalizedEvidence
from hephaestus.policy.diagnosis_policy import DIAGNOSTIC_DOMAIN_ORDER, DiagnosisPolicy
from hephaestus.schemas.diagnosis_contract import DiagnosticHypothesis


@dataclass(frozen=True, slots=True)
class RuleSpec:
    domain: str
    supporting_signals: frozenset[str]
    contradicting_signals: frozenset[str]
    required_tests: tuple[str, ...]
    interventions: tuple[str, ...]
    summary: str


_RULES = (
    RuleSpec(
        "evaluation_integrity",
        frozenset(
            {
                "eval_integrity_failed",
                "eval_pack_unverified",
                "deterministic_scorecard_missing",
                "evaluation_settings_mismatch",
            }
        ),
        frozenset({"eval_integrity_verified"}),
        (
            "verify frozen eval-pack identity and content hash",
            "rerun deterministic scorecard with recorded settings",
        ),
        ("repair_evaluation", "collect_more_evidence"),
        "Evaluation evidence may be incomplete, unverified, or inconsistent.",
    ),
    RuleSpec(
        "launch_or_reproducibility",
        frozenset(
            {
                "non_reproducible",
                "replay_failed",
                "seed_mismatch",
                "launch_config_mismatch",
            }
        ),
        frozenset({"reproducibility_verified"}),
        (
            "replay the recorded launch contract",
            "compare seeds, config, environment, and artifact references",
        ),
        ("collect_more_evidence", "change_training_recipe"),
        "Launch or replay evidence is consistent with a reproducibility problem.",
    ),
    RuleSpec(
        "runtime_or_system",
        frozenset({"runtime_failure", "hardware_interruption", "data_loader_failure"}),
        frozenset({"runtime_healthy"}),
        (
            "inspect the first runtime incident and process exit evidence",
            "repeat a bounded launch smoke test",
        ),
        ("collect_more_evidence", "stop"),
        "Recorded runtime evidence is consistent with a system or execution failure.",
    ),
    RuleSpec(
        "data_quality",
        frozenset(
            {
                "data_quality_failed",
                "contamination_detected",
                "malformed_data",
                "deduplication_failed",
            }
        ),
        frozenset({"data_quality_verified"}),
        (
            "rerun data-quality checks on the exact manifest version",
            "audit malformed, duplicate, and contamination records",
        ),
        ("repair_data", "replace_or_mix_dataset"),
        "Explicit data checks are consistent with a data-quality problem.",
    ),
    RuleSpec(
        "data_coverage",
        frozenset({"data_coverage_gap", "domain_missing"}),
        frozenset({"data_coverage_verified"}),
        ("measure capability coverage against manifest domains",),
        ("replace_or_mix_dataset", "collect_more_evidence"),
        "Explicit coverage evidence indicates a possible data-coverage gap.",
    ),
    RuleSpec(
        "data_format_or_wrapper",
        frozenset(
            {
                "wrapper_mismatch",
                "data_format_mismatch",
                "prompt_target_boundary_mismatch",
            }
        ),
        frozenset({"wrapper_compatible"}),
        ("round-trip representative samples through training and eval wrappers",),
        ("change_preprocessing", "repair_data"),
        "Recorded formatting evidence indicates a wrapper or boundary mismatch.",
    ),
    RuleSpec(
        "tokenizer",
        frozenset(
            {"tokenizer_mismatch", "tokenizer_incompatible", "special_token_mismatch"}
        ),
        frozenset({"tokenizer_compatible"}),
        (
            "compare tokenizer identity, vocabulary, and special-token IDs across all stages",
        ),
        ("change_tokenizer", "collect_more_evidence"),
        "Recorded compatibility evidence indicates a tokenizer mismatch.",
    ),
    RuleSpec(
        "checkpoint_integrity",
        frozenset(
            {
                "checkpoint_hash_mismatch",
                "checkpoint_corrupt",
                "resume_checkpoint_mismatch",
            }
        ),
        frozenset({"checkpoint_verified"}),
        ("verify checkpoint content hash and strict-load result",),
        ("rollback", "collect_more_evidence"),
        "Checkpoint evidence is consistent with corruption or resume mismatch.",
    ),
    RuleSpec(
        "architecture",
        frozenset(
            {
                "architecture_mismatch",
                "checkpoint_architecture_mismatch",
                "strict_loader_contract_failed",
            }
        ),
        frozenset({"architecture_compatible"}),
        ("strict-load the checkpoint against the recorded architecture contract",),
        ("change_model", "collect_more_evidence"),
        "Recorded loader evidence indicates a possible architecture-contract mismatch.",
    ),
    RuleSpec(
        "optimizer_or_scheduler",
        frozenset({"optimizer_pathology", "scheduler_misconfigured"}),
        frozenset({"optimizer_stable"}),
        ("inspect optimizer state, learning-rate trace, and scheduler boundaries",),
        ("change_training_recipe", "collect_more_evidence"),
        "Training evidence is consistent with an optimizer or scheduler problem.",
    ),
    RuleSpec(
        "numerical_instability",
        frozenset({"non_finite_loss", "non_finite_gradient", "numerical_overflow"}),
        frozenset({"numerically_stable"}),
        ("reproduce the first non-finite step with numeric telemetry",),
        ("change_training_recipe", "rollback", "stop"),
        "Recorded non-finite values indicate numerical instability.",
    ),
    RuleSpec(
        "undertraining",
        frozenset({"undertraining_detected", "training_budget_exhausted"}),
        frozenset({"training_sufficient"}),
        ("compare learning curves against a controlled longer run",),
        ("resume_training", "change_training_recipe"),
        "Explicit learning-curve evidence is consistent with undertraining.",
    ),
    RuleSpec(
        "overfitting",
        frozenset({"overfitting_detected", "train_eval_gap"}),
        frozenset({"no_overfitting"}),
        ("repeat evaluation on frozen held-out evidence",),
        ("change_training_recipe", "replace_or_mix_dataset"),
        "Explicit train/eval evidence is consistent with overfitting.",
    ),
    RuleSpec(
        "decoding",
        frozenset({"decoding_mismatch", "decoding_artifact"}),
        frozenset({"decoding_verified"}),
        ("evaluate the same checkpoint under identical recorded decoding settings",),
        ("collect_more_evidence",),
        "Recorded comparison evidence is consistent with a decoding artifact.",
    ),
    RuleSpec(
        "model_family_limitation",
        frozenset({"model_family_limitation_detected"}),
        frozenset({"model_family_adequate"}),
        ("run a controlled model-family comparison with all other variables fixed",),
        ("change_model", "branch"),
        "Controlled evidence may indicate a model-family limitation.",
    ),
)


def build_hypotheses(
    evidence: list[NormalizedEvidence],
    policy: DiagnosisPolicy,
    eval_integrity_verified: bool,
) -> list[DiagnosticHypothesis]:
    hypotheses: list[DiagnosticHypothesis] = []
    for rule in _RULES:
        supporting = [
            item
            for item in evidence
            if rule.supporting_signals.intersection(item.signals)
        ]
        if not supporting:
            continue
        contradicting = [
            item
            for item in evidence
            if rule.contradicting_signals.intersection(item.signals)
        ]
        support_by_ref = _best_by_ref(supporting)
        contradiction_by_ref = _best_by_ref(contradicting)
        score = _confidence(
            list(support_by_ref.values()), list(contradiction_by_ref.values()), policy
        )
        if not eval_integrity_verified and rule.domain not in {
            "evaluation_integrity",
            "runtime_or_system",
        }:
            score = min(score, policy.unverified_eval_downstream_confidence_ceiling)
        supporting_refs = sorted(support_by_ref)
        contradicting_refs = sorted(contradiction_by_ref)
        digest = hashlib.sha256(
            f"{rule.domain}|{'|'.join(supporting_refs)}|{'|'.join(contradicting_refs)}".encode()
        ).hexdigest()[:16]
        hypotheses.append(
            DiagnosticHypothesis(
                hypothesis_id=f"hyp-{digest}",
                failure_domain=rule.domain,
                summary=rule.summary,
                supporting_evidence_refs=supporting_refs,
                contradicting_evidence_refs=contradicting_refs,
                required_tests=list(rule.required_tests),
                recommended_intervention_kinds=list(rule.interventions),
                confidence=round(score, 4),
                metadata={
                    "statement_type": "hypothesis",
                    "causation_claimed": False,
                    "distinct_supporting_sources": len(support_by_ref),
                    "distinct_contradicting_sources": len(contradiction_by_ref),
                    "eval_integrity_confidence_cap_applied": (
                        not eval_integrity_verified
                        and rule.domain
                        not in {"evaluation_integrity", "runtime_or_system"}
                    ),
                },
            )
        )
    order = {domain: index for index, domain in enumerate(DIAGNOSTIC_DOMAIN_ORDER)}
    hypotheses.sort(
        key=lambda item: (
            -item.confidence,
            order.get(item.failure_domain, 999),
            item.hypothesis_id,
        )
    )
    return hypotheses


def inconclusive_hypothesis(missing_evidence: list[str]) -> DiagnosticHypothesis:
    digest = hashlib.sha256(
        "|".join(sorted(missing_evidence)).encode("utf-8")
    ).hexdigest()[:16]
    return DiagnosticHypothesis(
        hypothesis_id=f"hyp-{digest}",
        failure_domain="inconclusive",
        summary="Available evidence does not distinguish a likely failure domain.",
        required_tests=[f"collect {item}" for item in missing_evidence]
        or ["collect independent diagnostic evidence"],
        recommended_intervention_kinds=["collect_more_evidence"],
        confidence=0.0,
        metadata={"statement_type": "hypothesis", "causation_claimed": False},
    )


def _best_by_ref(evidence: list[NormalizedEvidence]) -> dict[str, NormalizedEvidence]:
    best: dict[str, NormalizedEvidence] = {}
    for item in evidence:
        existing = best.get(item.source_ref)
        if existing is None or item.confidence > existing.confidence:
            best[item.source_ref] = item
    return best


def _confidence(
    supporting: list[NormalizedEvidence],
    contradicting: list[NormalizedEvidence],
    policy: DiagnosisPolicy,
) -> float:
    strongest = max(item.confidence for item in supporting)
    score = 0.35 + (0.50 * strongest)
    score += policy.additional_independent_evidence_bonus * max(0, len(supporting) - 1)
    if contradicting:
        strongest_contradiction = max(item.confidence for item in contradicting)
        score -= policy.contradiction_penalty * strongest_contradiction
    return max(0.0, min(policy.maximum_hypothesis_confidence, score))
