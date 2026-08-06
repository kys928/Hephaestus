"""Evidence-backed deterministic failure classification."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from hephaestus.recovery.models import (
    FailureClassification,
    NormalizedFailureEvidence,
    RecoveryRequest,
)
from hephaestus.recovery.normalization import evidence_fingerprint
from hephaestus.schemas.contract_common import ContractIssue


@dataclass(frozen=True, slots=True)
class ClassificationRule:
    category: str
    domain: str
    supporting_signals: frozenset[str]
    contradicting_signals: frozenset[str]
    retryability: str
    safe_to_automate: bool
    requires_approval: bool = False


_RULES = (
    ClassificationRule(
        "transient_provider_outage",
        "runtime_or_system",
        frozenset({"provider_unavailable", "transient_provider_outage"}),
        frozenset({"provider_healthy"}),
        "retryable",
        True,
    ),
    ClassificationRule(
        "transient_network_download_interruption",
        "runtime_or_system",
        frozenset({"network_interruption", "download_interrupted"}),
        frozenset(),
        "retryable",
        True,
    ),
    ClassificationRule(
        "worker_lease_loss",
        "runtime_or_system",
        frozenset(
            {
                "lease_expired",
                "heartbeat_missing",
                "worker_process_missing",
                "duplicate_ownership",
                "late_completion",
                "stale_result",
            }
        ),
        frozenset(),
        "conditional",
        True,
    ),
    ClassificationRule(
        "process_crash",
        "runtime_or_system",
        frozenset({"process_crash"}),
        frozenset(),
        "conditional",
        True,
    ),
    ClassificationRule(
        "explicit_cancellation",
        "runtime_or_system",
        frozenset({"explicit_cancellation"}),
        frozenset(),
        "not_retryable",
        False,
    ),
    ClassificationRule(
        "operator_interruption",
        "runtime_or_system",
        frozenset({"operator_interruption"}),
        frozenset(),
        "conditional",
        False,
        True,
    ),
    ClassificationRule(
        "out_of_memory",
        "runtime_or_system",
        frozenset({"out_of_memory"}),
        frozenset(),
        "conditional",
        False,
        True,
    ),
    ClassificationRule(
        "resource_budget_exhaustion",
        "runtime_or_system",
        frozenset({"budget_exhausted"}),
        frozenset(),
        "not_retryable",
        False,
        True,
    ),
    ClassificationRule(
        "data_loader_failure",
        "runtime_or_system",
        frozenset({"data_loader_failure"}),
        frozenset(),
        "conditional",
        True,
    ),
    ClassificationRule(
        "malformed_or_contaminated_data",
        "data_quality",
        frozenset({"malformed_data", "contamination_detected"}),
        frozenset(),
        "not_retryable",
        False,
        True,
    ),
    ClassificationRule(
        "tokenizer_incompatibility",
        "tokenizer",
        frozenset({"tokenizer_incompatible"}),
        frozenset(),
        "not_retryable",
        False,
        True,
    ),
    ClassificationRule(
        "model_checkpoint_incompatibility",
        "architecture",
        frozenset({"model_checkpoint_incompatible"}),
        frozenset(),
        "not_retryable",
        False,
        True,
    ),
    ClassificationRule(
        "checkpoint_corruption",
        "checkpoint_integrity",
        frozenset({"checkpoint_corrupt", "checkpoint_hash_mismatch"}),
        frozenset({"checkpoint_verified"}),
        "not_retryable",
        False,
        True,
    ),
    ClassificationRule(
        "missing_checkpoint_evidence",
        "checkpoint_integrity",
        frozenset({"checkpoint_missing"}),
        frozenset({"checkpoint_verified"}),
        "conditional",
        False,
    ),
    ClassificationRule(
        "resume_token_corruption",
        "checkpoint_integrity",
        frozenset({"resume_token_corrupt"}),
        frozenset({"resume_token_valid"}),
        "not_retryable",
        False,
    ),
    ClassificationRule(
        "missing_metrics",
        "evaluation_integrity",
        frozenset({"metrics_missing"}),
        frozenset({"evaluation_complete"}),
        "retryable",
        True,
    ),
    ClassificationRule(
        "incomplete_evaluation",
        "evaluation_integrity",
        frozenset({"evaluation_incomplete"}),
        frozenset({"evaluation_complete"}),
        "retryable",
        True,
    ),
    ClassificationRule(
        "deterministic_regression",
        "evaluation_integrity",
        frozenset({"deterministic_regression"}),
        frozenset(),
        "not_retryable",
        False,
        True,
    ),
    ClassificationRule(
        "high_evaluation_variance",
        "evaluation_integrity",
        frozenset({"high_evaluation_variance"}),
        frozenset({"evaluation_variance_bounded"}),
        "conditional",
        True,
    ),
    ClassificationRule(
        "replay_failure",
        "launch_or_reproducibility",
        frozenset({"replay_failed"}),
        frozenset({"replay_verified"}),
        "conditional",
        False,
    ),
    ClassificationRule(
        "policy_or_approval_block",
        "inconclusive",
        frozenset({"policy_blocked", "approval_required"}),
        frozenset({"approval_verified"}),
        "not_retryable",
        False,
        True,
    ),
    ClassificationRule(
        "invalid_configuration",
        "launch_or_reproducibility",
        frozenset({"invalid_configuration"}),
        frozenset({"configuration_verified"}),
        "not_retryable",
        False,
        True,
    ),
    ClassificationRule(
        "permanent_unsupported_capability",
        "model_family_limitation",
        frozenset({"unsupported_capability"}),
        frozenset(),
        "not_retryable",
        False,
        True,
    ),
    ClassificationRule(
        "poisoned_or_deprecated_lineage",
        "model_family_limitation",
        frozenset({"lineage_poisoned", "lineage_deprecated", "lineage_archived"}),
        frozenset({"lineage_trusted"}),
        "not_retryable",
        False,
        True,
    ),
    ClassificationRule(
        "storage_integrity_failure",
        "runtime_or_system",
        frozenset({"storage_integrity_failure"}),
        frozenset(),
        "conditional",
        True,
    ),
    ClassificationRule(
        "state_persistence_failure",
        "runtime_or_system",
        frozenset({"state_persistence_failure"}),
        frozenset(),
        "conditional",
        True,
    ),
)

_HARD_PRECEDENCE = (
    "deterministic_regression",
    "high_evaluation_variance",
    "checkpoint_corruption",
    "poisoned_or_deprecated_lineage",
    "policy_or_approval_block",
)


def classify_failure(
    request: RecoveryRequest,
    evidence: list[NormalizedFailureEvidence],
) -> FailureClassification:
    candidates: list[tuple[ClassificationRule, float, list[str], list[str]]] = []
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
        score = _score(
            list(support_by_ref.values()), list(contradiction_by_ref.values())
        )
        candidates.append(
            (
                rule,
                score,
                sorted(support_by_ref),
                sorted(contradiction_by_ref),
            )
        )

    candidates.sort(key=lambda row: (-row[1], _rule_index(row[0].category)))
    selected = _select_candidate(candidates)
    fingerprint = evidence_fingerprint(evidence)
    issues: list[ContractIssue] = []
    alternatives = [
        row[0].category
        for row in candidates
        if selected is None or row[0] != selected[0]
    ]

    if selected is None:
        category = "unknown_inconclusive"
        domain = _diagnosis_domain(evidence) or "inconclusive"
        confidence = 0.0 if not candidates else round(candidates[0][1] * 0.5, 4)
        retryability = "unknown"
        evidence_refs = sorted({item.source_ref for item in evidence})
        contradicting_refs: list[str] = []
        safe_to_automate = False
        requires_approval = False
        issues.append(
            ContractIssue(
                code="recovery_classification_inconclusive",
                category="missing_evidence",
                message="Failure evidence does not distinguish one safe recovery category.",
                retryable=True,
                blocking=True,
                evidence_refs=evidence_refs,
            )
        )
    else:
        rule, confidence, evidence_refs, contradicting_refs = selected
        category = rule.category
        domain = rule.domain
        retryability = rule.retryability
        safe_to_automate = rule.safe_to_automate and not contradicting_refs
        requires_approval = rule.requires_approval
        confidence = round(confidence, 4)
        if contradicting_refs:
            issues.append(
                ContractIssue(
                    code="conflicting_recovery_evidence",
                    category="evaluation_inconclusive",
                    message="Contradicting evidence lowers recovery confidence and blocks automatic execution.",
                    retryable=True,
                    blocking=True,
                    evidence_refs=contradicting_refs,
                )
            )

    signature_payload = (
        f"{category}|{request.lineage_id}|{request.experiment_id}|"
        f"{request.run_id}|{request.operation_id}"
    )
    signature = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()
    classification_id = (
        f"rc-{hashlib.sha256((signature + fingerprint).encode()).hexdigest()[:16]}"
    )
    return FailureClassification(
        classification_id=classification_id,
        category=category,
        likely_failure_domain=domain,
        confidence=confidence,
        retryability=retryability,
        evidence_refs=evidence_refs,
        contradicting_evidence_refs=contradicting_refs,
        alternative_categories=alternatives,
        requires_new_evidence=(
            category == "unknown_inconclusive" or bool(contradicting_refs)
        ),
        requires_approval=requires_approval,
        safe_to_automate=safe_to_automate,
        failure_signature=signature,
        evidence_fingerprint=fingerprint,
        issues=issues,
        metadata={
            "classification_basis": "explicit deterministic signals",
            "causation_claimed": False,
            "candidate_count": len(candidates),
        },
    )


def _select_candidate(
    candidates: list[tuple[ClassificationRule, float, list[str], list[str]]],
) -> tuple[ClassificationRule, float, list[str], list[str]] | None:
    if not candidates:
        return None
    for category in _HARD_PRECEDENCE:
        for row in candidates:
            if row[0].category == category and row[1] >= 0.65:
                return row
    leading = candidates[0]
    if leading[1] < 0.6:
        return None
    if len(candidates) > 1 and leading[1] - candidates[1][1] < 0.1:
        return None
    return leading


def _score(
    supporting: list[NormalizedFailureEvidence],
    contradicting: list[NormalizedFailureEvidence],
) -> float:
    strongest = max(item.confidence for item in supporting)
    score = 0.4 + (0.5 * strongest) + min(0.08, 0.04 * (len(supporting) - 1))
    if contradicting:
        score -= 0.45 * max(item.confidence for item in contradicting)
    return max(0.0, min(0.98, score))


def _best_by_ref(
    evidence: list[NormalizedFailureEvidence],
) -> dict[str, NormalizedFailureEvidence]:
    best: dict[str, NormalizedFailureEvidence] = {}
    for item in evidence:
        if (
            item.source_ref not in best
            or item.confidence > best[item.source_ref].confidence
        ):
            best[item.source_ref] = item
    return best


def _diagnosis_domain(evidence: list[NormalizedFailureEvidence]) -> str | None:
    for item in evidence:
        if item.evidence_kind != "diagnosis_report":
            continue
        leading_id = str(item.payload.get("leading_hypothesis_id") or "")
        hypotheses = item.payload.get("hypotheses", [])
        if not isinstance(hypotheses, list):
            continue
        for hypothesis in hypotheses:
            if not isinstance(hypothesis, dict):
                continue
            if leading_id and str(hypothesis.get("hypothesis_id") or "") != leading_id:
                continue
            domain = str(hypothesis.get("failure_domain") or "").strip()
            if domain:
                return domain
    return None


def _rule_index(category: str) -> int:
    for index, rule in enumerate(_RULES):
        if rule.category == category:
            return index
    return 999
