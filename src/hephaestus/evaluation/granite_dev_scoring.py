"""Strict development-only scoring for the two *different* Granite Evaluator ontologies.

The production certified role-mastery v1 uses six decision labels; historical
Evaluator semantic-invariance research uses eight. These are DISTINCT tasks.
A score from one is never silently transported to the other.

Pure scorer: the caller must authenticate source/model/reference bytes separately.
It never handles protected final items, grants certification, or chooses a repair.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from typing import Mapping, Sequence

# The original frozen role response contract; this must not be changed merely
# to admit research-specific historical states.
REQUIRED_RESPONSE_KEYS = frozenset({
    "decision", "action", "primary_variable", "confidence",
    "evidence_refs", "uncertainties", "rationale",
})
PRODUCTION_ONTOLOGY = "frozen-role-mastery-evaluator.v1"
RESEARCH_ONTOLOGY = "historical-semantic-invariance-evaluator.v1"
PRODUCTION_DECISIONS = frozenset({
    "improved", "regressed", "no_material_change", "scientific_rejection",
    "incomplete_evidence", "evaluation_invalid",
})
RESEARCH_DECISIONS = frozenset({
    "scientific_rejection", "incomplete_evidence", "inconclusive",
    "recheck_required", "regressed", "equivalent", "certification_ready",
    "improved",
})
ONTOLOGY_DECISIONS = {
    PRODUCTION_ONTOLOGY: PRODUCTION_DECISIONS,
    RESEARCH_ONTOLOGY: RESEARCH_DECISIONS,
}
PERMITTED_PARTITIONS = frozenset({"exposed_development", "burned_diagnostic"})


class GraniteScoringError(ValueError):
    """A scientific identity or case-set contract was violated."""


def _unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"JSON constant is not finite: {value}")


def _finite(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(x) for x in value.values())
    if isinstance(value, list):
        return all(_finite(x) for x in value)
    return True


def _parse_generated(raw: object) -> tuple[dict[str, object] | None, str | None]:
    if not isinstance(raw, str) or not raw or len(raw) > 16384:
        return None, "empty_or_oversized_output"
    try:
        result = json.loads(raw, object_pairs_hook=_unique, parse_constant=_reject_constant)
    except (ValueError, TypeError, OverflowError):
        return None, "invalid_strict_json"
    if not isinstance(result, dict) or set(result) != REQUIRED_RESPONSE_KEYS:
        return None, "invalid_envelope"
    if not _finite(result):
        return None, "invalid_strict_json"
    if (type(result["confidence"]) not in (int, float)
            or not 0 <= result["confidence"] <= 1):
        return None, "invalid_envelope"
    if (not isinstance(result["action"], str)
            or not isinstance(result["primary_variable"], str)
            or not isinstance(result["rationale"], str)
            or not result["rationale"].strip()
            or len(result["rationale"]) > 2000):
        return None, "invalid_envelope"
    if not isinstance(result["decision"], str):
        return None, "invalid_envelope"
    for key in ("evidence_refs", "uncertainties"):
        arr = result[key]
        if not isinstance(arr, list) or not all(isinstance(x, str) for x in arr):
            return None, "invalid_envelope"
    return result, None


def _map_cases(items: Sequence[Mapping[str, object]], *,
               expected_keys: frozenset[str], ontology: str,
               kind: str) -> dict[str, Mapping[str, object]]:
    mapped: dict[str, Mapping[str, object]] = {}
    if not isinstance(items, (list, tuple)) or not items:
        raise GraniteScoringError(f"{kind} requires nonempty complete rows")
    for row in items:
        if not isinstance(row, Mapping) or set(row) != expected_keys:
            raise GraniteScoringError(f"{kind} row has missing/unexpected fields")
        case_id = row["case_id"]
        if not isinstance(case_id, str) or not case_id or len(case_id) > 256:
            raise GraniteScoringError(f"{kind} case ID invalid")
        if case_id in mapped:
            raise GraniteScoringError(f"{kind} duplicate case ID")
        if row["ontology"] != ontology:
            raise GraniteScoringError("cross-ontology scoring is forbidden")
        if row["partition"] not in PERMITTED_PARTITIONS:
            raise GraniteScoringError("only exposed development/burned diagnostic data admitted")
        mapped[case_id] = row
    return mapped


def score_development_decisions(
    reference_rows: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
    *, ontology: str,
) -> dict[str, object]:
    """Score strict generated decisions against independently supplied references.

    The source/labels MUST be independently byte-verified by a caller. This
    function does not establish that reference labels are scientifically true,
    or that model outputs/provenance were authentic; its returned claim flags
    are deliberately negative until those external checks are performed.
    """
    if ontology not in ONTOLOGY_DECISIONS:
        raise GraniteScoringError("unrecognized Evaluator ontology")
    labels = ONTOLOGY_DECISIONS[ontology]
    references = _map_cases(
        reference_rows, expected_keys=frozenset({
            "case_id", "ontology", "partition", "expected_decision",
        }), ontology=ontology, kind="reference",
    )
    predictions = _map_cases(
        prediction_rows, expected_keys=frozenset({
            "case_id", "ontology", "partition", "raw_generation",
        }), ontology=ontology, kind="prediction",
    )
    if references.keys() != predictions.keys():
        raise GraniteScoringError("case completeness mismatch: no missing or extra generations allowed")
    if any(
        reference["partition"] != predictions[cid]["partition"]
        for cid, reference in references.items()
    ):
        raise GraniteScoringError("reference/prediction partition mismatch")
    for ref in references.values():
        if not isinstance(ref["expected_decision"], str) or ref["expected_decision"] not in labels:
            raise GraniteScoringError("reference decision is outside declared ontology")

    confusion: dict[str, Counter[str]] = {key: Counter() for key in sorted(labels)}
    reasons: Counter[str] = Counter()
    correct = 0
    for cid in sorted(references):
        expected = str(references[cid]["expected_decision"])
        parsed, error = _parse_generated(predictions[cid]["raw_generation"])
        decision = str(parsed["decision"]) if parsed is not None else "<invalid>"
        if decision not in labels:
            error = error or "decision_outside_ontology"
            decision = "<invalid>"
        if error:
            reasons[error] += 1
        if decision == expected:
            correct += 1
        confusion[expected][decision] += 1

    total = len(references)
    return {
        "schema": "granite-evaluator-exposed-development-scoring.v1",
        "ontology": ontology,
        "partition_scope": "exposed_development_or_burned_diagnostic_only",
        "count": total,
        "decision_correct": correct,
        "decision_accuracy": correct / total,
        "per_state": {
            key: {
                "total": sum(confusion[key].values()),
                "correct": confusion[key][key],
                "accuracy": (confusion[key][key] / sum(confusion[key].values())
                             if sum(confusion[key].values()) else None),
                "confusion": dict(sorted(confusion[key].items())),
            }
            for key in sorted(labels)
        },
        "invalid_reasons": dict(sorted(reasons.items())),
        "caller_source_authentication_required": True,
        "reference_truth_independently_established": False,
        "model_execution_provenance_verified": False,
        "historical_exposed_only": True,
        "can_certify": False,
        "can_promote": False,
    }


def compare_matched_development(
    reference_rows: Sequence[Mapping[str, object]],
    parent_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    *, reference_ontology: str, parent_ontology: str, candidate_ontology: str,
) -> dict[str, object]:
    """Compare matching tasks only. Never translate historical research states."""
    if reference_ontology != parent_ontology or parent_ontology != candidate_ontology:
        raise GraniteScoringError("different ontology cannot be a matched model comparison")
    before = score_development_decisions(reference_rows, parent_rows, ontology=reference_ontology)
    after = score_development_decisions(reference_rows, candidate_rows, ontology=reference_ontology)
    return {
        "schema": "granite-evaluator-exposed-development-comparison.v1",
        "ontology": reference_ontology,
        "parent": before,
        "candidate": after,
        "absolute_decision_accuracy_gain": after["decision_accuracy"] - before["decision_accuracy"],
        "research_only": True,
        "independent_confirmation": False,
        "promotion_eligible": False,
    }
