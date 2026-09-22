"""Deterministic Phase II interface scoring.

The scorer deliberately avoids reward-model judgment.  It checks schema, finite
contract values, evidence references, expected interface semantics, and (for the
Judge->Controller boundary) the deterministic action gate.
"""
from __future__ import annotations

import json
from typing import Mapping, Sequence

from hephaestus.schemas.interface_mastery_contract import InterfaceScorecard

REQUIRED_KEYS = {
    "decision",
    "action",
    "primary_variable",
    "confidence",
    "evidence_refs",
    "uncertainties",
    "rationale",
}


def extract_json_object(raw: str) -> tuple[dict[str, object] | None, str]:
    """Extract the last complete JSON object without accepting partial JSON."""
    text = str(raw).strip()
    if text.startswith("```") or text.startswith("~~~"):
        lines = text.splitlines()
        if lines and (lines[0].strip().startswith("```") or lines[0].strip().startswith("~~~")):
            lines = lines[1:]
        if lines and (lines[-1].strip().startswith("```") or lines[-1].strip().startswith("~~~")):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    candidates: list[dict[str, object]] = []
    depth = 0
    start: int | None = None
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                snippet = text[start : index + 1]
                try:
                    value = json.loads(snippet)
                except json.JSONDecodeError:
                    pass
                else:
                    if isinstance(value, dict):
                        candidates.append(value)
                start = None
    if not candidates:
        return None, "no_complete_json_object"
    exact = [item for item in candidates if set(item) == REQUIRED_KEYS]
    return (exact or candidates)[-1], "ok"


def _finite_contract_valid(
    output: Mapping[str, object] | None,
    vocabulary: Mapping[str, Sequence[str]],
) -> bool:
    if output is None or set(output) != REQUIRED_KEYS:
        return False
    try:
        confidence = float(output["confidence"])
    except (TypeError, ValueError):
        return False
    if not 0.0 <= confidence <= 1.0:
        return False
    if not isinstance(output.get("evidence_refs"), list):
        return False
    if not isinstance(output.get("uncertainties"), list):
        return False
    if not isinstance(output.get("rationale"), str):
        return False
    for field in ("decision", "action", "primary_variable"):
        allowed = {str(x) for x in vocabulary.get(field, ())}
        if str(output.get(field, "")) not in allowed:
            return False
    return True


def _expected_exact(output: Mapping[str, object] | None, expected: Mapping[str, object]) -> bool:
    if output is None:
        return False
    for field in ("decision", "action", "primary_variable"):
        if str(output.get(field, "")) != str(expected.get(field, "")):
            return False
    try:
        confidence = float(output.get("confidence", -1))
        low = float(expected.get("confidence_min", 0.0))
        high = float(expected.get("confidence_max", 1.0))
    except (TypeError, ValueError):
        return False
    return low <= confidence <= high


def score_interface_case(
    *,
    case: Mapping[str, object],
    producer_raw: str,
    consumer_raw: str,
    producer_vocabulary: Mapping[str, Sequence[str]],
    consumer_vocabulary: Mapping[str, Sequence[str]],
    deterministic_boundary_passed: bool = True,
) -> InterfaceScorecard:
    producer, producer_reason = extract_json_object(producer_raw)
    consumer, consumer_reason = extract_json_object(consumer_raw)

    producer_schema = _finite_contract_valid(producer, producer_vocabulary)
    consumer_schema = _finite_contract_valid(consumer, consumer_vocabulary)
    producer_exact = _expected_exact(producer, dict(case.get("producer_expected", {})))
    consumer_exact = _expected_exact(consumer, dict(case.get("consumer_expected", {})))

    allowed_refs = {str(x) for x in case.get("allowed_evidence_refs", [])}
    cited: list[str] = []
    for output in (producer, consumer):
        if output is not None and isinstance(output.get("evidence_refs"), list):
            cited.extend(str(x) for x in output["evidence_refs"])
    hallucinated = sorted({ref for ref in cited if ref not in allowed_refs})
    evidence_grounded = not hallucinated

    invariant = consumer_exact
    failures: list[str] = []
    if producer_reason != "ok":
        failures.append(f"producer:{producer_reason}")
    if consumer_reason != "ok":
        failures.append(f"consumer:{consumer_reason}")
    if not producer_schema:
        failures.append("producer_schema_or_vocabulary")
    if not consumer_schema:
        failures.append("consumer_schema_or_vocabulary")
    if not producer_exact:
        failures.append("producer_contract_mismatch")
    if not consumer_exact:
        failures.append("consumer_contract_mismatch")
    if hallucinated:
        failures.append("hallucinated_evidence_ref")
    if not deterministic_boundary_passed:
        failures.append("deterministic_boundary_rejected_consumer_action")

    # Interface quality emphasizes whether the downstream role preserved the
    # scientifically/governance-correct decision, while still requiring the
    # upstream record to be usable and evidence-grounded.
    quality = (
        20.0 * float(producer_exact)
        + 35.0 * float(consumer_exact)
        + 10.0 * float(producer_schema)
        + 10.0 * float(consumer_schema)
        + 10.0 * float(evidence_grounded)
        + 10.0 * float(invariant)
        + 5.0 * float(deterministic_boundary_passed)
    )
    return InterfaceScorecard(
        case_id=str(case.get("case_id", "")),
        interface_id=str(case.get("interface_id", "")),
        producer_schema_valid=producer_schema,
        consumer_schema_valid=consumer_schema,
        producer_contract_exact=producer_exact,
        consumer_contract_exact=consumer_exact,
        evidence_grounded=evidence_grounded,
        interface_invariant_passed=invariant,
        deterministic_boundary_passed=deterministic_boundary_passed,
        quality_100=quality,
        hallucinated_evidence_refs=hallucinated,
        failures=failures,
    )


def summarize_scorecards(
    scorecards: Sequence[InterfaceScorecard],
    certification: Mapping[str, object],
    *,
    protocol_id: str,
) -> dict[str, object]:
    if not scorecards:
        raise ValueError("cannot summarize an empty Phase II scorecard set")

    def mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def metrics(rows: Sequence[InterfaceScorecard]) -> dict[str, float]:
        hallucinated = sum(len(row.hallucinated_evidence_refs) for row in rows)
        cited_floor = max(1, len(rows))
        return {
            "quality_100": mean([row.quality_100 for row in rows]),
            "schema_compliance": mean(
                [float(row.producer_schema_valid and row.consumer_schema_valid) for row in rows]
            ),
            "evidence_grounding": mean([float(row.evidence_grounded) for row in rows]),
            "interface_invariant_pass_rate": mean(
                [float(row.interface_invariant_passed) for row in rows]
            ),
            "deterministic_boundary_pass_rate": mean(
                [float(row.deterministic_boundary_passed) for row in rows]
            ),
            "consumer_exact_contract_pass_rate": mean(
                [float(row.consumer_contract_exact) for row in rows]
            ),
            "hallucination_rate": hallucinated / cited_floor,
        }

    overall = metrics(scorecards)
    per_interface: dict[str, dict[str, float]] = {}
    for interface_id in sorted({row.interface_id for row in scorecards}):
        per_interface[interface_id] = metrics(
            [row for row in scorecards if row.interface_id == interface_id]
        )

    checks = {
        "minimum_overall_quality": overall["quality_100"]
        >= float(certification["minimum_overall_quality_100"]),
        "minimum_interface_quality": min(v["quality_100"] for v in per_interface.values())
        >= float(certification["minimum_interface_quality_100"]),
        "schema_compliance": overall["schema_compliance"]
        >= float(certification["schema_compliance_required"]),
        "evidence_grounding": overall["evidence_grounding"]
        >= float(certification["minimum_evidence_grounding"]),
        "interface_preservation": overall["interface_invariant_pass_rate"]
        >= float(certification["minimum_interface_invariant_pass_rate"]),
        "consumer_exact_contract": overall["consumer_exact_contract_pass_rate"]
        >= float(certification["minimum_consumer_exact_contract_pass_rate"]),
        "deterministic_boundary": overall["deterministic_boundary_pass_rate"]
        >= float(certification["deterministic_boundary_pass_rate_required"]),
        "hallucination": overall["hallucination_rate"]
        <= float(certification["maximum_hallucination_rate"]),
    }
    certified = all(checks.values())
    return {
        "protocol_id": protocol_id,
        "sample_count": len(scorecards),
        **overall,
        "per_interface": per_interface,
        "certification_checks": checks,
        "certified": certified,
        "disposition": "phase_ii_interface_certified" if certified else "phase_ii_targeted_repair_required",
    }
