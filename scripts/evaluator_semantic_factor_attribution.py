"""CPU-only observational semantic-factor attribution for persisted Evaluator outputs.

IMPORTANT: Predictions contain only a final decision, not six independently
reported factors. Minimal latent edits are *counterfactual explanations*, not
proof of what the model perceived or which internal step failed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Mapping

import build_evaluator_semantic_invariance_repair_v1 as frozen
from hephaestus.schemas.evaluator_factors import ScientificFactors, combination_space
from hephaestus.policy.evaluator_semantic_oracle import scientific_state

FIELDS = tuple(ScientificFactors.__dataclass_fields__)
VALID_STATES = frozenset(frozen.STATE_ORDER)
SPACE = combination_space()


def reference_factors(case: Mapping) -> ScientificFactors:
    """Decode only *frozen reference labels*, never a model's latent beliefs."""
    state = case["expected"]["decision"]
    if state not in VALID_STATES or case["semantic_root"] != state:
        raise ValueError("reference state mismatch")
    factors = ScientificFactors(**frozen._canonical_latent(state))
    facts = case["verified_facts"]
    mapping = {
        "gate_pass": "deterministic_gate_passed",
        "complete": "evidence_complete",
        "provenance_valid": "provenance_valid",
        "recheck_satisfied": "recheck_satisfied",
    }
    if any(facts[machine] is not getattr(factors, factor) for factor, machine in mapping.items()):
        raise ValueError("reference factor/machine truth mismatch")
    if scientific_state(factors) != state:
        raise ValueError("reference factor/oracle truth mismatch")
    return factors



def decode_frozen_development_evidence(case: Mapping) -> ScientificFactors:
    """Independently decode known frozen template sentences, not model beliefs.

    This detects contradictory/missing synthetic reference facts. Template
    decodability does not prove natural-language understanding or lack of
    ambiguity outside this exact generated diagnostic.
    """
    bank = frozen.DEV_PHRASES
    categories = {
        "gate_pass":("gate_pass", True), "gate_fail":("gate_pass", False),
        "complete":("complete", True), "incomplete":("complete", False),
        "prov_valid":("provenance_valid", True),
        "prov_invalid":("provenance_valid", False),
        "effect_positive":("effect", "positive"),
        "effect_neutral":("effect", "neutral"),
        "effect_negative":("effect", "negative"),
        "variance_stable":("variance_stable", True),
        "variance_high":("variance_stable", False),
        "recheck_yes":("recheck_satisfied", True),
        "recheck_no":("recheck_satisfied", False),
    }
    vocabulary = {}
    for category, value in categories.items():
        for phrase in bank[category]:
            if phrase in vocabulary:
                raise ValueError("ambiguous reference vocabulary")
            vocabulary[phrase] = value
    extracted = {}
    evidence = case["evidence"]
    if len(evidence) != 6:
        raise ValueError("frozen development case does not have six facts")
    for record in evidence:
        lookup = vocabulary.get(record["fact"])
        if lookup is None:
            raise ValueError("unrecognized frozen development evidence sentence")
        key, value = lookup
        if key in extracted:
            raise ValueError("contradictory/repeated scientific factor evidence")
        extracted[key] = value
    if set(extracted) != set(FIELDS):
        raise ValueError("incomplete scientific factor evidence")
    return ScientificFactors(**extracted)


def nearest_counterfactuals(factors: ScientificFactors, predicted_state: str) -> dict:
    """Minimal oracle input edits making a wrong decision logically possible."""
    if predicted_state not in VALID_STATES:
        return dict(known_state=False, minimum_factor_edits=None, necessary_factors=[],
                    possible_factors=[], alternative_minimal_explanations=0)
    distances = []
    for candidate in SPACE:
        if scientific_state(candidate) != predicted_state:
            continue
        changed = frozenset(k for k in FIELDS if getattr(factors, k) != getattr(candidate, k))
        distances.append(changed)
    if not distances:
        raise ValueError("oracle has no support for predicted state")
    least = min(map(len, distances))
    minimal = sorted({d for d in distances if len(d) == least}, key=lambda d: sorted(d))
    return dict(known_state=True, minimum_factor_edits=least,
                necessary_factors=sorted(set.intersection(*(set(x) for x in minimal))),
                possible_factors=sorted(set.union(*(set(x) for x in minimal))),
                alternative_minimal_explanations=len(minimal))


def decision_from_persisted_row(row: Mapping) -> str:
    score = row["score"]
    parsed = score.get("parsed")
    if not isinstance(parsed, dict):
        return "<invalid>"
    state = parsed.get("decision")
    return state if isinstance(state, str) and state in VALID_STATES else "<invalid>"


def analyze_observed(cases: list[dict], rows: list[dict], label: str) -> dict:
    """Aggregate *only observed* and identity-verified samples."""
    idx = {c["case_id"]: c for c in cases}
    ids = [r["case_id"] for r in rows]
    if not rows or len(ids) != len(set(ids)) or not set(ids) <= set(idx):
        raise ValueError("observed evidence has missing/duplicate/unknown sample identity")
    confusion = Counter()
    correctness = Counter()
    predictions = Counter()
    by_state = defaultdict(lambda: Counter())
    by_handoff = defaultdict(lambda: Counter())
    by_factor_value = defaultdict(lambda: Counter())
    closest = Counter()
    necessary = Counter()
    possible = Counter()
    ambiguous = 0
    invalid = 0
    for row in rows:
        case = idx[row["case_id"]]
        factors = reference_factors(case)
        expected = case["expected"]["decision"]
        predicted = decision_from_persisted_row(row)
        right = expected == predicted
        if right != bool(row["score"]["decision_exact"]):
            raise ValueError("reported score disagrees with output")
        predictions[predicted] += 1
        confusion[f"{expected} -> {predicted}"] += 1
        correctness["correct" if right else "wrong"] += 1
        st = by_state[expected]
        st["total"] += 1
        st["correct"] += int(right)
        strat = "advisory" if case.get("upstream_output") is not None else "direct"
        by_handoff[strat]["total"] += 1
        by_handoff[strat]["correct"] += int(right)
        for factor in FIELDS:
            value = getattr(factors, factor)
            pair = (factor, str(value))
            by_factor_value[pair]["total"] += 1
            by_factor_value[pair]["correct"] += int(right)
        if not right:
            hypothesis = nearest_counterfactuals(factors, predicted)
            if hypothesis["minimum_factor_edits"] is None:
                invalid += 1
                continue
            closest[str(hypothesis["minimum_factor_edits"])] += 1
            for factor in hypothesis["necessary_factors"]:
                necessary[factor] += 1
            for factor in hypothesis["possible_factors"]:
                possible[factor] += 1
            ambiguous += hypothesis["alternative_minimal_explanations"] > 1
    state_result = {s: dict(total=by_state[s]["total"], correct=by_state[s]["correct"]) for s in frozen.STATE_ORDER}
    return {
        "label": label, "observed_cases": len(rows), "correct": correctness["correct"],
        "wrong": correctness["wrong"], "observed_accuracy": correctness["correct"]/len(rows),
        "state_counts": state_result,
        "predicted_state_counts": dict(sorted(predictions.items())),
        "decision_confusion": dict(sorted(confusion.items())),
        "by_advisory_presence": {s:dict(by_handoff[s]) for s in ("direct", "advisory")},
        "by_reference_factor_value": {
            factor: {value:dict(by_factor_value[(factor,value)])
                     for value in (("positive","neutral","negative") if factor=="effect" else ("True","False"))}
            for factor in FIELDS
        },
        "minimal_counterfactual_factor_edits": dict(sorted(closest.items())),
        "factors_necessary_in_every_minimal_explanation": dict(sorted(necessary.items())),
        "factors_possible_in_at_least_one_minimal_explanation": dict(sorted(possible.items())),
        "ambiguous_minimal_explanations": ambiguous,
        "invalid_or_nonstate_predictions": invalid,
        "actual_factor_extraction_error_observed": False,
        "actual_precedence_error_observed": False,
        "attribution_identifiability": "Unidentifiable: existing generations report only final state, not independently extracted factors.",
    }


def paired_comparison(reference: list[dict], candidate: list[dict], cases: list[dict], label: str) -> dict:
    idx = {c["case_id"] for c in cases}
    ref = {r["case_id"]: r for r in reference}
    cand = {r["case_id"]: r for r in candidate}
    if len(ref)!=len(reference) or len(cand)!=len(candidate) or not cand or not set(cand) <= set(ref) <= idx:
        raise ValueError("unmatched or duplicated paired evidence")
    pair = Counter()
    for caseid,row in cand.items():
        truth = next(c["expected"]["decision"] for c in cases if c["case_id"] == caseid)
        a = decision_from_persisted_row(ref[caseid]) == truth
        b = decision_from_persisted_row(row) == truth
        pair[("candidate_" + str(b), "reference_" + str(a))] += 1
    return {"label":label,"identical_case_count":len(cand),
            "reference_correct":sum(decision_from_persisted_row(ref[k])==
                                    next(c["expected"]["decision"] for c in cases if c["case_id"]==k) for k in cand),
            "candidate_correct":sum(decision_from_persisted_row(cand[k])==
                                    next(c["expected"]["decision"] for c in cases if c["case_id"]==k) for k in cand),
            "paired_outcomes":{"_".join(k):v for k,v in sorted(pair.items())},
            "comparison_is_same_case_not_full_diagnostic":len(cand)<len(cases)}


def score_explicit_factor_probe(expected: ScientificFactors, model_record: Mapping) -> dict:
    """Optional *future* probe scoring: never infer factors from final decision alone.

    This different output contract requires separate inference/approval; useful
    for distinguishing explicit extraction vs oracle-application errors, but
    cannot prove hidden internal reasoning and may change model behavior.
    """
    if not isinstance(model_record, Mapping) or set(model_record) != {"factors", "decision"}:
        return {"status":"invalid_probe_contract"}
    raw = model_record.get("factors")
    if not isinstance(raw, Mapping) or set(raw) != set(FIELDS):
        return {"status":"invalid_probe_contract"}
    try:
        reported = ScientificFactors(**raw)
    except (TypeError, ValueError):
        return {"status":"invalid_probe_contract"}
    got = model_record.get("decision")
    if not isinstance(got,str) or got not in VALID_STATES:
        return {"status":"invalid_probe_contract"}
    expected_state = scientific_state(expected)
    reported_oracle = scientific_state(reported)
    extraction_correct = reported == expected
    decision_correct = got == expected_state
    internally_coherent = got == reported_oracle
    if extraction_correct and internally_coherent:
        status = "explicitly_consistent_correct"
    elif extraction_correct and not internally_coherent:
        status = "reported_precedence_mismatch"
    elif not extraction_correct and internally_coherent:
        status = "reported_factor_mismatch_with_coherent_decision"
    else:
        status = "compound_or_inconsistent_mismatch"
    return {
        "status":status,"expected_state":expected_state,
        "reported_factor_correctness":{k:getattr(expected,k)==getattr(reported,k) for k in FIELDS},
        "all_reported_factors_correct":extraction_correct,
        "reported_decision_correct":decision_correct,
        "reported_decision_consistent_with_reported_factors":internally_coherent,
        "oracle_from_reported_factors":reported_oracle,
        "limitations":"Explicit probe outputs are model reports, not proof of internal cognitive stages; contract may change behavior.",
    }
