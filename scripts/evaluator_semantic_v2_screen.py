"""Bounded, balanced development screen for D; no new certification claims.

This is a fixed subset of the existing frozen development diagnostic, never a
new test partition. The full original 256-case diagnostic stays immutable.
Screening can only *reject* an unreachable checkpoint, never certify it.
"""
from collections import Counter
import hashlib
import json

STATES = (
    "scientific_rejection", "incomplete_evidence", "improved", "regressed",
    "equivalent", "inconclusive", "recheck_required", "certification_ready",
)
ROBUST = {"scientific_rejection", "incomplete_evidence", "regressed", "certification_ready"}
SCREEN_PER_STATE = 8
FULL_PER_STATE = 32
FULL_TOTAL = 256
MIN_OVERALL_CORRECT = 244  # ceil(0.95 * 256)
MIN_WEAK_CORRECT = 29      # ceil(0.90 * 32)
MIN_ROBUST_CORRECT = 31    # ceil(0.95 * 32)


def screen_cases(full_cases):
    """Exactly eight cases per state, identity-stable and independent of outputs."""
    cases = list(full_cases)
    if len(cases) != FULL_TOTAL or len({r["case_id"] for r in cases}) != FULL_TOTAL:
        raise ValueError("the original frozen 256-case diagnostic is required")
    groups = {s: [r for r in cases if r["expected"]["decision"] == s] for s in STATES}
    if any(len(v) != FULL_PER_STATE for v in groups.values()):
        raise ValueError("frozen diagnostic has incorrect state distribution")
    for state, rows in groups.items():
        rows.sort(key=lambda r: hashlib.sha256(("evaluator-D-screen-v1:" + r["case_id"]).encode()).hexdigest())
    selected = [groups[state][i] for i in range(SCREEN_PER_STATE) for state in STATES]
    assert len(selected) == SCREEN_PER_STATE * len(STATES)
    return selected


def screen_hash(selected):
    return hashlib.sha256(json.dumps([r["case_id"] for r in selected], separators=(",", ":")).encode()).hexdigest()


def verdict(selected, rows):
    """Reject only when perfect remaining answers cannot satisfy frozen accuracy gates."""
    selected = list(selected)
    rows = list(rows)
    required = {r["case_id"]: r for r in selected}
    if len(selected) != 64 or len(rows) != 64 or len({r["case_id"] for r in rows}) != 64:
        raise ValueError("screen requires exactly 64 unique outputs")
    if set(r["case_id"] for r in rows) != set(required):
        raise ValueError("screen identities do not match the frozen selection")
    correct = Counter()
    count = Counter()
    for r in rows:
        state = required[r["case_id"]]["expected"]["decision"]
        count[state] += 1
        correct[state] += bool(r["score"]["decision_exact"])
    if any(count[s] != SCREEN_PER_STATE for s in STATES):
        raise ValueError("unbalanced screen scores")
    observed = sum(correct.values())
    upper = observed + (FULL_TOTAL - len(rows))
    reasons = []
    if upper < MIN_OVERALL_CORRECT:
        reasons.append("overall_95pct_mathematically_unreachable")
    per_state = {}
    for state in STATES:
        upper_correct = correct[state] + (FULL_PER_STATE - count[state])
        minimum = MIN_ROBUST_CORRECT if state in ROBUST else MIN_WEAK_CORRECT
        if upper_correct < minimum:
            reasons.append("state_gate_unreachable:" + state)
        per_state[state] = {
            "correct": correct[state], "total": count[state],
            "full_upper_correct": upper_correct, "required_full_correct": minimum,
        }
    return {
        "status": "screen_rejected" if reasons else "screen_gate_reachable",
        "scientific_improvement_demonstrated": False,
        "case_count": len(rows),
        "correct": observed,
        "screen_accuracy": observed / len(rows),
        "full_accuracy_upper_bound": upper / FULL_TOTAL,
        "full_accuracy_gate": MIN_OVERALL_CORRECT / FULL_TOTAL,
        "screen_case_order_sha256": screen_hash(selected),
        "per_state": per_state,
        "unreachable_gates": reasons,
        "screen_is_subset_of_frozen_development": True,
        "screen_is_not_certification": True,
    }
