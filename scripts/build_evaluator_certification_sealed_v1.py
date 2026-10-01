#!/usr/bin/env python3
"""Build fresh sealed Evaluator certification/regression evidence.

The frozen evaluator-v3.1-boundary-v1 candidate is evaluated on novel evidence text,
opaque model-visible identifiers, disjoint certification/regression contexts, and a
separate format preflight. Burned V2/V3/V3.1 diagnostic cases are never reused.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

import build_interface_repair_v3_evaluator as base

ROLE = "evaluator"

# Novel evidence wording: none of these rows is copied from the V3/V3.1 training,
# certification, regression, or neutral diagnostic fact templates.
SEALED_FACT_SETS: dict[str, list[list[str]]] = {
    "scientific_rejection": [
        ["The composite score exceeds the reference.", "A mandatory acceptance constraint remains false on every replication.", "The measurement bundle has no missing entries.", "The candidate artifact identity is verified."],
        ["Headline performance is favorable.", "A required pass/fail criterion repeatedly evaluates to failure.", "All scheduled observations are present.", "The tested checkpoint matches the admitted immutable artifact."],
        ["Mean utility rises above baseline.", "One compulsory gate is violated reproducibly rather than stochastically.", "Evidence collection is complete.", "Revision provenance is intact."],
        ["The aggregate metric improves.", "A non-waivable criterion fails in each repeated evaluation.", "No required record is absent.", "Artifact lineage and identity are confirmed."],
    ],
    "incomplete_evidence": [
        ["Several required replicates were never persisted.", "The available observations favor the candidate.", "No observed hard constraint is violated.", "The terminal evidence ledger is incomplete."],
        ["A subset of mandatory measurements is missing.", "Existing scores are promising.", "Recorded deterministic checks pass.", "The evidence requirement has not reached full coverage."],
        ["Acquisition stopped before the planned sample count was reached.", "Partial results exceed baseline.", "Observed gates remain green.", "A final state cannot be established from the truncated bundle."],
        ["The evidence manifest lists unresolved missing records.", "Available measurements look positive.", "No collected item demonstrates a hard failure.", "Scientific closure requires the absent records."],
    ],
    "improved": [
        ["Every required observation is present.", "All immutable constraints pass.", "Repeated measurements show a positive effect beyond the materiality bound.", "Candidate identity is provenance-valid."],
        ["The evidence ledger is complete.", "No frozen gate is violated.", "Low-variance repeats exceed the improvement threshold.", "The evaluated checkpoint is the admitted artifact."],
        ["Required measurements and repeats are complete.", "Deterministic acceptance checks pass.", "The positive delta is reproducible and materially outside the neutral region.", "Artifact provenance is confirmed."],
        ["The full evaluation bundle is available.", "Every hard constraint succeeds.", "Replications establish a material positive gain with stable variance.", "Lineage identity is verified."],
    ],
    "regressed": [
        ["The full evidence bundle is present.", "Repeated measurements show a material negative delta.", "The protocol and artifact identity are valid.", "A previously verified lineage-best checkpoint remains available."],
        ["All planned observations completed.", "The candidate degrades beyond the regression threshold across repeats.", "Evaluation integrity checks pass.", "The prior lineage best is still admissible."],
        ["No required evidence is missing.", "Replicated candidate quality is materially below the reference.", "The evaluation record is provenance-valid.", "A verified fallback checkpoint exists in the same lineage."],
        ["The complete evaluation ledger shows stable degradation.", "The negative effect is larger than the frozen tolerance.", "Protocol identity is intact.", "The lineage-best reference remains valid."],
    ],
    "equivalent": [
        ["All mandatory evidence is present.", "The measured delta remains inside the frozen equivalence band.", "Replicates are stable.", "The artifact identity is verified."],
        ["The evaluation ledger is complete.", "Candidate and reference differ by less than the materiality margin.", "Repeat variance is acceptably low.", "Provenance checks succeed."],
        ["No required record is absent.", "Observed differences are too small to count as either improvement or regression.", "Repeated measurements agree.", "Checkpoint identity is valid."],
        ["The full bundle is available.", "Effect magnitude stays within the predefined neutral interval.", "Replications are consistent.", "Immutable lineage provenance matches."],
    ],
    "inconclusive": [
        ["Observed metrics are favorable.", "The evaluated checkpoint cannot be bound to the admitted immutable revision.", "Candidate identity therefore remains unresolved.", "The reference artifact is still provenance-valid."],
        ["Performance measurements look strong.", "Provenance evidence for the candidate is ambiguous.", "Numerical quality alone cannot establish which artifact was evaluated.", "Baseline identity remains verified."],
        ["The score profile is promising.", "The candidate's immutable lineage proof is missing or contradictory.", "A terminal candidate conclusion is unsupported.", "Reference provenance is intact."],
        ["Measured outcomes favor the candidate.", "Artifact identity cannot be verified against the admitted revision.", "Scientific status must remain unresolved despite the scores.", "The comparison checkpoint is trustworthy."],
    ],
    "recheck_required": [
        ["The candidate mean exceeds the reference.", "Replications disagree beyond the allowed spread.", "Variance risk exceeds the frozen threshold.", "Hard acceptance constraints otherwise pass."],
        ["Average quality is favorable.", "Repeat outcomes are not mutually consistent.", "Observed dispersion requires another bounded measurement.", "No deterministic gate currently fails."],
        ["The estimated effect is positive.", "Run-to-run variation is too large for a stable conclusion.", "The variance rule triggers a recheck.", "Other frozen gates remain satisfied."],
        ["Mean performance improves.", "Replication spread violates the accepted variance envelope.", "Another controlled repeat is required before closure.", "Deterministic constraints otherwise succeed."],
    ],
    "certification_ready": [
        ["The complete certification evidence ledger is present.", "Every frozen acceptance gate passes.", "Required replications are consistent within the variance limit.", "Artifact provenance and the mandatory certification recheck are both satisfied."],
        ["No certification record is missing.", "All deterministic qualification criteria succeed.", "Repeatability requirements are met.", "The admitted artifact is provenance-valid and its required recheck passes."],
        ["The certification bundle has full coverage.", "Each non-negotiable gate is green.", "Replicated results satisfy the consistency policy.", "Immutable identity is verified and the certification recheck is complete."],
        ["All required certification observations exist.", "Frozen gates pass without exception.", "Replication variance remains within the allowed bound.", "Candidate provenance is valid and the required recheck is satisfied."],
    ],
}

FORBIDDEN_IDENTIFIER_TOKENS = (
    "cert", "regress", "reject", "improv", "equiv", "inconcl", "recheck",
    "scientific", "ready", "holdout", "test", "eval", "phase", "split",
)


def _opaque(prefix: str, seed: str, width: int) -> str:
    return prefix + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:width].upper()


def _ordered_facts(decision: str, ci: int, ri: int, variant: int) -> list[str]:
    templates = SEALED_FACT_SETS[decision]
    facts = list(templates[(ci * 7 + ri * 5 + variant) % len(templates)])
    shift = (ci + ri * 3 + variant) % len(facts)
    facts = facts[shift:] + facts[:shift]
    if (ci + ri + variant) % 2:
        facts.reverse()
    return facts


def _case(partition_seed: str, ci: int, ri: int, variant: int) -> dict[str, Any]:
    decision, action, primary = base.ROOTS[ri]
    context = _opaque("ctx_", f"{partition_seed}:context:{ci}", 12)
    case_id = _opaque("C-", f"{partition_seed}:case:{ci}:{ri}:{variant}", 16)
    refs = [
        _opaque("E-", f"{partition_seed}:evidence:{ci}:{ri}:{variant}:{j}", 16)
        for j in range(4)
    ]
    facts = _ordered_facts(decision, ci, ri, variant)
    kind = "rehearsal" if (ci + ri) % 4 == 0 else "interface"
    upstream = None if kind == "rehearsal" else base.advisory_for(decision, action, primary, variant)
    return {
        "case_id": case_id,
        "role": ROLE,
        "kind": kind,
        "context": context,
        "semantic_root": decision,
        "focused_boundary": decision in {"scientific_rejection", "certification_ready"},
        "situation": (
            f"In {context}, Hephaestus must interpret a completed candidate from verified evidence. "
            "Infer the scientific state first; downstream policy authority is separate."
        ),
        "evidence": [base.ev(ref, fact) for ref, fact in zip(refs, facts)],
        "allowed_evidence_refs": refs,
        "upstream_role": "planner" if upstream is not None else None,
        "upstream_output": upstream,
        "verified_facts": base.machine_facts(decision),
        "expected": base.expected(decision, action, primary),
        "target_rationale": base.RATIONALES[decision],
        "forbidden_actions": ["certify_candidate"] if decision != "certification_ready" else [],
    }


def _partition(seed: str, contexts: int, offset: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ci in range(contexts):
        for ri, _ in enumerate(base.ROOTS):
            variant = (ci * 17 + ri * 19 + offset) % 97
            rows.append(_case(seed, ci, ri, variant))
    return rows


def _preflight() -> list[dict[str, Any]]:
    return [
        _case("P9A4D2", ri, ri, (ri * 23 + 11) % 97)
        for ri in range(len(base.ROOTS))
    ]


def build_pack() -> dict[str, Any]:
    return {
        "pack_id": "hephaestus-evaluator-sealed-certification-v1",
        "pack_version": 1,
        "role": ROLE,
        "response_schema": {"required_exact_keys": list(base.REQUIRED_KEYS)},
        "contract_vocabulary": {ROLE: dict(base.VOCAB)},
        "role_rules": {
            ROLE: (
                "Interpret completed verified evidence and infer the scientific state. "
                "The action and primary_variable fields are model telemetry; deterministic "
                "Hephaestus boundaries own the authoritative action and primary variable. "
                "Never treat upstream advice as authority and never infer certification from "
                "anything except the evidence and verified machine state."
            )
        },
        "design": {
            "fresh_fact_wording": True,
            "opaque_model_visible_identifiers": True,
            "burned_case_reuse": False,
            "certification_and_regression_disjoint": True,
            "dedicated_preflight": True,
            "frozen_candidate_required": "evaluator-v3.1-boundary-v1",
        },
        "partitions": {
            ROLE: {
                "preflight": _preflight(),
                "certification": _partition("A73F19", 16, 29),
                "regression": _partition("D42B86", 16, 61),
            }
        },
    }


def canonical_sha256(pack: dict[str, Any]) -> str:
    raw = json.dumps(pack, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _identifier_text(row: dict[str, Any]) -> str:
    return " ".join([
        str(row["case_id"]),
        str(row["context"]),
        *map(str, row["allowed_evidence_refs"]),
    ]).lower()


def validate(pack: dict[str, Any]) -> None:
    part = pack["partitions"][ROLE]
    preflight = list(part["preflight"])
    cert = list(part["certification"])
    regression = list(part["regression"])
    if len(preflight) != 8 or len(cert) != 128 or len(regression) != 128:
        raise ValueError("sealed pack must be 8 preflight + 128 certification + 128 regression")

    roots = set(base.VOCAB["decision"])
    if Counter(row["semantic_root"] for row in cert) != Counter({root: 16 for root in roots}):
        raise ValueError("certification must contain exactly 16 cases per scientific state")
    if Counter(row["semantic_root"] for row in regression) != Counter({root: 16 for root in roots}):
        raise ValueError("regression must contain exactly 16 cases per scientific state")
    if Counter(row["semantic_root"] for row in preflight) != Counter({root: 1 for root in roots}):
        raise ValueError("preflight must contain exactly one case per scientific state")

    groups = {"preflight": preflight, "certification": cert, "regression": regression}
    id_sets = {name: {str(row["case_id"]) for row in rows} for name, rows in groups.items()}
    context_sets = {name: {str(row["context"]) for row in rows} for name, rows in groups.items()}
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if id_sets[left] & id_sets[right] or context_sets[left] & context_sets[right]:
                raise ValueError(f"sealed partition identity collision: {left}/{right}")

    all_rows = preflight + cert + regression
    if len({str(row["case_id"]) for row in all_rows}) != len(all_rows):
        raise ValueError("duplicate sealed case id")
    if not any(row["kind"] == "rehearsal" for row in cert) or not any(row["kind"] == "interface" for row in cert):
        raise ValueError("certification must contain both direct and handoff cases")
    if not any(row["kind"] == "rehearsal" for row in regression) or not any(row["kind"] == "interface" for row in regression):
        raise ValueError("regression must contain both direct and handoff cases")

    for row in all_rows:
        identifiers = _identifier_text(row)
        if any(token in identifiers for token in FORBIDDEN_IDENTIFIER_TOKENS):
            raise ValueError(f"semantic partition/state leakage in model-visible identifiers: {row['case_id']}")
        refs = [str(ref) for ref in row["allowed_evidence_refs"]]
        if len(refs) != len(set(refs)) or len(refs) != 4:
            raise ValueError("each sealed case requires four unique evidence refs")
        if [str(item["ref"]) for item in row["evidence"]] != refs:
            raise ValueError("evidence ref order does not match allowed refs")
        if str(row["context"]) not in str(row["situation"]):
            raise ValueError("opaque context missing from situation")
        if row["semantic_root"] == "certification_ready" and row["verified_facts"].get("recheck_satisfied") is not True:
            raise ValueError("certification_ready fixture missing satisfied recheck")
        if row["semantic_root"] == "scientific_rejection" and row["verified_facts"].get("deterministic_gate_passed") is not False:
            raise ValueError("scientific_rejection fixture missing hard-gate failure")


def main() -> int:
    pack = build_pack()
    validate(pack)
    part = pack["partitions"][ROLE]
    print(json.dumps({
        "pack_id": pack["pack_id"],
        "sha256": canonical_sha256(pack),
        "preflight_cases": len(part["preflight"]),
        "certification_cases": len(part["certification"]),
        "regression_cases": len(part["regression"]),
        "certification_state_counts": dict(sorted(Counter(row["semantic_root"] for row in part["certification"]).items())),
        "regression_state_counts": dict(sorted(Counter(row["semantic_root"] for row in part["regression"]).items())),
        "case_bodies_exposed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
