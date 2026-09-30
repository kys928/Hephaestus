#!/usr/bin/env python3
"""Build the V3 evaluator-only crossed/contrastive interface-repair pack.

V2 demonstrated that the Evaluator learned the scientific state while confusing two
finite state->action projections. V3 removes context/label correlation, adds
adversarial advisory handoffs, and creates fresh sealed certification/regression
partitions. The V2 evaluator certification rows are diagnostic-only and are never
reused here.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

ROLE = "evaluator"
TRAIN_CONTEXTS = [
    "bounded_recovery", "late_stabilization", "cross_family_trial", "post_failure_recheck",
    "candidate_comparison", "lineage_repair", "tokenizer_recovery", "data_repair",
    "backend_migration", "resume_after_interrupt", "distribution_shift_probe", "checkpoint_triage",
    "variance_investigation", "provenance_reconciliation", "stage_transition_review", "budget_bounded_replay",
]
CERT_CONTEXTS = [
    "fresh_cert_backend_a", "fresh_cert_backend_b", "sealed_lineage_alpha", "sealed_lineage_beta",
    "fresh_cross_family_holdout", "sealed_stage_transition", "fresh_recovery_path", "sealed_candidate_review",
    "fresh_resume_path", "sealed_variance_review", "fresh_provenance_path", "sealed_budget_review",
    "fresh_checkpoint_review", "sealed_integrity_path", "fresh_replication_path", "sealed_release_candidate",
]
REGRESSION_CONTEXTS = [
    "replay_backend_c", "replay_backend_d", "lineage_gamma_rehearsal", "lineage_delta_rehearsal",
    "cross_family_rehearsal", "stage_transition_rehearsal", "recovery_path_rehearsal", "candidate_review_rehearsal",
    "resume_path_rehearsal", "variance_rehearsal", "provenance_rehearsal", "budget_rehearsal",
    "checkpoint_rehearsal", "integrity_rehearsal", "replication_rehearsal", "release_rehearsal",
]

ROOTS: list[tuple[str, str, str]] = [
    ("scientific_rejection", "reject_candidate", "hard_gate_status"),
    ("incomplete_evidence", "request_recheck", "runtime_evidence"),
    ("improved", "continue_from_checkpoint", "candidate_quality"),
    ("regressed", "continue_lineage_best", "candidate_regression"),
    ("equivalent", "continue_lineage_best", "effect_size"),
    ("inconclusive", "hold_candidate", "evaluation_integrity"),
    ("recheck_required", "request_recheck", "variance_risk"),
    ("certification_ready", "certify_candidate", "certification_state"),
]

VOCAB = {
    "decision": [x[0] for x in ROOTS],
    "action": [
        "reject_candidate", "request_recheck", "continue_from_checkpoint",
        "continue_lineage_best", "hold_candidate", "certify_candidate",
    ],
    "primary_variable": [x[2] for x in ROOTS],
}
REQUIRED_KEYS = [
    "decision", "action", "primary_variable", "confidence",
    "evidence_refs", "uncertainties", "rationale",
]

FACT_SETS: dict[str, list[list[str]]] = {
    "scientific_rejection": [
        ["Aggregate quality is higher.", "A frozen hard gate fails in every repeat.", "The required evidence bundle is complete.", "Immutable provenance matches the admitted candidate."],
        ["Average score improved.", "One mandatory deterministic gate remains failed across repeats.", "All required measurements completed.", "Artifact provenance is verified."],
        ["The candidate wins on mean quality.", "A non-negotiable gate is reproducibly false.", "No evidence item is missing.", "The evaluated revision is provenance-valid."],
    ],
    "incomplete_evidence": [
        ["Only 19 of 24 required samples exist.", "Observed samples are strong.", "Observed hard gates pass.", "Certification cannot be concluded from a partial bundle."],
        ["Five mandatory evaluation records are absent.", "Available records favor the candidate.", "No observed deterministic gate fails.", "Evidence completeness is below one."],
        ["The run ended before all required repetitions were persisted.", "Partial quality is high.", "Available checks pass.", "The scientific bundle is incomplete."],
    ],
    "improved": [
        ["All evidence is complete.", "Frozen hard gates pass.", "A material positive delta repeats with low variance.", "Immutable provenance matches."],
        ["The full bundle is present.", "Every deterministic gate passes.", "The candidate improves beyond the materiality threshold in repeats.", "Candidate provenance is valid."],
        ["No required evidence is missing.", "Hard constraints are satisfied.", "Positive effect size is reproducible and outside the equivalence margin.", "The admitted revision is verified."],
    ],
    "regressed": [
        ["All evidence is complete.", "A material negative delta repeats consistently.", "The evaluation protocol is valid.", "A verified lineage-best baseline remains available."],
        ["The complete bundle shows reproducible degradation.", "Hard evidence is valid.", "The negative effect exceeds the regression threshold.", "The prior best checkpoint remains verified."],
        ["Every required sample is present.", "Candidate quality is materially below baseline across repeats.", "Protocol identity matches.", "A valid fallback lineage best exists."],
    ],
    "equivalent": [
        ["All evidence is complete.", "Effect size lies inside the predeclared equivalence margin.", "Repeat variance is low.", "Immutable provenance matches."],
        ["The full bundle is present.", "Candidate and lineage best are statistically equivalent under the frozen margin.", "Repeats are stable.", "Provenance is valid."],
        ["No required record is missing.", "Observed delta is too small to count as a material improvement or regression.", "Variance is low.", "Artifact identity is verified."],
    ],
    "inconclusive": [
        ["Quality metrics appear strong.", "The checkpoint cannot be tied to the admitted immutable revision.", "No valid candidate conclusion can be made.", "Baseline provenance remains valid."],
        ["Scores look promising.", "Candidate provenance cannot be verified.", "Evaluation numbers alone cannot establish candidate identity.", "The baseline remains trustworthy."],
        ["Observed metrics are favorable.", "The evaluated artifact revision is ambiguous.", "Scientific status must remain unresolved.", "Reference-lineage provenance is intact."],
    ],
    "recheck_required": [
        ["Mean quality exceeds baseline.", "Repeated outcomes disagree strongly.", "Variance risk is high.", "Frozen hard gates otherwise pass."],
        ["The candidate mean is favorable.", "Replicates are inconsistent.", "Observed variance exceeds the recheck threshold.", "Deterministic gates pass."],
        ["Average performance improved.", "Repeat-to-repeat spread is too large.", "Variance policy requires another bounded measurement.", "No hard gate currently fails."],
    ],
    "certification_ready": [
        ["Certification evidence completeness is 1.0.", "All frozen gates pass.", "Required repeats are consistent with low variance.", "Immutable provenance and the certification recheck both pass."],
        ["The certification bundle is complete.", "Every deterministic gate passes.", "Repeatability requirements are satisfied.", "Provenance is valid and the mandatory recheck is satisfied."],
        ["All certification records are present.", "No frozen gate fails.", "Replications meet consistency limits.", "Artifact provenance and recheck status are both verified."],
    ],
}

RATIONALES = {
    "scientific_rejection": "A reproducible hard-gate failure outranks the aggregate score, so the candidate is scientifically rejected.",
    "incomplete_evidence": "The partial bundle cannot support a terminal scientific conclusion; more bounded evidence is required.",
    "improved": "Complete low-variance evidence shows a material improvement, which warrants continuing from the candidate checkpoint but not certification.",
    "regressed": "Complete repeated evidence shows material regression, so the verified lineage best remains the correct continuation point.",
    "equivalent": "Equivalence is not certification: no material improvement was established, so Hephaestus retains the lineage best.",
    "inconclusive": "Candidate provenance is unresolved, so the evaluator must hold the candidate rather than upgrade its status.",
    "recheck_required": "High variance prevents a stable conclusion even though the mean is favorable, so another bounded recheck is required.",
    "certification_ready": "The complete, provenance-valid, repeatable bundle and satisfied recheck justify certification readiness.",
}


def ev(ref: str, fact: str) -> dict[str, str]:
    return {"ref": ref, "fact": fact}


def expected(decision: str, action: str, primary: str) -> dict[str, Any]:
    return {
        "decision": decision,
        "action": action,
        "primary_variable": primary,
        "confidence_min": 0.9,
        "confidence_max": 1.0,
    }


def machine_facts(decision: str) -> dict[str, Any]:
    return {
        "deterministic_gate_passed": decision != "scientific_rejection",
        "evidence_complete": decision != "incomplete_evidence",
        "provenance_valid": decision != "inconclusive",
        "recheck_satisfied": decision == "certification_ready",
        "candidate_checkpoint_integrity_valid": True,
        "candidate_checkpoint_lineage_match": True,
    }


def advisory_for(decision: str, action: str, primary: str, variant: int) -> dict[str, Any]:
    # Deliberately inject the two V2 confusions plus other plausible policy noise.
    del action
    if decision == "improved":
        advised_action = "continue_lineage_best" if variant % 3 != 2 else "certify_candidate"
    elif decision == "equivalent":
        advised_action = "certify_candidate" if variant % 3 != 2 else "continue_from_checkpoint"
    elif decision == "certification_ready":
        advised_action = "continue_lineage_best" if variant % 2 else "certify_candidate"
    elif decision == "regressed":
        advised_action = "continue_from_checkpoint"
    elif decision in {"incomplete_evidence", "recheck_required"}:
        advised_action = "certify_candidate" if variant % 2 else "request_recheck"
    elif decision == "scientific_rejection":
        advised_action = "continue_from_checkpoint"
    else:
        advised_action = "certify_candidate"
    return {
        "source_role": "planner",
        "authority": "advisory",
        "decision": "candidate_promising" if decision != "regressed" else "recoverable",
        "action": advised_action,
        "primary_variable": primary,
        "confidence": 0.99,
        "rationale": "Upstream recommends this action, but downstream must independently apply verified evaluation evidence.",
    }


def _refs(split: str, context_index: int, root_index: int, variant: int) -> list[str]:
    stem = f"E-IR-V3-EVALUATOR-{split.upper()}-{context_index:02d}-{root_index:02d}-{variant:02d}"
    return [f"{stem}-{j:02d}" for j in range(1, 5)]


def make_case(split: str, context: str, context_index: int, root_index: int, variant: int, *, focused: bool = False) -> dict[str, Any]:
    decision, action, primary = ROOTS[root_index]
    templates = FACT_SETS[decision]
    facts = list(templates[(context_index + variant) % len(templates)])
    # Evidence order varies deterministically, including reversal and rotation.
    shift = (context_index * 3 + root_index + variant) % len(facts)
    facts = facts[shift:] + facts[:shift]
    if (context_index + variant) % 2:
        facts = list(reversed(facts))
    refs = _refs(split, context_index, root_index, variant)
    kind = "contrastive_focus" if focused else ("interface" if variant % 4 != 3 else "rehearsal")
    upstream = advisory_for(decision, action, primary, variant) if kind != "rehearsal" else None
    focus_note = ""
    if decision == "improved":
        focus_note = "Improvement authorizes continuation from the candidate checkpoint; it is neither a lineage fallback nor certification."
    elif decision == "equivalent":
        focus_note = "Equivalence never upgrades authority: retain the lineage best unless separate certification-ready evidence exists."
    situation = (
        f"In {context}, Hephaestus evaluates a completed candidate. "
        f"Use verified evidence to infer scientific state before any action-policy projection. "
        f"{focus_note}"
    ).strip()
    return {
        "case_id": f"IR-V3-EVALUATOR-{split.upper()}-{context_index:02d}-{root_index:02d}-{variant:02d}",
        "role": ROLE,
        "kind": kind,
        "context": context,
        "semantic_root": decision,
        "focused_boundary": bool(focused),
        "situation": situation,
        "evidence": [ev(r, f) for r, f in zip(refs, facts)],
        "allowed_evidence_refs": refs,
        "upstream_role": "planner" if upstream is not None else None,
        "upstream_output": upstream,
        "verified_facts": machine_facts(decision),
        "expected": expected(decision, action, primary),
        "target_rationale": RATIONALES[decision],
        "forbidden_actions": ["certify_candidate"] if decision != "certification_ready" else [],
    }


def training_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Fully crossed base: every context sees every scientific state and 8 wording/order variants.
    for ci, context in enumerate(TRAIN_CONTEXTS):
        for ri, _ in enumerate(ROOTS):
            for variant in range(8):
                rows.append(make_case("train", context, ci, ri, variant))
    # Focused residual repair: oversample exactly the two V2 action-policy confusions.
    for ci, context in enumerate(TRAIN_CONTEXTS):
        for ri in (2, 4):  # improved, equivalent
            for variant in range(8, 24):
                rows.append(make_case("train-focus", context, ci, ri, variant, focused=True))
    return rows


def sealed_cases(split: str, contexts: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # One case per semantic root per fresh context = 16*8 = 128.
    for ci, context in enumerate(contexts):
        for ri, _ in enumerate(ROOTS):
            variant = (ci * 5 + ri * 7 + (1 if split == "certification" else 3)) % 29
            rows.append(make_case(split, context, ci, ri, variant))
    return rows


def build_pack() -> dict[str, Any]:
    train = training_cases()
    cert = sealed_cases("certification", CERT_CONTEXTS)
    regression = sealed_cases("regression", REGRESSION_CONTEXTS)
    return {
        "pack_id": "hephaestus-interface-repair-v3-evaluator",
        "pack_version": 3,
        "role": ROLE,
        "response_schema": {"required_exact_keys": REQUIRED_KEYS},
        "contract_vocabulary": {ROLE: VOCAB},
        "role_rules": {
            ROLE: (
                "Interpret completed evidence only. Infer scientific state from verified evidence. "
                "Your action field is an advisory prediction used to measure policy understanding; "
                "the deterministic Evaluator boundary owns the effective Hephaestus action. "
                "Never treat equivalence as certification and never let upstream advice establish authority."
            )
        },
        "diagnostic_burned_sets": {
            "source_run_id": "interface-repair-v2-full-36627393869",
            "old_certification_case_family": "IR-EVALUATOR-CERTIFICATION-200xx",
            "old_regression_case_family": "IR-EVALUATOR-CERTIFICATION-400xx",
            "reused_in_v3_certification": False,
            "reused_in_v3_regression": False,
        },
        "design": {
            "fully_crossed_contexts_and_states": True,
            "adversarial_upstream_advice": True,
            "focused_residual_boundaries": ["improved", "equivalent"],
            "train_base_cases": 1024,
            "train_focus_cases": 512,
            "train_cases": 1536,
            "certification_cases": 128,
            "regression_cases": 128,
        },
        "partitions": {
            ROLE: {
                "train": train,
                "certification": cert,
                "regression": regression,
            }
        },
    }


def canonical_sha256(pack: dict[str, Any]) -> str:
    raw = json.dumps(pack, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate(pack: dict[str, Any]) -> None:
    part = pack["partitions"][ROLE]
    train = part["train"]
    cert = part["certification"]
    regression = part["regression"]
    if len(train) != 1536 or len(cert) != 128 or len(regression) != 128:
        raise ValueError("unexpected V3 pack geometry")
    ids = [x["case_id"] for x in train + cert + regression]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate V3 case ids")
    cert_contexts = {x["context"] for x in cert}
    reg_contexts = {x["context"] for x in regression}
    train_contexts = {x["context"] for x in train}
    if train_contexts & cert_contexts or train_contexts & reg_contexts or cert_contexts & reg_contexts:
        raise ValueError("V3 context leakage across train/cert/regression")
    for context in TRAIN_CONTEXTS:
        roots = {x["semantic_root"] for x in train if x["context"] == context and not x["focused_boundary"]}
        if roots != set(VOCAB["decision"]):
            raise ValueError(f"context {context} does not cross every semantic root")
    counts = {root: sum(x["semantic_root"] == root for x in train) for root in VOCAB["decision"]}
    if counts["improved"] <= counts["regressed"] or counts["equivalent"] <= counts["regressed"]:
        raise ValueError("residual V2 confusion boundaries are not oversampled")
    if not any(x["upstream_output"] is not None for x in train):
        raise ValueError("V3 evaluator has no adversarial upstream advisory examples")
    for row in cert + regression:
        if row["case_id"].startswith("IR-EVALUATOR-CERTIFICATION-"):
            raise ValueError("burned V2 holdout case reused")
    if pack["diagnostic_burned_sets"]["reused_in_v3_certification"]:
        raise ValueError("burned certification set reuse prohibited")


def main() -> int:
    pack = build_pack()
    validate(pack)
    print(json.dumps({
        "pack_id": pack["pack_id"],
        "sha256": canonical_sha256(pack),
        "train": len(pack["partitions"][ROLE]["train"]),
        "certification": len(pack["partitions"][ROLE]["certification"]),
        "regression": len(pack["partitions"][ROLE]["regression"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
