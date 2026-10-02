#!/usr/bin/env python3
"""Build a concept-level Evaluator semantic-invariance repair curriculum.

The burned sealed certification cases are diagnosis only. This builder does not copy
any burned prompt or evidence sentence. Instead it factorizes the eight scientific
states into independent scientific variables, creates controlled contrast pairs,
balances direct and Planner-handoff cases, and reserves a lexically disjoint
non-certification development diagnostic for checkpoint selection.

No training, certification, or promotion is performed by this file.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

import build_evaluator_certification_sealed_v1 as burned
import build_interface_repair_v3_evaluator as old

ROLE = "evaluator"
STATE_ORDER = [x[0] for x in old.ROOTS]
ROOT_BY_STATE = {d: (a, p) for d, a, p in old.ROOTS}
REQUIRED_KEYS = list(old.REQUIRED_KEYS)

TRAIN_PHRASES: dict[str, list[str]] = {
    "gate_pass": [
        "Every non-waivable acceptance check resolves successfully.",
        "The mandatory constraint set contains no failed check.",
        "All fixed pass/fail requirements are satisfied.",
        "No compulsory acceptance condition is violated.",
        "The immutable qualification checks are all green.",
        "Each hard acceptance condition evaluates to success.",
    ],
    "gate_fail": [
        "At least one non-waivable acceptance check is reproducibly false.",
        "A compulsory qualification condition fails on repeated evaluation.",
        "The fixed pass/fail set contains a persistent failed requirement.",
        "One mandatory acceptance constraint remains violated across repeats.",
        "A hard qualification condition is consistently not satisfied.",
        "Repeated verification confirms a failed non-negotiable check.",
    ],
    "complete": [
        "The planned observation ledger contains every required entry.",
        "All scheduled measurement records are present in the evidence ledger.",
        "The acquisition plan has full record coverage.",
        "Every required observation has been persisted.",
        "The measurement manifest has no missing required slot.",
        "Required evidence collection reached its planned total.",
    ],
    "incomplete": [
        "The planned observation ledger still has required empty entries.",
        "Some mandatory measurement records were never persisted.",
        "Evidence acquisition stopped before full record coverage.",
        "At least one required observation is absent from the ledger.",
        "The measurement manifest contains unresolved required gaps.",
        "Required evidence collection did not reach its planned total.",
    ],
    "prov_valid": [
        "The measured artifact is cryptographically bound to the admitted revision.",
        "Artifact identity matches the immutable candidate admitted to evaluation.",
        "The evidence can be traced unambiguously to the registered checkpoint.",
        "Lineage and revision identity are both verified for the measured candidate.",
        "The evaluated artifact fingerprint matches the admitted candidate record.",
        "Candidate provenance resolves uniquely to the registered immutable revision.",
    ],
    "prov_invalid": [
        "The measured artifact cannot be bound uniquely to the admitted revision.",
        "Artifact identity does not resolve to a verified immutable candidate.",
        "The evidence cannot be traced unambiguously to the registered checkpoint.",
        "Candidate lineage proof is missing or contradictory for the measured artifact.",
        "The evaluated artifact fingerprint cannot be verified against the admitted record.",
        "Candidate provenance leaves the measured revision unresolved.",
    ],
    "effect_positive": [
        "Across repeats, the candidate clears the predefined practical-gain threshold.",
        "Repeated measurements place the positive delta beyond the materiality boundary.",
        "The replicated effect is positive and larger than the minimum meaningful gain.",
        "Candidate performance exceeds the reference by more than the practical margin.",
        "The repeated quality delta lies materially on the favorable side of the threshold.",
        "Replicated measurements establish a practically meaningful positive effect.",
    ],
    "effect_neutral": [
        "The repeated delta remains inside the predefined indifference interval.",
        "Observed differences stay within the no-material-change band.",
        "The replicated effect does not leave the practical equivalence region.",
        "Candidate and reference remain inside the frozen neutral-effect interval.",
        "The measured difference is smaller than the minimum meaningful-change margin.",
        "Repeated measurements remain within the practical no-change bounds.",
    ],
    "effect_negative": [
        "Across repeats, the candidate falls beyond the tolerated degradation threshold.",
        "Repeated measurements place the negative delta outside the allowed loss margin.",
        "The replicated effect is materially unfavorable relative to the reference.",
        "Candidate quality is below the reference by more than the regression tolerance.",
        "The repeated quality delta lies materially on the adverse side of the threshold.",
        "Replicated measurements establish a practically meaningful negative effect.",
    ],
    "variance_stable": [
        "Dispersion across repetitions remains inside the accepted stability envelope.",
        "Repeated outcomes agree within the frozen spread limit.",
        "Run-to-run variation stays below the instability threshold.",
        "Replication spread satisfies the predefined consistency requirement.",
        "The observed variance remains within the accepted repeatability bound.",
        "Replicates are mutually consistent under the fixed variance policy.",
    ],
    "variance_high": [
        "Dispersion across repetitions exceeds the accepted stability envelope.",
        "Repeated outcomes disagree beyond the frozen spread limit.",
        "Run-to-run variation breaches the instability threshold.",
        "Replication spread violates the predefined consistency requirement.",
        "The observed variance exceeds the accepted repeatability bound.",
        "Replicates are not mutually consistent under the fixed variance policy.",
    ],
    "recheck_yes": [
        "The mandatory confirmation cycle has completed successfully.",
        "The required final confirmation measurement is satisfied.",
        "The policy-mandated confirmation step is complete and passing.",
        "The required confirmation pass has been executed successfully.",
        "The terminal confirmation requirement is explicitly satisfied.",
        "The mandatory post-evaluation confirmation has passed.",
    ],
    "recheck_no": [
        "The mandatory confirmation cycle has not been completed successfully.",
        "The required final confirmation measurement is not yet satisfied.",
        "The policy-mandated confirmation step remains incomplete or unpassed.",
        "The required confirmation pass has not been established.",
        "The terminal confirmation requirement is not explicitly satisfied.",
        "The mandatory post-evaluation confirmation has not passed.",
    ],
}

DEV_PHRASES: dict[str, list[str]] = {
    "gate_pass": [
        "The compulsory decision checks all return pass.",
        "No fixed exclusion criterion is triggered.",
        "Every binding qualification rule succeeds.",
        "The hard-rule verdicts contain no failure.",
    ],
    "gate_fail": [
        "A compulsory decision check repeatedly returns fail.",
        "A fixed exclusion criterion is reproducibly triggered.",
        "At least one binding qualification rule does not succeed.",
        "The hard-rule verdicts contain a persistent failure.",
    ],
    "complete": [
        "The required acquisition roster is fully populated.",
        "No planned observation slot remains unfilled.",
        "The evidence inventory reaches complete planned coverage.",
        "All required measurement units are accounted for.",
    ],
    "incomplete": [
        "The required acquisition roster is not fully populated.",
        "One or more planned observation slots remain unfilled.",
        "The evidence inventory falls short of planned coverage.",
        "Some required measurement units are not accounted for.",
    ],
    "prov_valid": [
        "Identity attestation links the observations to exactly one admitted checkpoint.",
        "The measured revision is unambiguously the registered candidate.",
        "Artifact attestation resolves cleanly to the admitted lineage member.",
        "The observation bundle carries a valid identity binding to the candidate.",
    ],
    "prov_invalid": [
        "Identity attestation cannot link the observations to exactly one admitted checkpoint.",
        "The measured revision is not unambiguously the registered candidate.",
        "Artifact attestation does not resolve cleanly to an admitted lineage member.",
        "The observation bundle lacks a valid identity binding to the candidate.",
    ],
    "effect_positive": [
        "The replicated advantage exceeds the frozen smallest-worthwhile-effect bound.",
        "The candidate's repeated gain clears the practical significance cutoff.",
        "The measured benefit is consistently larger than the declared meaningful-effect floor.",
        "Replications put the candidate beyond the favorable practical-difference limit.",
    ],
    "effect_neutral": [
        "The replicated difference remains within the frozen smallest-worthwhile-effect bounds.",
        "The candidate's repeated delta stays below the practical significance cutoff in magnitude.",
        "The measured difference remains inside the declared negligible-effect region.",
        "Replications keep the candidate within the practical indifference limits.",
    ],
    "effect_negative": [
        "The replicated disadvantage exceeds the frozen tolerated-loss bound.",
        "The candidate's repeated loss crosses the practical degradation cutoff.",
        "The measured harm is consistently larger than the declared acceptable-loss ceiling.",
        "Replications put the candidate beyond the adverse practical-difference limit.",
    ],
    "variance_stable": [
        "Replicate dispersion satisfies the independent stability criterion.",
        "The repeat measurements remain inside the allowed scatter window.",
        "Observed replicate spread does not trigger the instability rule.",
        "The consistency check accepts the dispersion across runs.",
    ],
    "variance_high": [
        "Replicate dispersion violates the independent stability criterion.",
        "The repeat measurements exceed the allowed scatter window.",
        "Observed replicate spread triggers the instability rule.",
        "The consistency check rejects the dispersion across runs.",
    ],
    "recheck_yes": [
        "The required independent confirmation has returned a passing result.",
        "A mandatory final verification has been completed successfully.",
        "The required confirmation record exists and is passing.",
        "The final mandated confirmation criterion is fulfilled.",
    ],
    "recheck_no": [
        "The required independent confirmation has not returned a passing result.",
        "A mandatory final verification has not been completed successfully.",
        "The required confirmation record is absent or not passing.",
        "The final mandated confirmation criterion is not fulfilled.",
    ],
}

CONTRAST_EDGES: list[tuple[str, str, int]] = [
    ("improved", "equivalent", 24),
    ("improved", "certification_ready", 24),
    ("equivalent", "certification_ready", 24),
    ("inconclusive", "recheck_required", 24),
    ("inconclusive", "incomplete_evidence", 16),
    ("recheck_required", "scientific_rejection", 16),
    ("scientific_rejection", "regressed", 16),
    ("improved", "regressed", 16),
    ("equivalent", "regressed", 16),
    ("certification_ready", "incomplete_evidence", 16),
    ("inconclusive", "scientific_rejection", 16),
    ("recheck_required", "incomplete_evidence", 16),
    ("inconclusive", "incomplete_evidence", 8),
    ("recheck_required", "regressed", 8),
    ("incomplete_evidence", "scientific_rejection", 8),
    ("scientific_rejection", "regressed", 8),
]

# The edge multiset above is validated to contribute exactly 64 contrast examples per state.


def _opaque(prefix: str, seed: str, width: int = 14) -> str:
    return prefix + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:width].upper()


def _canonical_latent(state: str) -> dict[str, Any]:
    latent = {
        "gate_pass": True,
        "complete": True,
        "provenance_valid": True,
        "effect": "positive",
        "variance_stable": True,
        "recheck_satisfied": False,
    }
    if state == "scientific_rejection":
        latent["gate_pass"] = False
    elif state == "incomplete_evidence":
        latent["complete"] = False
    elif state == "regressed":
        latent["effect"] = "negative"
    elif state == "equivalent":
        latent["effect"] = "neutral"
    elif state == "inconclusive":
        latent["provenance_valid"] = False
    elif state == "recheck_required":
        latent["variance_stable"] = False
    elif state == "certification_ready":
        latent["recheck_satisfied"] = True
    elif state != "improved":
        raise ValueError(state)
    return latent


def _derive_state(latent: dict[str, Any]) -> str:
    if not latent["gate_pass"]:
        return "scientific_rejection"
    if not latent["complete"]:
        return "incomplete_evidence"
    if not latent["provenance_valid"]:
        return "inconclusive"
    if not latent["variance_stable"]:
        return "recheck_required"
    if latent["effect"] == "negative":
        return "regressed"
    if latent["effect"] == "neutral":
        return "equivalent"
    if latent["effect"] == "positive" and latent["recheck_satisfied"]:
        return "certification_ready"
    return "improved"


def _stress_latent(state: str, variant: int) -> dict[str, Any]:
    latent = _canonical_latent(state)
    # Introduce lower-priority distractor defects only where precedence preserves target state.
    if state == "scientific_rejection":
        latent["complete"] = variant % 3 != 0
        latent["provenance_valid"] = variant % 4 != 0
        latent["variance_stable"] = variant % 5 != 0
    elif state == "incomplete_evidence":
        latent["provenance_valid"] = variant % 3 != 0
        latent["variance_stable"] = variant % 4 != 0
    elif state == "inconclusive":
        latent["variance_stable"] = variant % 3 != 0
    if _derive_state(latent) != state:
        raise RuntimeError(f"stress precedence drift for {state}: {latent}")
    return latent


def _phrase(bank: dict[str, list[str]], key: str, seed: str) -> str:
    values = bank[key]
    index = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) % len(values)
    return values[index]


def _render_facts(latent: dict[str, Any], bank: dict[str, list[str]], seed: str) -> list[str]:
    keys = [
        "gate_pass" if latent["gate_pass"] else "gate_fail",
        "complete" if latent["complete"] else "incomplete",
        "prov_valid" if latent["provenance_valid"] else "prov_invalid",
        f"effect_{latent['effect']}",
        "variance_stable" if latent["variance_stable"] else "variance_high",
        "recheck_yes" if latent["recheck_satisfied"] else "recheck_no",
    ]
    facts = [_phrase(bank, key, seed) for key in keys]
    shift = int(hashlib.sha256(f"{seed}:order".encode()).hexdigest()[:4], 16) % len(facts)
    facts = facts[shift:] + facts[:shift]
    if int(hashlib.sha256(f"{seed}:reverse".encode()).hexdigest()[:2], 16) % 2:
        facts.reverse()
    return facts


def _wrong_action(expected_action: str, seed: str) -> str:
    choices = [x for x in old.VOCAB["action"] if x != expected_action]
    return choices[int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16) % len(choices)]


def _upstream(expected_action: str, seed: str, *, correct: bool) -> dict[str, Any]:
    action = expected_action if correct else _wrong_action(expected_action, seed)
    generic = ["candidate_promising", "candidate_uncertain", "candidate_recoverable", "candidate_reviewed"]
    decision = generic[int(hashlib.sha256(f"{seed}:decision".encode()).hexdigest()[:8], 16) % len(generic)]
    primary_choices = list(old.VOCAB["primary_variable"])
    primary = primary_choices[int(hashlib.sha256(f"{seed}:primary".encode()).hexdigest()[:8], 16) % len(primary_choices)]
    return {
        "source_role": "planner",
        "authority": "advisory",
        "decision": decision,
        "action": action,
        "primary_variable": primary,
        "confidence": 0.99,
        "rationale": "This upstream recommendation is advisory and may be right or wrong; the Evaluator must use verified evidence independently.",
    }


def _target_rationale(state: str) -> str:
    return {
        "scientific_rejection": "A binding hard-rule failure has scientific precedence over favorable secondary measurements.",
        "incomplete_evidence": "Missing required observations prevent a closed scientific judgment.",
        "improved": "Complete, valid, stable evidence establishes a material positive effect, while the mandatory confirmation is not yet satisfied.",
        "regressed": "Complete, valid, stable evidence establishes a material adverse effect.",
        "equivalent": "Complete, valid, stable evidence places the effect inside the practical no-change region.",
        "inconclusive": "The candidate artifact cannot be verified, so a trustworthy scientific status cannot be assigned.",
        "recheck_required": "Excess replicate dispersion prevents a stable conclusion and requires another bounded measurement.",
        "certification_ready": "A material positive effect is supported by complete, valid, stable evidence and the mandatory confirmation has passed.",
    }[state]


def _make_case(
    *,
    corpus: str,
    state: str,
    latent: dict[str, Any],
    family: str,
    index: int,
    bank: dict[str, list[str]],
    has_upstream: bool,
    upstream_correct: bool,
    context_seed: str | None = None,
    fact_seed: str | None = None,
    contrast_partner: str | None = None,
) -> dict[str, Any]:
    derived = _derive_state(latent)
    if derived != state:
        raise RuntimeError(f"latent state mismatch: {state} != {derived}")
    action, primary = ROOT_BY_STATE[state]
    base_seed = f"{corpus}:{family}:{index}:{context_seed or ''}"
    context = _opaque("ctx_", context_seed or base_seed, 16)
    case_id = _opaque("C-", f"{base_seed}:{state}", 18)
    facts = _render_facts(latent, bank, fact_seed or base_seed)
    refs = [_opaque("E-", f"{case_id}:ref:{j}", 18) for j in range(len(facts))]
    upstream = _upstream(action, f"{base_seed}:upstream", correct=upstream_correct) if has_upstream else None
    return {
        "case_id": case_id,
        "role": ROLE,
        "kind": "interface" if has_upstream else "rehearsal",
        "curriculum_family": family,
        "contrast_partner": contrast_partner,
        "context": context,
        "semantic_root": state,
        "situation": (
            f"In {context}, Hephaestus must classify a completed candidate from the supplied scientific evidence and verified facts. "
            "Infer exactly one scientific state. Upstream recommendations are non-authoritative."
        ),
        "evidence": [old.ev(ref, fact) for ref, fact in zip(refs, facts)],
        "allowed_evidence_refs": refs,
        "upstream_role": "planner" if upstream is not None else None,
        "upstream_output": upstream,
        "verified_facts": {
            "deterministic_gate_passed": latent["gate_pass"],
            "evidence_complete": latent["complete"],
            "provenance_valid": latent["provenance_valid"],
            "recheck_satisfied": latent["recheck_satisfied"],
            "candidate_checkpoint_integrity_valid": True,
            "candidate_checkpoint_lineage_match": True,
        },
        "expected": old.expected(state, action, primary),
        "target_rationale": _target_rationale(state),
        "forbidden_actions": ["certify_candidate"] if state != "certification_ready" else [],
    }


def _generalization_cases(corpus: str, bank: dict[str, list[str]], per_state: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in STATE_ORDER:
        for j in range(per_state):
            has_upstream = j % 2 == 1
            correct = ((j // 2) % 2 == 0)
            rows.append(_make_case(
                corpus=corpus, state=state, latent=_canonical_latent(state), family="factorial_generalization",
                index=j, bank=bank, has_upstream=has_upstream, upstream_correct=correct,
            ))
    return rows


def _stress_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in STATE_ORDER:
        for j in range(32):
            has_upstream = j % 2 == 1
            correct = ((j // 2) % 2 == 0)
            rows.append(_make_case(
                corpus="train", state=state, latent=_stress_latent(state, j), family="precedence_and_field_isolation",
                index=j, bank=TRAIN_PHRASES, has_upstream=has_upstream, upstream_correct=correct,
            ))
    return rows


def _contrast_cases() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    degree = Counter()
    global_pair = 0
    for left, right, pair_count in CONTRAST_EDGES:
        for local in range(pair_count):
            pair_seed = f"contrast:{left}:{right}:{local}:{global_pair}"
            has_upstream = global_pair % 2 == 1
            left_action, _ = ROOT_BY_STATE[left]
            right_action, _ = ROOT_BY_STATE[right]
            # Same advisory action for both pair members so the handoff cannot identify the target state.
            common_action = old.VOCAB["action"][int(hashlib.sha256(f"{pair_seed}:up".encode()).hexdigest()[:8], 16) % len(old.VOCAB["action"])]
            for state, other in ((left, right), (right, left)):
                row = _make_case(
                    corpus="train", state=state, latent=_canonical_latent(state), family="minimal_contrast",
                    index=global_pair, bank=TRAIN_PHRASES, has_upstream=False, upstream_correct=False,
                    context_seed=pair_seed, fact_seed=pair_seed, contrast_partner=other,
                )
                if has_upstream:
                    action, _ = ROOT_BY_STATE[state]
                    row["kind"] = "interface"
                    row["upstream_role"] = "planner"
                    row["upstream_output"] = {
                        "source_role": "planner", "authority": "advisory", "decision": "candidate_reviewed",
                        "action": common_action, "primary_variable": "candidate_quality", "confidence": 0.99,
                        "rationale": "The same advisory recommendation is attached to both members of this controlled contrast pair.",
                    }
                    row["contrast_upstream_action_correct"] = common_action == action
                rows.append(row)
                degree[state] += 1
            global_pair += 1
    if degree != Counter({state: 64 for state in STATE_ORDER}):
        raise RuntimeError(f"contrast graph is not state-balanced: {degree}")
    if global_pair != 256 or len(rows) != 512:
        raise RuntimeError("contrast curriculum must contain 256 pairs / 512 cases")
    return rows


def _preflight() -> list[dict[str, Any]]:
    return [
        _make_case(
            corpus="preflight", state=state, latent=_canonical_latent(state), family="format_preflight",
            index=i, bank=DEV_PHRASES, has_upstream=i % 2 == 1, upstream_correct=False,
        )
        for i, state in enumerate(STATE_ORDER)
    ]


def build_pack() -> dict[str, Any]:
    train = _generalization_cases("train", TRAIN_PHRASES, 96) + _contrast_cases() + _stress_cases()
    diagnostic = _generalization_cases("development", DEV_PHRASES, 32)
    preflight = _preflight()
    return {
        "pack_id": "hephaestus-evaluator-semantic-invariance-repair-v1",
        "pack_version": 1,
        "role": ROLE,
        "response_schema": {"required_exact_keys": REQUIRED_KEYS},
        "contract_vocabulary": {ROLE: dict(old.VOCAB)},
        "role_rules": {
            ROLE: (
                "Infer exactly one scientific-state decision from verified evidence and facts. "
                "The decision field must contain a scientific-state token, never an action token. "
                "Planner output is advisory and has zero authority. Model action and primary-variable outputs are telemetry only; "
                "deterministic Hephaestus boundaries own their authoritative values."
            )
        },
        "design": {
            "source_parent": "evaluator-v3.1-boundary-v1",
            "burned_sealed_cases_used_as_training_rows": False,
            "factorized_latent_ontology": True,
            "state_balanced_training": True,
            "direct_handoff_balance_per_state": True,
            "controlled_contrast_pairs": True,
            "upstream_advice_target_correlation": "balanced_or_pair_constant",
            "model_visible_identifiers_opaque": True,
            "old_exact_evidence_sentence_reuse": False,
            "development_diagnostic_is_non_certification": True,
            "future_certification_must_be_fresh_after_checkpoint_selection": True,
        },
        "partitions": {ROLE: {
            "train": train,
            "preflight": preflight,
            "development_diagnostic": diagnostic,
        }},
    }


def canonical_sha256(pack: dict[str, Any]) -> str:
    raw = json.dumps(pack, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _old_fact_sentences() -> set[str]:
    sentences: set[str] = set()
    for templates in old.FACT_SETS.values():
        for template in templates:
            sentences.update(map(str, template))
    for templates in burned.SEALED_FACT_SETS.values():
        for template in templates:
            sentences.update(map(str, template))
    return sentences


def validate(pack: dict[str, Any]) -> None:
    part = pack["partitions"][ROLE]
    train = list(part["train"])
    preflight = list(part["preflight"])
    diagnostic = list(part["development_diagnostic"])
    if len(train) != 1536 or len(preflight) != 8 or len(diagnostic) != 256:
        raise ValueError("curriculum must be 1536 train + 8 preflight + 256 development diagnostic")
    if Counter(x["semantic_root"] for x in train) != Counter({s: 192 for s in STATE_ORDER}):
        raise ValueError("training must contain exactly 192 cases per state")
    if Counter(x["semantic_root"] for x in diagnostic) != Counter({s: 32 for s in STATE_ORDER}):
        raise ValueError("development diagnostic must contain exactly 32 cases per state")
    if Counter(x["semantic_root"] for x in preflight) != Counter({s: 1 for s in STATE_ORDER}):
        raise ValueError("preflight must contain exactly one case per state")
    family = Counter((x["semantic_root"], x["curriculum_family"]) for x in train)
    for state in STATE_ORDER:
        expected = {
            "factorial_generalization": 96,
            "minimal_contrast": 64,
            "precedence_and_field_isolation": 32,
        }
        for name, n in expected.items():
            if family[(state, name)] != n:
                raise ValueError(f"{state}/{name} count drift")
        rows = [x for x in train if x["semantic_root"] == state]
        if Counter(x["kind"] for x in rows) != Counter({"interface": 96, "rehearsal": 96}):
            raise ValueError(f"{state} direct/handoff imbalance")
    old_sentences = _old_fact_sentences()
    train_sentences = {str(ev["fact"]) for x in train for ev in x["evidence"]}
    dev_sentences = {str(ev["fact"]) for x in diagnostic for ev in x["evidence"]}
    if train_sentences & old_sentences or dev_sentences & old_sentences:
        raise ValueError("new curriculum reuses an exact old/burned evidence sentence")
    if train_sentences & dev_sentences:
        raise ValueError("development diagnostic wording overlaps training wording")
    all_rows = train + preflight + diagnostic
    ids = [str(x["case_id"]) for x in all_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case id")
    forbidden = ("improv", "equiv", "regress", "reject", "inconcl", "recheck", "cert", "train", "dev", "split")
    for row in all_rows:
        identifier_text = " ".join([str(row["case_id"]), str(row["context"]), *map(str, row["allowed_evidence_refs"])]).lower()
        if any(token in identifier_text for token in forbidden):
            raise ValueError(f"semantic/split leakage in identifiers: {row['case_id']}")
        refs = list(map(str, row["allowed_evidence_refs"]))
        if len(refs) != 6 or len(refs) != len(set(refs)):
            raise ValueError("every case must have six unique evidence refs")
        if [str(x["ref"]) for x in row["evidence"]] != refs:
            raise ValueError("evidence refs/order mismatch")
    train_contexts = {str(x["context"]) for x in train}
    dev_contexts = {str(x["context"]) for x in diagnostic}
    if train_contexts & dev_contexts:
        raise ValueError("training/development context overlap")


def main() -> int:
    pack = build_pack()
    validate(pack)
    part = pack["partitions"][ROLE]
    train = part["train"]
    diagnostic = part["development_diagnostic"]
    print(json.dumps({
        "pack_id": pack["pack_id"],
        "sha256": canonical_sha256(pack),
        "train_cases": len(train),
        "preflight_cases": len(part["preflight"]),
        "development_diagnostic_cases": len(diagnostic),
        "train_state_counts": dict(sorted(Counter(x["semantic_root"] for x in train).items())),
        "development_state_counts": dict(sorted(Counter(x["semantic_root"] for x in diagnostic).items())),
        "train_family_counts": dict(sorted(Counter(x["curriculum_family"] for x in train).items())),
        "train_kind_counts": dict(sorted(Counter(x["kind"] for x in train).items())),
        "burned_case_reuse": False,
        "future_certification_generated": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
