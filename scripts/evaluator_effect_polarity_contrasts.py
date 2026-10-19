"""CPU-only independent decision-mapping contrasts for the Evaluator.

These are freshly constructed, NOT part of the frozen development pack.
They test the deterministic scientific oracle and the single-variable
experimental design only. No model prediction or GPU inference happens here.
"""
from __future__ import annotations
from dataclasses import replace
import hashlib
import json

from hephaestus.schemas.evaluator_factors import ScientificFactors
from hephaestus.policy.evaluator_semantic_oracle import scientific_state

DEFAULT = ScientificFactors(
    gate_pass=True, complete=True, provenance_valid=True,
    effect="neutral", variance_stable=True, recheck_satisfied=False,
)
RECHECK = replace(DEFAULT, effect="positive", variance_stable=False)

# Each transition changes one primary scientific factor only. Negative controls
# intentionally change a lower-priority factor with NO expected state change.
TRANSITIONS = {
    "equivalent_to_improved": (DEFAULT, "effect", "positive", "improved"),
    "equivalent_to_regressed": (DEFAULT, "effect", "negative", "regressed"),
    "equivalent_to_recheck": (DEFAULT, "variance_stable", False, "recheck_required"),
    "equivalent_to_inconclusive": (DEFAULT, "provenance_valid", False, "inconclusive"),
    "equivalent_to_incomplete": (DEFAULT, "complete", False, "incomplete_evidence"),
    "equivalent_to_rejection": (DEFAULT, "gate_pass", False, "scientific_rejection"),
    "equivalent_confirmation_irrelevant": (DEFAULT, "recheck_satisfied", True, "equivalent"),
    "recheck_to_improved": (RECHECK, "variance_stable", True, "improved"),
    "recheck_to_equivalent": (replace(RECHECK, effect="neutral"), "variance_stable", True, "equivalent"),
    "recheck_to_regressed": (replace(RECHECK, effect="negative"), "variance_stable", True, "regressed"),
    "recheck_to_inconclusive": (RECHECK, "provenance_valid", False, "inconclusive"),
    "recheck_to_incomplete": (RECHECK, "complete", False, "incomplete_evidence"),
    "recheck_to_rejection": (RECHECK, "gate_pass", False, "scientific_rejection"),
    "recheck_effect_irrelevant": (RECHECK, "effect", "negative", "recheck_required"),
    "recheck_confirmation_irrelevant": (RECHECK, "recheck_satisfied", True, "recheck_required"),
}

FACTOR_KEYS = (
    "gate_pass", "complete", "provenance_valid", "effect",
    "variance_stable", "recheck_satisfied",
)
# Use literal measurement fields and a separate optional throughput distractor.
# The factor oracle never reads the throughput observation.
EVIDENCE_KEYS = {
    "gate_pass":"hard_gates",
    "complete":"required_records_present",
    "provenance_valid":"artifact_attestation",
    "effect":"primary_quality_delta",
    "variance_stable":"replicate_spread",
    "recheck_satisfied":"independent_confirmation",
}
MARGIN=0.5
SPREAD_LIMIT=0.1


def render(f:ScientificFactors, *, distractor:bool, phrasing:str)->dict:
    if phrasing not in ("neutral_numbers","named_scientific_fields"):
        raise ValueError("unknown independently authored phrasing")
    delta={"positive":1.0,"neutral":0.0,"negative":-1.0}[f.effect]
    entries={
        "hard_gates": "PASS" if f.gate_pass else "FAIL",
        "required_records_present": "40/40" if f.complete else "37/40",
        "artifact_attestation": "VALID" if f.provenance_valid else "INVALID",
        "primary_quality_delta": f"{delta:+.1f}; threshold ±{MARGIN:.1f}",
        "replicate_spread": f"{0.05 if f.variance_stable else 0.2:.2f}; max {SPREAD_LIMIT:.2f}",
        "independent_confirmation": "PASS" if f.recheck_satisfied else "NOT_PASSED",
    }
    if phrasing=="named_scientific_fields":
        entries["effect_role"]="Only primary quality controls scientific direction."
    if distractor:
        entries["secondary_throughput_delta"]="A report-only throughput delta: "+f"{-delta:+.1f}"+", non-authoritative for quality"
    return entries


def contrast_panel()->list[dict]:
    out=[]
    for name,(baseline, factor, new_value, expected) in TRANSITIONS.items():
        modified=replace(baseline,**{factor:new_value})
        assert getattr(baseline,factor)!=new_value
        assert scientific_state(modified)==expected
        for phrasing in ("neutral_numbers","named_scientific_fields"):
            for distractor in (False,True):
                group=f"{name}:{phrasing}:throughput={int(distractor)}"
                common={
                    "id":"contrast-"+hashlib.sha256(group.encode()).hexdigest()[:16],
                    "transition":name, "changed_factor":factor,
                    "presentation":phrasing,"throughput_distractor":distractor,
                    "scientific_change_expected":scientific_state(baseline)!=scientific_state(modified),
                    "baseline_state":scientific_state(baseline),
                    "modified_state":scientific_state(modified),
                }
                out.append({**common,
                    "baseline_factors":baseline,
                    "modified_factors":modified,
                    "baseline_evidence":render(baseline,distractor=distractor,phrasing=phrasing),
                    "modified_evidence":render(modified,distractor=distractor,phrasing=phrasing),
                })
    return out


def validate_panel(rows=None):
    rows=contrast_panel() if rows is None else list(rows)
    assert len(rows)==60 and len({r["id"] for r in rows})==60
    for row in rows:
        before=row["baseline_factors"];after=row["modified_factors"]
        changes=[k for k in FACTOR_KEYS if getattr(before,k)!=getattr(after,k)]
        if changes!=[row["changed_factor"]]:
            raise ValueError("contrast changed multiple scientific factors")
        changed_evidence=[k for k in EVIDENCE_KEYS.values()
            if row["baseline_evidence"][k]!=row["modified_evidence"][k]]
        if changed_evidence!=[EVIDENCE_KEYS[row["changed_factor"]]]:
            raise ValueError("a contrast changed extra scientific evidence fields")
        if row["throughput_distractor"]:
            for label in ("baseline_evidence","modified_evidence"):
                evidence=row[label]
                # Negative throughput is deliberately independent of state oracle.
                assert "secondary_throughput_delta" in evidence
        if (scientific_state(before),scientific_state(after))!=(row["baseline_state"],row["modified_state"]):
            raise ValueError("oracle labels drift")
        if (row["baseline_state"]!=row["modified_state"])!=row["scientific_change_expected"]:
            raise ValueError("precedence negative controls drift")
    return dict(
        panel_size=len(rows),single_variable_pairs=len(rows),
        state_changes=sum(x["scientific_change_expected"] for x in rows),
        precedence_invariant_pairs=sum(not x["scientific_change_expected"] for x in rows),
        test_pairs_with_throughput=sum(x["throughput_distractor"] for x in rows),
        per_transition_presentations=4,
        oracle_only_no_model_inference=True,
        approval_for_model_inference=False,
        frozen_development_examples_reused=False,
    )


def inference_fixture_manifest():
    rows=contrast_panel();summary=validate_panel(rows)
    # Never put labels inside prompts. The returned fixture is only for
    # prospective separately-authorized inference; not for training or cert.
    prompt_records=[]
    for r in rows:
        for side in ("baseline","modified"):
            input_evidence=r[f"{side}_evidence"]
            prompt_records.append(dict(
                id=r["id"]+":"+side,
                observation=input_evidence,
                output_contract="one_evaluator_scientific_state",
            ))
    fingerprint=hashlib.sha256(json.dumps(prompt_records,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {**summary,"prompt_count":len(prompt_records),
        "unlabeled_prompt_identity_sha256":fingerprint,
        "authorized_inference":False,"paid_run":False}
