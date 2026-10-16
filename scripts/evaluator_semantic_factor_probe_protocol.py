"""Frozen CPU-only protocol for a future D checkpoint factor-only inference probe.

This module NEVER loads a model, allocates a pod, calls any API, or writes S3.
The probe uses the *existing* 64-case balanced D development screen to enable
same-case comparison with persisted D final decisions. It is exploratory, NOT
an independent evaluation partition or scientific certification.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Mapping

import build_evaluator_semantic_invariance_repair_v1 as frozen
from evaluator_semantic_v2_screen import screen_cases, screen_hash
from evaluator_semantic_factor_attribution import (
    FIELDS, VALID_STATES, factor_only_diagnostic_prompt,
    decode_frozen_development_evidence, score_explicit_factor_probe,
)
from hephaestus.schemas.evaluator_factors import ScientificFactors

D_CHECKPOINT_SHA256 = "5124d3e1e5ddfcb69f85e9746d2dbd737423834e46dc7eb4c56c786acebc02f1"
D_RUN_ID = "evaluator-semantic-v2-D-38945578442"
D_SOURCE_SHA = "ffa0458d5850f9d572a06d5ee6660fd63fc8acc6"
FROZEN_DEVELOPMENT_SHA256 = "ae8b2bd702d9b0751f9b1b04958d5a4559188971fcaf1dd818be4da233cd6912"
D_SCREEN_SHA256 = "98c3346fde66cc92d355bae019de811cb9c7614746073d1e4f93e6a069dbe11c"
SCOPE = "exploratory_saved-D-screen_factor-only_no-certification"
MAX_NEW_TOKENS = 128


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False)


def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def panel():
    """Freeze 64 prompts and verify identical case selection to paid D screen."""
    pack = frozen.build_pack()
    frozen.validate(pack)
    cases = pack["partitions"]["evaluator"]["development_diagnostic"]
    raw = canonical(cases)
    if sha(raw)!=FROZEN_DEVELOPMENT_SHA256:
        raise ValueError("original development pack identity drift")
    selected = screen_cases(cases)
    if screen_hash(selected)!=D_SCREEN_SHA256:
        raise ValueError("D screen identity or order drift")
    records = []
    for case in selected:
        # Oracle label and hidden gold factors must NEVER be in prompt.
        expected_factors = decode_frozen_development_evidence(case)
        prompt = factor_only_diagnostic_prompt(case)
        if case["case_id"] in prompt or canonical(case["expected"]) in prompt:
            raise ValueError("target or case identity leaked into probe")
        records.append({"case_id":case["case_id"],
                        "prompt":prompt,
                        "prompt_sha256":sha(prompt),
                        "reference_factors":expected_factors,
                        "target_state":case["expected"]["decision"],
                        "has_advisory":case.get("upstream_output") is not None})
    if len(records)!=64 or len({x["case_id"] for x in records})!=64:
        raise ValueError("panel geometry drift")
    by_state=Counter(x["target_state"] for x in records)
    if set(by_state)!=set(VALID_STATES) or set(by_state.values())!={8}:
        raise ValueError("screen is not eight-state balanced")
    return records


def plan_manifest(records):
    """Metadata-only manifest: NEVER persist target factors or dev prompts."""
    return {
        "protocol":"evaluator-six-factor-D-v1",
        "scope":SCOPE,
        "source_run_id":D_RUN_ID,
        "source_repo_sha":D_SOURCE_SHA,
        "checkpoint_sha256":D_CHECKPOINT_SHA256,
        "development_sha256":FROZEN_DEVELOPMENT_SHA256,
        "screen_sha256":D_SCREEN_SHA256,
        "prompt_identity_sha256":sha(canonical([(r["case_id"],r["prompt_sha256"]) for r in records])),
        "case_count":len(records),
        "case_count_per_state":8,
        "max_new_tokens":MAX_NEW_TOKENS,
        "sampling":False,
        "factor_output_contract":"six-key JSON-only, no final decision",
        "reference_decisions_reused":"persisted D step96, identical case IDs",
        "new_model_inference_executed":False,
        "new_spending_authorized":False,
        "runpod_allocation_allowed":False,
        "certification_allowed":False,
        "training_or_promotion_allowed":False,
    }


def parse_factor_json(raw):
    """Exact JSON object only: no extracting from reasoning or free text."""
    if not isinstance(raw,str):
        return None
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out:
                raise ValueError("duplicate key")
            out[k]=v
        return out
    try:
        value=json.loads(raw,object_pairs_hook=unique,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
        if not isinstance(value,dict) or set(value)!=set(FIELDS):
            return None
        return ScientificFactors(**value)
    except (ValueError,TypeError):
        return None


def evaluate_reports(records, reports, old_decisions):
    """Compare future explicit factor reports with *saved* separate decisions.

    reports: {"case_id", "prompt_sha256", "checkpoint_sha256", "raw_factors"}
    old_decisions: persisted D screen rows with exact case IDs and parsed state.
    """
    records=list(records);reports=list(reports);old_decisions=list(old_decisions)
    by_id={r["case_id"]:r for r in records}
    ids=set(by_id)
    if (len(records)!=64 or len(by_id)!=64 or len(reports)!=64 or
        len({r.get("case_id") for r in reports})!=64 or
        {r.get("case_id") for r in reports}!=ids):
        raise ValueError("factor probe reports incomplete/duplicated/mismatched")
    if (len(old_decisions)!=64 or len({r["case_id"] for r in old_decisions})!=64 or
        {r["case_id"] for r in old_decisions}!=ids):
        raise ValueError("paired prior D decisions incomplete/duplicated/mismatched")
    paired={r["case_id"]:r for r in old_decisions}
    by_state=Counter()
    by_factor=Counter()
    by_advice=Counter()
    statuses=Counter()
    invalid=0
    factors_fully_correct=0
    prior_correct=0
    pairs_correct=0
    for row in reports:
        ref=by_id[row["case_id"]]
        if (row.get("prompt_sha256")!=ref["prompt_sha256"] or
            row.get("checkpoint_sha256")!=D_CHECKPOINT_SHA256):
            raise ValueError("prompt/model lineage mismatch")
        parsed=parse_factor_json(row.get("raw_factors"))
        prior_row=paired[row["case_id"]]
        prediction=(prior_row.get("score",{}).get("parsed") or {}).get("decision")
        if prediction not in VALID_STATES:
            prediction="<invalid>"
        previously_right=prediction==ref["target_state"]
        prior_correct+=int(previously_right)
        kind="advisory" if ref["has_advisory"] else "direct"
        state=ref["target_state"]
        by_state[(state,"total")]+=1
        by_advice[(kind,"total")]+=1
        if parsed is None:
            invalid+=1
            statuses["invalid_factor_json"]+=1
            for factor in FIELDS:
                by_factor[(factor,"total")]+=1
            continue
        all_correct=parsed==ref["reference_factors"]
        factors_fully_correct+=int(all_correct)
        by_state[(state,"all_factors_correct")]+=int(all_correct)
        by_advice[(kind,"all_factors_correct")]+=int(all_correct)
        pairs_correct+=int(all_correct and previously_right)
        for factor in FIELDS:
            by_factor[(factor,"total")]+=1
            by_factor[(factor,"correct")]+=int(getattr(parsed,factor)==getattr(ref["reference_factors"],factor))
        if prediction=="<invalid>":
            statuses["saved_decision_invalid"]+=1
        else:
            inferred=score_explicit_factor_probe(ref["reference_factors"],{
                "factors":parsed.to_dict(),"decision":prediction})
            statuses[inferred["status"]]+=1
    return {
        "protocol":"evaluator-six-factor-D-v1",
        "total":64,"factor_report_valid_json":64-invalid,
        "factor_report_invalid_json":invalid,
        "all_six_factors_correct":factors_fully_correct,
        "saved_D_decisions_correct":prior_correct,
        "both_factors_and_D_decision_correct":pairs_correct,
        "by_state":{s:dict(total=by_state[(s,"total")],
                          all_factors_correct=by_state[(s,"all_factors_correct")]) for s in sorted(VALID_STATES)},
        "per_factor":{name:dict(total=by_factor[(name,"total")],
                                correct=by_factor[(name,"correct")]) for name in FIELDS},
        "by_handoff":{"direct":dict(total=by_advice[("direct","total")],all_factors_correct=by_advice[("direct","all_factors_correct")]),
                     "advisory":dict(total=by_advice[("advisory","total")],all_factors_correct=by_advice[("advisory","all_factors_correct")])},
        "observable_mismatch_categories":dict(sorted(statuses.items())),
        "limitations":"Probe factors are self-reports and use a changed prompt contract. Matched decisions are prior D generations, not simultaneous hidden intermediate states. 64 frozen development cases are diagnostic-only; no fresh independent certification or production promotion.",
        "model_inference_performed_by_this_function":False,
    }
