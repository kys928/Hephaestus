#!/usr/bin/env python3
"""Exploratory exact-fence normalization of already paid factor-only outputs.

NOT a change to the original strict 0/64 contract score; only interprets 64
unaltered, immutable saved generations under a separately labeled diagnostic.
"""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path

import run_interface_repair_v1 as s3
from evaluator_semantic_factor_probe_protocol import (
    panel,evaluate_reports,parse_factor_json,
)

RUN="factor-only-D-38962439250"
ROOT="hephaestus/scientific/v5/evaluator_factor_only_probe_v1/"+RUN+"/checkpoints/step-096/evaluation"
D_ROOT="hephaestus/scientific/v5/evaluator_semantic_repair_v2/evaluator-semantic-v2-D-38945578442/checkpoints/step-096/evaluation"

def normalize_exact_fenced_object(raw: str) -> str | None:
    """Remove only a complete Markdown JSON fence, not arbitrary surrounding text."""
    if not isinstance(raw,str):
        return None
    fence=re.escape(chr(96)*3)
    pattern=fence+r"(?:json)?[ \t]*\r?\n(\{[\s\S]*\})\r?\n"+fence+r"[ \t]*"
    match=re.fullmatch(pattern,raw.strip(),re.IGNORECASE)
    if match is None:
        return None
    candidate=match.group(1)
    return candidate if parse_factor_json(candidate) is not None else None



def normalize_fenced_singleton_array(raw: str) -> str | None:
    """Unwrap ONLY a full JSON code fence containing a single-object array.

    Strict 0/64 schema score is immutable; this is a separately labeled
    exploratory post-hoc representation normalization.
    """
    if not isinstance(raw,str):
        return None
    fence=re.escape(chr(96)*3)
    pattern=fence+r"json[ \t]*\r?\n([\s\S]*?)\r?\n"+fence+r"[ \t]*"
    match=re.fullmatch(pattern,raw.strip(),re.IGNORECASE)
    if match is None:
        return None
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out:
                raise ValueError("duplicate JSON key")
            out[k]=v
        return out
    try:
        value=json.loads(match.group(1),object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
    except ValueError:
        return None
    if not isinstance(value,list) or len(value)!=1 or not isinstance(value[0],dict):
        return None
    candidate=json.dumps(value[0],sort_keys=True,separators=(",",":"))
    return candidate if parse_factor_json(candidate) is not None else None


def get(client,key,receipts):
    resp=client.get_object(Bucket=s3.bucket(),Key=key)
    try:raw=resp["Body"].read()
    finally:resp["Body"].close()
    receipts.append(dict(key=key,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
    return json.loads(raw)


def main():
    client=s3.s3_client();client.head_bucket(Bucket=s3.bucket())
    receipts=[]
    terminal=get(client,ROOT+"/result.json",receipts)
    if terminal.get("status")!="factor_probe_complete" or terminal.get("completed")!=64:
        raise ValueError("factor probe source result not complete")
    source=[r for i in range(1,9) for r in get(client,ROOT+f"/samples/shard-{i:04d}.json",receipts)]
    old=[r for i in range(1,3) for r in get(client,D_ROOT+f"/samples/shard-{i:04d}.json",receipts)]
    if len(source)!=64 or len(old)!=64 or any(r.get("valid_json") for r in source):
        raise ValueError("must preserve originally invalid strict formatting")
    normalized=[]
    exact_object_fence=0
    singleton_array_fence=0
    for row in source:
        parsed=normalize_exact_fenced_object(row["raw_factors"])
        exact_object_fence+=parsed is not None
        if parsed is None:
            parsed=normalize_fenced_singleton_array(row["raw_factors"])
            singleton_array_fence+=parsed is not None
        normalized.append({
            "case_id":row["case_id"],"prompt_sha256":row["prompt_sha256"],
            "checkpoint_sha256":row["checkpoint_sha256"],
            "raw_factors":parsed if parsed is not None else ""
        })
    records=panel()
    score=evaluate_reports(records,normalized,old)
    from collections import Counter,defaultdict
    from evaluator_semantic_factor_probe_protocol import parse_factor_json
    state_factor=defaultdict(Counter)
    value_confusion=defaultdict(Counter)
    reference={r["case_id"]:r for r in records}
    for entry in normalized:
        ref=reference[entry["case_id"]]
        report=parse_factor_json(entry["raw_factors"])
        if report is None:
            continue
        state=ref["target_state"]
        for factor in ("gate_pass","complete","provenance_valid","effect","variance_stable","recheck_satisfied"):
            wanted=getattr(ref["reference_factors"],factor)
            observed=getattr(report,factor)
            state_factor[state][(factor,"total")]+=1
            state_factor[state][(factor,"correct")]+=int(wanted==observed)
            value_confusion[factor][str(wanted)+" -> "+str(observed)]+=1
    score["per_state_per_factor"]={
        state:{factor:dict(
            correct=state_factor[state][(factor,"correct")],
            total=state_factor[state][(factor,"total")]
        ) for factor in ("gate_pass","complete","provenance_valid","effect","variance_stable","recheck_satisfied")}
        for state in sorted(state_factor)}
    score["factor_value_confusion"]={factor:dict(sorted(matrix.items())) for factor,matrix in sorted(value_confusion.items())}
    score.update({
        "label":"exploratory_exact_fence_normalized_saved_factor_reports",
        "original_strict_valid_json":0,
        "exact_object_fence_normalizable":exact_object_fence,
        "fenced_singleton_array_normalizable":singleton_array_fence,
        "saved_raw_outputs_mutated":False,
        "normalization_restriction":"exact complete markdown JSON fence with either object or strictly singleton JSON array of one six-factor object; no target-derived repair, no new inference",
        "original_required_JSON_contract_passed":False,
        "same_case_reference_decisions":33,
        "read_receipts":receipts,
        "gpu_allocations":0,"s3_writes":0,
        "certification":False,
        "limitations":"Post-hoc format tolerance is NOT strict-format compliance or independent proof of model reasoning. Previously frozen development examples and independent saved D decisions only."
    })
    Path("evaluator-factor-probe-fence-normalized.json").write_text(json.dumps(score,sort_keys=True,indent=2)+"\n")
    compact={k:score[k] for k in ("total","original_strict_valid_json","exact_object_fence_normalizable","fenced_singleton_array_normalizable",
        "factor_report_valid_json","all_six_factors_correct","saved_D_decisions_correct",
        "per_factor","by_state","observable_mismatch_categories","by_handoff",
        "per_state_per_factor","factor_value_confusion")}
    print("EVALUATOR_FACTOR_FENCE_NORMALIZED "+json.dumps(compact,sort_keys=True))
    return score

if __name__=="__main__":
    main()
