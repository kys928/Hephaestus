#!/usr/bin/env python3
"""Aggregate all five post-training adversarial role audits."""
from __future__ import annotations
import json, os, time
from pathlib import Path
import sys

SCRIPTS=Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0,str(SCRIPTS))

import launch_first_bounded_scientific_training as storage

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"configs/experiments/hephaestus_post_training_redteam_v1.json").read_text())

def main()->int:
    run_id=(os.environ.get("HEPHAESTUS_POST_TRAINING_REDTEAM_RUN_ID") or "").strip()
    if not run_id:
        raise RuntimeError("missing HEPHAESTUS_POST_TRAINING_REDTEAM_RUN_ID")
    client=storage.s3_client()
    base=f"{CFG['execution']['s3_prefix'].rstrip('/')}/{run_id}"
    results={}
    for role in CFG["candidates"]:
        key=f"{base}/roles/{role}/result.json"
        try:
            results[role]=json.loads(storage.read_key(client,key))
        except Exception as exc:
            results[role]={"status":"missing","error":f"{type(exc).__name__}: {exc}"}

    complete=all(r.get("status")=="completed" for r in results.values())
    roles={}
    global_weak=[]
    for role,r in results.items():
        if r.get("status")!="completed":
            roles[role]={"status":r.get("status"),"error":r.get("error"),"error_type":r.get("error_type")}
            continue
        a=r["adapted_aggregate"]
        flags=r["audit_flags"]
        roles[role]={
            "status":"completed",
            "model_id":r["model_id"],
            "revision":r["revision"],
            "original_certification_quality_100":CFG["candidates"][role]["certification_quality_100"],
            "redteam_quality_100":a["quality_100"],
            "generalization_gap":r["generalization_gap_from_original_certification"],
            "exact_behavior_rate":a["exact_behavior_rate"],
            "strict_schema_rate":a["strict_schema_rate"],
            "semantic_schema_rate":a["semantic_schema_rate"],
            "evidence_precision":a["evidence_precision"],
            "evidence_recall":a["evidence_recall"],
            "hallucination_rate":a["hallucination_rate"],
            "pair_metrics":a["pair_metrics"],
            "stochastic_triplet_agreement":r["stochastic_consistency"]["triplet_agreement_rate"],
            "base_vs_adapter_mean_quality_delta":r["base_vs_adapter"]["mean_quality_delta"],
            "base_case_regression_rate":r["base_vs_adapter"]["case_regression_rate"],
            "audit_flags":flags,
            "weakest_dimensions":a["weakest_dimensions"],
            "weakest_adapter_lift_dimensions":r["base_vs_adapter"]["weakest_adapter_lift_dimensions"],
        }
        for item in a["weakest_dimensions"]:
            global_weak.append({"role":role,**item})
    aggregate={
        "result_version":"hephaestus-post-training-redteam-aggregate.v1",
        "run_id":run_id,
        "status":"completed" if complete else "incomplete",
        "all_roles_completed":complete,
        "pack_sha256":CFG["pack"]["canonical_sha256"],
        "training_performed":False,
        "weights_mutated":False,
        "production_certification_mutated":False,
        "roles":roles,
        "weakest_regions_across_stack":sorted(global_weak,key=lambda x:(x["quality_100"],x["role"],x["dimension"]))[:25],
        "completed_at_unix":time.time(),
    }
    raw=(json.dumps(aggregate,indent=2,sort_keys=True)+"\n").encode()
    client.put_object(Bucket=storage.VOLUME_ID,Key=f"{base}/result.json",Body=raw)
    if storage.read_key(client,f"{base}/result.json")!=raw:
        raise RuntimeError("aggregate S3 readback mismatch")
    print("POST_TRAINING_REDTEAM_AGGREGATE_JSON "+json.dumps(aggregate,sort_keys=True))
    return 0 if complete else 2

if __name__=="__main__":
    raise SystemExit(main())
