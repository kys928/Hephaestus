#!/usr/bin/env python3
"""Independent S3 read-only D factor-report evaluation; GPU allocation forbidden."""
import hashlib
import json
import os
from pathlib import Path

import run_interface_repair_v1 as s3
from evaluator_semantic_factor_probe_protocol import (
    panel,plan_manifest,evaluate_reports,D_RUN_ID,D_SOURCE_SHA,
    D_CHECKPOINT_SHA256,
)

D_SOURCE_PREFIX="hephaestus/scientific/v5/evaluator_semantic_repair_v2/"+D_RUN_ID
PROBE_PREFIX="hephaestus/scientific/v5/evaluator_factor_only_probe_v1"


def get(client,key,reads):
    resp=client.get_object(Bucket=s3.bucket(),Key=key)
    try:
        raw=resp["Body"].read()
    finally:
        resp["Body"].close()
    reads.append(dict(key=key,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    return json.loads(raw)


def main():
    runid=os.environ["HEPHAESTUS_FACTOR_PROBE_RUN_ID"]
    if not runid.startswith("factor-only-D-") or "/" in runid or ".." in runid:
        raise ValueError("untrusted factor-run identity")
    source=panel()
    manifest=plan_manifest(source)
    client=s3.s3_client()
    client.head_bucket(Bucket=s3.bucket())
    reads=[]
    folder=f"{PROBE_PREFIX}/{runid}/checkpoints/step-096/evaluation"
    terminal=get(client,folder+"/result.json",reads)
    if terminal["status"]!="factor_probe_complete" or terminal["run_id"]!=runid or terminal["checkpoint_sha256"]!=D_CHECKPOINT_SHA256:
        raise ValueError("factor-only terminal identity or completeness mismatch")
    metadata=get(client,folder+"/manifest.json",reads)
    for k in ("protocol","prompt_identity_sha256","checkpoint_sha256","screen_sha256","development_sha256"):
        if metadata.get(k)!=manifest[k]:
            raise ValueError(f"factor-probe manifest drift: {k}")
    records=[]
    for i in range(1,9):
        shard=get(client,folder+f"/samples/shard-{i:04d}.json",reads)
        if len(shard)!=8:
            raise ValueError("factor-only case count/shard drift")
        records+=shard
    if not all(r.get("checkpoint_sha256")==D_CHECKPOINT_SHA256 for r in records):
        raise ValueError("incorrect source adapter checkpoint")
    old=[]
    old_folder=D_SOURCE_PREFIX+"/checkpoints/step-096/evaluation"
    for i in range(1,3):
        old+=get(client,old_folder+f"/samples/shard-{i:04d}.json",reads)
    if len(old)!=64:
        raise ValueError("saved D reference decisions incomplete")
    report=evaluate_reports(source,records,old)
    report["source_run_id"]=D_RUN_ID
    report["source_repo_sha"]=D_SOURCE_SHA
    report["factor_probe_run_id"]=runid
    report["checked_object_hashes"]=reads
    report["frozen_reference_cases_unchanged"]=True
    report["new_gpu_allocations_from_this_assessment"]=0
    report["s3_writes_from_this_assessment"]=0
    report["no_scientific_certification"]=True
    Path("evaluator-factor-probe-assessment.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    small={k:report[k] for k in ("protocol","total","factor_report_valid_json","all_six_factors_correct","saved_D_decisions_correct","per_factor","by_state","observable_mismatch_categories")}
    print("EVALUATOR_FACTOR_PROBE_ASSESSMENT "+json.dumps(small,sort_keys=True))
    return report


if __name__=="__main__":
    main()
