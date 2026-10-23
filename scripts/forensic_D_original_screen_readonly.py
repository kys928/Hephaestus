#!/usr/bin/env python3
"""Compare same D step-96 checkpoint's prior 64-case, original-format screen.

CPU/S3 GET only. No model inference, GitHub modifications, S3 writes or RunPod access.
"""
import collections
import hashlib
import json
from pathlib import Path

import run_interface_repair_v1 as s3
from prepare_evaluator_semantic_repair_v2 import prepare
from evaluator_semantic_v2_screen import screen_cases,screen_hash
from evaluator_contrast_inference_protocol import D_SOURCE_RUN, D_SOURCE_SHA
from evaluator_semantic_factor_probe_protocol import D_CHECKPOINT_SHA256

BASE="hephaestus/scientific/v5/evaluator_semantic_repair_v2/"+D_SOURCE_RUN
OUTPUT=Path("D-contrast-original-D-screen-forensic")
OUTPUT.mkdir(exist_ok=True)

def get(client,path):
    value=client.get_object(Bucket=s3.bucket(),Key=path)
    try:
        body=value["Body"].read()
    finally:
        value["Body"].close()
    return json.loads(body),dict(key=path,sha256=hashlib.sha256(body).hexdigest(),bytes=len(body))

def main():
    client=s3.s3_client();client.head_bucket(Bucket=s3.bucket())
    receipts=[]
    source,r=get(client,BASE+"/training/result.json");receipts.append(r)
    if source.get("status")!="training_complete" or source.get("repo_sha")!=D_SOURCE_SHA:
        raise ValueError("D source training lineage/status mismatch")
    steps=[z["artifact"] for z in source["checkpoint_artifacts"] if z["optimizer_step"]==96]
    if len(steps)!=1 or steps[0]["sha256"]!=D_CHECKPOINT_SHA256:
        raise ValueError("D step96 source checkpoint SHA mismatch")
    screen,r=get(client,BASE+"/checkpoints/step-096/evaluation/screen/result.json")
    receipts.append(r)
    _,pack,_,_=prepare("D")
    selected=screen_cases(pack["partitions"]["evaluator"]["development_diagnostic"])
    if len(selected)!=64 or screen.get("screen_case_order_sha256")!=screen_hash(selected):
        raise ValueError("original D screen identity mismatch")
    rows=[]
    for i in (1,2):
        chunk,r=get(client,BASE+f"/checkpoints/step-096/evaluation/samples/shard-{i:04d}.json")
        receipts.append(r)
        rows.extend(chunk)
    if len(rows)!=64 or set(r["case_id"] for r in rows)!={r["case_id"] for r in selected}:
        raise ValueError("D original 64-case screen samples incomplete")
    reference={r["case_id"]:r for r in selected}
    grouped=collections.defaultdict(lambda:collections.Counter())
    confusion=collections.Counter()
    samples=[]
    for row in rows:
        case=reference[row["case_id"]]
        ref=case["expected"]["decision"]
        got=(row.get("score",{}).get("parsed") or {}).get("decision")
        correct=row["score"]["decision_exact"]
        grouped[ref]["total"]+=1
        grouped[ref]["correct"]+=int(correct)
        grouped[ref]["parseable"]+=int(isinstance(got,str))
        confusion[(ref,str(got))]+=1
        if (got!=ref and len(samples)<12):
            samples.append({"reference":ref,"predicted":got,
                            "raw_excerpt":str(row.get("generation",{}).get("projected_output"))[:350],
                            "evidence_excerpt":case["evidence"][:2]})
    total=sum(c["total"] for c in grouped.values())
    correct=sum(c["correct"] for c in grouped.values())
    if screen.get("correct")!=correct or screen.get("case_count")!=total:
        raise ValueError("D screen persisted summary diverges from exact saved rows")
    report={
        "GPU_allocations":0,"S3_writes":0,"CPU_only":True,
        "original_source_run":D_SOURCE_RUN,"training_repo_sha":D_SOURCE_SHA,
        "adapter_checkpoint_sha256":D_CHECKPOINT_SHA256,
        "screen_summary":screen,"screen_total":total,"screen_correct":correct,
        "accuracy":correct/total,
        "per_state":{k:dict(v) for k,v in sorted(grouped.items())},
        "confusion":[{"reference":k[0],"predicted":k[1],"count":v} for k,v in sorted(confusion.items())],
        "wrong_output_examples":samples,
        "S3_GET_receipts":receipts,
        "note":"Original-format D screen is a different frozen development subset from the 56 independently authored contrasts. Accuracy difference is observational, not a causal matched-prompt ablation."
    }
    (OUTPUT/"original-D-screen-comparison.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("D_FORENSIC_ORIGINAL_FORMAT_SCREEN "+json.dumps(
        {"total":total,"correct":correct,"accuracy":correct/total,
        "per_state":report["per_state"],"model_sha":D_CHECKPOINT_SHA256,
        "screen_status":screen.get("status")},sort_keys=True),flush=True)
    print("D_FORENSIC_ORIGINAL_FORMAT_WRONG_EXAMPLES "+json.dumps(samples[:3],sort_keys=True),flush=True)

if __name__=="__main__":main()
