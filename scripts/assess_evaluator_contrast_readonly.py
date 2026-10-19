#!/usr/bin/env python3
"""Independent CPU/S3 GET-only rescore of new matched-contrast D inference."""
from __future__ import annotations
import hashlib,json,math,os
from pathlib import Path
import run_interface_repair_v1 as s3
from evaluator_contrast_inference_protocol import panel,manifest,assess,PREFIX
from evaluator_semantic_factor_probe_protocol import D_CHECKPOINT_SHA256

def fetch(client,key,receipts):
    response=client.get_object(Bucket=s3.bucket(),Key=key)
    try:raw=response["Body"].read()
    finally:response["Body"].close()
    receipts.append(dict(key=key,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    return json.loads(raw)

def main():
    runid=os.environ["HEPHAESTUS_CONTRAST_RUN_ID"]
    if not runid.startswith("D-contrast-") or "/" in runid or ".." in runid:
        raise ValueError("unsafe immutable contrast run id")
    unique,pairs=panel();m=manifest()
    client=s3.s3_client();client.head_bucket(Bucket=s3.bucket())
    prefix=PREFIX+"/"+runid+"/evaluation";receipts=[]
    terminal=fetch(client,prefix+"/result.json",receipts)
    if (terminal.get("status")!="contrast_inference_complete" or terminal.get("run_id")!=runid
            or terminal.get("checkpoint_sha256")!=D_CHECKPOINT_SHA256
            or terminal.get("unique_prompts")!=len(unique)):
        raise ValueError("terminal result missing, incomplete or wrong checkpoint")
    stored_m=fetch(client,prefix+"/manifest.json",receipts)
    for k in ("protocol","D_checkpoint_sha256","prompts_sha256","pairs_sha256","paired_challenges"):
        if stored_m.get(k)!=m[k]:raise ValueError("independent scientific manifest identity drift: "+k)
    shard_size=8
    rows=[]
    for i in range(1,math.ceil(len(unique)/shard_size)+1):
        shard=fetch(client,prefix+f"/samples/shard-{i:04d}.json",receipts)
        if not shard or len(shard)>shard_size:raise ValueError("invalid persisted shard length")
        rows.extend(shard)
    if len(rows)!=len(unique):raise ValueError("incomplete persisted generations")
    independent=assess(unique,pairs,rows)
    remote=terminal["analysis"]
    if independent!=remote:raise ValueError("CPU-independent rescore diverges from GPU result")
    report=dict(run_id=runid,protocol=m["protocol"],
                result=independent,read_only_receipts=receipts,
                gpu_allocation=0,s3_writes=0,certification=False,
                scientific_gates_unchanged=True,
                note="120 appearances are not 120 independent samples: identical prompts were deduplicated with fixed greedy decoding.")
    Path("D-contrast-inference-independent.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    small={k:independent[k] for k in ("unique_prompt_count","pair_count",
        "strict_format_valid_unique","exploratory_state_parseable_unique",
        "both_ends_correct","pair_transition_consistency",
        "by_changed_factor","by_throughput_distractor","response_formats")}
    print("D_CONTRAST_INDEPENDENT_ASSESSMENT "+json.dumps(small,sort_keys=True),flush=True)

if __name__=="__main__":
    main()
