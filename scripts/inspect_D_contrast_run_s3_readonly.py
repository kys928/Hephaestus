#!/usr/bin/env python3
"""Read-only S3 forensics: failed D 60-pair GPU run 39299580607.

S3 operations allowed: list_objects_v2, head_object, get_object ONLY.
Output: local JSON report and bounded log summary, no S3 writes, no pod/API
credentials, no raw generated text, no activation or scientific promotion.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import run_interface_repair_v1 as s3
from evaluator_contrast_inference_protocol import (
    PREFIX, panel, manifest as reference_manifest, parse_decision,
)
from evaluator_semantic_factor_probe_protocol import D_CHECKPOINT_SHA256

RUN_ID = "D-contrast-39299580607"
ROOT = f"{PREFIX}/{RUN_ID}/evaluation"
APPROVAL_KEY = f"{PREFIX}/approvals/heph-D-contrasts-20261019-capacity-retry-1/once.json"
OBSERVED_POD_CREATE_UTC = "2026-10-19T18:05:29.002592+00:00"
OBSERVED_POD_DELETE_UTC = "2026-10-19T18:21:07.355249+00:00"
OUTPUT = Path("D-contrast-39299580607-S3-readonly-inspection.json")
MAX_JSON_BYTES = 2_000_000

def utc(value):
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)

def elapsed(first, later):
    if first is None or later is None:
        return None
    a=datetime.fromisoformat(first.replace("Z","+00:00"))
    b=datetime.fromisoformat(later.replace("Z","+00:00"))
    return round((b-a).total_seconds(), 3)

def digest(blob):
    return hashlib.sha256(blob).hexdigest()

def list_run(client, bucket):
    """LIST is optional for read-only credentials; known-key HEAD still works."""
    prefix=ROOT+"/"
    token=None
    found=[]
    while True:
        args={"Bucket":bucket,"Prefix":prefix,"MaxKeys":500}
        if token:
            args["ContinuationToken"]=token
        data=client.list_objects_v2(**args)
        for item in data.get("Contents", []):
            key=item["Key"]
            if key.startswith(prefix):
                found.append(dict(key=key,bytes=item.get("Size"),
                                  last_modified_utc=utc(item.get("LastModified")),
                                  etag=str(item.get("ETag",""))))
        token=data.get("NextContinuationToken")
        if not data.get("IsTruncated") or not token:
            break
        if len(found)>5000:
            raise RuntimeError("unexpectedly many files under isolated run prefix")
    return sorted(found,key=lambda x:x["key"])

def fetch_json(client, bucket, key, receipts):
    try:
        head=client.head_object(Bucket=bucket,Key=key)
    except Exception as exc:
        if _missing(exc):
            receipts.append(dict(key=key,present=False))
            return None
        raise
    n=int(head.get("ContentLength",-1))
    if n<0 or n>MAX_JSON_BYTES:
        raise ValueError(f"object too large or unknown length: {key}")
    obj=client.get_object(Bucket=bucket,Key=key)
    try:
        raw=obj["Body"].read(MAX_JSON_BYTES+1)
    finally:
        obj["Body"].close()
    if len(raw)!=n:
        raise ValueError("HEAD/GET byte size drift")
    receipt=dict(key=key,present=True,bytes=n,sha256=digest(raw),
                 last_modified_utc=utc(head.get("LastModified")))
    receipts.append(receipt)
    val=json.loads(raw)
    return val

def _missing(exc):
    response=getattr(exc,"response",{})
    if not isinstance(response,dict):
        return False
    code=str(response.get("Error",{}).get("Code",""))
    return code in ("404","NoSuchKey","NotFound")

def _summary_bootstrap(value):
    if not isinstance(value,dict):
        return None
    return {k:value.get(k) for k in ("stage","timestamp_unix","venv_ready",
            "hf_cache_root_exists","nvidia_smi_error") if k in value}

def _summary_progress(value):
    if not isinstance(value,dict):
        return None
    return {k:value.get(k) for k in ("stage","completed","total",
            "total_generation_seconds","valid_exploratory","elapsed_seconds")
            if k in value}

def _summary_result(value):
    if not isinstance(value,dict):
        return None
    return {k:value.get(k) for k in ("status","run_id","repo_sha","error_type",
            "error","checkpoint_sha256","unique_prompts","paired_challenges",
            "training_performed") if k in value}

def inspect(client, bucket):
    unique,pairs=panel()
    pinned=reference_manifest()
    expected={x["prompt_sha256"] for x in unique}
    receipts=[]
    listing_error=None
    try:
        listing=list_run(client,bucket)
    except Exception as exc:
        listing=[]
        listing_error=type(exc).__name__  # no secret-bearing exception strings
    bootstrap=fetch_json(client,bucket,ROOT+"/bootstrap/progress.json",receipts)
    manifest=fetch_json(client,bucket,ROOT+"/manifest.json",receipts)
    progress=fetch_json(client,bucket,ROOT+"/progress.json",receipts)
    result=fetch_json(client,bucket,ROOT+"/result.json",receipts)
    # Historical claim is read without mutation, to avoid any notion of retry.
    claim=fetch_json(client,bucket,APPROVAL_KEY,receipts)

    shard_data=[]
    shards=[]
    for i in range(1,math.ceil(len(unique)/8)+1):
        key=f"{ROOT}/samples/shard-{i:04d}.json"
        value=fetch_json(client,bucket,key,receipts)
        if value is None:
            continue
        if not isinstance(value,list) or not 0<len(value)<=8:
            raise ValueError(f"invalid saved sample shard: {key}")
        shards.append(dict(key=key,rows=len(value)))
        shard_data.extend(value)
    prompt_hashes=[]
    bad_checkpoint=0
    latency=[]
    token_counts=[]
    parse_formats=Counter()
    valid=0
    pred={}
    for row in shard_data:
        if not isinstance(row,dict):
            raise ValueError("invalid sample row")
        h=row.get("prompt_sha256")
        if h not in expected:
            raise ValueError("unknown prompt identity in saved samples")
        prompt_hashes.append(h)
        if row.get("checkpoint_sha256")!=D_CHECKPOINT_SHA256:
            bad_checkpoint+=1
        parsed=parse_decision(row.get("raw"))
        parse_formats[parsed["mode"]]+=1
        valid+=int(parsed["strict"] is not None)
        pred[h]=parsed["exploratory"]
        t=row.get("latency_seconds")
        if type(t) in (int,float) and math.isfinite(t) and t>=0:
            latency.append(t)
        n=row.get("generated_tokens")
        if type(n)==int and n>=0:
            token_counts.append(n)
    if len(prompt_hashes)!=len(set(prompt_hashes)):
        raise ValueError("duplicate prompt hashes in persisted sample shards")

    both_observed=both_correct=both_transition_consistent=0
    observed_pair_by_transition=Counter()
    for pair in pairs:
        a=pair["baseline_prompt_sha256"]
        b=pair["modified_prompt_sha256"]
        if a not in pred or b not in pred:
            continue
        both_observed+=1
        observed_pair_by_transition[pair["transition"]]+=1
        got_a,got_b=pred[a],pred[b]
        both_correct+=int(got_a==pair["baseline_state"] and
                          got_b==pair["modified_state"])
        both_transition_consistent+=int(
            got_a is not None and got_b is not None and
            ((got_a!=got_b)==pair["scientific_change_expected"])
        )
    manifest_id=None
    if isinstance(manifest,dict):
        manifest_id={k:manifest.get(k) for k in
            ("protocol","D_checkpoint_sha256","prompts_sha256",
             "pairs_sha256","unique_prompts","paired_challenges")}
    expected_id={k:pinned.get(k) for k in
            ("protocol","D_checkpoint_sha256","prompts_sha256",
             "pairs_sha256","unique_prompts","paired_challenges")}
    manifest_valid=manifest_id==expected_id if manifest_id else None
    by_key={x["key"]:x for x in receipts}
    listing_metas={x["key"]:x for x in listing}

    # Observed last writes yield lower/upper timing constraints, not detailed
    # model-init measurements. Bootstrap overwrites its earlier stages.
    boot_time=None
    if isinstance(bootstrap,dict) and type(bootstrap.get("timestamp_unix")) in (int,float):
        boot_time=datetime.fromtimestamp(bootstrap["timestamp_unix"],tz=timezone.utc).isoformat()
    manifest_time=by_key.get(ROOT+"/manifest.json",{}).get("last_modified_utc")
    progress_time=by_key.get(ROOT+"/progress.json",{}).get("last_modified_utc")
    saved_shard_times=[by_key[x["key"]]["last_modified_utc"] for x in shards]
    created=OBSERVED_POD_CREATE_UTC
    deleted=OBSERVED_POD_DELETE_UTC
    timeline=dict(
        pod_create_utc=created,pod_delete_utc=deleted,
        pod_lifetime_seconds=elapsed(created,deleted),
        last_saved_bootstrap_stage=_summary_bootstrap(bootstrap),
        last_saved_bootstrap_timestamp_utc=boot_time,
        manifest_write_utc=manifest_time,
        progress_last_write_utc=progress_time,
        shard_write_utc=saved_shard_times,
        pod_to_last_bootstrap_seconds=elapsed(created,boot_time),
        pod_to_manifest_seconds=elapsed(created,manifest_time),
        manifest_to_first_shard_seconds=elapsed(manifest_time,saved_shard_times[0]) if saved_shard_times else None,
        pod_to_last_shard_seconds=elapsed(created,saved_shard_times[-1]) if saved_shard_times else None,
    )
    if manifest is None:
        inference_readiness="no_model_loaded_manifest_seen; model init not proven"
    elif shards:
        inference_readiness="model_manifest_and_some_completed_samples_saved"
    elif progress is not None:
        inference_readiness="model_manifest_and_progress_seen_without_sample_shards"
    else:
        inference_readiness="model_manifest_seen_but_no_saved_sample_shards"

    claim_info=None
    if isinstance(claim,dict):
        claim_info={k:claim.get(k) for k in ("runid","sha","protocol")}
    return dict(
        run_id=RUN_ID,run_id_under_audit_only=True,read_only_s3=True,
        s3_operations=["list_objects_v2","head_object","get_object"],
        writes_to_s3=0,gpu_allocation=0,no_training_or_certification=True,
        expected_unique_prompts=len(unique),expected_pairs=len(pairs),
        s3_listing_error_type=listing_error,
        s3_keys=listing,read_receipts=receipts,
        claim=claim_info,bootstrap=_summary_bootstrap(bootstrap),
        manifest=manifest_id,manifest_matches_frozen_protocol=manifest_valid,
        progress=_summary_progress(progress),terminal_result=_summary_result(result),
        shard_inventory=shards,persisted_unique_generations=len(shard_data),
        distinct_prompt_hashes=len(set(prompt_hashes)),
        checkpoint_mismatch_count=bad_checkpoint,
        strict_format_valid_count=valid,parse_modes=dict(parse_formats),
        generated_tokens_sum=sum(token_counts) if token_counts else None,
        generation_latency_seconds_sum=round(sum(latency),3) if latency else None,
        generation_latency_seconds_max=max(latency) if latency else None,
        fully_observed_pairs=both_observed,
        fully_observed_pairs_both_correct=both_correct,
        fully_observed_pairs_transition_consistent=both_transition_consistent,
        observed_pair_transitions=dict(sorted(observed_pair_by_transition.items())),
        timeline=timeline,inference_readiness=inference_readiness,
        interpretive_limit="Missing shards do not prove no generations happened: unfinished batches of up to 7 are never persisted. Last bootstrap stage overwrites earlier timestamps. Partial pairs are selected and not a valid full 60-pair scientific score.",
    )

def main():
    client=s3.s3_client()
    bucket=s3.bucket()
    audit=inspect(client,bucket)
    OUTPUT.write_text(json.dumps(audit,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")
    concise={k:audit[k] for k in (
        "run_id","expected_unique_prompts","expected_pairs","s3_listing_error_type",
        "claim","bootstrap","manifest","manifest_matches_frozen_protocol",
        "progress","terminal_result","shard_inventory",
        "persisted_unique_generations","checkpoint_mismatch_count",
        "strict_format_valid_count","parse_modes","generated_tokens_sum",
        "generation_latency_seconds_sum","fully_observed_pairs",
        "fully_observed_pairs_both_correct","fully_observed_pairs_transition_consistent",
        "timeline","inference_readiness","interpretive_limit"
    )}
    print("D_CONTRAST_S3_READONLY_FORENSICS "+json.dumps(concise,sort_keys=True,default=str),flush=True)

if __name__=="__main__":
    main()
