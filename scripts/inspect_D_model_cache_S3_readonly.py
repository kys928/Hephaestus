#!/usr/bin/env python3
"""Zero-GPU HEAD/GET/LIST inventory of archived D checkpoint and HF cache.

A directory listing alone cannot prove a Hugging Face snapshot is complete,
since cached snapshot files can be symlinks in POSIX storage.
"""
from __future__ import annotations
import json
from pathlib import Path
import run_interface_repair_v1 as s3
from evaluator_contrast_inference_protocol import D_SOURCE_RUN, D_SOURCE_SHA
from evaluator_semantic_factor_probe_protocol import D_CHECKPOINT_SHA256

ROOT=Path(__file__).resolve().parents[1]
RESULT=f"hephaestus/scientific/v5/evaluator_semantic_repair_v2/{D_SOURCE_RUN}/training/result.json"
BASE_REV="51dd4bc2ade4059a6bd87649d68aa11e4fb2529b"
SNAPSHOT=f"hephaestus-cache/huggingface/hub/models--ibm-granite--granite-3.3-8b-instruct/snapshots/{BASE_REV}/"
ADAPTER=f"hephaestus-cache/evaluator-semantic-invariance-repair-v1/adapters/{D_CHECKPOINT_SHA256}/"
OUT=Path("D-contrast-S3-cache-readiness.json")

def list_hint(client,bucket,prefix):
    try:
        r=client.list_objects_v2(Bucket=bucket,Prefix=prefix,MaxKeys=100)
        contents=r.get("Contents",[])
        return dict(list_allowed=True,listed=len(contents),
           listing_truncated=bool(r.get("IsTruncated")),
           config_json_seen=any(x["Key"]==prefix+"config.json" for x in contents),
           archive_seen=any(x["Key"].endswith("parent-adapter.tar.gz") for x in contents))
    except Exception as exc:
        return dict(list_allowed=False,error_type=type(exc).__name__)

def inspect(client,bucket):
    r=client.get_object(Bucket=bucket,Key=RESULT)
    try: saved=json.loads(r["Body"].read())
    finally:r["Body"].close()
    if saved.get("repo_sha")!=D_SOURCE_SHA or saved.get("status")!="training_complete":
        raise ValueError("source D training identity/status mismatch")
    matches=[x["artifact"] for x in saved["checkpoint_artifacts"] if x["optimizer_step"]==96]
    if len(matches)!=1 or matches[0].get("sha256")!=D_CHECKPOINT_SHA256:
        raise ValueError("frozen D step96 checkpoint identity mismatch")
    ckpt=matches[0]
    meta=client.head_object(Bucket=bucket,Key=ckpt["s3_key"])
    n=int(meta["ContentLength"])
    if n!=int(ckpt["bytes"]):
        raise ValueError("frozen D archive size drift")
    return dict(
       status="source_checkpoint_s3_head_verified",
       s3_read_only=True,allocations=0,training=False,
       base_model="ibm-granite/granite-3.3-8b-instruct",
       base_revision=BASE_REV,
       D_checkpoint_sha256=D_CHECKPOINT_SHA256,
       archived_adapter_bytes=n,
       source_training_result_status=saved["status"],
       persistent_base_snapshot_hint=list_hint(client,bucket,SNAPSHOT),
       persistent_adapter_cache_hint=list_hint(client,bucket,ADAPTER),
       snapshot_completeness="UNKNOWN: S3 prefix listing is not a complete on-pod HF local_files_only test; POSIX symlinks and caches are not fully verified.",
       download_feasibility="UNKNOWN: full base model bytes and HF bandwidth are not measured via this read-only check.",
    )

def main():
    evidence=inspect(s3.s3_client(),s3.bucket())
    OUT.write_text(json.dumps(evidence,indent=2,sort_keys=True)+"\n")
    print("D_CONTRAST_CACHE_READONLY "+json.dumps(evidence,sort_keys=True))

if __name__=="__main__":
    main()
