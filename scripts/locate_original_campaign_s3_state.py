#!/usr/bin/env python3
"""Paginated HEAD-free key index of the assigned existing S3 network volume.

Locate actual persisted campaign stores, not a guessed path. No object GET,
no data content, no dataset write, no training/GPU launch, no approval grant.
A finite cap must be reported as incomplete, not absence of state.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import PurePosixPath, Path

import boto3
from botocore.config import Config

MAX_LISTED = 120_000
PREFIX = ""  # Include state paths outside hephaestus/scientific/v1.


def is_candidate(key:str)->bool:
    low=key.lower()
    base=PurePosixPath(low).name
    return (
        base.endswith((".sqlite",".sqlite3",".db",".sqlite.gz",".sqlite3.gz"))
        or ("staged_workflow_states" in low)
        or ("staged_workflow_records" in low)
        or ("governed_campaign_state" in low)
        or ("campaign-state" in base and base.endswith((".json",".jsonl",".gz")))
        or (base in {"production.sqlite3","state.sqlite","workflow-state.json"})
    )


def inspect(client,bucket:str)->dict:
    counted=0
    page_count=0
    continuation=None
    candidates=[]
    top=Counter()
    seen=set()
    truncated=False
    while counted<MAX_LISTED:
        kwargs={"Bucket":bucket,"MaxKeys":1000,"Prefix":PREFIX}
        if continuation:
            kwargs["ContinuationToken"]=continuation
        response=client.list_objects_v2(**kwargs)
        page_count+=1
        for obj in response.get("Contents",[]):
            key=str(obj["Key"])
            counted+=1
            top[key.split("/")[0]]+=1
            if is_candidate(key):
                # Path is output only for state-looking candidate keys, not
                # arbitrary volume contents, API tokens or unrelated secrets.
                if key not in seen:
                    candidates.append({"key":key,"bytes":int(obj["Size"]),
                                       "sha256_key":"sha256:"+sha256(key.encode()).hexdigest(),
                                       "metadata_only_not_verified_sqlite":True})
                    seen.add(key)
        continuation=response.get("NextContinuationToken")
        if not response.get("IsTruncated",False):
            truncated=False
            break
        if not continuation:
            raise ValueError("S3 listing was truncated but has no cursor")
        truncated=True
    return {
        "schema":"hephaestus-s3-historical-state-locator.v1",
        "objects_listed":counted,"list_pages":page_count,"limit":MAX_LISTED,
        "entire_bucket_listed":not truncated,
        "listing_incomplete":truncated,
        "top_prefix_object_counts":dict(top),
        "candidate_state_objects":candidates[:80],
        "candidate_count":len(candidates),
        "state_database_bytes_read":0,
        "candidate_authenticity_verified":False,
        "historical_receipts_recovered":False,
        "training_data_acquired":False,
        "permissions_changed":False,
        "gpu_used":False,
    }


def main():
    client=boto3.client("s3",
        endpoint_url=os.environ["RUNPOD_S3_ENDPOINT_URL"],
        region_name=os.environ["RUNPOD_DATACENTER_ID"],
        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4",retries={"max_attempts":4,"mode":"standard"}))
    report=inspect(client,os.environ["RUNPOD_NETWORK_VOLUME_ID"])
    Path("historical_state_s3_locator.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        k:report[k] for k in (
            "objects_listed","list_pages","entire_bucket_listed",
            "candidate_count","historical_receipts_recovered"
        )},sort_keys=True))


if __name__=="__main__":
    main()
