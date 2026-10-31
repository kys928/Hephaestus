#!/usr/bin/env python3
"""Read-only inspect actual SQLite bytes for all positive-promotion S3 candidates.

A fully paginated 45,990-key S3 metadata scan found 16 possible SQLite DBs
under the positive_promotion prefix. This opens only those bounded DB files,
checks schema and counts of the ORIGINAL staged workflow collection. It never
uploads DB bytes, reads training corpora, changes storage or asserts missing
history from unscanned paths.
"""
from __future__ import annotations
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import tempfile

import boto3
from botocore.config import Config

PREFIX="hephaestus/scientific/v1/positive_promotion/"
MAX_DB_FILES=32
MAX_FILE_BYTES=512*1024
STATE="staged_workflow_states"
RECORDS="staged_workflow_records"


def audit(client,bucket):
    marker=None
    objects=[]
    while True:
        kw={"Bucket":bucket,"Prefix":PREFIX,"MaxKeys":1000}
        if marker:kw["ContinuationToken"]=marker
        page=client.list_objects_v2(**kw)
        objects.extend([
            {"key":row["Key"],"size":int(row["Size"])}
            for row in page.get("Contents",[])
            if row["Key"].endswith("/production_state/production.sqlite3")
        ])
        if not page.get("IsTruncated"):break
        marker=page.get("NextContinuationToken")
        if not marker:raise ValueError("truncated state DB inventory has no cursor")
    if not objects or len(objects)>MAX_DB_FILES:
        raise ValueError("unexpected positive-promotion DB candidate count")
    scanned=[]
    matches=[]
    with tempfile.TemporaryDirectory(prefix="hephaestus-verified-state-only-") as tmp:
        for i,entry in enumerate(objects):
            if entry["size"]>MAX_FILE_BYTES:
                raise ValueError("state candidate exceeds bounded read")
            data=client.get_object(Bucket=bucket,Key=entry["key"])["Body"].read(MAX_FILE_BYTES+1)
            if len(data)!=entry["size"] or not data.startswith(b"SQLite format 3\x00"):
                raise ValueError("candidate sqlite byte identity or file signature failure")
            db=Path(tmp)/f"db-{i}.sqlite3"
            db.write_bytes(data)
            uri=db.as_uri()+"?mode=ro&immutable=1"
            with sqlite3.connect(uri,uri=True) as connection:
                connection.execute("PRAGMA query_only=ON")
                connection.execute("PRAGMA trusted_schema=OFF")
                tables={str(row[0]) for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
                count=0
                if "state_records" in tables:
                    count=int(connection.execute(
                        "SELECT count(*) FROM state_records WHERE collection IN (?,?)",
                        (STATE,RECORDS)).fetchone()[0])
            item={
                "path":entry["key"],
                "bytes":len(data),
                "sha256":"sha256:"+sha256(data).hexdigest(),
                "contains_expected_state_records_table":"state_records" in tables,
                "governed_campaign_collection_rows":count,
                "original_database_bytes_not_persisted":True,
            }
            if count:matches.append(item)
            scanned.append(item)
    return {
        "schema":"hephaestus-archived-positive-promotion-state-audit.v1",
        "prefix":PREFIX,"entire_prefix_scanned":True,
        "sqlite_files_read_only":len(scanned),
        "candidate_db_metadata":scanned,
        "databases_with_staged_workflow_records":len(matches),
        "matching_database_metadata":matches,
        "actual_11_stage_test_history_recovered":False,
        "source_11_stage_pytest_archive_contains_db":False,
        "no_original_state_uploaded":True,
        "no_dataset_acquisition_or_gpu":True,
    }


def main():
    client=boto3.client("s3",endpoint_url=os.environ["RUNPOD_S3_ENDPOINT_URL"],
        region_name=os.environ["RUNPOD_DATACENTER_ID"],
        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"))
    r=audit(client,os.environ["RUNPOD_NETWORK_VOLUME_ID"])
    Path("original_state_sqlite_readonly_audit.json").write_text(
        json.dumps(r,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "dbs_opened":r["sqlite_files_read_only"],
        "dbs_with_real_staged_campaign_rows":r["databases_with_staged_workflow_records"],
        "original_11_test_state_recovered":r["actual_11_stage_test_history_recovered"],
    },sort_keys=True))


if __name__=="__main__":
    main()
