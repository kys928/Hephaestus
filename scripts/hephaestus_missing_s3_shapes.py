#!/usr/bin/env python3
"""Summarize 66 S3 NoSuchKey cases by category, file type and object size, no keys printed."""
from collections import Counter
from pathlib import PurePosixPath
import json,os
import boto3
s3=boto3.client("s3",region_name="EU-CZ-1",endpoint_url="https://s3api-eu-cz-1.runpod.io/",aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"])
obj=s3.get_object(Bucket="cviwpryzao",Key="hephaestus/recovery-audit/source-get-probe-43259691731.json")
with obj["Body"] as stream:d=json.load(stream)
rows=d["issues"]
byroot=Counter();ext=Counter();size=Counter(); parent=Counter()
for row in rows:
 p=PurePosixPath(row["key"])
 byroot[str(p.parts[0])]+=1
 ext[p.suffix.lower() or "<no extension>"]+=1
 size["zero" if row["size"]==0 else "nonzero"]+=1
 parts=p.parts
 parent["/".join(parts[:3]) if len(parts)>=3 else "short"]+=1
print("MISSING_SOURCE_SHAPES="+json.dumps({"total":len(rows),"by_root":byroot,"by_extension":ext,"by_size_zero_or_not":size,"by_top_three_path_parts":parent},sort_keys=True))
