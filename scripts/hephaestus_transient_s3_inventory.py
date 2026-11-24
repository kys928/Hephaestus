#!/usr/bin/env python3
"""Classify original S3 top-level transient prefixes without exposing individual keys."""
import os,json
from collections import Counter
import boto3
PREFIXES=(".cache/","codex-worktrees/","hephaestus-tmp/")
def main():
 s3=boto3.client("s3",region_name="EU-CZ-1",endpoint_url="https://s3api-eu-cz-1.runpod.io/",aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"])
 result={}
 for root in PREFIXES:
  count=0;size=0;recent=0
  for page in s3.get_paginator("list_objects_v2").paginate(Bucket="cviwpryzao",Prefix=root):
   for item in page.get("Contents",[]):
    count+=1;size+=int(item["Size"])
  result[root]={"object_count":count,"total_bytes":size}
 print("ADDITIONAL_ROOTS="+json.dumps(result,sort_keys=True))
 print("ADDITIONAL_ROOTS_INVENTORY_COMPLETE=true")
if __name__=="__main__":main()
