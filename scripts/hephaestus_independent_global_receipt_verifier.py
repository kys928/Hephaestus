#!/usr/bin/env python3
"""Independent read-only verifier for Hephaestus original S3 -> Global Volume evidence receipt.

Does not modify, repair, or delete Global Volume content. Reads a previously
completed, hash-anchored receipt and rehashes every referenced Global file.
Emits only coarse progress to the private original source S3.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import re
import sys
import time
import stat
import boto3
from botocore.config import Config

SOURCE="cviwpryzao"
GLOBAL="cmw51pjsk000007l9a0mv1jga"
SOURCE_RUN="43248214667"
ROOT=Path("/workspace")
BASE=ROOT/"hephaestus/hf-restores/export-129d690ef9ae24ef141fbb0c"
RECEIPT=BASE/"S3_EVIDENCE_RECEIPT.json"
EXPECTED_ARCHIVE=ROOT/"hephaestus/artifacts/archives/original_cuda_45/cuda-json-45-original-five-20261030-r9.tar"
SHA=re.compile(r"^[0-9a-f]{64}$")
BUF=1024*1024
MAX_RECEIPT_BYTES=16*1024*1024

def stamp():
 return datetime.now(timezone.utc).isoformat()

def hash_file(path):
 h=hashlib.sha256();total=0
 with path.open("rb") as stream:
  while chunk:=stream.read(BUF):h.update(chunk);total+=len(chunk)
 return h.hexdigest(),total

def valid_relative(raw):
 if not isinstance(raw,str) or not raw or len(raw)>1024 or raw.startswith("/") or "\\" in raw or "//" in raw:
  raise ValueError("Unsafe evidence receipt relative path")
 bits=raw.split("/")
 if any(s in {".","..",""} for s in bits) or bits[0] not in {"scientific","research","legacy"}:
  raise ValueError("Invalid evidence receipt path component")
 return Path(*bits)

def main():
 if os.environ.get("RUNPOD_GLOBAL_VOLUME_ID")!=GLOBAL or not ROOT.is_mount() or not EXPECTED_ARCHIVE.is_file():
  raise RuntimeError("Wrong Global Volume mount identity")
 run=os.environ.get("VERIFY_RUN_ID","")
 if not run.isdecimal():raise ValueError("Missing verifier run identity")
 s3=boto3.client("s3",endpoint_url="https://s3api-eu-cz-1.runpod.io/",region_name="EU-CZ-1",
  aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
  config=Config(retries={"max_attempts":6,"mode":"standard"},connect_timeout=30,read_timeout=120))
 s3.head_bucket(Bucket=SOURCE)
 origin=s3.get_object(Bucket=SOURCE,Key="hephaestus/recovery-status/"+SOURCE_RUN+".json")
 with origin["Body"] as f: source_status=json.load(f)
 if source_status.get("status")!="COMPLETE" or source_status.get("run_id")!=SOURCE_RUN:
  raise RuntimeError("Original S3 restore has no completed receipt status")
 expected_sha=source_status.get("receipt_sha256")
 if not isinstance(expected_sha,str) or not SHA.fullmatch(expected_sha):
  raise RuntimeError("Original S3 restore receipt SHA missing")
 if str(RECEIPT)!=source_status.get("receipt"):
  raise RuntimeError("Expected receipt path disagrees with original progress status")
 if RECEIPT.is_symlink() or not RECEIPT.is_file() or RECEIPT.stat().st_size>MAX_RECEIPT_BYTES:
  raise RuntimeError("Missing/unsafe receipt")
 raw=RECEIPT.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=expected_sha:
  raise RuntimeError("Global Volume receipt does not match S3-anchored SHA-256")
 def unique_json_pairs(pairs):
  result={}
  for k,v in pairs:
   if k in result:raise ValueError("Duplicate JSON key")
   result[k]=v
  return result
 receipt=json.loads(raw,object_pairs_hook=unique_json_pairs)
 if receipt.get("schema")!="hephaestus.global-s3-evidence-parity.v1" or receipt.get("source_volume")!=SOURCE or receipt.get("global_volume")!=GLOBAL:
  raise RuntimeError("Unexpected receipt schema/volume identity")
 records=receipt.get("files")
 if not isinstance(records,list) or len(records)<1 or len(records)>14000 or len(records)!=receipt.get("verified_files"):
  raise RuntimeError("Invalid evidence file count in receipt")
 if receipt.get("verified_files")!=source_status.get("verified"):
  raise RuntimeError("Receipt file count disagrees with original S3 status")
 status_key="hephaestus/recovery-verification/"+run+".json"
 def publish(state,checked):
  payload={"schema":"hephaestus.global-independent-verification.v1","origin_run_id":SOURCE_RUN,
   "verify_run_id":run,"status":state,"checked":checked,"total":len(records),"updated_utc":stamp()}
  if state=="COMPLETE":
   payload.update({"receipt_sha256":expected_sha,"verified_bytes":receipt["verified_bytes"],"scope":"S3 source parity, NOT HF manifest parity"})
  s3.put_object(Bucket=SOURCE,Key=status_key,Body=json.dumps(payload,sort_keys=True).encode(),ContentType="application/json")
  print("VERIFY_PROGRESS="+str(checked)+"/"+str(len(records))+" "+state,flush=True)
 publish("RUNNING",0)
 seen=set();count=0;total=0
 for obj in records:
  if not isinstance(obj,dict):raise ValueError("Nonobject file receipt row")
  path=valid_relative(obj.get("path"))
  relative=path.as_posix()
  if relative in seen:raise RuntimeError("Duplicate receipt file path")
  seen.add(relative)
  expected_hash=obj.get("sha256");size=obj.get("bytes")
  if not isinstance(expected_hash,str) or not SHA.fullmatch(expected_hash) or type(size) is not int or size<0:
   raise RuntimeError("Invalid receipt hash or byte size")
  target=BASE/"evidence"/path
  if not target.is_relative_to(BASE/"evidence"):raise RuntimeError("Destination escaped evidence root")
  cursor=target
  while cursor!=ROOT:
   if cursor.is_symlink():raise RuntimeError("Symlink in Global path")
   cursor=cursor.parent
  try:st=target.lstat()
  except FileNotFoundError:raise RuntimeError("Missing restored evidence file")
  if not stat.S_ISREG(st.st_mode) or st.st_size!=size:raise RuntimeError("Non-file or wrong byte length in restored evidence")
  got,n=hash_file(target)
  if got!=expected_hash or n!=size:raise RuntimeError("Restored evidence SHA-256 mismatch")
  count+=1;total+=size
  if count%500==0 or count==len(records):publish("RUNNING",count)
 if count!=receipt["verified_files"] or total!=receipt["verified_bytes"]:
  raise RuntimeError("Receipt aggregate byte/count mismatch")
 if hash_file(RECEIPT)[0]!=expected_sha:
  raise RuntimeError("Receipt changed during independent verification")
 publish("COMPLETE",count)
 print("INDEPENDENT_GLOBAL_EVIDENCE_SHA256_VERIFIED=true",flush=True)
 print("INDEPENDENT_VERIFIED_FILES="+str(count),flush=True)
 print("SOURCE_PARITY_ONLY=true",flush=True)
if __name__=="__main__":
 try:main()
 except Exception as ex:
  print("INDEPENDENT_GLOBAL_VERIFICATION_FAILED="+type(ex).__name__+": "+str(ex)[:200],flush=True)
  sys.exit(1)
