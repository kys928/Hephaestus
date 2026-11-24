#!/usr/bin/env python3
"""Read-only S3 object inventory and storage-custody classification.
Does not read private file contents or log individual object keys.
Writes private inventory JSON to the original source S3 under recovery-audit/.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib, json, os
from pathlib import PurePosixPath
import boto3
from botocore.config import Config

SRC="cviwpryzao"
ROOTS=("hephaestus/scientific/","hephaestus/research/","Hephaestus/artifacts/","hephaestus-cache/")
MUTABLE_EXT=(".sqlite",".sqlite3",".db",".db-wal",".db-shm",".sqlite-wal",".sqlite-shm",".lock",".log",".tmp",".pid",".sock")
SENSITIVE_EXT=(".pem",".key",".p12",".pfx",".env")
ARCHIVE_EXT=(".zip",".tar",".tgz",".tar.gz",".7z",".zst")
MODEL_EXT=(".safetensors",".gguf",".onnx",".ckpt",".pt",".pth")
DATA_EXT=(".parquet",".arrow",".csv",".tsv",".jsonl",".npy",".npz")
CODE_EXT=(".py",".sh",".ipynb",".bat",".ps1")
OPER_PARTS={"logs","tmp","temp","worktrees","lock","locks","state","states","runtime","active"}
PRIV_PARTS={"private","protected","holdout","holdouts","secrets","credentials","keys","eval","evals","evaluation","evaluations","certification","certifications","eval_pack","eval_packs"}
MODEL_PARTS={"models","model","weights","adapter","adapters","tokenizers","tokenizer"}
DATA_PARTS={"data","dataset","datasets","train","training_data","corpus","corpora","preprocessed","trainable"}
SECRET_WORDS=("secret","credential","api_key","access_key","password","private_key","id_rsa","id_ed25519")
def classify(key):
 p=PurePosixPath(key);parts={x.lower() for x in p.parts};lower=key.lower();name=p.name.lower()
 if lower.startswith("hephaestus-cache/"):return "third_party_upstream_cache"
 if any(x in name for x in SECRET_WORDS) or name.startswith(".env") or name.endswith(SENSITIVE_EXT) or "secrets" in parts:return "confidential_or_credentials"
 if any(x in PRIV_PARTS for x in parts) or any(x in lower for x in ("holdout","private_eval","protected_eval","certification")):return "protected_evaluation_or_custody"
 if name.endswith(MUTABLE_EXT) or any(x in OPER_PARTS for x in parts) or any(x in name for x in ("state.sqlite","optimizer","scheduler","rng_state","trainer_state","heartbeat","worker.log","watchdog")):return "mutable_or_operational_state"
 if name.endswith(ARCHIVE_EXT):return "archives_requires_separate_validation"
 if name.endswith(CODE_EXT):return "code_or_development_from_github"
 if name.endswith(MODEL_EXT) or parts&MODEL_PARTS:return "model_or_adapter_candidates"
 if name.endswith(DATA_EXT) or parts&DATA_PARTS:return "dataset_candidates"
 if lower.startswith(("hephaestus/scientific/","hephaestus/research/")):return "research_evidence_or_unclassified"
 return "misc_unclassified"
def main():
 keys=("RUNPOD_S3_ACCESS_KEY_ID","RUNPOD_S3_SECRET_ACCESS_KEY","GITHUB_RUN_ID")
 if any(not os.environ.get(k) for k in keys):raise RuntimeError("Missing private S3 access or run identity")
 s3=boto3.client("s3",endpoint_url="https://s3api-eu-cz-1.runpod.io/",region_name="EU-CZ-1",aws_access_key_id=os.environ[keys[0]],aws_secret_access_key=os.environ[keys[1]],config=Config(retries={"max_attempts":7,"mode":"standard"},read_timeout=90))
 s3.head_bucket(Bucket=SRC)
 pages=s3.list_objects_v2(Bucket=SRC,Delimiter="/",MaxKeys=1000)
 top=sorted(x["Prefix"] for x in pages.get("CommonPrefixes",[]))
 inventory=[];summary={};totals=Counter();sizes=Counter()
 for prefix in ROOTS:
  counts=Counter();bytes_by=Counter();n=0;size=0
  for page in s3.get_paginator("list_objects_v2").paginate(Bucket=SRC,Prefix=prefix):
   for obj in page.get("Contents",[]):
    key=obj["Key"];category=classify(key);amount=int(obj["Size"]);n+=1;size+=amount
    counts[category]+=1;bytes_by[category]+=amount;totals[category]+=1;sizes[category]+=amount
    inventory.append({"key":key,"bytes":amount,"category":category,"etag":str(obj.get("ETag","")),"last_modified":obj.get("LastModified").isoformat() if obj.get("LastModified") else None})
  summary[prefix]={"files":n,"bytes":size,"categories":{cat:{"files":counts[cat],"bytes":bytes_by[cat]} for cat in sorted(counts)}}
 record={"schema":"hephaestus.original-s3-storage-inventory.v1","source_volume_id":SRC,"recorded_utc":datetime.now(timezone.utc).isoformat(),"top_level_prefixes":top,"prefix_summary":summary,"category_summary":{cat:{"files":totals[cat],"bytes":sizes[cat]} for cat in sorted(totals)},"objects":inventory,"warning":"Categories derived from path and extension only; not a rights, immutability, training admissibility or independent hash audit"}
 raw=(json.dumps(record,sort_keys=True,separators=(",",":"))+"\n").encode()
 dest="hephaestus/recovery-audit/source-inventory-"+os.environ["GITHUB_RUN_ID"]+".json"
 try:
  s3.head_object(Bucket=SRC,Key=dest)
 except s3.exceptions.ClientError as ex:
  if str(ex.response.get("Error",{}).get("Code","")) not in {"404","NoSuchKey","NotFound"}:raise
  s3.put_object(Bucket=SRC,Key=dest,Body=raw,ContentType="application/json")
 else:raise RuntimeError("Inventory snapshot key already exists")
 preview={"schema":record["schema"],"source_volume_id":SRC,"top_level_prefixes":top,"prefix_summary":summary,"category_summary":record["category_summary"],"private_inventory_s3_key":dest,"private_inventory_sha256":hashlib.sha256(raw).hexdigest()}
 print("SOURCE_STORAGE_AUDIT="+json.dumps(preview,sort_keys=True),flush=True)
 print("SOURCE_STORAGE_AUDIT_COMPLETE=true",flush=True)
if __name__=="__main__":main()
