#!/usr/bin/env python3
"""Resumable, single-writer Czech S3 evidence recovery to pinned Global Volume.
Read-only on source. Copies missing files; checks SHA256 and never overwrites
different destination bytes. This is an S3 parity receipt, NOT HF certification.
"""
import hashlib, json, os, re, sys, tempfile, time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import boto3
from botocore.config import Config

SOURCE="cviwpryzao"
GLOBAL="cmw51pjsk000007l9a0mv1jga"
ROOT=Path("/workspace")
BASE=ROOT/"hephaestus/hf-restores/export-129d690ef9ae24ef141fbb0c"
DEST=BASE/"evidence"
ARCHIVE=ROOT/"hephaestus/artifacts/archives/original_cuda_45/cuda-json-45-original-five-20261030-r9.tar"
ROOTS={"hephaestus/scientific/":"scientific","hephaestus/research/":"research","Hephaestus/artifacts/":"legacy/artifacts"}
BAD=set(".git .hg .svn .cache .local .venv venv __pycache__ node_modules site-packages tmp temp private protected holdout holdouts secrets secret credentials keys status locks worktrees state eval evals evaluation evaluations eval_pack eval_packs protected_evals certification certifications".split())
BAD_SUFFIX=(".lock",".log",".tmp",".sqlite",".sqlite3",".db",".pem",".key",".p12",".pfx",".zip",".tar",".tar.gz",".tgz",".7z",".pyc",".pyo",".pkl",".pickle")
BAD_BITS=("secret","credential","access_key","api_key","password","token_secret","worker.log","watchdog","heartbeat","private_key","account.json","holdout","certification","private_eval","protected_eval")
MODEL_EXT={".safetensors",".gguf",".onnx",".pt",".pth",".ckpt"}
MODEL_CONF=set("adapter_config.json config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt added_tokens.json tokenizer.model adapter_model.bin pytorch_model.bin model.safetensors.index.json".split())
DATA_EXT={".jsonl",".parquet",".arrow",".csv",".tsv",".npy",".npz",".bin"}
NON_ART={".py",".sh",".bat",".ps1",".ipynb",".so",".o",".exe",".dll"}
TEXT_EXT={".json",".jsonl",".txt",".md",".yml",".yaml",".toml",".csv",".tsv",".ini",".cfg"}
SECRET=re.compile(rb"(?:hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)",re.I)
BUF=1024*1024
def reject(p):
 n=p.name.lower()
 return (not n or n.startswith(".") or n in {".env","id_rsa","id_ed25519","known_hosts","authorized_keys","config.json.bak"} or n.endswith(BAD_SUFFIX)
         or any(x in n for x in BAD_BITS) or any(x.lower() in BAD for x in p.parts) or p.suffix.lower() in NON_ART)
def kind(p):
 parts=[x.lower() for x in p.parts];n=p.name.lower();s=p.suffix.lower()
 model=any(x in {"models","model","adapters","adapter","tokenizers","tokenizer","weights"} for x in parts)
 data=any(x in {"dataset","datasets","training_data","corpus","corpora","preprocessed","trainable"} for x in parts)
 if any(x in n for x in ("optimizer","scheduler","rng_state","trainer_state","scaler.pt")):return "skip"
 if n in MODEL_CONF and model:return "models"
 if s in MODEL_EXT:return "models" if model or s in {".safetensors",".gguf"} else "skip"
 if (data and s in DATA_EXT) or (n.startswith("trainable") and s in DATA_EXT) or (data and (n.endswith("manifest.json") or n.endswith("contract.json"))):return "datasets"
 return "evidence"
def sha_file(path):
 h=hashlib.sha256();size=0
 with path.open("rb") as f:
  while part:=f.read(BUF):h.update(part);size+=len(part)
 return h.hexdigest(),size
def run():
 if not ROOT.is_mount() or not ARCHIVE.is_file() or os.environ.get("RUNPOD_GLOBAL_VOLUME_ID")!=GLOBAL:
  raise RuntimeError("Wrong/unmounted Global Volume; refusing writes")
 s3=boto3.client("s3",endpoint_url="https://s3api-eu-cz-1.runpod.io/",region_name="EU-CZ-1",
  aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
  config=Config(retries={"max_attempts":6,"mode":"standard"},connect_timeout=30,read_timeout=120))
 s3.head_bucket(Bucket=SOURCE)
 tag=os.environ.get("RECOVERY_RUN_ID","")
 if not tag.isdigit():raise RuntimeError("Invalid recovery identity")
 status_key="hephaestus/recovery-status/"+tag+".json"
 def status(state,done,total,extra=None):
  message={"schema":"hephaestus.s3-evidence-progress.v1","run_id":tag,"status":state,"verified":done,"total":total,"updated_utc":datetime.now(timezone.utc).isoformat()}
  if extra:message.update(extra)
  s3.put_object(Bucket=SOURCE,Key=status_key,Body=json.dumps(message,sort_keys=True).encode("utf-8"),ContentType="application/json")
  print("RECOVERY_PROGRESS="+str(done)+"/"+str(total)+"; state="+state,flush=True)
 rows=[]
 for prefix,destroot in ROOTS.items():
  for page in s3.get_paginator("list_objects_v2").paginate(Bucket=SOURCE,Prefix=prefix):
   for obj in page.get("Contents",[]):
    key=obj["Key"]
    if not key.startswith(prefix) or key.endswith("/"):continue
    tail=key[len(prefix):]
    if (not tail or "//" in tail or "\\" in tail or any(part in ("",".","..") for part in tail.split("/"))):raise RuntimeError("Unsafe source path")
    rel=PurePosixPath(tail)
    if reject(rel) or kind(rel)!="evidence":continue
    rows.append((destroot+"/"+tail,key,int(obj["Size"]),str(obj.get("ETag",""))))
 rows.sort()
 if len(rows)>14000 or sum(x[2] for x in rows)>2*1024**3:raise RuntimeError("S3 inventory exceeds approved evidence bound")
 if len({x[0] for x in rows})!=len(rows):raise RuntimeError("Duplicate target path")
 status("STARTED",0,len(rows),{"source_candidate_bytes":sum(x[2] for x in rows)})
 DEST.mkdir(parents=True,exist_ok=True)
 audit=[];existing=0;copied=0;skipped_sensitive=0
 with tempfile.TemporaryDirectory(prefix="hephaestus-s3-recover-") as folder:
  temp=Path(folder)/"source"
  for index,(rel,key,size,etag) in enumerate(rows,1):
   h=hashlib.sha256();received=0;tail=b"";sensitive=False
   response=s3.get_object(Bucket=SOURCE,Key=key)
   with temp.open("wb") as out:
    stream=response["Body"]
    try:
     while block:=stream.read(BUF):
      out.write(block);h.update(block);received+=len(block)
      if PurePosixPath(rel).suffix.lower() in TEXT_EXT:
       if SECRET.search(tail+block):sensitive=True
       tail=block[-256:]
    finally:stream.close()
   if received!=size:raise RuntimeError("Source object changed size: "+rel)
   if sensitive:skipped_sensitive+=1;continue
   target=DEST/rel
   if target.is_symlink():raise RuntimeError("Symlink target refused")
   cursor=target.parent
   while cursor!=ROOT:
    if cursor.is_symlink():raise RuntimeError("Symlink parent refused")
    cursor=cursor.parent
   target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists():
    actual,n=sha_file(target)
    if (actual,n)!=(h.hexdigest(),size):raise RuntimeError("Existing Global file conflicts with S3 source: "+rel)
    existing+=1
   else:
    with temp.open("rb") as inp,target.open("xb") as out:
     while part:=inp.read(BUF):out.write(part)
    for attempt in range(3):
     if sha_file(target)==(h.hexdigest(),size):break
     time.sleep(2)
    else:raise RuntimeError("Global Volume readback mismatch: "+rel)
    copied+=1
   audit.append({"path":rel,"bytes":size,"sha256":h.hexdigest()})
   if index%50==0 or index==len(rows):status("RUNNING",len(audit),len(rows),{"existing":existing,"new":copied})
 for record in audit:
  if sha_file(DEST/record["path"])!=(record["sha256"],record["bytes"]):raise RuntimeError("FINAL_SHA_MISMATCH "+record["path"])
 if len(audit)!=8113:
  status("COUNT_MISMATCH",len(audit),len(rows),{"expected":8113,"excluded_sensitive":skipped_sensitive})
  raise RuntimeError("S3 evidence classifier count differs from HF export; refusing complete receipt")
 receipt=BASE/"S3_EVIDENCE_RECEIPT.json"
 report={"schema":"hephaestus.global-s3-evidence-parity.v1","source_volume":SOURCE,"global_volume":GLOBAL,"verified_files":len(audit),
  "verified_bytes":sum(r["bytes"] for r in audit),"preexisting":existing,"copied":copied,"files":audit,
  "note":"S3/source SHA256 parity only, not the Hugging Face export manifest verification"}
 raw=(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n").encode()
 if receipt.exists():
  if receipt.read_bytes()!=raw:raise RuntimeError("Existing S3 receipt differs; refusing overwrite")
 else:
  with receipt.open("xb") as out:out.write(raw)
 if receipt.read_bytes()!=raw:raise RuntimeError("Receipt readback failed")
 status("COMPLETE",len(audit),len(rows),{"receipt":str(receipt),"receipt_sha256":hashlib.sha256(raw).hexdigest()})
 print("S3_EVIDENCE_RESTORE_COMPLETE=true",flush=True)
if __name__=="__main__":
 try:run()
 except Exception as exc:
  print("S3_EVIDENCE_RESTORE_FAILED="+type(exc).__name__+": "+str(exc)[:180],flush=True)
  sys.exit(1)
