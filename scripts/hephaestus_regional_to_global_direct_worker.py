#!/usr/bin/env python3
"""Recover frozen 8,224 original S3 inventory evidence from regional filesystem to Global.

Read-only source regional mount, single Global writer. SHA256 checks source
and Global before reporting a receipt. No mutable state or protected material.
"""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path,PurePosixPath
import hashlib,json,os,stat,sys,time
import boto3
from botocore.config import Config

SOURCE_ID="cviwpryzao"
GLOBAL_ID="cmw51pjsk000007l9a0mv1jga"
SOURCE=Path("/workspace-source")
GLOBAL=Path("/workspace-global")
DEST=GLOBAL/"hephaestus/hf-restores/export-129d690ef9ae24ef141fbb0c"
EVIDENCE=DEST/"evidence"
ARCHIVE=GLOBAL/"hephaestus/artifacts/archives/original_cuda_45/cuda-json-45-original-five-20261030-r9.tar"
INVENTORY_KEY="hephaestus/recovery-audit/source-inventory-43256848721.json"
INVENTORY_SHA256="046dde9ebe9a712665081794586d502de5722f7f815a61567377414b680f5fb8"
ROOTS={"hephaestus/scientific/":"scientific","hephaestus/research/":"research","Hephaestus/artifacts/":"legacy/artifacts"}
BAD=set(".git .hg .svn .cache .local .venv venv __pycache__ node_modules site-packages tmp temp private protected holdout holdouts secrets secret credentials keys status locks worktrees state eval evals evaluation evaluations eval_pack eval_packs protected_evals certification certifications".split())
BAD_SUFFIX=(".lock",".log",".tmp",".sqlite",".sqlite3",".db",".pem",".key",".p12",".pfx",".zip",".tar",".tar.gz",".tgz",".7z",".pyc",".pyo",".pkl",".pickle")
BAD_BITS=("secret","credential","access_key","api_key","password","token_secret","worker.log","watchdog","heartbeat","private_key","account.json","holdout","certification","private_eval","protected_eval")
MODEL_EXT={".safetensors",".gguf",".onnx",".pt",".pth",".ckpt"}
MODEL_CONF=set("adapter_config.json config.json generation_config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.json merges.txt added_tokens.json tokenizer.model adapter_model.bin pytorch_model.bin model.safetensors.index.json".split())
DATA_EXT={".jsonl",".parquet",".arrow",".csv",".tsv",".npy",".npz",".bin"}
NON_ART={".py",".sh",".bat",".ps1",".ipynb",".so",".o",".exe",".dll"}
SECRET_SIGNS=(b"-----BEGIN PRIVATE KEY-----",b"github_pat_",b"ghp_",b"hf_")
BUF=4*1024*1024
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
def sha(path):
 h=hashlib.sha256();n=0
 with path.open("rb") as f:
  while blob:=f.read(BUF):h.update(blob);n+=len(blob)
 return h.hexdigest(),n
def safe_file(base,path,expected_size):
 if not path.is_relative_to(base) or path==base:raise RuntimeError("Path outside mounted volume")
 cursor=path
 while cursor!=base:
  if cursor.is_symlink():raise RuntimeError("Symlink in source/destination path")
  cursor=cursor.parent
 try:st=path.lstat()
 except FileNotFoundError:raise RuntimeError("Frozen regional source file disappeared or is missing")
 if not stat.S_ISREG(st.st_mode) or st.st_size!=expected_size:raise RuntimeError("Regional source file size/type changed")
def main():
 if os.environ.get("RUNPOD_NETWORK_VOLUME_ID")!=SOURCE_ID or os.environ.get("RUNPOD_GLOBAL_VOLUME_ID")!=GLOBAL_ID:
  raise RuntimeError("Wrong advertised volume identity")
 if not SOURCE.is_mount() or not GLOBAL.is_mount() or not ARCHIVE.is_file() or not (SOURCE/"hephaestus/scientific").is_dir():
  raise RuntimeError("Expected regional+global mounts not present or Global archive marker missing")
 run=os.environ.get("RECOVERY_RUN_ID","")
 if not run.isdecimal():raise RuntimeError("Invalid recovery run ID")
 s3=boto3.client("s3",endpoint_url="https://s3api-eu-cz-1.runpod.io/",region_name="EU-CZ-1",
  aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],config=Config(retries={"max_attempts":5,"mode":"standard"}))
 s3.head_bucket(Bucket=SOURCE_ID)
 def emit(state,checked,total,extra=None):
  rec={"schema":"hephaestus.regional-direct-recovery.v1","run_id":run,"status":state,"verified":checked,"total":total,"updated_utc":datetime.now(timezone.utc).isoformat()}
  if extra:rec.update(extra)
  s3.put_object(Bucket=SOURCE_ID,Key="hephaestus/recovery-status/"+run+".json",Body=json.dumps(rec,sort_keys=True).encode(),ContentType="application/json")
  print("DIRECT_RECOVERY_PROGRESS="+str(checked)+"/"+str(total)+" state="+state,flush=True)
 inv=s3.get_object(Bucket=SOURCE_ID,Key=INVENTORY_KEY)
 with inv["Body"] as f:raw=f.read()
 if hashlib.sha256(raw).hexdigest()!=INVENTORY_SHA256:raise RuntimeError("Frozen original S3 inventory digest changed")
 data=json.loads(raw)
 if data.get("schema")!="hephaestus.original-s3-storage-inventory.v1" or data.get("source_volume_id")!=SOURCE_ID:
  raise RuntimeError("Incorrect original S3 inventory")
 rows=[]
 for obj in data["objects"]:
  key=obj["key"]
  if not isinstance(key,str) or key.startswith("/") or "\\" in key or "//" in key or any(x in {"",".",".."} for x in key.split("/")):
   raise RuntimeError("Unsafe source inventory key")
  for prefix,name in ROOTS.items():
   if not key.startswith(prefix) or key.endswith("/"):continue
   tail=key[len(prefix):];rel=PurePosixPath(tail)
   if reject(rel) or kind(rel)!="evidence":break
   rows.append((name+"/"+tail,key,int(obj["bytes"])))
   break
 rows.sort()
 if len(rows)!=8224 or sum(item[2] for item in rows)!=607271450 or len({r[0] for r in rows})!=len(rows):
  raise RuntimeError("Frozen candidate set/size differs from independently audited 8,224 objects")
 emit("PREFLIGHT",0,len(rows))
 # Check the entire original frozen source *before* starting any new Global writes.
 for i,(rel,key,size) in enumerate(rows,1):
  safe_file(SOURCE,SOURCE/key,size)
 emit("SOURCE_PRESENT",0,len(rows))
 EVIDENCE.mkdir(parents=True,exist_ok=True)
 audit=[];existing=0;copied=0
 for i,(rel,key,size) in enumerate(rows,1):
  src=SOURCE/key
  safe_file(SOURCE,src,size)
  digest,n=sha(src)
  if n!=size:raise RuntimeError("Source bytes changed during local file read")
  target=EVIDENCE/rel
  if not target.is_relative_to(EVIDENCE):raise RuntimeError("Target path escaped")
  current=target.parent
  while current!=GLOBAL:
   if current.is_symlink():raise RuntimeError("Destination path contains a symlink")
   current=current.parent
  target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists() or target.is_symlink():
   if target.is_symlink() or not target.is_file() or sha(target)!=(digest,size):
    raise RuntimeError("Existing Global evidence differs from original source")
   existing+=1
  else:
   with src.open("rb") as f,target.open("xb") as out:
    while chunk:=f.read(BUF):out.write(chunk)
   for trial in range(3):
    if sha(target)==(digest,size):break
    time.sleep(3)
   else:raise RuntimeError("Global Volume SHA-256 readback failed")
   copied+=1
  audit.append({"path":rel,"bytes":size,"sha256":digest})
  if i%100==0 or i==len(rows):emit("RUNNING",len(audit),len(rows),{"existing":existing,"copied":copied})
 emit("FINAL_READBACK",len(audit),len(rows))
 for i,obj in enumerate(audit,1):
  target=EVIDENCE/obj["path"]
  if sha(target)!=(obj["sha256"],obj["bytes"]):raise RuntimeError("Final Global Volume hash mismatch")
  if i%1000==0 or i==len(audit):emit("FINAL_READBACK",i,len(audit),{"pass":"global_readback"})
 report={"schema":"hephaestus.global-s3-evidence-parity.v1","source_volume":SOURCE_ID,"global_volume":GLOBAL_ID,
  "source_access":"direct read-only regional filesystem mount","source_inventory_sha256":INVENTORY_SHA256,
  "verified_files":len(audit),"verified_bytes":sum(x["bytes"] for x in audit),
  "preexisting":existing,"copied":copied,"files":audit,"hf_export_expected_files":8113,
  "s3_source_count_differs_from_hf":True,"skipped_sensitive_files":0,
  "note":"Original frozen regional S3 inventory versus source regional mount and Global SHA256 readback; NOT Hugging Face export manifest parity"}
 encoded=(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n").encode()
 receipt=DEST/"S3_EVIDENCE_RECEIPT.json"
 if receipt.exists():
  if receipt.is_symlink() or receipt.read_bytes()!=encoded:raise RuntimeError("Existing receipt conflicts; refusing overwrite")
 else:
  with receipt.open("xb") as f:f.write(encoded)
 if receipt.read_bytes()!=encoded:raise RuntimeError("Receipt readback mismatch")
 emit("COMPLETE",len(audit),len(rows),{"receipt":str(receipt),"receipt_sha256":hashlib.sha256(encoded).hexdigest(),"scope":"regional mount parity, NOT HF parity"})
 print("REGIONAL_DIRECT_EVIDENCE_COMPLETE=true",flush=True)
if __name__=="__main__":
 try:main()
 except Exception as ex:
  print("REGIONAL_DIRECT_RECOVERY_FAILED="+type(ex).__name__,flush=True)
  run=os.environ.get("RECOVERY_RUN_ID","")
  if run.isdecimal():
   try:
    s3=boto3.client("s3",endpoint_url="https://s3api-eu-cz-1.runpod.io/",region_name="EU-CZ-1",
     aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"])
    raw=json.dumps({"schema":"hephaestus.regional-direct-recovery.v1","run_id":run,"status":"FAILED","error_class":type(ex).__name__,"failure_kind":str(ex)[:110],"updated_utc":datetime.now(timezone.utc).isoformat()},sort_keys=True).encode()
    s3.put_object(Bucket=SOURCE_ID,Key="hephaestus/recovery-status/"+run+".json",Body=raw,ContentType="application/json")
   except Exception as report_exc:print("FAILED_STATUS_WRITE_ERROR="+type(report_exc).__name__,flush=True)
  sys.exit(1)
