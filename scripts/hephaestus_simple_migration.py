#!/usr/bin/env python3
"""One-shot HF evidence reconciliation + Czech-only project archive.

Only one download pass for genuinely absent HF evidence. Models and training
datasets are never retransferred. Keeps original source and old receipts.
"""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import hashlib, json, os, re, shutil, stat, sys, tarfile, tempfile, time, urllib.request
import boto3
from botocore.config import Config

HF_OWNER="DavidDulovic"
HF_REPOS={
 "models":("model","hephaestus-trained-models","13f743df8bfb",869),
 "datasets":("dataset","hephaestus-training-data","78c6dbc4e08b",4),
 "evidence":("dataset","hephaestus-scientific-evidence","50cce2d27351",8113),
}
MANIFEST_SHA="129d690ef9ae24ef141fbb0c1ce48280f033b33f03b691073c20231d16677638"
GLOBAL_ID="cmw51pjsk000007l9a0mv1jga"
CZECH_ID="cviwpryzao"
GLOBAL=Path("/workspace-global")
CZECH=Path("/workspace-source")
RESTORE=GLOBAL/"hephaestus/hf-restores"/("export-"+MANIFEST_SHA[:24])
EXTRAS=GLOBAL/"hephaestus/migrations/hf-czech-20261113"
KNOWN_ARCHIVE=GLOBAL/"hephaestus/artifacts/archives/original_cuda_45/cuda-json-45-original-five-20261030-r9.tar"
PREFIXES=("hephaestus/scientific/","hephaestus/research/","Hephaestus/artifacts/")
SHA_RE=re.compile(r"^[0-9a-f]{64}$")
BUF=4*1024*1024
MAX_EXTRAS_BYTES=12*1024**3
MAX_EXTRAS_OBJECTS=14000
SKIP_PARTS={"cache","caches",".cache","tmp","temp","worktrees","__pycache__","node_modules",".git",".venv","venv"}
SKIP_NAMES=("id_rsa","id_ed25519",".env","credentials.json","authorized_keys")
SECRET_PATTERN=re.compile(r"(secret|credential|access[_-]?key|api[_-]?key|password|private[_-]?key|oauth[_-]?token|\\.pem$|\\.p12$|\\.pfx$)",re.I)

def sha(path):
 h=hashlib.sha256();n=0
 with path.open("rb") as f:
  while chunk:=f.read(BUF):h.update(chunk);n+=len(chunk)
 return h.hexdigest(),n

def safe_rel(p):
 if not isinstance(p,str) or not p or len(p)>1024 or p.startswith("/") or "\\" in p or "//" in p or "\x00" in p:
  raise ValueError("Invalid path")
 bits=p.split("/")
 if any(x in ("",".","..") for x in bits):raise ValueError("Invalid relative path")
 return Path(*bits)

def read_url(url,token):
 req=urllib.request.Request(url,headers={"Authorization":"Bearer "+token,"Accept":"application/json"})
 with urllib.request.urlopen(req,timeout=90) as response:return response.read()

def no_dup(pairs):
 out={}
 for key,value in pairs:
  if key in out:raise ValueError("Duplicate manifest JSON property")
  out[key]=value
 return out

def stable_receipt(path,data):
 raw=(json.dumps(data,sort_keys=True,separators=(",",":"))+"\n").encode()
 if path.is_symlink():raise RuntimeError("Unsafe receipt symlink")
 if path.exists():
  if path.read_bytes()!=raw:raise RuntimeError("Existing deterministic receipt conflicts")
 else:
  with path.open("xb") as out:out.write(raw)
 if path.read_bytes()!=raw:raise RuntimeError("Receipt reread mismatch")
 return hashlib.sha256(raw).hexdigest()

def main():
 run=os.environ.get("RECOVERY_RUN_ID","")
 if not run.isdecimal():raise RuntimeError("Invalid run identity")
 token=os.environ.get("HF_TOKEN")
 if not token:raise RuntimeError("Missing private HF read token")
 if (os.environ.get("RUNPOD_GLOBAL_VOLUME_ID")!=GLOBAL_ID or
     os.environ.get("RUNPOD_NETWORK_VOLUME_ID")!=CZECH_ID or
     not GLOBAL.is_mount() or not CZECH.is_mount() or not KNOWN_ARCHIVE.is_file()):
  raise RuntimeError("Expected source Czech and destination Global mounts not verified")
 s3=boto3.client("s3",endpoint_url="https://s3api-eu-cz-1.runpod.io/",region_name="EU-CZ-1",
  aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
  aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
  config=Config(retries={"max_attempts":5,"mode":"standard"},read_timeout=120))
 s3.head_bucket(Bucket=CZECH_ID)
 status_key="hephaestus/recovery-status/"+run+".json"
 def progress(phase,extra=None):
  record={"schema":"hephaestus.hf-plus-extras-copy.v1","run_id":run,
   "status":phase,"timestamp_utc":datetime.now(timezone.utc).isoformat()}
  if extra:record.update(extra)
  s3.put_object(Bucket=CZECH_ID,Key=status_key,
    Body=json.dumps(record,sort_keys=True).encode(),ContentType="application/json")
  summary={k:v for k,v in record.items() if k not in {"schema","run_id","timestamp_utc"}}
  print("MIGRATION_PROGRESS="+json.dumps(summary,sort_keys=True),flush=True)
 progress("STARTED")
 who=json.loads(read_url("https://huggingface.co/api/whoami-v2",token))
 if who.get("name")!=HF_OWNER:raise RuntimeError("Wrong private Hugging Face account")
 revisions={}
 for group,(repo_type,name,prefix,count) in HF_REPOS.items():
  kind="models" if repo_type=="model" else "datasets"
  obj=json.loads(read_url("https://huggingface.co/api/"+kind+"/"+HF_OWNER+"/"+name,token))
  if not obj.get("private") or not str(obj.get("sha","")).startswith(prefix):
   raise RuntimeError("HF repository revision changed")
  revisions[group]=obj["sha"]
 raw=read_url("https://huggingface.co/datasets/"+HF_OWNER+
  "/hephaestus-scientific-evidence/resolve/"+revisions["evidence"]+"/EXPORT_MANIFEST.json",token)
 if hashlib.sha256(raw).hexdigest()!=MANIFEST_SHA:raise RuntimeError("Pinned HF export manifest mismatch")
 manifest=json.loads(raw,object_pairs_hook=no_dup)
 if manifest.get("schema")!="hephaestus-hf-export-v1" or manifest.get("source_volume_id")!=CZECH_ID:
  raise RuntimeError("HF export source identity mismatch")
 all_entries=manifest.get("artifacts")
 if not isinstance(all_entries,list) or len(all_entries)!=8986:raise RuntimeError("Unexpected HF export inventory")
 counts=Counter();source_keys=set();evidence=[]
 for row in all_entries:
  name=row.get("repo");typ=row.get("repo_type")
  group=next((g for g,(k,n,r,count) in HF_REPOS.items() if name==HF_OWNER+"/"+n and typ==k),None)
  path=row.get("path");n=row.get("bytes");digest=row.get("sha256")
  if (group is None or not isinstance(n,int) or n<0 or not isinstance(digest,str)
      or not SHA_RE.fullmatch(digest) or row.get("remote_verification")!="verified"):
   raise RuntimeError("Malformed HF file metadata")
  p=safe_rel(path)
  if p.parts[0] in {"scientific","research"}:source_key="hephaestus/"+p.as_posix()
  elif p.parts[0]=="legacy":source_key="Hephaestus/artifacts/"+"/".join(p.parts[1:])
  else:raise RuntimeError("Unexpected HF source prefix")
  if source_key in source_keys:raise RuntimeError("Duplicate curated source key")
  source_keys.add(source_key);counts[group]+=1
  if group=="evidence":evidence.append((p,n,digest))
 if any(counts[group]!=meta[3] for group,meta in HF_REPOS.items()):
  raise RuntimeError("Curated HF group counts changed")
 evidence.sort(key=lambda row:row[0].as_posix())
 progress("EVIDENCE_SCAN",{"expected_evidence":len(evidence),"models_reused":869,"datasets_reused":4})
 stage_root=Path("/tmp/hephaestus-hf-evidence")
 # Model and dataset groups are intentionally not touched.
 evidence_root=RESTORE/"evidence"
 if evidence_root.is_symlink():raise RuntimeError("Unsafe evidence destination root")
 evidence_root.mkdir(parents=True,exist_ok=True)
 missing=[];total_evidence_bytes=0
 for index,(rel,size,digest) in enumerate(evidence,1):
  target=evidence_root/rel
  if not target.is_relative_to(evidence_root):raise RuntimeError("Evidence path escaped")
  cursor=target
  while cursor!=evidence_root:
   if cursor.is_symlink():raise RuntimeError("Symlink evidence destination")
   cursor=cursor.parent
  total_evidence_bytes+=size
  if target.exists():
   if not target.is_file() or sha(target)!=(digest,size):
    raise RuntimeError("Existing Global evidence differs from HF manifest")
  else:missing.append((rel,size,digest))
  if index%1000==0:progress("EVIDENCE_SCAN",{"checked":index,"total":len(evidence),"missing":len(missing)})
 progress("EVIDENCE_SCAN_DONE",{"existing":len(evidence)-len(missing),"missing":len(missing)})
 if missing:
  from huggingface_hub import hf_hub_download
  stage_root.mkdir(parents=True,exist_ok=True)
  def download_one(row):
   rel,size,digest=row
   path=Path(hf_hub_download(repo_id=HF_OWNER+"/hephaestus-scientific-evidence",repo_type="dataset",
      revision=revisions["evidence"],filename=rel.as_posix(),token=token,local_dir=str(stage_root)))
   if path.is_symlink() or not path.is_file() or sha(path)!=(digest,size):
    raise RuntimeError("Downloaded private HF evidence hash differs")
   return row,path
  restored=0
  for start in range(0,len(missing),12):
   with ThreadPoolExecutor(max_workers=6) as pool:
    futures=[pool.submit(download_one,row) for row in missing[start:start+12]]
    for fut in as_completed(futures):
     (rel,size,digest),path=fut.result()
     dest=evidence_root/rel
     cursor=dest.parent
     while cursor!=evidence_root:
      if cursor.is_symlink():raise RuntimeError("Unsafe evidence parent")
      cursor=cursor.parent
     dest.parent.mkdir(parents=True,exist_ok=True)
     if dest.exists() or dest.is_symlink():
      if sha(dest)!=(digest,size):raise RuntimeError("Concurrent Global evidence conflict")
     else:
      with path.open("rb") as reader,dest.open("xb") as writer:
       shutil.copyfileobj(reader,writer,length=BUF)
      if sha(dest)!=(digest,size):raise RuntimeError("Global missing-file write failed checksum")
     restored+=1
     path.unlink()
   if restored%60==0 or restored==len(missing):
    progress("EVIDENCE_REPAIR",{"downloaded":restored,"missing_total":len(missing)})
 progress("EVIDENCE_READY",{"verified":8113,"downloaded":len(missing),"bytes":total_evidence_bytes})
 evidence_receipt={
  "schema":"hephaestus.hf-evidence-presence.v1","manifest_sha256":MANIFEST_SHA,
  "evidence_repo_revision":revisions["evidence"],"group":"evidence",
  "verified_files":8113,"verified_bytes":total_evidence_bytes,
  "verification":"one full SHA256 per existing file; two checks for newly downloaded files",
  "scope":"Hugging Face evidence group; models and datasets retained from earlier completed Global verification"
 }
 evidence_receipt_path=RESTORE/("HF_EVIDENCE_"+MANIFEST_SHA[:24]+".json")
 evidence_receipt_sha=stable_receipt(evidence_receipt_path,evidence_receipt)
 progress("HF_EVIDENCE_COMPLETE",{"evidence_files":8113,"receipt_sha256":evidence_receipt_sha})
 # Czech-only immutable cold snapshot. Never use it as a live SQLite/checkpoint mount.
 excluded=[];included=[]
 for prefix in PREFIXES:
  for page in s3.get_paginator("list_objects_v2").paginate(Bucket=CZECH_ID,Prefix=prefix):
   for item in page.get("Contents",[]):
    key=item["Key"]
    if key.endswith("/") or key in source_keys:continue
    rel=safe_rel(key)
    lower=key.lower();parts={x.lower() for x in rel.parts}
    if (parts&SKIP_PARTS or rel.name.lower() in SKIP_NAMES
        or SECRET_PATTERN.search(lower)):
     excluded.append({"key_sha256":hashlib.sha256(key.encode()).hexdigest(),"reason":"cache_temp_or_secret"})
     continue
    included.append((key,int(item["Size"])))
 if len(included)>MAX_EXTRAS_OBJECTS or sum(x[1] for x in included)>MAX_EXTRAS_BYTES:
  raise RuntimeError("Uncurated Czech-only snapshot exceeds planned 12-GiB bound")
 included.sort()
 progress("CZECH_EXTRAS_PACKAGING",{"candidate_files":len(included),"candidate_bytes":sum(x[1] for x in included),
     "excluded_nonessential":len(excluded)})
 archive_path=Path("/tmp/hephaestus-czech-extras.tar")
 copied=0;source_missing=0;symlinks=0;live_files=[]
 with tarfile.open(archive_path,"w",dereference=False,format=tarfile.PAX_FORMAT) as tar:
  for index,(key,listed_size) in enumerate(included,1):
   file_path=CZECH/key
   cursor=file_path.parent
   bad_parent=False
   while cursor!=CZECH:
    if cursor.is_symlink():bad_parent=True;break
    cursor=cursor.parent
   if bad_parent:
    excluded.append({"key_sha256":hashlib.sha256(key.encode()).hexdigest(),"reason":"symlinked_parent"})
    continue
   try:st=file_path.lstat()
   except FileNotFoundError:
    source_missing+=1
    excluded.append({"key_sha256":hashlib.sha256(key.encode()).hexdigest(),"reason":"source_absent"})
    continue
   if not (stat.S_ISREG(st.st_mode) or stat.S_ISLNK(st.st_mode)):
    excluded.append({"key_sha256":hashlib.sha256(key.encode()).hexdigest(),"reason":"nonregular_source"})
    continue
   tar.add(file_path,arcname=key,recursive=False)
   copied+=1
   if stat.S_ISLNK(st.st_mode):symlinks+=1
   live_files.append({"key":key,"source_listed_bytes":listed_size,"source_stat_bytes":st.st_size,
                     "type":"symlink" if stat.S_ISLNK(st.st_mode) else "regular"})
   if index%1500==0:progress("CZECH_EXTRAS_PACKAGING",{"processed":index,"candidates":len(included)})
 manifest_meta={
  "schema":"hephaestus.czech-only-cold-snapshot.v1",
  "hf_manifest_sha256":MANIFEST_SHA,"source_network_volume":CZECH_ID,
  "included":live_files,"omissions":excluded,"source_missing_count":source_missing,
  "symlinks_saved_without_following":symlinks,
  "note":"Cold snapshot only. Mutable DB/WAL consistency not asserted; do not extract unsafe symlinks or activate protected eval material automatically."
 }
 metadata=(json.dumps(manifest_meta,sort_keys=True,separators=(",",":"))+"\n").encode()
 metadata_path=Path("/tmp/hephaestus-extras-manifest.json")
 metadata_path.write_bytes(metadata)
 # Store the manifest next to the tar on Global, not into a tar that has closed.
 tar_hash,tar_bytes=sha(archive_path)
 EXTRAS.mkdir(parents=True,exist_ok=True)
 destination=EXTRAS/("czech-only-"+tar_hash[:24]+".tar")
 if destination.is_symlink():raise RuntimeError("Snapshot destination is symlink")
 if destination.exists():
  if sha(destination)!=(tar_hash,tar_bytes):raise RuntimeError("Existing Global snapshot hash mismatch")
 else:
  with archive_path.open("rb") as reader,destination.open("xb") as writer:
   shutil.copyfileobj(reader,writer,length=BUF)
  if sha(destination)!=(tar_hash,tar_bytes):raise RuntimeError("Global Czech snapshot readback mismatch")
 metadata_hash=hashlib.sha256(metadata).hexdigest()
 stable_receipt(EXTRAS/("czech-only-inventory-"+metadata_hash[:24]+".json"),manifest_meta)
 archive_receipt={
  "schema":"hephaestus.czech-only-bulk-copy.v1",
  "source_volume":CZECH_ID,"global_volume":GLOBAL_ID,
  "hf_manifest_sha256":MANIFEST_SHA,"archive":destination.name,
  "archive_sha256":tar_hash,"archive_bytes":tar_bytes,
  "files_in_archive":copied,"symlink_entries":symlinks,
  "source_absent":source_missing,
  "excluded_nonessential_or_unsafe":len(excluded),
  "inventory_sha256":metadata_hash,
  "note":"Cold snapshot, NOT a verified live SQLite backup"
 }
 bulk_receipt_sha=stable_receipt(EXTRAS/("CZECH_EXTRAS_"+tar_hash[:24]+".json"),archive_receipt)
 progress("COMPLETE",{"hf_evidence_verified":8113,"hf_missing_downloaded":len(missing),
  "czech_archive_sha256":tar_hash,"czech_archive_files":copied,
  "czech_archive_bytes":tar_bytes,"czech_omitted":len(excluded),
  "czech_bulk_receipt_sha256":bulk_receipt_sha,
  "hf_receipt_sha256":evidence_receipt_sha})
 print("HEPHAESTUS_SIMPLE_MIGRATION_COMPLETE=true",flush=True)

if __name__=="__main__":
 try:main()
 except Exception as exc:
  print("HEPHAESTUS_SIMPLE_MIGRATION_FAILED="+type(exc).__name__,flush=True)
  run=os.environ.get("RECOVERY_RUN_ID","")
  if run.isdecimal():
   try:
    client=boto3.client("s3",endpoint_url="https://s3api-eu-cz-1.runpod.io/",region_name="EU-CZ-1",
      aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
      aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"])
    rec={"schema":"hephaestus.hf-plus-extras-copy.v1","run_id":run,
         "status":"FAILED","error_class":type(exc).__name__}
    client.put_object(Bucket=CZECH_ID,Key="hephaestus/recovery-status/"+run+".json",
        Body=json.dumps(rec,sort_keys=True).encode(),ContentType="application/json")
   except Exception:
    print("FAILED_STATUS_PERSISTENCE=true",flush=True)
  sys.exit(1)
