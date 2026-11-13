#!/usr/bin/env python3
"""Read-only audit of curated HF export and original Czech S3 inventory.

No Pod allocation, storage mutation, credential printing, or large downloads.
"""
from __future__ import annotations
from collections import Counter
import hashlib, json, os, urllib.request
from decimal import Decimal
import boto3

EXPECTED_MANIFEST="129d690ef9ae24ef141fbb0c1ce48280f033b33f03b691073c20231d16677638"
HF_OWNER="DavidDulovic"
HF={
 "models":("model","hephaestus-trained-models","13f743df8bfb",869),
 "datasets":("dataset","hephaestus-training-data","78c6dbc4e08b",4),
 "evidence":("dataset","hephaestus-scientific-evidence","50cce2d27351",8113)
}
INVENTORY_KEY="hephaestus/recovery-audit/source-inventory-43256848721.json"
INVENTORY_SHA="046dde9ebe9a712665081794586d502de5722f7f815a61567377414b680f5fb8"
SOURCE="cviwpryzao"

def req(url,token=None,body=None):
 headers={"Accept":"application/json"}
 if token:headers["Authorization"]="Bearer "+token
 if body is not None:headers["Content-Type"]="application/json"
 request=urllib.request.Request(url,data=json.dumps(body).encode() if body is not None else None,headers=headers)
 with urllib.request.urlopen(request,timeout=60) as resp:return resp.read()

def main():
 token=os.environ.get("HF_TOKEN","")
 if not token:raise RuntimeError("GitHub HF_TOKEN secret missing")
 user=json.loads(req("https://huggingface.co/api/whoami-v2",token))
 if user.get("name")!=HF_OWNER:raise RuntimeError("Wrong HF account")
 print("HF_ACCOUNT_OK=true",flush=True)
 revisions={}
 for group,(kind,name,prefix,count) in HF.items():
  url="https://huggingface.co/api/"+("models" if kind=="model" else "datasets")+"/"+HF_OWNER+"/"+name
  obj=json.loads(req(url,token))
  sha=obj.get("sha","")
  if not sha.startswith(prefix) or not obj.get("private"):
   raise RuntimeError("Private HF repository head changed from the previously verified export")
  revisions[group]=sha
  print("HF_PIN_"+group.upper()+"="+sha[:12],flush=True)
 manifest_url="https://huggingface.co/datasets/"+HF_OWNER+"/hephaestus-scientific-evidence/resolve/"+revisions["evidence"]+"/EXPORT_MANIFEST.json"
 raw=req(manifest_url,token)
 manifest_sha=hashlib.sha256(raw).hexdigest()
 if manifest_sha!=EXPECTED_MANIFEST:raise RuntimeError("HF manifest digest changed: "+manifest_sha)
 data=json.loads(raw)
 if data.get("schema")!="hephaestus-hf-export-v1" or data.get("source_volume_id")!=SOURCE:raise RuntimeError("Export identity mismatch")
 entries=data.get("artifacts",[])
 counts=Counter()
 hf_source_keys=set()
 bygroup={}
 for row in entries:
  p=row.get("path");rt=row.get("repo_type");repo=row.get("repo");size=row.get("bytes");digest=row.get("sha256")
  g=next((x for x, (kind,name,rev,n) in HF.items() if repo==HF_OWNER+"/"+name and rt==kind),None)
  if not g or not isinstance(p,str) or not isinstance(size,int) or size<0 or not isinstance(digest,str) or len(digest)!=64 or row.get("remote_verification")!="verified":
   raise RuntimeError("Malformed HF export entry")
  if p.startswith("scientific/") or p.startswith("research/"):key="hephaestus/"+p
  elif p.startswith("legacy/"):key="Hephaestus/artifacts/"+p[len("legacy/"):]
  else:raise RuntimeError("Unrecognized curated export path prefix")
  if key in hf_source_keys:raise RuntimeError("Duplicate Czech source key in export")
  hf_source_keys.add(key)
  counts[g]+=1
 if len(entries)!=8986 or any(counts[g]!=HF[g][3] for g in HF):
  raise RuntimeError("Curated HF export counts changed")
 print("HF_EXPORT_MANIFEST_SHA256="+manifest_sha,flush=True)
 print("HF_EXPORT_COUNTS="+json.dumps(dict(counts),sort_keys=True),flush=True)
 s3=boto3.client("s3",endpoint_url="https://s3api-eu-cz-1.runpod.io/",region_name="EU-CZ-1",
  aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"])
 with s3.get_object(Bucket=SOURCE,Key=INVENTORY_KEY)["Body"] as body: inv_raw=body.read()
 if hashlib.sha256(inv_raw).hexdigest()!=INVENTORY_SHA:raise RuntimeError("Original S3 inventory snapshot changed")
 snapshot=json.loads(inv_raw)
 all_objects={x["key"]:x for x in snapshot["objects"]}
 preserved=Counter();bytes_=Counter();matched=Counter()
 for key,obj in all_objects.items():
  if key in hf_source_keys:
   matched[obj["category"]]+=1
  else:
   preserved[obj["category"]]+=1;bytes_[obj["category"]]+=int(obj["bytes"])
 print("HF_SOURCE_KEYS_LISTED_IN_ORIGINAL_S3="+str(len(hf_source_keys & all_objects.keys())),flush=True)
 print("HF_SOURCE_KEYS_NOT_LISTED_IN_ORIGINAL_S3="+str(len(hf_source_keys - all_objects.keys())),flush=True)
 print("CZECH_NOT_IN_CURATED_HF_BY_CATEGORY="+json.dumps({k:{"files":preserved[k],"bytes":bytes_[k]} for k in sorted(preserved)},sort_keys=True),flush=True)
 print("CZECH_CURATED_HF_MATCHES_BY_CATEGORY="+json.dumps(dict(matched),sort_keys=True),flush=True)
 # Check RunPod balance without spending credit or exposing credentials in logs.
 request={"query":"query HephaestusPreflight { myself { clientBalance currentSpendPerHr isAutoPayEnabled } }"}
 try:
  raw=req("https://api.runpod.io/graphql",os.environ["RUNPOD_API_KEY"],request)
  myself=json.loads(raw).get("data",{}).get("myself",{})
  balance=Decimal(str(myself["clientBalance"]))
  if not balance.is_finite():raise RuntimeError("Non-finite provider credit")
  print("RUNPOD_PREPAID_ABOVE_2_USD="+str(balance>=Decimal("2")).lower(),flush=True)
  print("RUNPOD_AUTOPAY_ON="+str(myself.get("isAutoPayEnabled") is True).lower(),flush=True)
 except Exception as exc:
  print("RUNPOD_BUDGET_CHECK_FAILED="+type(exc).__name__,flush=True)
 print("CURATED_AND_SOURCE_INVENTORY_PREFLIGHT_COMPLETE=true",flush=True)

if __name__=="__main__":
 try:main()
 except Exception as exc:
  print("CURATED_PREFLIGHT_FAILED="+type(exc).__name__+": "+str(exc)[:160],flush=True)
  raise SystemExit(1)
