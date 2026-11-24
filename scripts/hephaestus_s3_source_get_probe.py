#!/usr/bin/env python3
"""Read-only diagnosis of RunPod S3 listed source objects that return NoSuchKey on GET.
Uses an existing private inventory and probes only a one-byte Range per object.
Reports keys privately in original S3; never prints private key names to GitHub logs.
"""
import hashlib,json,os,sys
from pathlib import PurePosixPath
from concurrent.futures import ThreadPoolExecutor,as_completed
import boto3
from botocore.exceptions import ClientError
sys.path.insert(0,"scripts")
from hephaestus_global_s3_evidence_worker import ROOTS,reject,kind
ROOT="cviwpryzao"
INV="hephaestus/recovery-audit/source-inventory-43256848721.json"
def main():
 s3=boto3.client("s3",region_name="EU-CZ-1",endpoint_url="https://s3api-eu-cz-1.runpod.io/",
  aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"])
 result=s3.get_object(Bucket=ROOT,Key=INV)
 with result["Body"] as f:inventory=json.load(f)
 rows=[]
 for obj in inventory["objects"]:
  key=obj["key"]
  for prefix,group in ROOTS.items():
   if not key.startswith(prefix) or key.endswith("/"):continue
   tail=key[len(prefix):]
   if not tail or "//" in tail or "\\" in tail or any(x in ("",".","..") for x in tail.split("/")):continue
   p=PurePosixPath(tail)
   if not reject(p) and kind(p)=="evidence":rows.append((group+"/"+tail,key,int(obj["bytes"])))
   break
 rows.sort()
 print("FROZEN_AUDIT_EVIDENCE_CANDIDATES="+str(len(rows)),flush=True)
 issues=[]
 def probe(idx,row):
  rel,key,size=row
  try:
   if size:
    r=s3.get_object(Bucket=ROOT,Key=key,Range="bytes=0-0")
    with r["Body"] as f:f.read(1)
   else:s3.head_object(Bucket=ROOT,Key=key)
   return None
  except ClientError as ex:
   code=str(ex.response.get("Error",{}).get("Code","Unknown"))
   return {"index":idx,"key":key,"key_sha256":hashlib.sha256(key.encode()).hexdigest(),"code":code,"size":size}
 with ThreadPoolExecutor(max_workers=12) as pool:
  futs=[pool.submit(probe,i,row) for i,row in enumerate(rows,1)]
  for f in as_completed(futs):
   res=f.result()
   if res:issues.append(res)
 issues.sort(key=lambda x:x["index"])
 report={"schema":"hephaestus.s3-listed-object-get-probe.v1","source_inventory_key":INV,"probe_count":len(rows),"issues":issues}
 target="hephaestus/recovery-audit/source-get-probe-"+os.environ["GITHUB_RUN_ID"]+".json"
 s3.put_object(Bucket=ROOT,Key=target,Body=json.dumps(report,sort_keys=True).encode(),ContentType="application/json")
 stats={}
 for item in issues:stats[item["code"]]=stats.get(item["code"],0)+1
 print("S3_RANGE_GET_PROBE_ISSUES="+json.dumps(stats,sort_keys=True),flush=True)
 print("S3_RANGE_GET_PROBE_INDEXES="+json.dumps([x["index"] for x in issues[:30]]),flush=True)
 print("PRIVATE_GET_PROBE_REPORT="+target,flush=True)
 print("S3_RANGE_GET_PROBE_COMPLETE=true",flush=True)
if __name__=="__main__":main()
