#!/usr/bin/env python3
"""Prepaid-capped independent Global Volume receipt readback controller."""
import json,os,time,urllib.request
from decimal import Decimal
from botocore.exceptions import ClientError
import boto3
GLOBAL="cmw51pjsk000007l9a0mv1jga"
SOURCE="cviwpryzao"
ORIGIN_RUN="43261952175"
VERIFIER_SHA="0aabb3796e2e435a14588b11e4a8f96d5b94b9db"
RATE_CEILING=Decimal("0.60")
SPEND_CEILING=Decimal("3.00")
MIN_RESERVE=Decimal("0.50")
def request(method,path,body=None):
 payload=None if body is None else json.dumps(body).encode()
 req=urllib.request.Request("https://api.runpod.io/v2"+path,data=payload,method=method,
  headers={"Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"],"Content-Type":"application/json"})
 with urllib.request.urlopen(req,timeout=45) as resp:
  raw=resp.read();return json.loads(raw) if raw else {}
def pods():
 ps=request("GET","/pods")
 rows=ps if isinstance(ps,list) else ps.get("pods",ps.get("data",[]))
 if not isinstance(rows,list):raise RuntimeError("Malformed Pods list")
 return rows
def account():
 gql={"query":"query HephaestusReadOnlyBudget { myself { clientBalance currentSpendPerHr isAutoPayEnabled } }"}
 req=urllib.request.Request("https://api.runpod.io/graphql",data=json.dumps(gql).encode(),
  headers={"Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"],"Content-Type":"application/json"})
 with urllib.request.urlopen(req,timeout=30) as r:obj=json.load(r)
 if obj.get("errors"):raise RuntimeError("No live budget observation")
 info=obj["data"]["myself"]
 if info["isAutoPayEnabled"] is not False:raise RuntimeError("Autopay enabled/unknown: no new Pod")
 credit=Decimal(str(info["clientBalance"]));spend=Decimal(str(info["currentSpendPerHr"]))
 if not credit.is_finite() or not spend.is_finite() or min(credit,spend)<0:raise RuntimeError("Invalid prepaid amounts")
 return credit,spend
def status_client():
 return boto3.client("s3",endpoint_url="https://s3api-eu-cz-1.runpod.io/",region_name="EU-CZ-1",
  aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"])
def main():
 run=os.environ["GITHUB_RUN_ID"]
 if not run.isdecimal():raise RuntimeError("Missing GitHub verifier ID")
 name="hephaestus-independent-global-verify-"+run
 if [p for p in pods() if p.get("name")==name]:raise RuntimeError("Previous verifier Pod exists; refuse double billing")
 s3=status_client()
 original=s3.get_object(Bucket=SOURCE,Key="hephaestus/recovery-status/"+ORIGIN_RUN+".json")
 with original["Body"] as f:state=json.load(f)
 if state.get("status")!="COMPLETE" or not state.get("receipt_sha256"):raise RuntimeError("S3 restore receipt missing")
 if [p for p in pods() if p.get("name")=="hephaestus-regional-global-recovery-"+ORIGIN_RUN]:raise RuntimeError("Original recovery Pod still exists")
 credit,spend=account()
 if credit<Decimal("2.00"):raise RuntimeError("RunPod remaining balance below prepaid $2 guard")
 url="https://raw.githubusercontent.com/kys928/Hephaestus/"+VERIFIER_SHA+"/scripts/hephaestus_independent_global_receipt_verifier.py"
 cmd="python -m pip install --no-cache-dir -q 'boto3>=1.35,<2' && python -c \"import urllib.request;open('/tmp/verify.py','wb').write(urllib.request.urlopen('"+url+"',timeout=60).read())\" && python -u /tmp/verify.py"
 data={"name":name,"cpu":{"id":"cpu3c","vcpuCount":2},"cloud":"SECURE","image":"python:3.12-slim","disk":20,
  "mounts":{"global":[{"volumeId":GLOBAL,"path":"/workspace"}]},
  "cmd":["bash","-lc",cmd],
  "env":{"RUNPOD_GLOBAL_VOLUME_ID":GLOBAL,"VERIFY_RUN_ID":run,
   "RUNPOD_S3_ACCESS_KEY_ID":os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
   "RUNPOD_S3_SECRET_ACCESS_KEY":os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"]}}
 pod_id=None
 try:
  try:created=request("POST","/pods",data)
  except Exception:
   matches=[p for p in pods() if p.get("name")==name]
   if len(matches)!=1:raise RuntimeError("Ambiguous Pod allocation; no automatic repeat")
   created=matches[0]
  pod_id=created.get("id")
  if not isinstance(pod_id,str) or not pod_id:raise RuntimeError("Pod not identified")
  print("INDEPENDENT_VERIFIER_POD="+pod_id,flush=True)
  live=request("GET","/pods/"+pod_id)
  if live.get("id")!=pod_id or not any(x.get("volumeId")==GLOBAL for x in live.get("mounts",{}).get("global",[])):
   raise RuntimeError("Verifier Pod mounted the wrong volume")
  price=Decimal(str(live.get("costPerHr",live.get("cost","-1"))))
  if not price.is_finite() or price<=0 or price>RATE_CEILING:raise RuntimeError("Unexpected paid CPU price")
  credit,spend=account()
  secs=int(min(Decimal(3*3600),((min(SPEND_CEILING,credit-MIN_RESERVE))/(price+spend))*3600))
  if secs<600:raise RuntimeError("Insufficient remaining prepaid budget")
  print("INDEPENDENT_VERIFIER_WALLCLOCK_CAP_SECONDS="+str(secs),flush=True)
  end=time.monotonic()+secs
  previous=None
  while time.monotonic()<end:
   try:
    r=s3.get_object(Bucket=SOURCE,Key="hephaestus/recovery-verification/"+run+".json")
    with r["Body"] as f:progress=json.load(f)
   except ClientError as ex:
    if ex.response.get("Error",{}).get("Code") not in {"404","NoSuchKey","NotFound"}:raise
    progress={}
   observation=(progress.get("status"),progress.get("checked"),progress.get("total"))
   if observation!=previous:print("INDEPENDENT_VERIFIER_PROGRESS="+str(observation),flush=True);previous=observation
   if progress.get("verify_run_id")==run and progress.get("status")=="COMPLETE":
    if progress.get("receipt_sha256")!=state["receipt_sha256"]:raise RuntimeError("Independent result receipt hash changed")
    print("INDEPENDENT_VERIFIER_COMPLETE=true",flush=True)
    return
   actual=request("GET","/pods/"+pod_id)
   if actual.get("desiredStatus",actual.get("status")) in {"EXITED","TERMINATED"}:raise RuntimeError("Verifier worker stopped without accepted receipt")
   time.sleep(45)
  raise RuntimeError("Verifier reached prepaid time cap without completion")
 finally:
  matches=[p for p in pods() if p.get("name")==name and p.get("id")==pod_id]
  if matches:
   request("DELETE","/pods/"+pod_id)
   print("INDEPENDENT_VERIFIER_POD_DELETE_REQUESTED=true",flush=True)
if __name__=="__main__":
 try:main()
 except Exception as ex:
  print("INDEPENDENT_VERIFIER_CONTROLLER_FAILED="+type(ex).__name__+": "+str(ex)[:220],flush=True)
  raise SystemExit(1)
