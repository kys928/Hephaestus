#!/usr/bin/env python3
"""Bounded, single-Pod RunPod CPU recovery controller run by GitHub Actions."""
import os,json,time,urllib.request,urllib.error
from decimal import Decimal
from botocore.exceptions import ClientError
import boto3
API="https://api.runpod.io/v2"
SOURCE="cviwpryzao"
GLOBAL="cmw51pjsk000007l9a0mv1jga"
WORKER_SHA="a62df545655e3fd1ab22fba1b3523257105b2f96"
MAX_HOURS=4
RATE_CEILING=Decimal("0.60")
TOTAL_COST_CEILING=Decimal("3.00")
RESERVE=Decimal("0.50")
def req(method,path,body=None):
 raw=None if body is None else json.dumps(body).encode()
 r=urllib.request.Request(API+path,method=method,data=raw,headers={"Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"],"Content-Type":"application/json"})
 with urllib.request.urlopen(r,timeout=60) as response:
  data=response.read()
  return json.loads(data) if data else {}
def pods():
 x=req("GET","/pods")
 out=x if isinstance(x,list) else x.get("pods",x.get("data",[]))
 if not isinstance(out,list):raise RuntimeError("Unknown Pod list response")
 return out
def balance():
 body={"query":"query HephaestusBudget { myself { clientBalance currentSpendPerHr isAutoPayEnabled } }"}
 r=urllib.request.Request("https://api.runpod.io/graphql",data=json.dumps(body).encode(),headers={"Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"],"Content-Type":"application/json"})
 with urllib.request.urlopen(r,timeout=30) as response: user=json.load(response).get("data",{}).get("myself",{})
 if user.get("isAutoPayEnabled") is not False:raise RuntimeError("Unknown or active RunPod autopay; refuse")
 credit=Decimal(str(user["clientBalance"]))
 spending=Decimal(str(user["currentSpendPerHr"]))
 if not credit.is_finite() or not spending.is_finite() or credit<0 or spending<0:raise RuntimeError("Invalid prepaid account observation")
 return credit,spending
def source_s3():
 return boto3.client("s3",endpoint_url="https://s3api-eu-cz-1.runpod.io/",region_name="EU-CZ-1",
  aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"])
def main():
 run_id=os.environ["GITHUB_RUN_ID"]
 if not run_id.isdigit():raise RuntimeError("Invalid GitHub run ID")
 name="hephaestus-hf-global-recovery-"+run_id
 s3=source_s3();s3.head_bucket(Bucket=SOURCE)
 status_key="hephaestus/recovery-status/"+run_id+".json"
 existing=[p for p in pods() if p.get("name")==name]
 if existing:raise RuntimeError("Identical recovery Pod already present; no duplicate launch")
 credit,spend=balance()
 if credit<Decimal("2.00"):raise RuntimeError("Prepaid RunPod credit below $2 safety gate")
 script_url="https://raw.githubusercontent.com/kys928/Hephaestus/"+WORKER_SHA+"/scripts/hephaestus_global_s3_evidence_worker.py"
 bootstrap="python -m pip install --no-cache-dir -q 'boto3>=1.35,<2' && python -c \"import urllib.request;open('/tmp/restore.py','wb').write(urllib.request.urlopen('"+script_url+"',timeout=60).read())\" && python -u /tmp/restore.py"
 body={"name":name,"cpu":{"id":"cpu3c","vcpuCount":2},"cloud":"SECURE","image":"python:3.12-slim","disk":20,
  "mounts":{"global":[{"volumeId":GLOBAL,"path":"/workspace"}]},"cmd":["bash","-lc",bootstrap],
  "env":{"RUNPOD_GLOBAL_VOLUME_ID":GLOBAL,"RECOVERY_RUN_ID":run_id,
   "RUNPOD_S3_ACCESS_KEY_ID":os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
   "RUNPOD_S3_SECRET_ACCESS_KEY":os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"]}}
 pod_id=None
 try:
  # In case POST response is ambiguous, reconcile by exact name; never re-POST blindly.
  try: created=req("POST","/pods",body)
  except Exception:
   matches=[p for p in pods() if p.get("name")==name]
   if len(matches)!=1:raise RuntimeError("Ambiguous Pod allocation: reconcile by name; no automatic retry")
   created=matches[0]
  pod_id=created.get("id")
  if not isinstance(pod_id,str) or not pod_id:raise RuntimeError("Pod create yielded no ID; cleanup job must reconcile exact name")
  print("RECOVERY_POD_ID="+pod_id,flush=True)
  live=req("GET","/pods/"+pod_id)
  if live.get("id")!=pod_id:raise RuntimeError("Live Pod ID mismatch")
  mounts=live.get("mounts",{}).get("global",[])
  if not any(isinstance(m,dict) and m.get("volumeId")==GLOBAL for m in mounts):raise RuntimeError("Wrong Global Volume")
  price=Decimal(str(live.get("costPerHr",live.get("cost","-1"))))
  if not price.is_finite() or not Decimal("0")<price<=RATE_CEILING:raise RuntimeError("Live CPU price missing or exceeds $0.60/hr; Pod will be deleted")
  credit,spend=balance()
  allowed=min(Decimal(MAX_HOURS*3600),(min(TOTAL_COST_CEILING,credit-RESERVE)/(price+spend))*Decimal(3600))
  seconds=int(allowed)
  if seconds<900:raise RuntimeError("Insufficient prepaid credits for safe 15-minute recovery")
  print("RECOVERY_WALLCLOCK_CAP_SECONDS="+str(seconds),flush=True)
  print("RECOVERY_COST_CEILING_USD="+str(TOTAL_COST_CEILING),flush=True)
  end=time.monotonic()+seconds
  previous=None
  while time.monotonic()<end:
   try:
    obj=s3.get_object(Bucket=SOURCE,Key=status_key)
    with obj["Body"] as stream: state=json.load(stream)
   except ClientError as exc:
    if str(exc.response.get("Error",{}).get("Code","")) not in {"404","NoSuchKey","NotFound"}:raise
    state=None
   if state and state.get("run_id")==run_id:
    progress=(state.get("status"),state.get("verified"),state.get("total"))
    if progress!=previous:print("RECOVERY_PROGRESS="+str(progress),flush=True);previous=progress
    if state.get("status")=="COMPLETE":
     print("RECOVERY_COMPLETE_RECEIPT="+str(state.get("receipt")),flush=True)
     print("RECOVERY_CONTROLLER_SUCCESS=true",flush=True)
     return
    if state.get("status")=="COUNT_MISMATCH":raise RuntimeError("S3 evidence count does not match HF; no false success")
   live=req("GET","/pods/"+pod_id)
   if live.get("desiredStatus",live.get("status")) in {"EXITED","TERMINATED"}:raise RuntimeError("Worker exited before COMPLETE receipt")
   print("RECOVERY_MONITOR_ACTIVE=true",flush=True)
   time.sleep(40)
  raise RuntimeError("Prepaid wall-clock cap reached; all verified Global Volume files preserved")
 finally:
  # Delete only the exact Pod created for this workflow. A separate always()
  # GitHub Actions cleanup job covers unexpected runner death/ambiguous create.
  matches=[p for p in pods() if p.get("name")==name and p.get("id")==pod_id]
  if matches:
   req("DELETE","/pods/"+pod_id)
   print("RECOVERY_POD_DELETE_REQUESTED=true",flush=True)
if __name__=="__main__":
 try:main()
 except Exception as exc:
  print("RECOVERY_CONTROLLER_FAILED="+type(exc).__name__+": "+str(exc)[:180],flush=True)
  raise SystemExit(1)
