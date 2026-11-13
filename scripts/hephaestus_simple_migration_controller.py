#!/usr/bin/env python3
"""One small bounded RunPod CPU Pod for HF evidence plus Czech-only cold snapshot.

GitHub Actions controller: account/volume guards, launch, progress monitor,
and exact Pod teardown. No GPU, no model runs, no source deletion.
"""
import json,os,re,time,urllib.request,urllib.error
from decimal import Decimal
import boto3
from botocore.exceptions import ClientError

GLOBAL="cmw51pjsk000007l9a0mv1jga"
SOURCE="cviwpryzao"
API="https://api.runpod.io/v2"
WORKER_SHA="2b4e3335e784f6e437e8029670894933c823fdc6"
RATE_CEILING=Decimal("0.60")
TOTAL_BUDGET=Decimal("1.50")
RESERVE=Decimal("0.30")
MAX_SECONDS=3*3600

def req(method,path,body=None):
 data=None if body is None else json.dumps(body).encode()
 request=urllib.request.Request(API+path,data=data,method=method,headers={
  "Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"],
  "Content-Type":"application/json"})
 with urllib.request.urlopen(request,timeout=40) as r:
  raw=r.read()
  return json.loads(raw) if raw else {}

def pods():
 out=req("GET","/pods")
 ps=out if isinstance(out,list) else out.get("pods",out.get("data",[]))
 if not isinstance(ps,list):raise RuntimeError("Unknown RunPod Pod list")
 return ps

def account():
 body={"query":"query SimpleMigrationBudget { myself { clientBalance currentSpendPerHr isAutoPayEnabled } }"}
 request=urllib.request.Request("https://api.runpod.io/graphql",data=json.dumps(body).encode(),
  headers={"Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"],"Content-Type":"application/json"})
 with urllib.request.urlopen(request,timeout=30) as r:payload=json.load(r)
 if payload.get("errors"):raise RuntimeError("RunPod account API query failed")
 obj=payload["data"]["myself"]
 if obj.get("isAutoPayEnabled") is not False:raise RuntimeError("Autopay disabled/unknown guard failed")
 credit=Decimal(str(obj["clientBalance"]));spend=Decimal(str(obj["currentSpendPerHr"]))
 if not credit.is_finite() or not spend.is_finite() or min(credit,spend)<0:
  raise RuntimeError("Invalid prepaid account observation")
 return credit,spend

def s3_client():
 return boto3.client("s3",endpoint_url="https://s3api-eu-cz-1.runpod.io/",region_name="EU-CZ-1",
  aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
  aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"])

def main():
 rid=os.environ.get("GITHUB_RUN_ID","")
 if not rid.isdecimal():raise RuntimeError("Invalid workflow run identity")
 if not os.environ.get("HF_TOKEN"):raise RuntimeError("HF_TOKEN missing from GitHub Actions")
 name="hephaestus-hf-czech-simple-"+rid
 if any(p.get("name","").startswith("hephaestus-hf-czech-simple-") for p in pods()):
  raise RuntimeError("A simple Hephaestus migration Pod already exists; do not allocate another")
 credit,spend=account()
 print("ACCOUNT_CREDIT_ABOVE_0_75="+str(credit>=Decimal("0.75")).lower(),flush=True)
 if credit<Decimal("0.75"):raise RuntimeError("Insufficient RunPod prepaid credit for safe CPU migration")
 s3=s3_client()
 s3.head_bucket(Bucket=SOURCE)
 # Use a pinned, code-reviewed worker. No self-modifying scripts and no
 # GitHub Actions runner sleeping on an ephemeral Pod without cleanup.
 url=("https://raw.githubusercontent.com/kys928/Hephaestus/"+WORKER_SHA+
      "/scripts/hephaestus_simple_migration.py")
 bootstrap=("python -m pip install --no-cache-dir -q 'boto3>=1.35,<2' 'huggingface_hub>=0.36,<2' && "+
  "python -c \"import urllib.request;open('/tmp/migrate.py','wb').write(urllib.request.urlopen('"+
  url+"',timeout=60).read())\" && "+
  "python -u /tmp/migrate.py && exec sleep infinity")
 body={
  "name":name,"cpu":{"id":"cpu3c","vcpuCount":2},"cloud":"SECURE",
  "dataCenterIds":["EU-CZ-1"],"image":"python:3.12-slim","disk":24,
  "mounts":{"network":[{"volumeId":SOURCE,"path":"/workspace-source"}],
            "global":[{"volumeId":GLOBAL,"path":"/workspace-global"}]},
  "cmd":["bash","-lc",bootstrap],
  "env":{"HF_TOKEN":os.environ["HF_TOKEN"],"RECOVERY_RUN_ID":rid,
   "RUNPOD_NETWORK_VOLUME_ID":SOURCE,"RUNPOD_GLOBAL_VOLUME_ID":GLOBAL,
   "RUNPOD_S3_ACCESS_KEY_ID":os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
   "RUNPOD_S3_SECRET_ACCESS_KEY":os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"]}
 }
 pod_id=None
 try:
  try:created=req("POST","/pods",body)
  except Exception:
   matches=[p for p in pods() if p.get("name")==name]
   if len(matches)!=1:raise RuntimeError("Pod launch ambiguous; no duplicate POST retry")
   created=matches[0]
  pod_id=created.get("id")
  if not isinstance(pod_id,str) or not pod_id:raise RuntimeError("Pod identity unavailable")
  print("SIMPLE_MIGRATION_POD_ID="+pod_id,flush=True)
  live=req("GET","/pods/"+pod_id)
  if live.get("id")!=pod_id:raise RuntimeError("Pod identity mismatch")
  mounts=live.get("mounts",{})
  if not any(x.get("volumeId")==SOURCE and x.get("path")=="/workspace-source" for x in mounts.get("network",[])):
   raise RuntimeError("Original Czech Network Volume mount not confirmed")
  if not any(x.get("volumeId")==GLOBAL and x.get("path")=="/workspace-global" for x in mounts.get("global",[])):
   raise RuntimeError("Target Global Volume mount not confirmed")
  if live.get("dataCenterId") and live["dataCenterId"]!="EU-CZ-1":
   raise RuntimeError("Wrong Czech region")
  price=Decimal(str(live.get("costPerHr",live.get("cost","-1"))))
  if not price.is_finite() or not Decimal("0")<price<=RATE_CEILING:
   raise RuntimeError("Unexpected CPU hourly rate")
  credit,spend=account()
  allowed=min(Decimal(MAX_SECONDS),
   (min(TOTAL_BUDGET,credit-RESERVE)/(price+spend))*3600)
  seconds=int(allowed)
  if seconds<900:raise RuntimeError("Remaining prepaid balance permits less than 15 minutes")
  print("MAXIMUM_MIGRATION_WALL_SECONDS="+str(seconds),flush=True)
  print("COMPUTE_SPEND_ESTIMATE_CEILING_USD="+str(TOTAL_BUDGET),flush=True)
  key="hephaestus/recovery-status/"+rid+".json"
  end=time.monotonic()+seconds
  last=None
  while time.monotonic()<end:
   try:
    response=s3.get_object(Bucket=SOURCE,Key=key)
    with response["Body"] as f:state=json.load(f)
   except ClientError as exc:
    if str(exc.response.get("Error",{}).get("Code")) not in {"404","NoSuchKey","NotFound"}:raise
    state={}
   if state.get("run_id")==rid:
    event=(state.get("status"),state.get("checked"),state.get("missing"),
           state.get("verified"),state.get("candidate_files"))
    if event!=last:
     print("MIGRATION_STATE="+str(event),flush=True)
     last=event
    if state.get("status")=="FAILED":
     raise RuntimeError("Migration worker reported failure: "+str(state.get("error_class")))
    if state.get("status")=="COMPLETE":
     if (state.get("hf_evidence_verified")!=8113 or
         not re.fullmatch(r"[0-9a-f]{64}",str(state.get("czech_archive_sha256",""))) or
         type(state.get("czech_archive_files")) is not int):
      raise RuntimeError("Incomplete final HF evidence or Czech-only archive receipt")
     print("HF_EVIDENCE_VERIFIED=8113",flush=True)
     print("CZECH_EXTRA_ARCHIVE_SHA256="+state["czech_archive_sha256"],flush=True)
     print("CZECH_EXTRA_ARCHIVE_FILES="+str(state["czech_archive_files"]),flush=True)
     print("HEPHAESTUS_STORAGE_COPY_COMPLETE=true",flush=True)
     return
   live=req("GET","/pods/"+pod_id)
   if live.get("desiredStatus",live.get("status")) in {"EXITED","TERMINATED"}:
    raise RuntimeError("Migration Pod exited before final success status")
   time.sleep(35)
  raise RuntimeError("Prepaid migration time budget reached; completed Global files persist")
 finally:
  # Only this exact ephemeral migration Pod is eligible for teardown.
  if pod_id:
   candidates=[p for p in pods() if p.get("id")==pod_id and p.get("name")==name]
   if candidates:
    try:req("DELETE","/pods/"+pod_id)
    except urllib.error.HTTPError as exc:
     if exc.code!=404:raise
    print("SIMPLE_MIGRATION_POD_TERMINATION_REQUESTED=true",flush=True)

if __name__=="__main__":
 try:main()
 except Exception as exc:
  print("SIMPLE_MIGRATION_CONTROLLER_FAILED="+type(exc).__name__+": "+str(exc)[:170],flush=True)
  raise SystemExit(1)
