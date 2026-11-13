#!/usr/bin/env python3
"""RunPod CPU Hephaestus application boot controller.

Create only one exact named Pod. Check both source and Global mounts, monitor
real Hephaestus boot status from original regional S3, and leave a healthy
read-only-local console running until its independent GitHub lease expires.
No GPU training, no promotion, no public control API.
"""
from __future__ import annotations
import json,os,re,time,urllib.request,urllib.error
from decimal import Decimal
import boto3
from botocore.exceptions import ClientError

API="https://api.runpod.io/v2"
SOURCE="cviwpryzao"
GLOBAL="cmw51pjsk000007l9a0mv1jga"
BOOT_WORKER_SHA="613b58ae131a18832e46293964f9c56ecc5d9617"
HEPHAESTUS_SOURCE_SHA="f27882fc08aa6d7c162a4b12c5dc522e9cdb69e2"
EXPECTED_ROLES=5
MAX_BOOT_SECONDS=17*60
RATE_CEILING=Decimal("0.60")

def request(method,path,body=None):
 data=json.dumps(body).encode() if body is not None else None
 r=urllib.request.Request(API+path,method=method,data=data,headers={
  "Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"],"Content-Type":"application/json"})
 with urllib.request.urlopen(r,timeout=45) as resp:
  raw=resp.read()
  return json.loads(raw) if raw else {}

def pods():
 v=request("GET","/pods")
 rows=v if isinstance(v,list) else v.get("pods",v.get("data",[]))
 if not isinstance(rows,list):raise RuntimeError("Invalid Pod list")
 return rows

def main():
 rid=os.environ.get("GITHUB_RUN_ID","")
 if not rid.isdecimal():raise RuntimeError("Missing run id")
 name="hephaestus-global-live-"+rid
 if any(p.get("name")==name for p in pods()):
  raise RuntimeError("Pod with exact boot identity already exists")
 s3=boto3.client("s3",endpoint_url="https://s3api-eu-cz-1.runpod.io/",region_name="EU-CZ-1",
  aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
  aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"])
 s3.head_bucket(Bucket=SOURCE)
 url=("https://raw.githubusercontent.com/kys928/Hephaestus/"+BOOT_WORKER_SHA+
   "/scripts/hephaestus_boot_global_pod.py")
 cmd=("python -m pip install --no-cache-dir -q 'boto3>=1.35,<2' && "+
      "python -c \"import urllib.request;open('/tmp/boot.py','wb').write(urllib.request.urlopen('"+
      url+"',timeout=90).read())\" && python -u /tmp/boot.py")
 body={
  "name":name,"cpu":{"id":"cpu3c","vcpuCount":2},"cloud":"SECURE",
  "dataCenterIds":["EU-CZ-1"],"image":"python:3.12-slim","disk":20,
  "mounts":{"network":[{"volumeId":SOURCE,"path":"/workspace-state"}],
            "global":[{"volumeId":GLOBAL,"path":"/workspace-global"}]},
  "cmd":["bash","-lc",cmd],
  "env":{"BOOT_RUN_ID":rid,"HEPHAESTUS_BOOT_CODE_SHA":HEPHAESTUS_SOURCE_SHA,
         "RUNPOD_GLOBAL_VOLUME_ID":GLOBAL,"RUNPOD_NETWORK_VOLUME_ID":SOURCE,
         "RUNPOD_S3_ACCESS_KEY_ID":os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
         "RUNPOD_S3_SECRET_ACCESS_KEY":os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"]}
 }
 pid=None;ready=False
 try:
  try:created=request("POST","/pods",body)
  except Exception as exc:
   m=[x for x in pods() if x.get("name")==name]
   if len(m)!=1:
    if isinstance(exc,urllib.error.HTTPError):
     print("BOOT_CREATE_HTTP_STATUS="+str(exc.code),flush=True)
     try:
      raw=exc.read().decode(errors="replace").lower()
      tags=["cpu","disk","volume","region","capacity","balance","invalid","not available"]
      print("BOOT_CREATE_REJECTION_KINDS="+",".join(t for t in tags if t in raw),flush=True)
     except Exception:pass
    raise RuntimeError("CPU boot Pod allocation failed; no duplicate POST") from exc
   created=m[0]
  pid=created.get("id")
  if not isinstance(pid,str) or not re.fullmatch(r"[A-Za-z0-9_-]{5,80}",pid):raise RuntimeError("Invalid new Pod identity")
  print("HEPHAESTUS_BOOT_POD_ID="+pid,flush=True)
  live=request("GET","/pods/"+pid)
  if live.get("id")!=pid:raise RuntimeError("Pod identity changed")
  mounts=live.get("mounts",{})
  if not any(m.get("volumeId")==GLOBAL and m.get("path")=="/workspace-global" for m in mounts.get("global",[])):
   raise RuntimeError("Global Volume not mounted at expected path")
  if not any(m.get("volumeId")==SOURCE and m.get("path")=="/workspace-state" for m in mounts.get("network",[])):
   raise RuntimeError("Czech state volume not mounted at expected path")
  if live.get("dataCenterId") and live["dataCenterId"]!="EU-CZ-1":raise RuntimeError("Wrong datacenter")
  price=Decimal(str(live.get("costPerHr",live.get("cost","-1"))))
  if not price.is_finite() or not Decimal("0")<price<=RATE_CEILING:
   raise RuntimeError("Unexpected CPU Pod rate")
  print("BOOT_POD_PRICE_USD_PER_HOUR="+str(price),flush=True)
  deadline=time.monotonic()+MAX_BOOT_SECONDS
  prior=None
  while time.monotonic()<deadline:
   try:
    obj=s3.get_object(Bucket=SOURCE,Key="hephaestus/boot-status/"+rid+".json")
    with obj["Body"] as stream:report=json.load(stream)
   except ClientError as exc:
    if str(exc.response.get("Error",{}).get("Code")) not in {"404","NoSuchKey","NotFound"}:raise
    report={}
   if report.get("run_id")==rid:
    phase=report.get("phase")
    if phase!=prior:
     print("HEPHAESTUS_BOOT_PHASE="+str(phase),flush=True)
     prior=phase
    if phase=="FAILED":raise RuntimeError("Boot worker failed: "+str(report.get("error_class"))+" "+str(report.get("error_summary",""))[:100])
    if phase=="READY":
     if report.get("global_volume")!=GLOBAL or report.get("regional_volume")!=SOURCE:
      raise RuntimeError("Wrong boot resource identity")
     if report.get("model_scope")!="five original LoRA adapters materialized; no base-model CUDA inference":
      raise RuntimeError("Actual trained adapter loading was not reported")
     print("HEPHAESTUS_APP_READY=true",flush=True)
     print("HEPHAESTUS_CONSOLE_LOCAL_URL=http://127.0.0.1:8765",flush=True)
     print("HEPHAESTUS_BOOT_CONSOLE_ONLY_LOCALHOST=true",flush=True)
     print("HEPHAESTUS_BOOT_REGIONAL_BACKUP=hephaestus/runtime/boot-"+rid,flush=True)
     ready=True
     return
   live=request("GET","/pods/"+pid)
   if live.get("desiredStatus",live.get("status")) in {"EXITED","TERMINATED"}:raise RuntimeError("Pod stopped before Hephaestus READY")
   time.sleep(18)
  raise RuntimeError("Hephaestus boot timed out before READY")
 finally:
  if not ready:
   m=[p for p in pods() if p.get("name")==name and p.get("id")==pid]
   if m:
    try:request("DELETE","/pods/"+pid)
    except urllib.error.HTTPError as e:
     if e.code!=404:raise
    print("FAILED_HEPHAESTUS_BOOT_POD_DELETE_REQUESTED=true",flush=True)

if __name__=="__main__":
 try:main()
 except Exception as exc:
  print("HEPHAESTUS_BOOT_CONTROLLER_FAILED="+type(exc).__name__+": "+str(exc)[:160],flush=True)
  raise SystemExit(1)
