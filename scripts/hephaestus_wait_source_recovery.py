#!/usr/bin/env python3
"""Wait for original S3 recovery to finish AND terminate before starting independent readback."""
import json,os,time,urllib.request
from botocore.exceptions import ClientError
import boto3
ORIGIN_RUN="43262879222"
SOURCE="cviwpryzao"
POD_NAME="hephaestus-regional-global-recovery-"+ORIGIN_RUN
def get(url,token):
 req=urllib.request.Request(url,headers={"Authorization":"Bearer "+token,"Accept":"application/vnd.github+json"})
 with urllib.request.urlopen(req,timeout=30) as resp:return json.load(resp)
def main():
 token=os.environ["RUNPOD_API_KEY"];gh=os.environ["GH_RECOVERY_TOKEN"]
 s3=boto3.client("s3",endpoint_url="https://s3api-eu-cz-1.runpod.io/",region_name="EU-CZ-1",
  aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"])
 deadline=time.monotonic()+4*3600+600
 previous=None
 while time.monotonic()<deadline:
  run=get("https://api.github.com/repos/kys928/Hephaestus/actions/runs/"+ORIGIN_RUN,gh)
  ps=get("https://api.runpod.io/v2/pods",token)
  entries=ps if isinstance(ps,list) else ps.get("pods",ps.get("data",[]))
  if not isinstance(entries,list):raise RuntimeError("Unknown live RunPod pod list shape")
  matches=[p for p in entries if isinstance(p,dict) and p.get("name")==POD_NAME]
  try:
   obj=s3.get_object(Bucket=SOURCE,Key="hephaestus/recovery-status/"+ORIGIN_RUN+".json")
   with obj["Body"] as stream:status=json.load(stream)
  except ClientError as exc:
   if exc.response.get("Error",{}).get("Code") not in {"404","NoSuchKey","NotFound"}:raise
   status={}
  state=(run.get("status"),run.get("conclusion"),status.get("status"),status.get("verified"),status.get("total"),len(matches))
  if state!=previous:print("SOURCE_RECOVERY_DEPENDENCY="+str(state),flush=True);previous=state
  if run.get("status")=="completed" and run.get("conclusion")!="success":
   raise RuntimeError("Original S3 recovery failed; independent validation cannot start")
  if (run.get("status")=="completed" and run.get("conclusion")=="success" and
      status.get("status") in {"COMPLETE","COMPLETE_WITH_SOURCE_EXCEPTIONS"} and status.get("receipt_sha256") and not matches):
   print("ORIGINAL_S3_EVIDENCE_RESTORE_COMPLETE_AND_STOPPED=true",flush=True)
   return
  time.sleep(80)
 raise RuntimeError("Dependency wait expired: no completed S3 recovery receipt and confirmed Pod teardown")
if __name__=="__main__":
 try:main()
 except Exception as exc:
  print("RECOVERY_DEPENDENCY_BLOCKED="+type(exc).__name__+": "+str(exc)[:220],flush=True)
  raise SystemExit(1)
