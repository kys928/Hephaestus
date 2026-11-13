#!/usr/bin/env python3
"""One paid, bounded CUDA inference verification for original Hephaestus Controller.

Runs on any datacenter because it only reads pinned Global data and public HF
weights; the Czech source volume is not mounted. No training or promotion.
"""
import json,os,re,time,urllib.error,urllib.request
from decimal import Decimal
from botocore.exceptions import ClientError
import boto3

API="https://api.runpod.io/v2"
GLOBAL="cmw51pjsk000007l9a0mv1jga"
SOURCE="cviwpryzao"
WORKER_SHA="48b32ce4f35738252d8e8abf559bad3518eb9495"
GPU_CANDIDATES=("NVIDIA GeForce RTX 5090","NVIDIA RTX PRO 6000 Blackwell Server Edition")
MAX_PRICE=Decimal("3.50")
MAX_RUN_SEC=3300

def call(method,path,body=None):
 raw=None if body is None else json.dumps(body).encode()
 q=urllib.request.Request(API+path,method=method,data=raw,
  headers={"Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"],"Content-Type":"application/json"})
 with urllib.request.urlopen(q,timeout=50) as response:
  data=response.read()
  return json.loads(data) if data else {}

def pods():
 val=call("GET","/pods")
 rows=val if isinstance(val,list) else val.get("pods",val.get("data",[]))
 if not isinstance(rows,list):raise RuntimeError("Invalid Pod list")
 return rows

def balance():
 query={"query":"query HephaestusCudaSmokeBudget { myself { clientBalance currentSpendPerHr isAutoPayEnabled } }"}
 q=urllib.request.Request("https://api.runpod.io/graphql",data=json.dumps(query).encode(),
  headers={"Authorization":"Bearer "+os.environ["RUNPOD_API_KEY"],"Content-Type":"application/json"})
 with urllib.request.urlopen(q,timeout=30) as r:data=json.load(r)["data"]["myself"]
 if data.get("isAutoPayEnabled") is not False:raise RuntimeError("Autopay status not as observed")
 credit=Decimal(str(data["clientBalance"]))
 if not credit.is_finite() or credit<Decimal("4.00"):raise RuntimeError("Not enough remaining prepaid credit for GPU smoke")
 return credit,Decimal(str(data["currentSpendPerHr"]))

def main():
 rid=os.environ.get("GITHUB_RUN_ID","")
 if not rid.isdecimal() or not os.environ.get("HF_TOKEN"):raise RuntimeError("GPU smoke run ID / HF token missing")
 name="hephaestus-global-controller-cuda-"+rid
 if [p for p in pods() if p.get("name")==name]:raise RuntimeError("Exact smoke Pod already exists")
 credit,spend=balance()
 print("GPU_SMOKE_PREPAID_USD="+str(credit),flush=True)
 s3=boto3.client("s3",endpoint_url="https://s3api-eu-cz-1.runpod.io/",region_name="EU-CZ-1",
  aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
  aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"])
 s3.head_bucket(Bucket=SOURCE)
 url="https://raw.githubusercontent.com/kys928/Hephaestus/"+WORKER_SHA+"/scripts/hephaestus_one_role_cuda_smoke.py"
 start=("python -m pip install --no-cache-dir -q 'transformers==5.6.2' 'peft==0.21.2' "+
        "'accelerate>=1.3,<2' 'boto3>=1.35,<2' 'huggingface_hub[hf_xet]>=0.36,<2' && "+
        "python -c \"import urllib.request;open('/tmp/cuda_smoke.py','wb').write(urllib.request.urlopen('"+
        url+"',timeout=60).read())\" && python -u /tmp/cuda_smoke.py && exec sleep infinity")
 podid=None
 try:
  for gpu in GPU_CANDIDATES:
   body={"name":name,"gpu":{"id":gpu,"count":1},"cloud":"SECURE",
    "image":"runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404","disk":50,
    "mounts":{"global":[{"volumeId":GLOBAL,"path":"/workspace-global"}]},
    "cmd":["bash","-lc",start],
    "env":{"HF_TOKEN":os.environ["HF_TOKEN"],"GPU_SMOKE_RUN_ID":rid,
     "RUNPOD_GLOBAL_VOLUME_ID":GLOBAL,
     "RUNPOD_S3_ACCESS_KEY_ID":os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
     "RUNPOD_S3_SECRET_ACCESS_KEY":os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"]}}
   try:created=call("POST","/pods",body)
   except urllib.error.HTTPError as err:
    matches=[p for p in pods() if p.get("name")==name]
    if len(matches)==1:
     created=matches[0]
    else:
     print("GPU_ALLOCATION_REJECTED="+gpu+"; http="+str(err.code),flush=True)
     if err.code not in (400,409,429,503):raise
     continue
   podid=created.get("id")
   if not isinstance(podid,str) or not re.fullmatch(r"[A-Za-z0-9_-]{4,90}",podid):
    raise RuntimeError("Unexpected GPU Pod identity")
   print("GPU_SMOKE_POD_ID="+podid,flush=True)
   print("GPU_SMOKE_SELECTED_HARDWARE="+gpu,flush=True)
   break
  if not podid:raise RuntimeError("Neither specified single-GPU type could be allocated on Secure Cloud")
  live=call("GET","/pods/"+podid)
  if live.get("id")!=podid or not any(x.get("volumeId")==GLOBAL and x.get("path")=="/workspace-global" for x in live.get("mounts",{}).get("global",[])):
   raise RuntimeError("Wrong mounted Global Volume on GPU Pod")
  actual=Decimal(str(live.get("costPerHr",live.get("cost","-1"))))
  if not actual.is_finite() or actual<=0 or actual>MAX_PRICE:raise RuntimeError("Actual GPU hourly price unacceptable")
  print("GPU_SMOKE_ACTUAL_PRICE_PER_HOUR="+str(actual),flush=True)
  credit,spend=balance()
  # Global is durable and account credit also funds an unrelated running Pod.
  wall=int(min(Decimal(MAX_RUN_SEC),((credit-Decimal("2"))/(actual+spend))*3600))
  if wall<600:raise RuntimeError("Available prepaid funds cannot support minimum CUDA verification")
  until=time.monotonic()+wall
  last=None
  while time.monotonic()<until:
   try:
    obj=s3.get_object(Bucket=SOURCE,Key="hephaestus/gpu-smoke-status/"+rid+".json")
    with obj["Body"] as f:state=json.load(f)
   except ClientError as exc:
    if str(exc.response.get("Error",{}).get("Code")) not in ("404","NoSuchKey","NotFound"):raise
    state={}
   if state.get("run_id")==rid:
    phase=state.get("status")
    if phase!=last:print("GPU_MODEL_INFERENCE_PHASE="+str(phase),flush=True);last=phase
    if phase=="FAILED":raise RuntimeError("CUDA inference worker failed: "+str(state.get("error_class"))+" "+str(state.get("error_summary",""))[:100])
    if phase=="COMPLETE":
     if state.get("global_volume_id")!=GLOBAL or state.get("base_and_lora_loaded_on_cuda") is not True or state.get("generated_tokens",0)<1:
      raise RuntimeError("GPU result not an actual CUDA PEFT generation")
     print("ACTUAL_HEPHAESTUS_CONTROLLER_CUDA_INFERENCE=PASS",flush=True)
     print("CUDA_OUTPUT_TOKEN_SHA256="+str(state.get("token_ids_sha256")),flush=True)
     return
   live=call("GET","/pods/"+podid)
   if live.get("desiredStatus",live.get("status")) in ("EXITED","TERMINATED"):raise RuntimeError("GPU Pod stopped without valid CUDA inference")
   time.sleep(35)
  raise RuntimeError("GPU verification exhausted its safe remaining credit/time window")
 finally:
  if podid:
   matches=[p for p in pods() if p.get("name")==name and p.get("id")==podid]
   if matches:
    try:call("DELETE","/pods/"+podid)
    except urllib.error.HTTPError as exc:
     if exc.code!=404:raise
    print("EXACT_GPU_SMOKE_POD_DELETE_REQUESTED=true",flush=True)

if __name__=="__main__":
 try:main()
 except Exception as exc:
  print("HEPHAESTUS_CUDA_VERIFIER_FAILED="+type(exc).__name__+": "+str(exc)[:160],flush=True)
  raise SystemExit(1)
