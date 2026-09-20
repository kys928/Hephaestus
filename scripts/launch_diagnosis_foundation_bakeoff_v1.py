#!/usr/bin/env python3
"""Render or launch one bounded paid Diagnosis Foundation Bakeoff V1 pod."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
from typing import Any
SCRIPTS=Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:sys.path.insert(0,str(SCRIPTS))
import launch_adaptation_elasticity_v1 as lifecycle
import launch_first_bounded_scientific_training as storage
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig,RunPodExecutionAdapter
ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/experiments/hephaestus_diagnosis_foundation_bakeoff_v1.json"
AUTH="HEPHAESTUS_DIAG_FOUNDATION_LAUNCH_AUTHORIZED"
TERMINAL={"EXITED","FAILED","TERMINATED","STOPPED"}
def req(k):
 v=(os.environ.get(k) or "").strip()
 if not v:raise RuntimeError("missing "+k)
 return v
def load():return json.loads(CFG.read_text())
def shell()->str:
 return r'''set -Eeuo pipefail
export HF_HOME=/opt/hephaestus-cache/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME=/opt/hephaestus-cache/xdg
export TMPDIR=/opt/hephaestus-tmp
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"
python - <<'PYGPU'
import json,subprocess
p=subprocess.run(['nvidia-smi','--query-gpu=driver_version,name,memory.total,uuid,compute_cap','--format=csv,noheader'],capture_output=True,text=True)
print('DIAG_FOUNDATION_GPU_JSON '+json.dumps({'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()},sort_keys=True),flush=True)
if p.returncode!=0:raise SystemExit('nvidia-smi failed')
PYGPU
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates python3-venv
rm -rf /var/lib/apt/lists/* /opt/hephaestus-src /opt/hephaestus-venv
git clone --filter=blob:none https://github.com/kys928/Hephaestus.git /opt/hephaestus-src
cd /opt/hephaestus-src
git checkout --detach "$HEPHAESTUS_REPO_SHA"
python -m venv --system-site-packages /opt/hephaestus-venv
PY=/opt/hephaestus-venv/bin/python
"$PY" -m pip install --no-cache-dir --disable-pip-version-check -e '.[s3]'  'transformers==5.17.0' 'accelerate==1.15.0' 'safetensors==0.8.0'  'huggingface-hub==1.31.0' 'peft==0.21.0' 'mistral-common==1.11.7'
"$PY" -m pip uninstall -y hf-xet >/dev/null 2>&1 || true
"$PY" - <<'PYCUDA'
import json,torch
d={'torch':str(torch.__version__),'cuda':str(torch.version.cuda),'available':torch.cuda.is_available(),'count':torch.cuda.device_count(),'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'capability':list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None}
print('DIAG_FOUNDATION_CUDA_JSON '+json.dumps(d,sort_keys=True),flush=True)
if not d['available'] or d['count']!=1:raise SystemExit('CUDA preflight failed')
x=torch.ones(1,device='cuda');torch.cuda.synchronize();print('DIAG_FOUNDATION_CUDA_OK '+str(float(x.item())),flush=True)
PYCUDA
"$PY" scripts/validate_diagnosis_foundation_bakeoff_v1.py
"$PY" -m py_compile scripts/run_diagnosis_foundation_bakeoff_v1.py
"$PY" scripts/run_diagnosis_foundation_bakeoff_v1.py
'''
def env(repo_sha,run_id,real=False):
 d={"HEPHAESTUS_REPO_SHA":repo_sha,"HEPHAESTUS_DIAG_FOUNDATION_RUN_ID":run_id,"RUNPOD_S3_ACCESS_KEY_ID":"<secret>","RUNPOD_S3_SECRET_ACCESS_KEY":"<secret>","RUNPOD_S3_ENDPOINT_URL":"<endpoint>","RUNPOD_DATACENTER_ID":"<region>","RUNPOD_NETWORK_VOLUME_ID":"<bucket>","PYTHONUNBUFFERED":"1"}
 if real:
  for k in ["RUNPOD_S3_ACCESS_KEY_ID","RUNPOD_S3_SECRET_ACCESS_KEY","RUNPOD_S3_ENDPOINT_URL","RUNPOD_DATACENTER_ID","RUNPOD_NETWORK_VOLUME_ID"]:d[k]=req(k)
 return d
def body(c,name,e):
 x=c["execution"];b={"name":name[:180],"computeType":"GPU","gpuCount":1,"gpuTypeIds":x["gpu_type_ids"],"gpuTypePriority":"availability","cloudType":x["cloud_type"],"imageName":x["image"],"containerDiskInGb":x["container_disk_gb"],"dockerStartCmd":["bash","-lc",shell()],"interruptible":False,"env":e}
 if any(y in str(b["gpuTypeIds"]) for y in ["H100","H200","B200"]):raise ValueError("expensive fallback forbidden")
 if "networkVolumeId" in b:raise ValueError("network volume attachment forbidden")
 return b
def execute(c):
 if os.environ.get(AUTH)!="YES" or c["governance"]["paid_launch_allowed_now"] is not True:raise RuntimeError("paid bakeoff not authorized")
 repo=req("GITHUB_SHA");gid=req("GITHUB_RUN_ID");run_id=f"diagnosis-foundation-v1-{gid}";prefix=f"{c['execution']['s3_prefix']}/{run_id}";result_key=f"{prefix}/result.json";progress_key=f"{prefix}/progress.json"
 ex=RunPodExecutionAdapter(RunPodConfig.from_env(),EnvironmentSecretsProvider());client=storage.s3_client();b=body(c,"hephaestus-"+run_id,env(repo,run_id,True))
 pod_id=None;started=time.monotonic();rec={"launcher_version":"diagnosis-foundation-launcher.v1","run_id":run_id,"repo_sha":repo,"result_key":result_key,"status":"starting","gpu_type_ids":b["gpuTypeIds"]}
 try:
  pod=ex._create_pod(b);pod_id=str(pod["id"]);rec["pod_id"]=pod_id
  while True:
   elapsed=time.monotonic()-started
   if elapsed>=float(c["execution"]["hard_wall_seconds"]):raise TimeoutError("bakeoff hard wall reached")
   snap=storage.pod_snapshot(ex,pod_id)
   if snap is None:raise RuntimeError("pod disappeared before terminal result")
   if snap.get("costPerHr") is not None:
    hourly=float(snap["costPerHr"]);estimated=hourly*elapsed/3600;rec.update(cost_per_hr=hourly,estimated_cost_usd=estimated)
    if hourly>float(c["execution"]["max_hourly_usd"]):raise RuntimeError(f"hourly cost ceiling exceeded: {hourly}")
    if estimated>float(c["execution"]["max_estimated_total_usd"]):raise RuntimeError(f"total cost ceiling exceeded: {estimated}")
   raw=storage.maybe_read_key(client,progress_key)
   if raw:
    p=json.loads(raw);rec["last_progress"]=p
    if p.get("stage")=="materializing_model":
     age=time.time()-float(p.get("timestamp_unix",time.time()));rec["materialization_age_seconds"]=age
     if age>float(c["execution"]["materialization_stall_seconds"]):raise TimeoutError(f"model materialization stalled {age:.1f}s")
   terminal=storage.maybe_read_key(client,result_key)
   if terminal:
    rec["result"]=json.loads(terminal);rec["status"]="completed";print("DIAG_FOUNDATION_LAUNCH_RESULT_JSON "+json.dumps(rec,sort_keys=True),flush=True);return rec
   if str(snap.get("desiredStatus","")).upper() in TERMINAL:raise RuntimeError("pod became terminal before S3 result")
   time.sleep(float(c["execution"]["poll_seconds"]))
 finally:
  if pod_id:
   td=lifecycle.delete_pod_with_retries(ex,pod_id);print("DIAG_FOUNDATION_TEARDOWN_JSON "+json.dumps(td,sort_keys=True),flush=True)
   if not td.get("verified_absent"):raise RuntimeError("pod teardown unverified")
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--execute",action="store_true");args=ap.parse_args();c=load();repo=os.environ.get("GITHUB_SHA","<sha>");rid=f"diagnosis-foundation-v1-{os.environ.get('GITHUB_RUN_ID','<run>')}";b=body(c,"hephaestus-"+rid,env(repo,rid))
 if not args.execute:print(json.dumps({"mode":"render_only","protocol_id":c["protocol_id"],"request":b,"limits":c["execution"]},indent=2));return
 execute(c)
if __name__=="__main__":main()
