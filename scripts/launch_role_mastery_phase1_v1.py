#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
SCRIPTS=Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path: sys.path.insert(0,str(SCRIPTS))
import launch_adaptation_elasticity_v1 as lifecycle
import launch_first_bounded_scientific_training as storage
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig,RunPodExecutionAdapter
import role_mastery_common_v1 as common
ROOT=Path(__file__).resolve().parents[1];MARKER=ROOT/"configs/experiments/role_mastery_phase1_v1.launch.json";AUTH="HEPHAESTUS_ROLE_MASTERY_LAUNCH_AUTHORIZED";TERMINAL={"EXITED","FAILED","TERMINATED","STOPPED"}
def req(k):
 v=(os.environ.get(k) or "").strip()
 if not v: raise RuntimeError("missing "+k)
 return v
def marker(): return json.loads(MARKER.read_text())
def shell():
 return r'''set -Eeuo pipefail
export HF_HOME=/opt/hephaestus-cache/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME=/opt/hephaestus-cache/xdg
export TMPDIR=/opt/hephaestus-tmp
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates python3-venv
rm -rf /var/lib/apt/lists/* /opt/hephaestus-src /opt/hephaestus-venv
git clone --filter=blob:none https://github.com/kys928/Hephaestus.git /opt/hephaestus-src
cd /opt/hephaestus-src
git checkout --detach "$HEPHAESTUS_REPO_SHA"
python -m venv --system-site-packages /opt/hephaestus-venv
PY=/opt/hephaestus-venv/bin/python
"$PY" -m pip install --no-cache-dir --disable-pip-version-check -e '.[s3]' \
 'transformers==5.17.0' 'accelerate==1.15.0' 'safetensors==0.8.0' 'huggingface-hub==1.31.0' 'peft==0.21.0' 'mistral-common==1.11.7'
"$PY" -m pip uninstall -y hf-xet >/dev/null 2>&1 || true
export PYTHONPATH=src:scripts
"$PY" scripts/validate_role_mastery_phase1_v1.py
"$PY" -m py_compile scripts/role_mastery_common_v1.py scripts/run_role_mastery_phase1_v1.py
"$PY" scripts/run_role_mastery_phase1_v1.py
'''
def env(repo_sha,run_id,role,real=False):
 d={"HEPHAESTUS_REPO_SHA":repo_sha,"HEPHAESTUS_ROLE_MASTERY_RUN_ID":run_id,"HEPHAESTUS_ROLE_MASTERY_ROLE":role,"RUNPOD_S3_ACCESS_KEY_ID":"<secret>","RUNPOD_S3_SECRET_ACCESS_KEY":"<secret>","RUNPOD_S3_ENDPOINT_URL":"<endpoint>","RUNPOD_DATACENTER_ID":"<region>","RUNPOD_NETWORK_VOLUME_ID":"<bucket>","PYTHONUNBUFFERED":"1"}
 if real:
  for k in ["RUNPOD_S3_ACCESS_KEY_ID","RUNPOD_S3_SECRET_ACCESS_KEY","RUNPOD_S3_ENDPOINT_URL","RUNPOD_DATACENTER_ID","RUNPOD_NETWORK_VOLUME_ID"]: d[k]=req(k)
 return d
def body(c,name,e):
 x=c["execution"];return {"name":name[:180],"computeType":"GPU","gpuCount":1,"gpuTypeIds":x["gpu_type_ids"],"gpuTypePriority":"availability","cloudType":x["cloud_type"],"imageName":x["image"],"containerDiskInGb":x["container_disk_gb"],"dockerStartCmd":["bash","-lc",shell()],"interruptible":False,"env":e}
def maybe_read(client,key):
 try:return storage.maybe_read_key(client,key)
 except Exception:return None
def execute(role):
 c=common.load_cfg();m=marker()
 if os.environ.get(AUTH)!="YES" or c["governance"].get("paid_launch_allowed") is not True or m.get("authorized") is not True: raise RuntimeError("Phase I paid launch not authorized")
 if role not in c["candidate_registry"]: raise RuntimeError("unknown role")
 repo=req("GITHUB_SHA");gid=req("GITHUB_RUN_ID");run_id=f"role-mastery-phase1-{gid}-{role}"
 prefix=f"{c['execution']['s3_prefix'].rstrip('/')}/{run_id}";result_key=f"{prefix}/result.json";progress_key=f"{prefix}/progress.json"
 ex=RunPodExecutionAdapter(RunPodConfig.from_env(),EnvironmentSecretsProvider());client=storage.s3_client();b=body(c,"hephaestus-"+run_id,env(repo,run_id,role,True))
 pod_id=None;started=time.monotonic();rec={"run_id":run_id,"role":role,"repo_sha":repo,"status":"starting"}
 try:
  pod=ex._create_pod(b);pod_id=str(pod["id"]);rec["pod_id"]=pod_id
  while True:
   elapsed=time.monotonic()-started
   if elapsed>=float(c["execution"]["hard_wall_seconds"]): raise TimeoutError("role mastery hard wall")
   snap=storage.pod_snapshot(ex,pod_id)
   if snap is None: raise RuntimeError("pod disappeared")
   if snap.get("costPerHr") is not None:
    hourly=float(snap["costPerHr"]);cost=hourly*elapsed/3600;rec.update(cost_per_hr=hourly,estimated_cost_usd=cost)
    if hourly>float(c["execution"]["max_hourly_usd"]): raise RuntimeError("hourly cost ceiling exceeded")
    if cost>float(c["execution"]["max_estimated_role_usd"]): raise RuntimeError("role cost ceiling exceeded")
   raw=maybe_read(client,progress_key)
   if raw: rec["last_progress"]=json.loads(raw)
   raw=maybe_read(client,result_key)
   if raw:
    obj=json.loads(raw)
    if obj.get("status")=="failed": raise RuntimeError(f"scientific runner failed: {obj.get('error_type')}: {obj.get('error')}")
    if obj.get("status")=="completed": rec.update(status="completed",result=obj);print("ROLE_MASTERY_LAUNCH_RESULT_JSON "+json.dumps(rec,sort_keys=True),flush=True);return
   if str(snap.get("desiredStatus","")).upper() in TERMINAL: raise RuntimeError("pod terminal before result")
   time.sleep(float(c["execution"]["poll_seconds"]))
 finally:
  if pod_id:
   td=lifecycle.delete_pod_with_retries(ex,pod_id);print("ROLE_MASTERY_TEARDOWN_JSON "+json.dumps({"role":role,**td},sort_keys=True),flush=True)
   if not td.get("verified_absent"): raise RuntimeError("pod teardown unverified")
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--role",required=True);ap.add_argument("--execute",action="store_true");a=ap.parse_args();c=common.load_cfg()
 rid=f"role-mastery-phase1-{os.environ.get('GITHUB_RUN_ID','<run>')}-{a.role}"
 if not a.execute:
  print(json.dumps({"mode":"render_only","role":a.role,"authorized":marker().get("authorized"),"paid_launch_allowed":c["governance"].get("paid_launch_allowed"),"request":body(c,"hephaestus-"+rid,env(os.environ.get("GITHUB_SHA","<sha>"),rid,a.role))},indent=2));return
 execute(a.role)
if __name__=="__main__":main()
