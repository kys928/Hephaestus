#!/usr/bin/env python3
"""Render or explicitly launch the frozen Planner/Judge Bakeoff V1."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
SCRIPTS=Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path: sys.path.insert(0,str(SCRIPTS))
import launch_adaptation_elasticity_v1 as lifecycle
import launch_first_bounded_scientific_training as storage
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig,RunPodExecutionAdapter
ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/experiments/hephaestus_planner_judge_bakeoff_v1.json"
MARKER=ROOT/"configs/experiments/planner_judge_bakeoff_v1.launch.json"
AUTH="HEPHAESTUS_PJ_BAKEOFF_LAUNCH_AUTHORIZED"
TERMINAL={"EXITED","FAILED","TERMINATED","STOPPED"}

def req(k):
    v=(os.environ.get(k) or "").strip()
    if not v: raise RuntimeError("missing "+k)
    return v
def load(): return json.loads(CFG.read_text())
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
python - <<'PYGPU'
import json,subprocess
p=subprocess.run(['nvidia-smi','--query-gpu=driver_version,name,memory.total,uuid,compute_cap','--format=csv,noheader'],capture_output=True,text=True)
print('PJ_BAKEOFF_GPU_JSON '+json.dumps({'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()},sort_keys=True),flush=True)
if p.returncode!=0: raise SystemExit('nvidia-smi failed')
PYGPU
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates python3-venv
rm -rf /var/lib/apt/lists/* /opt/hephaestus-src /opt/hephaestus-venv
git clone --filter=blob:none https://github.com/kys928/Hephaestus.git /opt/hephaestus-src
cd /opt/hephaestus-src
git checkout --detach "$HEPHAESTUS_REPO_SHA"
python -m venv --system-site-packages /opt/hephaestus-venv
PY=/opt/hephaestus-venv/bin/python
"$PY" -m pip install --no-cache-dir --disable-pip-version-check -e '.[s3]' \
  'transformers==5.17.0' 'accelerate==1.15.0' 'safetensors==0.8.0' \
  'huggingface-hub==1.31.0' 'peft==0.21.0' 'mistral-common==1.11.7'
"$PY" -m pip uninstall -y hf-xet >/dev/null 2>&1 || true
"$PY" scripts/validate_planner_judge_bakeoff_v1.py
"$PY" -m py_compile scripts/run_planner_judge_bakeoff_v1.py
"$PY" scripts/run_planner_judge_bakeoff_v1.py
'''

def env(repo_sha,run_id,real=False):
    d={"HEPHAESTUS_REPO_SHA":repo_sha,"HEPHAESTUS_PJ_BAKEOFF_RUN_ID":run_id,"RUNPOD_S3_ACCESS_KEY_ID":"<secret>","RUNPOD_S3_SECRET_ACCESS_KEY":"<secret>","RUNPOD_S3_ENDPOINT_URL":"<endpoint>","RUNPOD_DATACENTER_ID":"<region>","RUNPOD_NETWORK_VOLUME_ID":"<bucket>","PYTHONUNBUFFERED":"1"}
    if real:
        for k in ["RUNPOD_S3_ACCESS_KEY_ID","RUNPOD_S3_SECRET_ACCESS_KEY","RUNPOD_S3_ENDPOINT_URL","RUNPOD_DATACENTER_ID","RUNPOD_NETWORK_VOLUME_ID"]: d[k]=req(k)
    return d

def body(c,name,e):
    x=c["execution"]
    return {"name":name[:180],"computeType":"GPU","gpuCount":1,"gpuTypeIds":x["gpu_type_ids"],"gpuTypePriority":"availability","cloudType":x["cloud_type"],"imageName":x["image"],"containerDiskInGb":x["container_disk_gb"],"dockerStartCmd":["bash","-lc",shell()],"interruptible":False,"env":e}

def maybe_read(client,key):
    for i in range(5):
        try:return storage.maybe_read_key(client,key)
        except Exception as exc:
            if type(exc).__name__!="FlexibleChecksumError":raise
            print("PJ_BAKEOFF_S3_CHECKSUM_RETRY "+json.dumps({"key":key,"attempt":i+1,"error":str(exc)},sort_keys=True),flush=True);time.sleep(min(2**i,8))
    return None

def execute(c):
    m=marker()
    if os.environ.get(AUTH)!="YES" or c["governance"]["paid_launch_allowed"] is not True or m.get("authorized") is not True:
        raise RuntimeError("paid Planner/Judge bakeoff is not explicitly authorized")
    repo=req("GITHUB_SHA");gid=req("GITHUB_RUN_ID");run_id=f"planner-judge-bakeoff-v1-{gid}"
    prefix=f"{c['execution']['s3_prefix'].rstrip('/')}/{run_id}";result_key=f"{prefix}/result.json";progress_key=f"{prefix}/progress.json"
    ex=RunPodExecutionAdapter(RunPodConfig.from_env(),EnvironmentSecretsProvider());client=storage.s3_client();b=body(c,"hephaestus-"+run_id,env(repo,run_id,True))
    pod_id=None;started=time.monotonic();rec={"launcher_version":"planner-judge-bakeoff-launcher.v1","run_id":run_id,"repo_sha":repo,"result_key":result_key,"status":"starting","gpu_type_ids":b["gpuTypeIds"]}
    try:
        pod=ex._create_pod(b);pod_id=str(pod["id"]);rec["pod_id"]=pod_id
        while True:
            elapsed=time.monotonic()-started
            if elapsed>=float(c["execution"]["hard_wall_seconds"]): raise TimeoutError("bakeoff hard wall reached")
            snap=storage.pod_snapshot(ex,pod_id)
            if snap is None: raise RuntimeError("pod disappeared before terminal result")
            if snap.get("costPerHr") is not None:
                hourly=float(snap["costPerHr"]);estimated=hourly*elapsed/3600;rec.update(cost_per_hr=hourly,estimated_cost_usd=estimated)
                if hourly>float(c["execution"]["max_hourly_usd"]): raise RuntimeError("hourly cost ceiling exceeded")
                if estimated>float(c["execution"]["max_estimated_total_usd"]): raise RuntimeError("total cost ceiling exceeded")
            raw=maybe_read(client,progress_key)
            if raw: rec["last_progress"]=json.loads(raw)
            terminal=maybe_read(client,result_key)
            if terminal:
                obj=json.loads(terminal)
                if obj.get("repo_sha")==repo and obj.get("status")=="failed": raise RuntimeError(f"scientific runner failed: {obj.get('error_type')}: {obj.get('error')}")
                if obj.get("status")=="completed": rec["result"]=obj;rec["status"]="completed";print("PJ_BAKEOFF_LAUNCH_RESULT_JSON "+json.dumps(rec,sort_keys=True),flush=True);return rec
            if str(snap.get("desiredStatus","")).upper() in TERMINAL: raise RuntimeError("pod became terminal before S3 result")
            time.sleep(float(c["execution"]["poll_seconds"]))
    finally:
        if pod_id:
            td=lifecycle.delete_pod_with_retries(ex,pod_id);print("PJ_BAKEOFF_TEARDOWN_JSON "+json.dumps(td,sort_keys=True),flush=True)
            if not td.get("verified_absent"): raise RuntimeError("pod teardown unverified")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--execute",action="store_true");args=ap.parse_args();c=load()
    repo=os.environ.get("GITHUB_SHA","<sha>");rid=f"planner-judge-bakeoff-v1-{os.environ.get('GITHUB_RUN_ID','<run>')}"
    rendered={"mode":"render_only","protocol_id":c["protocol_id"],"authorized":marker().get("authorized"),"paid_launch_allowed":c["governance"]["paid_launch_allowed"],"request":body(c,"hephaestus-"+rid,env(repo,rid)),"limits":c["execution"]}
    if not args.execute: print(json.dumps(rendered,indent=2));return
    execute(c)
if __name__=="__main__":main()
