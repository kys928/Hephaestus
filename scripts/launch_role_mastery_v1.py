#!/usr/bin/env python3
"""Launch and monitor one Phase I role-mastery pod."""
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
CFG=ROOT/"configs/experiments/hephaestus_role_mastery_v1.json"
MARKER=ROOT/"configs/experiments/role_mastery_v1.launch.json"
AUTH="HEPHAESTUS_ROLE_MASTERY_LAUNCH_AUTHORIZED"
TERMINAL={"EXITED","FAILED","TERMINATED","STOPPED"}

def req(k:str)->str:
    v=(os.environ.get(k) or "").strip()
    if not v: raise RuntimeError("missing "+k)
    return v

def cfg()->dict: return json.loads(CFG.read_text())
def marker()->dict: return json.loads(MARKER.read_text())

def shell(role:str)->str:
    return f'''set -Eeuo pipefail
export HF_HOME=/opt/hephaestus-cache/huggingface
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME=/opt/hephaestus-cache/xdg
export TMPDIR=/opt/hephaestus-tmp
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"
nvidia-smi
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
"$PY" scripts/build_role_mastery_v1.py
"$PY" -m py_compile scripts/run_role_mastery_v1.py
"$PY" scripts/run_role_mastery_v1.py --role {role}
'''

def env(repo_sha:str,run_id:str,role:str)->dict[str,str]:
    return {
      "HEPHAESTUS_REPO_SHA":repo_sha,"HEPHAESTUS_ROLE_MASTERY_RUN_ID":run_id,"HEPHAESTUS_ROLE":role,
      "RUNPOD_S3_ACCESS_KEY_ID":req("RUNPOD_S3_ACCESS_KEY_ID"),
      "RUNPOD_S3_SECRET_ACCESS_KEY":req("RUNPOD_S3_SECRET_ACCESS_KEY"),
      "RUNPOD_S3_ENDPOINT_URL":req("RUNPOD_S3_ENDPOINT_URL"),
      "RUNPOD_DATACENTER_ID":req("RUNPOD_DATACENTER_ID"),
      "RUNPOD_NETWORK_VOLUME_ID":req("RUNPOD_NETWORK_VOLUME_ID"),
      "PYTHONUNBUFFERED":"1",
    }

def body(c:dict,role:str,repo_sha:str,run_id:str)->dict:
    x=c["execution"]
    return {"name":f"hephaestus-role-mastery-{role}-{run_id}"[:180],"computeType":"GPU","gpuCount":1,
      "gpuTypeIds":x["gpu_type_ids"],"gpuTypePriority":"availability","cloudType":x["cloud_type"],
      "imageName":x["image"],"containerDiskInGb":x["container_disk_gb"],
      "dockerStartCmd":["bash","-lc",shell(role)],"interruptible":False,"env":env(repo_sha,run_id,role)}

def maybe(client,key):
    for i in range(5):
        try:return storage.maybe_read_key(client,key)
        except Exception as exc:
            if type(exc).__name__!="FlexibleChecksumError":raise
            time.sleep(min(2**i,8))
    return None

def execute(role:str)->dict:
    c=cfg();m=marker()
    if role not in c["candidates"]: raise RuntimeError("unknown role "+role)
    if os.environ.get(AUTH)!="YES" or m.get("authorized") is not True or c["governance"]["training_authorized"] is not True:
        raise RuntimeError("role mastery launch is not explicitly authorized")
    repo_sha=req("GITHUB_SHA");gid=req("GITHUB_RUN_ID");run_id=f"role-mastery-v1-{gid}"
    prefix=f"{c['execution']['s3_prefix'].rstrip('/')}/{run_id}/roles/{role}"
    result_key=f"{prefix}/result.json";progress_key=f"{prefix}/progress.json"
    ex=RunPodExecutionAdapter(RunPodConfig.from_env(),EnvironmentSecretsProvider());client=storage.s3_client()
    pod_id=None;started=time.monotonic();rec={"role":role,"run_id":run_id,"repo_sha":repo_sha,"status":"starting"}
    try:
        pod=ex._create_pod(body(c,role,repo_sha,run_id));pod_id=str(pod["id"]);rec["pod_id"]=pod_id
        print("ROLE_MASTERY_POD_JSON "+json.dumps(rec,sort_keys=True),flush=True)
        while True:
            elapsed=time.monotonic()-started
            if elapsed>=float(c["execution"]["hard_wall_seconds"]): raise TimeoutError("role mastery hard wall reached")
            snap=storage.pod_snapshot(ex,pod_id)
            if snap is None: raise RuntimeError("pod disappeared before terminal result")
            if snap.get("costPerHr") is not None:
                hourly=float(snap["costPerHr"]);cost=hourly*elapsed/3600;rec.update(cost_per_hr=hourly,estimated_cost_usd=cost)
                if hourly>float(c["execution"]["max_hourly_usd"]): raise RuntimeError("hourly cost ceiling exceeded")
                if cost>float(c["execution"]["max_estimated_usd_per_role"]): raise RuntimeError("per-role cost ceiling exceeded")
            raw=maybe(client,progress_key)
            if raw:
                rec["last_progress"]=json.loads(raw)
                print("ROLE_MASTERY_MONITOR_JSON "+json.dumps({"role":role,"pod_id":pod_id,"cost_per_hr":rec.get("cost_per_hr"),"estimated_cost_usd":rec.get("estimated_cost_usd"),"progress":rec["last_progress"]},sort_keys=True),flush=True)
            terminal=maybe(client,result_key)
            if terminal:
                obj=json.loads(terminal)
                if obj.get("repo_sha")!=repo_sha: raise RuntimeError("role result repo SHA mismatch")
                if obj.get("status")=="failed": raise RuntimeError(f"scientific runner failed: {obj.get('error_type')}: {obj.get('error')}")
                if obj.get("status")=="completed":
                    rec["status"]="completed";rec["result"]=obj
                    print("ROLE_MASTERY_LAUNCH_RESULT_JSON "+json.dumps(rec,sort_keys=True),flush=True);return rec
            if str(snap.get("desiredStatus","")).upper() in TERMINAL: raise RuntimeError("pod terminal before S3 result")
            time.sleep(float(c["execution"]["poll_seconds"]))
    finally:
        if pod_id:
            td=lifecycle.delete_pod_with_retries(ex,pod_id)
            print("ROLE_MASTERY_TEARDOWN_JSON "+json.dumps({"role":role,**td},sort_keys=True),flush=True)
            if not td.get("verified_absent"): raise RuntimeError("pod teardown unverified")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--role",required=True);ap.add_argument("--execute",action="store_true");a=ap.parse_args()
    if not a.execute:
        c=cfg();print(json.dumps({"role":a.role,"authorized":marker().get("authorized"),"candidate":c["candidates"].get(a.role),"execution":c["execution"]},indent=2));return
    execute(a.role)
if __name__=="__main__":main()
