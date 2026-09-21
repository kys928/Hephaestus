#!/usr/bin/env python3
"""Launch and monitor one inference-only post-training red-team role audit."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path

SCRIPTS=Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0,str(SCRIPTS))

import launch_adaptation_elasticity_v1 as lifecycle
import launch_first_bounded_scientific_training as storage
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig,RunPodExecutionAdapter

ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/experiments/hephaestus_post_training_redteam_v1.json"
MARKER=ROOT/"configs/experiments/post_training_redteam_v1.launch.json"
AUTH="HEPHAESTUS_POST_TRAINING_REDTEAM_AUTHORIZED"
TERMINAL={"EXITED","FAILED","TERMINATED","STOPPED"}

def req(name:str)->str:
    value=(os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError("missing required environment variable: "+name)
    return value

def cfg()->dict:
    return json.loads(CFG.read_text())

def marker()->dict:
    return json.loads(MARKER.read_text())

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
"$PY" scripts/build_post_training_redteam_v1.py
"$PY" -m py_compile scripts/run_post_training_redteam_v1.py
"$PY" scripts/run_post_training_redteam_v1.py --role {role}
'''

def env(repo_sha:str,run_id:str,role:str)->dict[str,str]:
    return {
        "HEPHAESTUS_REPO_SHA":repo_sha,
        "HEPHAESTUS_POST_TRAINING_REDTEAM_RUN_ID":run_id,
        "HEPHAESTUS_POST_TRAINING_REDTEAM_ROLE":role,
        "RUNPOD_S3_ACCESS_KEY_ID":req("RUNPOD_S3_ACCESS_KEY_ID"),
        "RUNPOD_S3_SECRET_ACCESS_KEY":req("RUNPOD_S3_SECRET_ACCESS_KEY"),
        "RUNPOD_S3_ENDPOINT_URL":req("RUNPOD_S3_ENDPOINT_URL"),
        "RUNPOD_DATACENTER_ID":req("RUNPOD_DATACENTER_ID"),
        "RUNPOD_NETWORK_VOLUME_ID":req("RUNPOD_NETWORK_VOLUME_ID"),
        "PYTHONUNBUFFERED":"1",
    }

def body(c:dict,role:str,repo_sha:str,run_id:str)->dict:
    x=c["execution"]
    return {
        "name":f"hephaestus-post-redteam-{role}-{run_id}"[:180],
        "computeType":"GPU",
        "gpuCount":1,
        "gpuTypeIds":x["gpu_type_ids"],
        "gpuTypePriority":"availability",
        "cloudType":x["cloud_type"],
        "imageName":x["image"],
        "containerDiskInGb":x["container_disk_gb"],
        "dockerStartCmd":["bash","-lc",shell(role)],
        "interruptible":False,
        "env":env(repo_sha,run_id,role),
    }

def maybe(client,key):
    for i in range(5):
        try:
            return storage.maybe_read_key(client,key)
        except Exception as exc:
            if type(exc).__name__!="FlexibleChecksumError":
                raise
            time.sleep(min(2**i,8))
    return None

def execute(role:str, run_id_override:str|None=None)->dict:
    c=cfg();m=marker()
    if role not in c["candidates"]:
        raise RuntimeError("unknown role "+role)
    if os.environ.get(AUTH)!="YES" or m.get("authorized") is not True:
        raise RuntimeError("post-training red-team launch is not explicitly authorized")
    if c["governance"]["inference_only"] is not True or c["governance"]["training_allowed"] is not False:
        raise RuntimeError("post-training red-team governance is not inference-only")
    if m.get("pack_sha256")!=c["pack"]["canonical_sha256"]:
        raise RuntimeError("launch marker pack hash mismatch")

    repo_sha=req("GITHUB_SHA")
    gid=req("GITHUB_RUN_ID")
    run_id=run_id_override or f"post-training-redteam-v1-{gid}"
    prefix=f"{c['execution']['s3_prefix'].rstrip('/')}/{run_id}/roles/{role}"
    result_key=f"{prefix}/result.json"
    progress_key=f"{prefix}/progress.json"
    ex=RunPodExecutionAdapter(RunPodConfig.from_env(),EnvironmentSecretsProvider())
    client=storage.s3_client()
    if run_id_override:
        if os.environ.get("HEPHAESTUS_POST_TRAINING_REDTEAM_RECOVERY_AUTHORIZED")!="YES":
            raise RuntimeError("recovery run-id override is not authorized")
        raw_existing=maybe(client,result_key)
        if raw_existing:
            prior=json.loads(raw_existing)
            if prior.get("status")!="failed":
                raise RuntimeError("recovery is allowed only over a failed prior role result")
            if prior.get("pack_sha256")!=c["pack"]["canonical_sha256"]:
                raise RuntimeError("recovery prior result pack hash mismatch")
            if prior.get("error_type")!="FileNotFoundError" or "selected-adapter.tar.gz." not in str(prior.get("error","")):
                raise RuntimeError("recovery prior failure is not the known adapter download-directory bug")
            client.delete_object(Bucket=storage.VOLUME_ID,Key=result_key)
            client.delete_object(Bucket=storage.VOLUME_ID,Key=progress_key)
    pod_id=None
    started=time.monotonic()
    rec={"role":role,"run_id":run_id,"repo_sha":repo_sha,"status":"starting"}
    try:
        observations=[]
        pod=lifecycle.retry_transient(
            lambda: ex._create_pod(body(c,role,repo_sha,run_id)),
            label=f"create_post_training_redteam_{role}",
            observations=observations,
        )
        pod_id=str(pod["id"])
        rec["pod_id"]=pod_id
        rec["create_observations"]=observations
        print("POST_TRAINING_REDTEAM_POD_JSON "+json.dumps(rec,sort_keys=True),flush=True)
        while True:
            elapsed=time.monotonic()-started
            if elapsed>=float(c["execution"]["hard_wall_seconds"]):
                raise TimeoutError("post-training red-team hard wall reached")
            snap=storage.pod_snapshot(ex,pod_id)
            if snap is None:
                raise RuntimeError("pod disappeared before terminal result")
            if snap.get("costPerHr") is not None:
                hourly=float(snap["costPerHr"])
                cost=hourly*elapsed/3600
                rec.update(cost_per_hr=hourly,estimated_cost_usd=cost)
                if hourly>float(c["execution"]["max_hourly_usd"]):
                    raise RuntimeError("hourly cost ceiling exceeded")
                if cost>float(c["execution"]["max_estimated_usd_per_role"]):
                    raise RuntimeError("per-role cost ceiling exceeded")
            raw=maybe(client,progress_key)
            if raw:
                progress=json.loads(raw)
                rec["last_progress"]=progress
                print("POST_TRAINING_REDTEAM_MONITOR_JSON "+json.dumps({
                    "role":role,"pod_id":pod_id,
                    "cost_per_hr":rec.get("cost_per_hr"),
                    "estimated_cost_usd":rec.get("estimated_cost_usd"),
                    "progress":progress,
                },sort_keys=True),flush=True)
            terminal=maybe(client,result_key)
            if terminal:
                obj=json.loads(terminal)
                if obj.get("repo_sha")!=repo_sha:
                    raise RuntimeError("red-team result repo SHA mismatch")
                if obj.get("status")=="failed":
                    raise RuntimeError(f"scientific runner failed: {obj.get('error_type')}: {obj.get('error')}")
                if obj.get("status")=="completed":
                    rec["status"]="completed"
                    rec["result"]=obj
                    print("POST_TRAINING_REDTEAM_LAUNCH_RESULT_JSON "+json.dumps(rec,sort_keys=True),flush=True)
                    return rec
            if str(snap.get("desiredStatus","")).upper() in TERMINAL:
                raise RuntimeError("pod terminal before S3 result")
            time.sleep(float(c["execution"]["poll_seconds"]))
    finally:
        if pod_id:
            td=lifecycle.delete_pod_with_retries(ex,pod_id)
            print("POST_TRAINING_REDTEAM_TEARDOWN_JSON "+json.dumps({"role":role,**td},sort_keys=True),flush=True)
            if not td.get("verified_absent"):
                raise RuntimeError("post-training red-team pod teardown unverified")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--role",required=True)
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--run-id")
    args=ap.parse_args()
    if not args.execute:
        c=cfg()
        print(json.dumps({"role":args.role,"authorized":marker().get("authorized"),"candidate":c["candidates"].get(args.role),"execution":c["execution"]},indent=2))
        return
    execute(args.role,args.run_id)

if __name__=="__main__":
    main()
