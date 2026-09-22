#!/usr/bin/env python3
"""Guarded paid launcher for the fresh adversarial Phase II-B recertification."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
from typing import Any
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig,RunPodExecutionAdapter

ROOT=Path(__file__).resolve().parents[1]
CFG_PATH=ROOT/"configs/experiments/hephaestus_interface_mastery_v1b.json"
MARKER_PATH=ROOT/"configs/experiments/interface_mastery_v1b.launch.json"
AUTH_ENV="HEPHAESTUS_PHASE_II_B_LAUNCH_AUTHORIZED"
TERMINAL={"EXITED","FAILED","TERMINATED","STOPPED"}

def required(name):
    v=(os.environ.get(name) or "").strip()
    if not v:raise RuntimeError("missing required environment variable: "+name)
    return v

def load_json(path):
    v=json.loads(Path(path).read_text())
    if not isinstance(v,dict):raise RuntimeError(f"{path} must be object")
    return v

def authorize(cfg,marker):
    if not cfg["governance"]["paid_launch_allowed"]:raise RuntimeError("Phase II-B paid launch disabled")
    if not cfg["repair_gate"].get("required_repair_run_id"):raise RuntimeError("Phase II-B repair run not frozen")
    if marker.get("authorized") is not True:raise RuntimeError("Phase II-B marker unauthorized")
    phrase=str(marker.get("authorization_phrase",""))
    if not phrase or required(AUTH_ENV)!=phrase:raise RuntimeError("Phase II-B runtime authorization mismatch")

def pod_shell():
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
print('INTERFACE_MASTERY_V1B_GPU_JSON '+json.dumps({'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()},sort_keys=True),flush=True)
if p.returncode != 0:raise SystemExit('nvidia-smi failed')
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
"$PY" scripts/build_interface_mastery_v1b.py
"$PY" scripts/run_interface_mastery_v1b.py
'''

def request_body(cfg,repo_sha,run_id,real_env):
    e=cfg["execution"];env={
      "HEPHAESTUS_REPO_SHA":repo_sha,"HEPHAESTUS_INTERFACE_MASTERY_V1B_RUN_ID":run_id,
      "RUNPOD_S3_ACCESS_KEY_ID":"<secret>","RUNPOD_S3_SECRET_ACCESS_KEY":"<secret>",
      "RUNPOD_S3_ENDPOINT_URL":"<endpoint>","RUNPOD_DATACENTER_ID":"<region>","RUNPOD_NETWORK_VOLUME_ID":"<volume>",
      "PYTHONUNBUFFERED":"1"}
    if real_env:
      for k in ("RUNPOD_S3_ACCESS_KEY_ID","RUNPOD_S3_SECRET_ACCESS_KEY","RUNPOD_S3_ENDPOINT_URL","RUNPOD_DATACENTER_ID","RUNPOD_NETWORK_VOLUME_ID"):env[k]=required(k)
    return {"name":f"hephaestus-interface-mastery-v1b-{run_id}"[:180],"computeType":"GPU","gpuCount":1,
      "gpuTypeIds":list(e["gpu_type_ids"]),"gpuTypePriority":"custom","cloudType":e["cloud_type"],"imageName":e["image"],
      "containerDiskInGb":int(e["container_disk_gb"]),"dockerStartCmd":["bash","-lc",pod_shell()],"interruptible":False,"env":env}

def s3_client():
    import boto3
    from botocore.config import Config
    return boto3.client("s3",endpoint_url=required("RUNPOD_S3_ENDPOINT_URL").rstrip("/"),region_name=required("RUNPOD_DATACENTER_ID"),aws_access_key_id=required("RUNPOD_S3_ACCESS_KEY_ID"),aws_secret_access_key=required("RUNPOD_S3_SECRET_ACCESS_KEY"),config=Config(retries={"mode":"standard","max_attempts":10}))

def maybe_result(client,key):
    try:r=client.get_object(Bucket=required("RUNPOD_NETWORK_VOLUME_ID"),Key=key)
    except Exception as exc:
      response=getattr(exc,"response",{});code=str(response.get("Error",{}).get("Code","")) if isinstance(response,dict) else ""
      if code in {"404","NoSuchKey","NotFound"}:return None
      raise
    try:v=json.loads(r["Body"].read().decode())
    finally:r["Body"].close()
    return v if isinstance(v,dict) else None

def delete_with_retries(execution,pod_id):
    errors=[]
    for attempt in range(1,6):
      try:execution.delete_pod(pod_id)
      except Exception as exc:errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
      time.sleep(min(attempt,3))
      try:execution.get_pod(pod_id)
      except Exception:return {"deleted":True,"verified_absent":True,"attempts":attempt,"errors":errors}
    return {"deleted":False,"verified_absent":False,"attempts":5,"errors":errors}

def execute(cfg,marker):
    authorize(cfg,marker);repo_sha=required("GITHUB_SHA");run_id=f"interface-mastery-v1b-{required('GITHUB_RUN_ID')}"
    key=f"{cfg['execution']['s3_prefix'].rstrip('/')}/{run_id}/result.json"
    client=s3_client();client.head_bucket(Bucket=required("RUNPOD_NETWORK_VOLUME_ID"))
    execution=RunPodExecutionAdapter(RunPodConfig.from_env(),EnvironmentSecretsProvider())
    started=time.monotonic();pod_id=None;record={"launcher_version":"interface-mastery-v1b-launcher.v1","run_id":run_id,"repo_sha":repo_sha,"status":"starting","result_key":key}
    try:
      pod=execution._create_pod(request_body(cfg,repo_sha,run_id,True));pod_id=str(pod["id"]);record["pod_id"]=pod_id
      while True:
        elapsed=time.monotonic()-started
        if elapsed>=float(cfg["execution"]["hard_wall_seconds"]):raise TimeoutError("Phase II-B hard wall reached")
        snap=execution.get_pod(pod_id);cost=snap.get("adjustedCostPerHr",snap.get("costPerHr"))
        if cost is not None:
          hourly=float(cost);estimated=hourly*elapsed/3600;record.update(cost_per_hr=hourly,estimated_cost_usd=estimated)
          if hourly>float(cfg["execution"]["max_hourly_usd"]):raise RuntimeError("Phase II-B hourly cost ceiling exceeded")
          if estimated>float(cfg["execution"]["max_estimated_total_usd"]):raise RuntimeError("Phase II-B cost ceiling exceeded")
        terminal=maybe_result(client,key)
        if terminal is not None:
          record["result"]=terminal
          if terminal.get("status")=="completed":record["status"]="completed";return record
          if terminal.get("status")=="failed":raise RuntimeError(f"Phase II-B runner failed: {terminal.get('error_type')}: {terminal.get('error')}")
        if str(snap.get("desiredStatus","")).upper() in TERMINAL:raise RuntimeError("Phase II-B pod terminal before result")
        time.sleep(float(cfg["execution"]["poll_seconds"]))
    finally:
      if pod_id:
        td=delete_with_retries(execution,pod_id);record["teardown"]=td
        print("INTERFACE_MASTERY_V1B_TEARDOWN_JSON "+json.dumps(td,sort_keys=True),flush=True)
        if not td.get("verified_absent"):raise RuntimeError("Phase II-B teardown not verified")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--execute",action="store_true");args=ap.parse_args()
    cfg=load_json(CFG_PATH);marker=load_json(MARKER_PATH);repo_sha=os.environ.get("GITHUB_SHA","<sha>");run_id=f"interface-mastery-v1b-{os.environ.get('GITHUB_RUN_ID','<run>')}"
    rendered={"mode":"execute" if args.execute else "render_only","protocol_id":cfg["protocol_id"],"authorized":marker.get("authorized") is True,
      "repair_run_id":cfg["repair_gate"].get("required_repair_run_id"),"request":request_body(cfg,repo_sha,run_id,False),"limits":cfg["execution"]}
    if not args.execute:print(json.dumps(rendered,indent=2,sort_keys=True));return 0
    record=execute(cfg,marker);print("INTERFACE_MASTERY_V1B_LAUNCH_RESULT_JSON "+json.dumps(record,sort_keys=True),flush=True);return 0
if __name__=="__main__":raise SystemExit(main())
