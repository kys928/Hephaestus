"""One-time, *off-pod*, audited RunPod launch for Prometheus decision-first research.

Run only on the authorized same-repo GitHub Actions label event, paired with
the independent Prometheus watchdog workflow. Never reuse a historical GPU
grant or fall back to unpinned role models. No arbitrary dollar ceiling is
assumed; the operational lease is bounded by actual prepaid balance and 2h.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import time

from hephaestus.development.prometheus_scientific_decision import (
    _verify, PARENT_SHA, D_SHA,
)
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod.config import RunPodConfig
from hephaestus.providers.runpod.cost_intelligence import ReadOnlyRunPodPricing
from hephaestus.providers.runpod.execution import RunPodExecutionAdapter, AmbiguousRunPodOperation
from hephaestus.providers.runpod.storage import RunPodNetworkVolumeStorage

ID = "prometheus-certified-granite-decision-first-v1"
GPU_PREFERENCES = (
    "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA A100 80GB PCIe",
)
IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
MAX_LEASE_SECONDS = 7200
MIN_LEASE_SECONDS = 900
RESERVE_USD = Decimal("0.40")
OTHER_SPEND_PER_HOUR = Decimal("0.05")
MAX_IDLE_SECONDS = 900


def _source_scope(sha: str) -> tuple[str,str]:
    if not re.fullmatch(r"[0-9a-f]{40}",sha):
        raise ValueError("original GitHub source SHA must be fully pinned")
    return ("hephaestus/scientific/prometheus/" + ID + "/" + sha,
            "prometheus-" + sha[:16])


def _identity_source(path: Path, expected: str) -> bytes:
    _verify(path,expected)
    return path.read_bytes()


def _s3_existing(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket,Key=key)
    except Exception as exc:
        code=str(getattr(exc,"response",{}).get("Error",{}).get("Code",""))
        if code in ("404","NoSuchKey","NotFound"):
            return False
        raise
    return True


def _put_once(s3, bucket: str, key: str, content: bytes) -> dict:
    if _s3_existing(s3,bucket,key):
        raise FileExistsError("Prometheus campaign source key already exists; no overwrite or duplicate launch")
    s3.put_object(Bucket=bucket,Key=key,Body=content)
    item=s3.get_object(Bucket=bucket,Key=key)
    try:
        readback=item["Body"].read(len(content)+1)
    finally:
        item["Body"].close()
    digest=hashlib.sha256(content).hexdigest()
    if readback!=content or hashlib.sha256(readback).hexdigest()!=digest:
        raise PermissionError("fresh campaign S3 source write/read verification failed")
    return {"key":key,"bytes":len(content),"sha256":digest}


def _wallet_lease(account, quote) -> int:
    # User has not imposed an arbitrary monetary cap, but paid Pod time must
    # still be finite and prepaid-only. Never rely on provider auto-expiry flags.
    if account.autopay_enabled:
        raise PermissionError("unlimited credit-card/autopay spending not permitted by research governance")
    spendable=account.credit_balance_usd-RESERVE_USD
    price=quote.price_usd_per_hour+account.already_spending_usd_per_hour+OTHER_SPEND_PER_HOUR
    if spendable<=0 or price<=0:
        raise PermissionError("no available prepaid RunPod GPU budget")
    seconds=min(MAX_LEASE_SECONDS,int((spendable*Decimal(3600)/price).to_integral_value(
        rounding=ROUND_FLOOR)))
    if seconds<MIN_LEASE_SECONDS:
        raise PermissionError("prepaid balance cannot safely support one bounded model-inference window")
    return seconds


def _pod_command(prefix: str, bundle_hash: str, duration: int) -> tuple[list[str],dict[str,str]]:
    if not re.fullmatch(r"[0-9a-f]{64}",bundle_hash) or not 900<=duration<=MAX_LEASE_SECONDS:
        raise ValueError("code bundle or finite GPU lease identity invalid")
    root="/workspace/"+prefix
    loc="/tmp/prometheus-verified-src"
    q=shlex.quote
    cmd=(
        "set -Eeuo pipefail; umask 077; "
        f"mkdir -p {q(root+'/status')}; "
        f"trap 'status=$?; printf \"%s\\n\" \"$status\" > {q(root+'/status/worker-exit.txt')}; sync' EXIT; "
        f"exec > {q(root+'/worker.log')} 2>&1; "
        f"mkdir -p {q(loc)}; "
        f"test \"$(sha256sum {q(root+'/source/code.tar.gz')} | cut -d' ' -f1)\" = {q(bundle_hash)}; "
        f"tar -xzf {q(root+'/source/code.tar.gz')} -C {q(loc)}; "
        f"cd {q(loc)}; test -f pyproject.toml; "
        "python -m pip install -q --ignore-installed 'cryptography>=44,<47'; "
        "python -m pip install -q -e '.[role-inference,peft-training,s3]'; "
        "export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=0; "
        f"timeout --signal=TERM --kill-after=30s {max(120,duration-180)} "
        "python -u -m hephaestus.development.prometheus_gpu_campaign "
        f"--parent-report {q(root+'/source/original-parent-dev.json')} "
        f"--d-report {q(root+'/source/historical-d-dev.json')} "
        f"--s3-endpoint https://s3api-eu-cz-1.runpod.io/ "
        f"--s3-region EU-CZ-1 --s3-bucket {q('cviwpryzao')} "
        f"--adapter-cache {q(root+'/private/adapters')} "
        "--hf-cache /tmp/prometheus-hf-cache "
        f"--out {q(root+'/private/cycle-001')}; "
        "echo 'PROMETHEUS_GPU_WORKER_FINISHED=true'; sync"
    )
    env={
        "RUNPOD_S3_ACCESS_KEY_ID":os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
        "RUNPOD_S3_SECRET_ACCESS_KEY":os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
        "RUNPOD_NETWORK_VOLUME_ID":"cviwpryzao",
        "HF_HUB_DISABLE_TELEMETRY":"1",
        "HF_HUB_ENABLE_HF_TRANSFER":"0",
        "PROMETHEUS_CAMPAIGN_ID":prefix,
    }
    return ["bash","-lc",cmd],env


def _read_exit(s3,bucket,prefix):
    key=prefix+"/status/worker-exit.txt"
    try:
        object_=s3.get_object(Bucket=bucket,Key=key)
        value=object_["Body"].read(256).decode().strip()
        object_["Body"].close()
    except Exception as exc:
        code=str(getattr(exc,"response",{}).get("Error",{}).get("Code",""))
        if code in ("404","NoSuchKey","NotFound"):
            return None
        raise
    if not value.isdecimal() or not 0<=int(value)<=255:
        raise PermissionError("corrupt worker exit status")
    return int(value)


def launch(args, *, config, adapter, provider, s3):
    sha=args.source_sha
    prefix,name=_source_scope(sha)
    if config.datacenter_id!="EU-CZ-1" or config.network_volume_id!="cviwpryzao":
        raise PermissionError("Prometheus GPU region/volume scope differs from original verified research")
    if adapter.state_root.is_relative_to(Path("/tmp")) is False:
        raise PermissionError("provider intent store path was not admitted")
    parent=_identity_source(args.parent_report,PARENT_SHA)
    d=_identity_source(args.d_report,D_SHA)
    bundle=args.code_bundle.read_bytes()
    bundle_sha=hashlib.sha256(bundle).hexdigest()
    if not bundle or len(bundle)>16*1024*1024:
        raise PermissionError("Prometheus code bundle absent or unexpectedly large")
    if _s3_existing(s3,config.network_volume_id,prefix+"/launch-intent.json"):
        raise PermissionError("Prometheus launch intent already spent; duplicate paid allocation forbidden")
    already=[p for p in adapter.list_pods()
             if str(p.get("name","")).startswith(name+"-")]
    if already:
        raise PermissionError("Prometheus provider resource already exists; reconcile before launching another")
    gpu_id=adapter.select_gpu(list(GPU_PREFERENCES))
    account=provider.account()
    quote=provider.quote_gpu(gpu_id=gpu_id,
                             datacenter_id=config.datacenter_id,cloud="SECURE")
    if quote.gpu_memory_gb<48:
        raise PermissionError("certified role base cannot be safely loaded on sub-48GB device")
    lease=_wallet_lease(account,quote)
    originals=[
        _put_once(s3,config.network_volume_id,prefix+"/source/original-parent-dev.json",parent),
        _put_once(s3,config.network_volume_id,prefix+"/source/historical-d-dev.json",d),
        _put_once(s3,config.network_volume_id,prefix+"/source/code.tar.gz",bundle),
    ]
    intent={
        "schema":"prometheus-paid-one-time-launch-intent.v1",
        "source_sha":sha,"prefix":prefix,"pod_basename":name,
        "model_study":"original-certified-Granite-Evaluator",
        "decision_roles":["diagnosis","planner"],
        "bounded_gpu_id":gpu_id,
        "gpu_price_usd_per_hour_observed":str(quote.price_usd_per_hour),
        "maximum_gpu_seconds":lease,
        "prepaid_balance_observed_usd":str(account.credit_balance_usd),
        "autopay_enabled":False,
        "code_sha256":bundle_sha,
        "source_receipts":originals,
        "new_model_training_allowed":False,
        "final_holdout_allowed":False,
        "production_activation_allowed":False,
        "created_at":datetime.now(timezone.utc).isoformat(),
    }
    _put_once(s3,config.network_volume_id,prefix+"/launch-intent.json",
              json.dumps(intent,sort_keys=True).encode())
    command,env=_pod_command(prefix,bundle_sha,lease)
    pod_id=None
    status="provider_not_allocated"
    try:
        pod=adapter.create_bounded_gpu_pod(
            name=name,image_name=IMAGE,gpu_type_ids=[gpu_id],
            docker_start_cmd=command,env=env,container_disk_in_gb=120,
            operation_id=ID+"-"+sha,
        )
        pod_id=pod["id"]
        print("PROMETHEUS_PROVIDER_POD_ALLOCATED="+pod_id,flush=True)
        deadline=time.monotonic()+lease
        idle_start=time.monotonic()
        while time.monotonic()<deadline:
            time.sleep(25)
            code=_read_exit(s3,config.network_volume_id,prefix)
            if code is not None:
                status="completed" if code==0 else "worker_failed"
                break
            current=adapter.get_pod(pod_id)
            if current.get("status") == "RUNNING":
                idle_start=None
            if idle_start is not None and time.monotonic()-idle_start>MAX_IDLE_SECONDS:
                status="pod_not_ready"
                break
        else:
            status="bounded_deadline_reached"
    finally:
        if pod_id:
            # Termination retries are safe and idempotent. The separate
            # off-pod watchdog independently enforces the same finite lease.
            for attempt in range(5):
                try:
                    adapter.delete_pod(pod_id)
                    print("PROMETHEUS_POD_DELETED="+pod_id,flush=True)
                    break
                except Exception:
                    if attempt==4:
                        raise
                    time.sleep(10)
    print("PROMETHEUS_LAUNCH_OUTCOME="+status,flush=True)
    if status!="completed":
        raise RuntimeError("Prometheus genuine GPU operation did not complete; no science claim")
    return {"status":status,"pod_id":pod_id,"prefix":prefix,"source_sha":sha}


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha",required=True)
    parser.add_argument("--code-bundle",type=Path,required=True)
    parser.add_argument("--parent-report",type=Path,required=True)
    parser.add_argument("--d-report",type=Path,required=True)
    parser.add_argument("--state-root",type=Path,required=True)
    args=parser.parse_args()
    config=RunPodConfig.from_env()
    if config.api_base_url!="https://api.runpod.io/v2":
        raise PermissionError("unsafe RunPod API endpoint")
    if not all(os.environ.get(k) for k in (
        "RUNPOD_API_KEY","RUNPOD_S3_ACCESS_KEY_ID","RUNPOD_S3_SECRET_ACCESS_KEY"
    )):
        raise PermissionError("genuine provider credentials unavailable")
    secrets=EnvironmentSecretsProvider()
    adapter=RunPodExecutionAdapter(config,secrets,state_root=args.state_root)
    s3=RunPodNetworkVolumeStorage.from_boto3(config,secrets).client
    provider=ReadOnlyRunPodPricing(bearer_token=os.environ["RUNPOD_API_KEY"])
    launch(args,config=config,adapter=adapter,provider=provider,s3=s3)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
