#!/usr/bin/env python3
"""Separate off-Pod, non-model watchdog for the exact user-delegated GPU45.

This secondary controller has independent RunPod API access and never
re-creates a Pod. It deletes only exact tagged campaign Pods when the prepaid
balance reserve is reached, full science completes, or hard deadline expires.
The 4h cap applies even if the primary controller/SSH/Python worker dies.
"""
from datetime import datetime,timezone
from decimal import Decimal
import argparse,os,time,boto3
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod.config import RunPodConfig
from hephaestus.providers.runpod.execution import RunPodExecutionAdapter
from hephaestus.providers.runpod.cost_intelligence import ReadOnlyRunPodPricing

RESERVE=Decimal("0.30")
MAX_LIVE_SECONDS=4*3600
MAX_WAIT_FOR_FIRST_POD_SECONDS=45*60
PREFIX="hephaestus-original-five-fullcuda45-"
VOLUME="cviwpryzao"


def exact_pods(adapter, run_tag):
    name=PREFIX+run_tag
    return [x for x in adapter.list_pods()
            if str(x.get("name","")).startswith(name+"-")
            and str(x.get("networkVolumeId",x.get("network_volume_id","")))==VOLUME]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-tag",required=True)
    a=p.parse_args()
    if not a.run_tag.isdecimal():
        raise PermissionError("original GPU watchdog must be pinned to exact GitHub run")
    cfg=RunPodConfig.from_env()
    if cfg.network_volume_id!=VOLUME or cfg.datacenter_id!="EU-CZ-1":
        raise PermissionError("watchdog cannot touch unrelated user resources")
    provider=ReadOnlyRunPodPricing(bearer_token=os.environ["RUNPOD_API_KEY"])
    adapter=RunPodExecutionAdapter(cfg,EnvironmentSecretsProvider(),
        state_root=__import__("pathlib").Path(os.environ.get("RUNNER_TEMP","/tmp"))
                   /"original-gpu-watchdog-read-only-state")
    start=time.monotonic()
    seen=False
    while time.monotonic()-start<MAX_LIVE_SECONDS+900:
        try:
            pods=exact_pods(adapter,a.run_tag)
            if len(pods)>1:
                # Two matching Pods means a bug. Delete ALL own Pods immediately.
                for item in pods:
                    adapter.delete_pod(str(item["id"]))
                raise RuntimeError("duplicate original GPU pods detected and terminated")
            if pods:
                seen=True
                latest=provider.account()
                shutdown=(latest.autopay_enabled
                  or latest.credit_balance_usd<=RESERVE
                  or time.monotonic()-start>=MAX_LIVE_SECONDS)
                if shutdown:
                    adapter.delete_pod(str(pods[0]["id"]))
                    print("INDEPENDENT_ORIGINAL_CUDA_POD_HARD_TERMINATED=true",flush=True)
                    return
            elif seen:
                print("INDEPENDENT_ORIGINAL_CUDA_WATCHDOG_CONFIRMED_NO_POD=true",flush=True)
                return
            elif time.monotonic()-start>MAX_WAIT_FOR_FIRST_POD_SECONDS:
                print("NO_CUDA_POD_ALLOCATION_SEEN_BEFORE_SUPERVISOR_TIMEOUT=true")
                return
        except Exception as exc:
            # Safety-critical visibility. Transient errors retry without
            # ever creating Pods; primary controller remains responsible too.
            print("ORIGINAL_CUDA_WATCHDOG_RECONCILIATION_ERROR="+type(exc).__name__,
                  flush=True)
        time.sleep(45)
    for p in exact_pods(adapter,a.run_tag):
        adapter.delete_pod(str(p["id"]))
    print("ORIGINAL_CUDA_WATCHDOG_FINAL_HARD_STOP=true",flush=True)


if __name__=="__main__":
    main()
