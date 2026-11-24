#!/usr/bin/env python3
"""Independent GitHub-hosted RunPod spending watchdog for one triad campaign.

Runs outside the GPU Pod and outside its primary controller, with its own
RunPod credentials. Never creates a Pod, never touches unrelated resources.
"""
from __future__ import annotations

from decimal import Decimal
from datetime import datetime,timezone
import argparse,json,os,re,time

from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod.config import RunPodConfig
from hephaestus.providers.runpod.execution import RunPodExecutionAdapter
from hephaestus.providers.runpod.cost_intelligence import ReadOnlyRunPodPricing
from hephaestus.providers.runpod.storage import RunPodNetworkVolumeStorage
from pathlib import Path

MAX_LIVE=75*60
MAX_FIRST_POD_WAIT=25*60
RESERVE=Decimal("1.00")
BASE="hephaestus/operations/ministral-triad"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-tag",required=True)
    a=p.parse_args()
    if not re.fullmatch(r"[0-9]{6,20}",a.run_tag):
        raise PermissionError("watchdog run id must be exact GitHub Actions run number")
    cfg=RunPodConfig.from_env()
    if cfg.network_volume_id!="cviwpryzao" or cfg.datacenter_id!="EU-CZ-1":
        raise PermissionError("independent watchdog refused unrelated GPU assets")
    provider=ReadOnlyRunPodPricing(bearer_token=os.environ["RUNPOD_API_KEY"])
    adapter=RunPodExecutionAdapter(cfg,EnvironmentSecretsProvider(),
        state_root=Path(os.environ["RUNNER_TEMP"])/"triad-independent-watchdog")
    s3=RunPodNetworkVolumeStorage.from_boto3(cfg,EnvironmentSecretsProvider()).client
    target="hephaestus-ministral-triad-"+a.run_tag+"-"
    marker=BASE+"/"+a.run_tag+"/watchdog-heartbeat.json"
    started=time.monotonic()
    first_seen=None
    missing_after_seen=False
    while time.monotonic()-started < MAX_FIRST_POD_WAIT+MAX_LIVE+360:
        remaining=time.monotonic()-started
        if first_seen is None and remaining>MAX_FIRST_POD_WAIT:
            print("TRIAD_WATCHDOG_NO_POD_BEFORE_START_CEILING=true",flush=True)
            return
        s3.put_object(Bucket=cfg.network_volume_id,Key=marker,
            Body=json.dumps({
                "schema":"external-triad-watchdog-heartbeat.v1",
                "run_tag":a.run_tag,
                "heartbeat_epoch":int(time.time()),
                "remaining_until_hard_stop":max(0,int(MAX_FIRST_POD_WAIT+MAX_LIVE+360-remaining)),
            },sort_keys=True).encode())
        own=[pod for pod in adapter.list_pods()
             if isinstance(pod.get("name"),str)
             and pod["name"].startswith(target)
             and isinstance(pod.get("id"),str)]
        if len(own)>1:
            print("TRIAD_WATCHDOG_DUPLICATE_PODS_FOUND=true",flush=True)
            for pod in own:
                adapter.delete_pod(pod["id"])
            raise RuntimeError("duplicate single-campaign Pod forbidden; both terminated")
        if own:
            if first_seen is None:
                first_seen=time.monotonic()
                print("TRIAD_WATCHDOG_REAL_GPU_POD_SEEN=true",flush=True)
            account=provider.account()
            expired=(time.monotonic()-first_seen >= MAX_LIVE
                     or time.monotonic()-started >= MAX_FIRST_POD_WAIT+MAX_LIVE
                     or account.autopay_enabled
                     or account.credit_balance_usd <= RESERVE)
            if expired:
                adapter.delete_pod(own[0]["id"])
                print("TRIAD_WATCHDOG_HARD_POD_TERMINATION=true",flush=True)
                return
        elif first_seen is not None:
            print("TRIAD_WATCHDOG_CONFIRMED_CONTROLLER_POD_DELETION=true",flush=True)
            return
        time.sleep(35)
    # Absolute last-resort bounded cleanup after primary job disappears.
    for pod in adapter.list_pods():
        if str(pod.get("name","")).startswith(target) and pod.get("id"):
            adapter.delete_pod(str(pod["id"]))
    print("TRIAD_WATCHDOG_FINAL_HARD_STOP=true",flush=True)


if __name__=="__main__":
    main()
