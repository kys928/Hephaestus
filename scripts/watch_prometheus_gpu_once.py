"""Independent off-pod Prometheus GPU watchdog.

This program NEVER allocates a Pod. It writes a start-ready receipt before the
launcher's provider-create request and watches only the exact campaign resource.
A cancelled launcher cannot cancel this independent GitHub Actions workflow.
The watchdog deletes an admitted Pod after worker exit or at its finite deadline.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import time

from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod.config import RunPodConfig
from hephaestus.providers.runpod.execution import RunPodExecutionAdapter
from hephaestus.providers.runpod.storage import RunPodNetworkVolumeStorage
from scripts.launch_prometheus_gpu_once import (
    ID, MAX_LEASE_SECONDS, _source_scope, _read_exit, _s3_existing, _put_once, IMAGE,
)

START_GRACE_SECONDS=1200
WATCHDOG_GRACE_SECONDS=480


def _matching(adapter, *, prefix, basename, config):
    pods=[]
    for p in adapter.list_pods():
        if not str(p.get("name","")).startswith(basename+"-"):
            continue
        # A model-created or unrelated Pod with an imitated name cannot be
        # confidently deleted. Refuse and escalate instead of guessing.
        if (p.get("dataCenterId")!=config.datacenter_id
            or p.get("cloud")!="SECURE"
            or p.get("image")!=IMAGE
            or p.get("env",{}).get("PROMETHEUS_CAMPAIGN_ID")!=prefix
            or p.get("gpu",{}).get("count")!=1
            or p.get("mounts")!={"network":[{"volumeId":config.network_volume_id,"path":"/workspace"}]}):
            raise PermissionError("matched-name Pod has an unexpected immutable source or volume identity")
        pods.append(p)
    return pods


def watch(args, *, config,adapter,s3,clock=time):
    prefix, basename=_source_scope(args.source_sha)
    if config.datacenter_id!="EU-CZ-1" or config.network_volume_id!="cviwpryzao":
        raise PermissionError("wrong network volume/datacenter for off-pod cleanup")
    ready_key=prefix+"/status/watchdog-ready.json"
    if _s3_existing(s3,config.network_volume_id,ready_key):
        raise PermissionError("watchdog already started for this campaign; duplicate attempt refused")
    start=datetime.now(timezone.utc)
    hard_deadline=start+timedelta(seconds=MAX_LEASE_SECONDS+WATCHDOG_GRACE_SECONDS)
    ready={
        "schema":"prometheus-independent-offpod-watchdog.v1",
        "source_sha":args.source_sha,"prefix":prefix,
        "started_at":start.isoformat(),"hard_deadline":hard_deadline.isoformat(),
        "owner":"separately_scheduled_github_actions_watchdog",
        "provider":"RunPod_v2","network_volume_id":config.network_volume_id,
    }
    _put_once(s3,config.network_volume_id,ready_key,json.dumps(ready,sort_keys=True).encode())
    print("PROMETHEUS_WATCHDOG_READY=true",flush=True)
    seen=False
    first=time.monotonic()
    while datetime.now(timezone.utc)<hard_deadline:
        pods=_matching(adapter,prefix=prefix,basename=basename,config=config)
        if len(pods)>1:
            # Delete every *individually* authenticated duplicate campaign Pod
            # before reporting the dangerous allocation-state violation.
            for pod in pods:
                adapter.delete_pod(pod["id"])
            raise RuntimeError("duplicate paid Prometheus resources terminated by independent watchdog")
        if pods:
            seen=True
            worker_exit=_read_exit(s3,config.network_volume_id,prefix)
            if worker_exit is not None:
                adapter.delete_pod(pods[0]["id"])
                print("PROMETHEUS_WATCHDOG_CLEANED_WORKER_EXIT="+str(worker_exit),flush=True)
                return
        if not seen and time.monotonic()-first>START_GRACE_SECONDS:
            # Final full-provider reconciliation before a clean no-allocation exit.
            if not _matching(adapter,prefix=prefix,basename=basename,config=config):
                print("PROMETHEUS_WATCHDOG_NO_ALLOCATION=true",flush=True)
                return
        time.sleep(25)
    # Hard deadline: a worker can have been OOM-killed, a runner killed, or a
    # status write lost; do not assume process termination means Pod deletion.
    pods=_matching(adapter,prefix=prefix,basename=basename,config=config)
    for pod in pods:
        for attempt in range(5):
            try:
                adapter.delete_pod(pod["id"])
                break
            except Exception:
                if attempt==4:
                    raise
                time.sleep(10)
    print("PROMETHEUS_WATCHDOG_DEADLINE_CLEANUP="+str(len(pods)),flush=True)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha",required=True)
    parser.add_argument("--state-root",type=Path,required=True)
    args=parser.parse_args()
    for key in ("RUNPOD_API_KEY","RUNPOD_S3_ACCESS_KEY_ID","RUNPOD_S3_SECRET_ACCESS_KEY"):
        if not os.environ.get(key):
            raise PermissionError("independent RunPod watchdog credentials unavailable")
    config=RunPodConfig.from_env()
    secrets=EnvironmentSecretsProvider()
    adapter=RunPodExecutionAdapter(config,secrets,state_root=args.state_root)
    s3=RunPodNetworkVolumeStorage.from_boto3(config,secrets).client
    watch(args,config=config,adapter=adapter,s3=s3)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
