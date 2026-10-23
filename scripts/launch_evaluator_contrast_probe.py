#!/usr/bin/env python3
"""One user-approved, instrumented D inference. $1.80 max estimate, no retries."""
from __future__ import annotations
import json,os,time,math
from pathlib import Path
import launch_evaluator_factor_probe as bootstrap
import launch_evaluator_semantic_repair_v2 as prior
import launch_interface_repair_v2 as infra
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig,RunPodExecutionAdapter
from evaluator_contrast_inference_protocol import PREFIX,PROTOCOL,manifest
from evaluator_semantic_factor_probe_protocol import D_CHECKPOINT_SHA256
import preflight_D_contrast_region_stock as regional
from evaluator_D_bootstrap_instrumentation import instrument as instrument_bootstrap

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/experiments/evaluator_contrast_probe_execution.json"
APPROVAL=ROOT/"configs/experiments/evaluator_contrast_probe_approval.json"

def authorize(cfg):
    a=json.loads(APPROVAL.read_text())
    # Former $0.75 approval has been consumed by the deleted pod. This
    # new user-approved, distinct one-use token is required for any allocation.
    if cfg.get("runpod_launch_enabled") is not True:
        raise PermissionError("new paid GPU launch not yet approved")
    if (a.get("status")!="authorized_once"
        or a.get("additional_allocations_authorized") is not True
        or a["approval_id"]!=cfg["approval_id"]
        or a["max_total_estimated_gpu_usd"]!=1.80
        or cfg["max_total_estimated_gpu_usd"]!=1.80
        or cfg["max_hourly_usd"]!=a["max_gpu_hourly_usd"]
        or cfg["max_hourly_usd"]>2.49
        or cfg["gpu_type_id"]!=regional.GPU
        or a["max_gpu_hourly_usd"]!=regional.MAX_HOURLY_USD
        or cfg.get("required_gpu_datacenter")!="EU-CZ-1"
        or cfg.get("storage_datacenter")!="EU-CZ-1"
        or cfg.get("attach_network_volume") is not True
        or cfg.get("s3_remote_volume_access") is not False
        or a["approval_id"]=="heph-D-contrasts-20261019-capacity-retry-1"
        or a.get("max_pods")!=1 or a.get("max_launch_attempts")!=1
        or a["source_D_step96_sha256"]!=D_CHECKPOINT_SHA256
        or not cfg["one_pod"] or cfg["extra_runs"]
        or any(a.get(k) for k in ("contingency_authorized","model_retraining_authorized","certification_authorized"))
        or os.environ.get("HEPHAESTUS_CONTRAST_AUTHORIZED")!="SINGLE_D_CONTRAST_PROBE"):
        raise PermissionError("contrast-only single-run approval boundary")
    return a

def pod_request(cfg,sha,runid,*,real_env):
    compatible=dict(cfg,initial_estimated_ceiling_usd=cfg["max_total_estimated_gpu_usd"],
        exceptional_hard_ceiling_usd=cfg["max_total_estimated_gpu_usd"])
    req=bootstrap.pod_request(compatible,sha,runid,real_env=real_env)
    old="scripts/run_evaluator_factor_probe.py --execute"
    new="scripts/run_evaluator_contrast_probe.py"
    count=[0]
    def rewrite(x):
        if isinstance(x,str):
            count[0]+=x.count(old)
            return x.replace(old,new).replace("factor-only-D-","D-contrast-")
        if isinstance(x,list):return [rewrite(v) for v in x]
        if isinstance(x,dict):return {k:rewrite(v) for k,v in x.items()}
        return x
    req=rewrite(req)
    if count[0]!=1 or old in json.dumps(req):raise ValueError("pod bootstrap command drift")
    if (req.get("gpuCount")!=1 or req.get("cloudType")!="SECURE"
            or req.get("gpuTypeIds")!=["NVIDIA RTX PRO 6000 Blackwell Server Edition"]):
        raise ValueError("unexpected single-GPU bootstrap template")
    # Reuse the same EU-CZ-1 volume and immutable checkpoint/model cache.
    req["gpuTypeIds"]=[cfg["gpu_type_id"]]
    # Prefer the RunPod-maintained Blackwell-compatible CUDA 12.8 image over
    # the previously stalled generic CUDA13 runtime. This changes only
    # container/bootstrap infrastructure, never checkpoint or evaluation.
    if (cfg.get("image")!="runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
            or cfg.get("container_disk_gb")!=100):
        raise ValueError("unvalidated container image or ephemeral disk")
    req["imageName"]=cfg["image"]
    req["containerDiskInGb"]=cfg["container_disk_gb"]
    if real_env and (req.get("dataCenterIds")!=[cfg["storage_datacenter"]]
                     or req.get("networkVolumeId")!=infra.required("RUNPOD_NETWORK_VOLUME_ID")
                     or infra.required("RUNPOD_DATACENTER_ID")!=cfg["storage_datacenter"]):
        raise ValueError("unexpected base volume or S3 storage region")
    req["dataCenterIds"]=[cfg["required_gpu_datacenter"]]
    if cfg.get("attach_network_volume") is not True or cfg["required_gpu_datacenter"]!=cfg["storage_datacenter"]:
        raise ValueError("cross-region volume mount is forbidden")
    # Preserve existing matched EU-CZ-1 network-volume mount and cached HF data.
    env=req["env"]
    env["HEPHAESTUS_GPU_DATACENTER_ID"]=cfg["required_gpu_datacenter"]
    # Same-region mounted volume; S3 is used for immutable event receipts.
    # CUDA12.8 base image must never inherit the CUDA13 virtualenv.
    old_venv="interface-repair-v2-py312-cu130-r2"
    shell=req["dockerStartCmd"][2]
    if shell.count(old_venv)!=1:
        raise ValueError("persistent venv bootstrap path changed")
    req["dockerStartCmd"][2]=shell.replace(old_venv,"D-contrast-py312-cu128-torch280-r1")
    req["dockerStartCmd"]=instrument_bootstrap(req["dockerStartCmd"])
    for key in ("HEPHAESTUS_FACTOR_PROBE_AUTHORIZED","HEPHAESTUS_FACTOR_PROBE_RUN_ID",
                "HEPHAESTUS_EVALUATOR_V2_AUTHORIZED"):
        env.pop(key,None)
    env["HEPHAESTUS_CONTRAST_AUTHORIZED"]="SINGLE_D_CONTRAST_PROBE"
    env["HEPHAESTUS_CONTRAST_RUN_ID"]=runid
    env["HEPHAESTUS_REPO_SHA"]=sha
    env["HEPHAESTUS_BOOTSTRAP_PROGRESS_KEY"]=f"{PREFIX}/{runid}/evaluation/bootstrap/progress.json"
    req["name"]=f"hephaestus-D-counterfactuals-{runid}"[:180]
    return req

def guard(cfg,elapsed,price):
    if not isinstance(price,(int,float)) or not math.isfinite(price) or price<=0 or price>cfg["max_hourly_usd"]:
        raise ValueError("actual GPU rate unverified or too high")
    projected=price*(elapsed+cfg["teardown_reserve_seconds"])/3600
    if projected>=cfg["max_total_estimated_gpu_usd"]:raise TimeoutError("approved $1.80 budget guard")
    return projected

def execute(cfg,client,execution,sha,runid):
    authorize(cfg)
    if not runid.startswith("D-contrast-") or len(sha)!=40 or any(k in runid for k in ("/","..")):
        raise ValueError("execution source/run identity invalid")
    key=f"{PREFIX}/{runid}/evaluation/result.json"
    if infra.maybe_json(client,key) is not None:raise RuntimeError("run already complete")
    # Validate the complete regional volume and GPU request BEFORE using up
    # the one-shot S3 claim. No extra pod can be created after this attempt.
    body=pod_request(cfg,sha,runid,real_env=True)
    client.put_object(Bucket=infra.required("RUNPOD_NETWORK_VOLUME_ID"),
        Key=f"{PREFIX}/approvals/{cfg['approval_id']}/once.json",IfNoneMatch="*",
        Body=json.dumps(dict(sha=sha,runid=runid,protocol=PROTOCOL,gpu_type_id=cfg["gpu_type_id"]),sort_keys=True).encode())
    podid=None;start=time.monotonic()
    record=dict(status="started",repo_sha=sha,run_id=runid,approval_id=cfg["approval_id"])
    try:
        pod=infra.create_with_capacity_retries(execution,body,1,0)
        podid=str(pod["id"]);record["pod_id"]=podid
        while True:
            elapsed=time.monotonic()-start
            if elapsed>=cfg["launcher_hard_wall_seconds"]:raise TimeoutError("launcher hard wall")
            state=execution.get_pod(podid)
            rate=float(state.get("adjustedCostPerHr",state.get("costPerHr")))
            guard(cfg,elapsed,rate)
            record["rate_usd_per_hour"]=rate
            progress=infra.maybe_json(client,f"{PREFIX}/{runid}/evaluation/progress.json")
            if progress is not None and progress.get("stage")!="contrast_inference":
                raise ValueError("unexpected progress stage")
            bootstrap_record=infra.maybe_json(client,f"{PREFIX}/{runid}/evaluation/bootstrap/progress.json")
            if (not bootstrap_record and elapsed>=cfg["bootstrap_heartbeat_max_wait_seconds"]):
                raise TimeoutError("no S3-visible container bootstrap in 10 minutes")
            if bootstrap_record and bootstrap_record.get("stage")=="bootstrap_failed":
                raise RuntimeError("bootstrap failure, fail closed")
            terminal=infra.maybe_json(client,key)
            if terminal:
                if terminal.get("run_id")!=runid or terminal.get("repo_sha")!=sha or terminal.get("status")!="contrast_inference_complete":
                    raise ValueError("incomplete or non-matching terminal evidence")
                record.update(status="contrast_inference_complete",result=terminal)
                return record
            if str(state.get("desiredStatus","")).upper() in infra.TERMINAL:
                raise RuntimeError("pod stopped before experiment completed")
            time.sleep(cfg["poll_seconds"])
    finally:
        if podid:
            deleted=prior.delete_verified(execution,podid)
            record["teardown"]=deleted
            elapsed=time.monotonic()-start
            record["elapsed_with_teardown_seconds"]=elapsed
            if "rate_usd_per_hour" in record:
                record["estimated_gpu_usd"]=record["rate_usd_per_hour"]*elapsed/3600
            print("D_CONTRAST_POD_TEARDOWN "+json.dumps({k:record.get(k) for k in
                ("pod_id","run_id","estimated_gpu_usd","elapsed_with_teardown_seconds","teardown")},sort_keys=True),flush=True)
            if not deleted.get("verified_absent"):raise RuntimeError("GPU pod deletion not verified")

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--execute",action="store_true")
    args=ap.parse_args()
    cfg=json.loads(CONFIG.read_text())
    if not args.execute:
        req=pod_request(cfg,"0"*40,"D-contrast-dryrun",real_env=False)
        assert "scripts/run_evaluator_contrast_probe.py" in json.dumps(req)
        print("D_CONTRAST_DRYRUN "+json.dumps(dict(paid=False,
           protocol=manifest()["protocol"],unique_prompts=manifest()["unique_prompts"],
           max_usd=cfg["max_total_estimated_gpu_usd"],actual_allocation=False),sort_keys=True))
        raise SystemExit(0)
    authorize(cfg)
    # Requery the exact A40 1x Secure Cloud GPU in EU-SE-1 immediately
    # before the EU-CZ-1 S3 atomic claim and single provisioning request.
    quote=regional.inspect_response(regional.query())
    print("D_CONTRAST_PRICE_PREFLIGHT "+json.dumps(quote,sort_keys=True),flush=True)
    if (not quote["confident_capacity_signal"] or quote["gpu_type"]!=cfg["gpu_type_id"]
            or quote["quoted_usd_per_hour"]>cfg["max_hourly_usd"]):
        raise ValueError("new selected GPU not safely available at approved price")
    record=execute(cfg,prior.s3_client(),
       RunPodExecutionAdapter(RunPodConfig.from_env(),EnvironmentSecretsProvider()),
       infra.required("GITHUB_SHA"),infra.required("HEPHAESTUS_CONTRAST_RUN_ID"))
    print("D_CONTRAST_LAUNCH_RESULT "+json.dumps(record,sort_keys=True),flush=True)
