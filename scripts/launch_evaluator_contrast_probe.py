#!/usr/bin/env python3
"""One immutable, paid D contrast inference. Hard $0.75 ceiling, no retries."""
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

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/"configs/experiments/evaluator_contrast_probe_execution.json"
APPROVAL=ROOT/"configs/experiments/evaluator_contrast_probe_approval.json"

def authorize(cfg):
    a=json.loads(APPROVAL.read_text())
    # Approval from the previous timed-out pod is consumed. The prepared
    # proposal is intentionally disabled until a NEW explicit user command.
    if cfg.get("runpod_launch_enabled") is not True:
        raise PermissionError("new paid GPU launch not yet approved")
    if (a.get("status")!="authorized_once"
        or a.get("additional_allocations_authorized") is not True
        or a["approval_id"]!=cfg["approval_id"]
        or a["max_total_estimated_gpu_usd"]!=0.75
        or cfg["max_total_estimated_gpu_usd"]!=0.75
        or cfg["max_hourly_usd"]!=a["max_gpu_hourly_usd"]
        or cfg["max_hourly_usd"]>2.49
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
    env=req["env"]
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
    if projected>=cfg["max_total_estimated_gpu_usd"]:raise TimeoutError("$0.75 budget guard")
    return projected

def execute(cfg,client,execution,sha,runid):
    authorize(cfg)
    if not runid.startswith("D-contrast-") or len(sha)!=40 or any(k in runid for k in ("/","..")):
        raise ValueError("execution source/run identity invalid")
    key=f"{PREFIX}/{runid}/evaluation/result.json"
    if infra.maybe_json(client,key) is not None:raise RuntimeError("run already complete")
    # Atomic claim is unique across workflow run IDs for this one approved experiment.
    client.put_object(Bucket=infra.required("RUNPOD_NETWORK_VOLUME_ID"),
        Key=f"{PREFIX}/approvals/{cfg['approval_id']}/once.json",IfNoneMatch="*",
        Body=json.dumps(dict(sha=sha,runid=runid,protocol=PROTOCOL),sort_keys=True).encode())
    body=pod_request(cfg,sha,runid,real_env=True)
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
    old=json.loads((ROOT/"configs/experiments/hephaestus_evaluator_semantic_repair_v2.json").read_text())
    quote=prior.quote_gpu(old)
    print("D_CONTRAST_PRICE_PREFLIGHT "+json.dumps(quote,sort_keys=True),flush=True)
    if quote["catalog_floor_usd_per_hour"]>cfg["max_hourly_usd"]:raise ValueError("GPU catalog price too high")
    record=execute(cfg,prior.s3_client(),
       RunPodExecutionAdapter(RunPodConfig.from_env(),EnvironmentSecretsProvider()),
       infra.required("GITHUB_SHA"),infra.required("HEPHAESTUS_CONTRAST_RUN_ID"))
    print("D_CONTRAST_LAUNCH_RESULT "+json.dumps(record,sort_keys=True),flush=True)
