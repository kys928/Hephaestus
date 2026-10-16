#!/usr/bin/env python3
"""Exactly one approved RunPod factor-only D inference allocation.

No training, no checkpoint selection, no replay, no certification. Initial $0.75
ceiling, $1.50 hard contingency ONLY with measured saved progress/need.
"""
from __future__ import annotations
import copy,json,os,time
from pathlib import Path

import launch_evaluator_semantic_repair_v2 as previous
import launch_interface_repair_v2 as infrastructure
from evaluator_semantic_factor_probe_protocol import D_CHECKPOINT_SHA256
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig,RunPodExecutionAdapter
import run_interface_repair_v1 as storage

ROOT=Path(__file__).resolve().parents[1]
APPROVAL=ROOT/"configs/experiments/evaluator_factor_probe_D_approval.json"
CONFIG=ROOT/"configs/experiments/evaluator_factor_probe_D_execution.json"
PREFIX="hephaestus/scientific/v5/evaluator_factor_only_probe_v1"
SUFFIX="checkpoints/step-096/evaluation"


def validate_approval(cfg):
    approved=json.loads(APPROVAL.read_text())
    if (approved.get("status")!="approved_for_exactly_one_inference_only_run"
        or cfg["approval_id"]!=approved["approval_id"]
        or cfg["initial_estimated_ceiling_usd"]!=approved["initial_total_estimated_ceiling_usd"]
        or cfg["exceptional_hard_ceiling_usd"]!=approved["exceptional_hard_ceiling_usd"]
        or cfg["max_hourly_usd"]!=approved["max_hourly_usd"]
        or cfg["gpu_type_id"]!=approved["gpu_type_id"]
        or approved["checkpoint_sha256"]!=D_CHECKPOINT_SHA256
        or cfg["pod_count_max"]!=1 or cfg["allow_retraining"] or cfg["repeated_run_authorized"]):
        raise PermissionError("the approved inference-only boundaries do not match this launcher")
    if os.environ.get("HEPHAESTUS_FACTOR_PROBE_AUTHORIZED")!="ONE_APPROVED_FACTOR_INFERENCE_ONLY":
        raise PermissionError("explicit probe launch authorization missing")
    return approved


def pod_request(cfg,repo_sha,run_id,*,real_env):
    # Exact established D loader/dependency/network-volume bootstrap; replace
    # only the original D inference runner with an approved factor-only runner.
    established=json.loads((ROOT/"configs/experiments/hephaestus_evaluator_semantic_repair_v2.json").read_text())
    adapted=copy.deepcopy(established)
    adapted["execution"]["s3_prefix"]=PREFIX
    adapted["execution"]["max_estimated_total_usd_evaluate"]=cfg["initial_estimated_ceiling_usd"]
    body=previous.request(adapted,repo_sha,run_id,"evaluate",real_env=real_env,arm="D")
    old="scripts/run_evaluator_semantic_repair_v2.py --mode evaluate --step 96 --arm D"
    new="scripts/run_evaluator_factor_probe.py --execute"
    observations=[0]
    def walk(x):
        if isinstance(x,str):
            observations[0]+=x.count(old)
            return x.replace(old,new)
        if isinstance(x,list):return [walk(v) for v in x]
        if isinstance(x,dict):return {k:walk(v) for k,v in x.items()}
        return x
    body=walk(body)
    if observations[0]!=1 or old in json.dumps(body):
        raise ValueError("original D pod bootstrap shape changed; refuse allocation")
    if not isinstance(body.get("env"),dict):
        raise ValueError("unexpected pod environment")
    env=body["env"]
    env.pop("HEPHAESTUS_EVALUATOR_V2_AUTHORIZED",None)
    env["HEPHAESTUS_FACTOR_PROBE_AUTHORIZED"]="ONE_APPROVED_FACTOR_INFERENCE_ONLY"
    env["HEPHAESTUS_FACTOR_PROBE_RUN_ID"]=run_id
    env["HEPHAESTUS_REPO_SHA"]=repo_sha
    env["HEPHAESTUS_BOOTSTRAP_PROGRESS_KEY"]=f"{PREFIX}/{run_id}/{SUFFIX}/bootstrap/progress.json"
    body["name"]=f"hephaestus-factor-only-D-{run_id}"[:180]
    return body


def capacity_and_cost_guard(cfg,progress,elapsed,hourly,extended):
    import math
    if not math.isfinite(hourly) or hourly<=0 or hourly>cfg["max_hourly_usd"]:
        raise ValueError("invalid/over-ceiling GPU hourly price")
    initial=cfg["initial_estimated_ceiling_usd"]
    hard=cfg["exceptional_hard_ceiling_usd"]
    reserve=cfg["teardown_reserve_seconds"]
    estimated=hourly*elapsed/3600
    projected_finish=None
    completed=0
    if isinstance(progress,dict):
        completed=progress.get("completed",0)
        if type(completed) is not int or not 0<=completed<=64:
            raise ValueError("invalid progress evidence")
    # Only authorize the contingency if enough independent S3-persisted
    # generations exist and measured extrapolation exceeds $0.75.
    if not extended and completed>=cfg["minimum_progress_for_contingency"]:
        projected_finish=hourly*(elapsed*64/completed+reserve)/3600
        if initial < projected_finish <= hard and completed<64:
            extended=True
    ceiling=hard if extended else initial
    if hourly*(elapsed+reserve)/3600 >= ceiling:
        raise TimeoutError("factor-only pod would exceed bounded inference estimate")
    return dict(estimated_cost_usd=estimated,
                contingency_activated=extended,
                completed=completed,projected_full_cost_usd=projected_finish,
                active_ceiling_usd=ceiling)


def execute(cfg,client,execution,repo_sha,run_id):
    validate_approval(cfg)
    if not repo_sha or len(repo_sha)!=40 or not run_id.startswith("factor-only-D-") or any(x in run_id for x in ("/","..")):
        raise ValueError("unsafe immutable run identity")
    key=f"{PREFIX}/{run_id}/{SUFFIX}/result.json"
    if infrastructure.maybe_json(client,key) is not None:
        raise RuntimeError("terminal probe evidence already exists; no replay")
    # Atomic run ID claim, fail before pod allocation if S3 does not support it.
    claim_key=f"{PREFIX}/approvals/{cfg['approval_id']}/single-inference-claim.json"
    client.put_object(Bucket=infrastructure.required("RUNPOD_NETWORK_VOLUME_ID"),Key=claim_key,
        Body=json.dumps(dict(run_id=run_id,repo_sha=repo_sha,checkpoint_sha256=D_CHECKPOINT_SHA256),sort_keys=True).encode(),
        IfNoneMatch="*")
    pod=None;started=time.monotonic();extended=False
    record=dict(run_id=run_id,repo_sha=repo_sha,status="starting",approval_id=cfg["approval_id"])
    try:
        body=pod_request(cfg,repo_sha,run_id,real_env=True)
        # No capacity retries: user approved one pod, not repeated speculative launches.
        instance=infrastructure.create_with_capacity_retries(execution,body,1,0)
        pod=str(instance["id"])
        record["pod_id"]=pod
        while True:
            elapsed=time.monotonic()-started
            if elapsed>=cfg["launcher_hard_wall_seconds"]:
                raise TimeoutError("launcher hard wall exceeded")
            state=execution.get_pod(pod)
            price=state.get("adjustedCostPerHr",state.get("costPerHr"))
            if price is None:
                raise ValueError("no verified pod price: fail closed")
            rate=float(price)
            progress=infrastructure.maybe_json(client,f"{PREFIX}/{run_id}/{SUFFIX}/progress.json")
            if progress is not None and progress.get("stage")!="factor_probe_inference":
                raise ValueError("invalid S3 factor-only progress stage")
            guard=capacity_and_cost_guard(cfg,progress,elapsed,rate,extended)
            if guard["contingency_activated"] and not extended:
                print("EVALUATOR_FACTOR_PROBE_CONTINGENCY "+json.dumps({
                    "completed":guard["completed"],
                    "projected_full_cost_usd":guard["projected_full_cost_usd"],
                    "initial_ceiling_usd":cfg["initial_estimated_ceiling_usd"],
                    "hard_ceiling_usd":cfg["exceptional_hard_ceiling_usd"],
                    "reason":"measured generation progress requires strictly bounded continuation"},sort_keys=True),flush=True)
            extended=guard["contingency_activated"]
            record.update(guard,cost_per_hr=rate)
            bootstrap=infrastructure.maybe_json(client,f"{PREFIX}/{run_id}/{SUFFIX}/bootstrap/progress.json")
            if bootstrap and bootstrap.get("stage")=="bootstrap_failed":
                raise RuntimeError("model/bootstrap failure; do not invoke contingency")
            terminal=infrastructure.maybe_json(client,key)
            if terminal:
                if terminal.get("run_id")!=run_id or terminal.get("repo_sha")!=repo_sha:
                    raise ValueError("terminal probe identity mismatch")
                if terminal.get("status")!="factor_probe_complete":
                    raise RuntimeError("factor-only runner failed")
                record.update(status="factor_probe_complete",result=terminal)
                return record
            if str(state.get("desiredStatus","")).upper() in infrastructure.TERMINAL:
                raise RuntimeError("pod stopped before complete factor-only evidence")
            time.sleep(cfg["poll_seconds"])
    finally:
        if pod:
            teardown=previous.delete_verified(execution,pod)
            record["teardown"]=teardown
            elapsed=time.monotonic()-started
            record["elapsed_including_teardown_seconds"]=elapsed
            if "cost_per_hr" in record:
                record["estimated_cost_usd"]=record["cost_per_hr"]*elapsed/3600
            print("EVALUATOR_FACTOR_PROBE_TEARDOWN "+json.dumps({
                "pod_id":pod,"run_id":run_id,"elapsed_including_teardown_seconds":elapsed,
                "estimated_cost_usd":record.get("estimated_cost_usd"),**teardown},sort_keys=True),flush=True)
            if not teardown.get("verified_absent"):
                raise RuntimeError("factor-only pod deletion not verified")


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    cfg=json.loads(CONFIG.read_text())
    sha=os.environ.get("GITHUB_SHA","0"*40)
    runid=os.environ.get("HEPHAESTUS_FACTOR_PROBE_RUN_ID","factor-only-D-dryrun")
    # Dry-run is allowed without accessing credentials.
    if not args.execute:
        body=pod_request(cfg,sha,runid,real_env=False)
        print("EVALUATOR_FACTOR_PROBE_DRY_RUN "+json.dumps({
            "status":"dry_run","approval_enforced_on_execute":True,
            "gpu_request_name":body["name"],
            "no_pod_allocated":True,
            "initial_usd":cfg["initial_estimated_ceiling_usd"],"hard_usd":cfg["exceptional_hard_ceiling_usd"]},sort_keys=True))
        return 0
    validate_approval(cfg)
    # Reuse original guarded catalog preflight without allocating a GPU.
    standard=json.loads((ROOT/"configs/experiments/hephaestus_evaluator_semantic_repair_v2.json").read_text())
    print("EVALUATOR_FACTOR_PROBE_PRICE_PREFLIGHT "+json.dumps(previous.quote_gpu(standard),sort_keys=True),flush=True)
    record=execute(cfg,previous.s3_client(),
        RunPodExecutionAdapter(RunPodConfig.from_env(),EnvironmentSecretsProvider()),
        infrastructure.required("GITHUB_SHA"),
        infrastructure.required("HEPHAESTUS_FACTOR_PROBE_RUN_ID"))
    print("EVALUATOR_FACTOR_PROBE_LAUNCH "+json.dumps(record,sort_keys=True),flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
