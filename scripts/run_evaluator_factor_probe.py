#!/usr/bin/env python3
"""Single approved, immutable inference-only D step-96 factor reporting probe."""
from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path

from evaluator_semantic_factor_probe_protocol import (
    D_CHECKPOINT_SHA256, D_RUN_ID, D_SOURCE_SHA, MAX_NEW_TOKENS,
    panel,plan_manifest,parse_factor_json,sha,
)
from interface_repair_v2_bootstrap import materialize_parent_cached
import run_interface_repair_v1 as runtime
import build_evaluator_semantic_repair_v2 as curriculum
from prepare_evaluator_semantic_repair_v2 import prepare

ROOT=Path(__file__).resolve().parents[1]
APPROVAL=ROOT/"configs/experiments/evaluator_factor_probe_D_approval.json"
PREFIX="hephaestus/scientific/v5/evaluator_factor_only_probe_v1"
D_RESULT="hephaestus/scientific/v5/evaluator_semantic_repair_v2/"+D_RUN_ID+"/training/result.json"


def authorize(config, run_id, repo_sha):
    approval=json.loads(APPROVAL.read_text())
    if approval.get("status")!="approved_for_exactly_one_inference_only_run":
        raise PermissionError("factor-only approval inactive")
    if config.get("approval_id")!=approval.get("approval_id") or approval.get("checkpoint_sha256")!=D_CHECKPOINT_SHA256:
        raise PermissionError("approval/checkpoint identity mismatch")
    if config.get("max_hourly_usd")!=2.49 or config.get("initial_estimated_ceiling_usd")!=0.75 or config.get("exceptional_hard_ceiling_usd")!=1.50:
        raise PermissionError("experiment spending guard mismatch")
    if os.getenv("HEPHAESTUS_FACTOR_PROBE_AUTHORIZED")!="ONE_APPROVED_FACTOR_INFERENCE_ONLY":
        raise PermissionError("no independent factor-probe authorization")
    if any(approval.get(x) for x in ("gpu_training_authorized","repeat_inference_authorized","certification_authorized","production_promotion_authorized","sealed_pack_use_authorized")):
        raise PermissionError("forbidden scientific authority")
    if not run_id.startswith("factor-only-D-") or len(repo_sha)!=40 or any(x in run_id for x in ("/","..")):
        raise PermissionError("bad immutable execution identity")
    return approval


class StopOnComplete:
    def __init__(self,tokenizer,starting_width,deadline):
        self.tokenizer=tokenizer
        self.starting_width=starting_width
        self.deadline=deadline
        self.complete=False
        self.deadline_hit=False
        self.last_check=0

    def __call__(self,input_ids,scores,**kwargs):
        del scores,kwargs
        if time.monotonic()>=self.deadline:
            self.deadline_hit=True
            return True
        tokens=input_ids[0][self.starting_width:]
        n=int(tokens.shape[0])
        if n>=7 and n-self.last_check>=3:
            self.last_check=n
            raw=self.tokenizer.decode(tokens,skip_special_tokens=True).strip()
            if parse_factor_json(raw) is not None:
                self.complete=True
                return True
        return False


def generate(model,tok,settings,prompt,seed,deadline):
    import torch
    from transformers import StoppingCriteriaList
    encoded=runtime.encode(tok,settings["roles"]["evaluator"]["load_kind"],prompt)
    width=encoded["input_ids"].shape[-1]
    stopper=StopOnComplete(tok,width,deadline)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    start=time.perf_counter()
    with torch.inference_mode():
        outputs=model.generate(
            **encoded,do_sample=False,max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,pad_token_id=tok.pad_token_id,
            eos_token_id=runtime.stop_ids(model,tok),
            stopping_criteria=StoppingCriteriaList([stopper]),
            return_dict_in_generate=True,
        )
    raw=tok.decode(outputs.sequences[0][width:],skip_special_tokens=True).strip()
    seconds=time.perf_counter()-start
    return {
        "raw_factors":raw, "valid_json":parse_factor_json(raw) is not None,
        "generated_tokens":int(outputs.sequences[0].shape[-1]-width),
        "prompt_tokens":int(width),"latency_seconds":round(seconds,4),
        "deadline_hit":stopper.deadline_hit,"early_stop_valid_json":stopper.complete,
    }


def run(config,client,run_id,repo_sha):
    authorize(config,run_id,repo_sha)
    cfg, pack, _, _ = prepare("D")
    source=runtime.load_stack if False else None  # no source stack loading
    source_result=curriculum.common.s3_json(client,D_RESULT)
    if source_result.get("repo_sha")!=D_SOURCE_SHA or source_result.get("status")!="training_complete":
        raise RuntimeError("D source training lineage or completion mismatch")
    artifacts=[x["artifact"] for x in source_result["checkpoint_artifacts"] if x["optimizer_step"]==96]
    if len(artifacts)!=1 or artifacts[0]["sha256"]!=D_CHECKPOINT_SHA256:
        raise RuntimeError("D step-96 adapter checkpoint identity mismatch")
    parent=json.loads((ROOT/cfg["parent_candidate_manifest"]).read_text())
    spec=curriculum.common.parent_spec(cfg,parent)
    if (spec["revision"]!=cfg["required_base_revision"] or
        source_result.get("parent_adapter_sha256")!=cfg["required_parent_adapter_sha256"]):
        raise RuntimeError("D source/base model revision drift")
    spec=dict(spec,adapter=artifacts[0])
    records=panel()
    manifest=plan_manifest(records)
    output_prefix=PREFIX+"/"+run_id+"/checkpoints/step-096/evaluation"
    root=Path("/opt/hephaestus-factor-probe")/run_id
    root.mkdir(parents=True,exist_ok=False)
    hard_wall=float(config["runner_hard_wall_seconds"])
    deadline=time.monotonic()+hard_wall
    base,adapter=materialize_parent_cached(client,spec,root/"model")
    model,tok,loaded=runtime.load_model("evaluator",cfg,spec,base,adapter)
    if loaded["gpu"]!=config["gpu_type_id"] or loaded["revision"]!=cfg["required_base_revision"]:
        raise RuntimeError("GPU/model identity differs from approved D probe")
    for p in model.parameters():p.requires_grad_(False)
    model.eval()
    if sum(p.numel() for p in model.parameters() if p.requires_grad)!=0:
        raise RuntimeError("frozen probe has trainable weights")
    # The original predictions are independently persisted, and *not regenerated*.
    meta={**manifest,"status":"factor_inference_started","run_id":run_id,"repo_sha":repo_sha,
          "model":loaded,"source_training_result_sha256":sha(json.dumps(source_result,sort_keys=True))}
    runtime.put_json(client,output_prefix+"/manifest.json",meta)
    rows=[]
    total=len(records)
    for i,r in enumerate(records,1):
        if time.monotonic()>=deadline:
            raise TimeoutError("factor probe runner hard wall before all 64 cases")
        generation=generate(model,tok,cfg,r["prompt"],int(cfg["model_protocol"]["seed"])+i,deadline)
        rows.append({k:v for k,v in r.items() if k in ("case_id","prompt_sha256")})
        rows[-1].update(checkpoint_sha256=D_CHECKPOINT_SHA256,**generation)
        if i%8==0:
            runtime.put_json(client,output_prefix+f"/samples/shard-{(i+7)//8:04d}.json",rows[-8:])
            runtime.put_json(client,output_prefix+"/progress.json",{
                "completed":i,"total":total,"valid_json":sum(x["valid_json"] for x in rows),
                "generated_tokens":sum(x["generated_tokens"] for x in rows),
                "total_generation_seconds":round(sum(x["latency_seconds"] for x in rows),4),
                "elapsed_seconds":round(hard_wall-(deadline-time.monotonic()),4),
                "stage":"factor_probe_inference"})
        if generation["deadline_hit"]:
            raise TimeoutError("factor probe generation exceeded approved hard wall")
    terminal=dict(status="factor_probe_complete",run_id=run_id,repo_sha=repo_sha,
        protocol=manifest["protocol"],checkpoint_sha256=D_CHECKPOINT_SHA256,
        completed=len(rows),valid_json=sum(x["valid_json"] for x in rows),
        inference_only=True,scientific_improvement_demonstrated=False,
        certified=False,promoted=False,s3_shards=8)
    runtime.put_json(client,output_prefix+"/result.json",terminal)
    return terminal


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    settings=json.loads((ROOT/"configs/experiments/evaluator_factor_probe_D_execution.json").read_text())
    if not args.execute:
        print(json.dumps(dict(status="dry_run",authorized_spend_used=False,checkpoint_sha256=D_CHECKPOINT_SHA256),sort_keys=True))
        return 0
    run_id=runtime.required("HEPHAESTUS_FACTOR_PROBE_RUN_ID")
    repo_sha=runtime.required("HEPHAESTUS_REPO_SHA")
    authorize(settings,run_id,repo_sha)
    client=runtime.s3_client()
    result_key=PREFIX+"/"+run_id+"/checkpoints/step-096/evaluation/result.json"
    try:
        result=run(settings,client,run_id,repo_sha)
    except Exception as e:
        # Fail closed but preserve already persisted sample shards / progress.
        runtime.put_json(client,result_key,dict(status="failed",run_id=run_id,repo_sha=repo_sha,
            error_type=type(e).__name__,error=str(e)[:400],inference_only=True))
        raise
    print("EVALUATOR_FACTOR_ONLY_RESULT "+json.dumps(result,sort_keys=True),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
