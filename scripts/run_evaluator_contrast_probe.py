#!/usr/bin/env python3
"""Single approved D step-96 matched-contrast INFERENCE ONLY runner."""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
import run_interface_repair_v1 as runtime
import build_evaluator_semantic_repair_v2 as curriculum
from prepare_evaluator_semantic_repair_v2 import prepare
from interface_repair_v2_bootstrap import materialize_parent_cached
from evaluator_contrast_inference_protocol import (
    panel,manifest,assess,parse_decision,PREFIX,PROTOCOL,
    D_SOURCE_RUN,D_SOURCE_SHA,
)
from evaluator_semantic_factor_probe_protocol import D_CHECKPOINT_SHA256

ROOT=Path(__file__).resolve().parents[1]
APPROVAL=ROOT/"configs/experiments/evaluator_contrast_probe_approval.json"
CONFIG=ROOT/"configs/experiments/evaluator_contrast_probe_execution.json"
D_RESULT="hephaestus/scientific/v5/evaluator_semantic_repair_v2/"+D_SOURCE_RUN+"/training/result.json"

def check(cfg,runid,repo_sha):
    a=json.loads(APPROVAL.read_text())
    if a.get("status")!="authorized_once" or cfg["approval_id"]!=a["approval_id"]:
        raise PermissionError("new contrast-only authorization absent")
    if (a["max_total_estimated_gpu_usd"]!=cfg["max_total_estimated_gpu_usd"] or
        a["max_gpu_hourly_usd"]!=cfg["max_hourly_usd"] or
        a["source_D_step96_sha256"]!=D_CHECKPOINT_SHA256 or
        cfg["max_total_estimated_gpu_usd"]!=0.75):
        raise PermissionError("scientific model or spending approval drift")
    if os.getenv("HEPHAESTUS_CONTRAST_AUTHORIZED")!="SINGLE_D_CONTRAST_PROBE":
        raise PermissionError("contrast-only environment token missing")
    if (not runid.startswith("D-contrast-") or "/" in runid or ".." in runid or
        len(repo_sha)!=40 or any(a.get(k) for k in ("model_retraining_authorized","certification_authorized","contingency_authorized"))):
        raise PermissionError("invalid execution identity or scope")
    return a

def generate(model,tok,runtime_cfg,text,deadline,max_tokens):
    import torch
    from transformers import StoppingCriteriaList
    class Deadline:
        hit=False
        def __call__(self,input_ids,scores,**kwargs):
            del input_ids,scores,kwargs
            self.hit=time.monotonic()>=deadline
            return self.hit
    enc=runtime.encode(tok,runtime_cfg["load_kind"],text)
    width=enc["input_ids"].shape[-1]
    stop=Deadline()
    clock=time.perf_counter()
    with torch.inference_mode():
        out=model.generate(**enc,do_sample=False,max_new_tokens=max_tokens,
            use_cache=True,pad_token_id=tok.pad_token_id,
            eos_token_id=runtime.stop_ids(model,tok),
            stopping_criteria=StoppingCriteriaList([stop]),return_dict_in_generate=True)
    raw=tok.decode(out.sequences[0][width:],skip_special_tokens=True).strip()
    parsed=parse_decision(raw)
    return dict(raw=raw,generated_tokens=int(out.sequences[0].shape[-1]-width),
        latency_seconds=round(time.perf_counter()-clock,4),
        deadline_hit=stop.hit,
        strict_state=parsed["strict"],exploratory_state=parsed["exploratory"],
        response_mode=parsed["mode"])

def run(cfg,runid,repo_sha,client):
    check(cfg,runid,repo_sha)
    selected,pairs=panel()
    m=manifest()
    assert m["paired_challenges"]==60 and m["unique_prompts"]==len(selected)
    source=curriculum.common.s3_json(client,D_RESULT)
    if source.get("status")!="training_complete" or source.get("repo_sha")!=D_SOURCE_SHA:
        raise ValueError("saved D source training was not completed")
    adapters=[x["artifact"] for x in source["checkpoint_artifacts"] if x["optimizer_step"]==96]
    if len(adapters)!=1 or adapters[0]["sha256"]!=D_CHECKPOINT_SHA256:
        raise ValueError("saved D step96 SHA mismatch")
    oldcfg,_,_,_=prepare("D")
    parent=json.loads((ROOT/oldcfg["parent_candidate_manifest"]).read_text())
    spec=curriculum.common.parent_spec(oldcfg,parent)
    if spec["revision"]!=oldcfg["required_base_revision"] or source["parent_adapter_sha256"]!=oldcfg["required_parent_adapter_sha256"]:
        raise ValueError("base or parent adapter drift")
    spec=dict(spec,adapter=adapters[0])
    output=PREFIX+"/"+runid+"/evaluation"
    root=Path("/opt/hephaestus-contrast-probe")/runid
    root.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+cfg["runner_hard_wall_seconds"]
    base,adapter=materialize_parent_cached(client,spec,root/"model")
    model,tok,info=runtime.load_model("evaluator",oldcfg,spec,base,adapter)
    if info["gpu"]!=cfg["gpu_type_id"] or info["revision"]!=oldcfg["required_base_revision"]:
        raise ValueError("GPU or base revision mismatch")
    for p in model.parameters():p.requires_grad_(False)
    model.eval()
    if any(p.requires_grad for p in model.parameters()):raise ValueError("trainable parameter on inference-only run")
    runtime.put_json(client,output+"/manifest.json",dict(**m,repo_sha=repo_sha,run_id=runid,
         source_training_sha256=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),
         model_info=info))
    rows=[]
    for i,item in enumerate(selected,1):
        if time.monotonic()>=deadline:raise TimeoutError("contrast model inference wall reached")
        gen=generate(model,tok,oldcfg["roles"]["evaluator"],item["prompt"],deadline,cfg["generation_max_new_tokens"])
        rows.append(dict(prompt_sha256=item["prompt_sha256"],checkpoint_sha256=D_CHECKPOINT_SHA256,
                         **gen))
        if i%cfg["shard_size"]==0 or i==len(selected):
            key=output+f"/samples/shard-{(i+cfg['shard_size']-1)//cfg['shard_size']:04d}.json"
            runtime.put_json(client,key,rows[-cfg["shard_size"]:])
            runtime.put_json(client,output+"/progress.json",dict(stage="contrast_inference",
                completed=i,total=len(selected),
                total_generation_seconds=round(sum(r["latency_seconds"] for r in rows),4),
                valid_exploratory=sum(r["exploratory_state"] is not None for r in rows)))
        if gen["deadline_hit"]:raise TimeoutError("generation exceeded hard wall")
    report=assess(selected,pairs,rows)
    result=dict(status="contrast_inference_complete",repo_sha=repo_sha,run_id=runid,
        checkpoint_sha256=D_CHECKPOINT_SHA256,paired_challenges=60,
        unique_prompts=len(selected),analysis=report,
        training_performed=False,certification_performed=False,
        production_promotion_performed=False)
    runtime.put_json(client,output+"/result.json",result)
    print("EVALUATOR_CONTRAST_REMOTE_SUMMARY "+json.dumps({k:report[k] for k in
        ("unique_prompt_count","pair_count","strict_format_valid_unique",
        "exploratory_state_parseable_unique","both_ends_correct",
        "pair_transition_consistency")},sort_keys=True),flush=True)
    return result

if __name__=="__main__":
    cfg=json.loads(CONFIG.read_text())
    runid=runtime.required("HEPHAESTUS_CONTRAST_RUN_ID")
    repo_sha=runtime.required("HEPHAESTUS_REPO_SHA")
    check(cfg,runid,repo_sha)
    client=runtime.s3_client()
    key=PREFIX+"/"+runid+"/evaluation/result.json"
    try:
        result=run(cfg,runid,repo_sha,client)
    except Exception as e:
        runtime.put_json(client,key,dict(status="failed",run_id=runid,repo_sha=repo_sha,error_type=type(e).__name__,error=str(e)[:350],training_performed=False))
        raise
    print("EVALUATOR_CONTRAST_REMOTE_RESULT "+json.dumps({
        "status":result["status"],"run_id":runid,"unique_prompts":result["unique_prompts"],
        "paired_challenges":result["paired_challenges"]},sort_keys=True),flush=True)
