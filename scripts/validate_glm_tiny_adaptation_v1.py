#!/usr/bin/env python3
"""Validate GLM Tiny Adaptation V1 without launching paid compute."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/experiments/hephaestus_glm_tiny_adaptation_v1.json"
SCREEN = ROOT / "configs/experiments/hephaestus_glm_cheap_screen_v1.json"
RECOVERY = ROOT / "configs/experiments/hephaestus_diagnostic_scaling_recovery_v1.json"

def req(cond: bool, msg: str) -> None:
    if not cond: raise RuntimeError(msg)

def validate() -> dict[str, Any]:
    p=json.loads(PATH.read_text())
    s=json.loads(SCREEN.read_text())
    r=json.loads(RECOVERY.read_text())
    req(p["protocol_id"]=="hephaestus_glm_tiny_adaptation_v1","protocol drifted")
    scope=p["scientific_scope"]
    req(scope["screening_only"] is True and scope["adaptation_probe_only"] is True,"scope drifted")
    req(scope["certification_claim_allowed"] is False and scope["promotion_allowed"] is False and scope["lineage_mutation_allowed"] is False,"governance scope drifted")
    m=p["model"]
    req(m["model_id"]=="zai-org/GLM-4.7-Flash","model drifted")
    req(m["revision"]=="7dd20894a642a0aa287e9827cb1a1f7f91386b67","revision drifted")
    req(m["precision"]=="bfloat16" and m["quantization"] is False and m["cpu_or_disk_offload"] is False,"precision/offload drifted")
    c=next(x for x in r["candidates"] if x["model_id"]==m["model_id"])
    req(all(c[k]==m[k] for k in ("revision","model_type","loader")),"recovery identity mismatch")
    t=p["training"]
    req(t["role"]=="diagnosis","tiny adaptation must train diagnosis only")
    req(int(t["optimizer_steps"])==3,"tiny adaptation must remain three steps")
    req(int(t["gradient_accumulation_steps"])==4 and int(t["training_examples_consumed"])==12,"tiny adaptation sample geometry drifted")
    req(int(t["max_seq_length"])==1024,"sequence geometry drifted")
    for key in ("rank","alpha","dropout","bias","learning_rate","weight_decay","max_grad_norm","model_seed","shuffle_seed"):
        req(t[key]==r["training"][key],f"training setting differs from recovery: {key}")
    req(t["source_training_dataset_sha256"]==r["sources"]["source_training_dataset_sha256"],"dataset SHA drifted")
    req(t["reuse_full_recovery_target_surface"] is True,"target surface must be reused")
    post=p["post_adaptation_probes"]
    req(post["reuse_cheap_screen_probe_set_exactly"] is True and int(post["required_probe_count"])==len(s["probes"])==5,"post-probe set drifted")
    req(sum(int(x["max_new_tokens"]) for x in s["probes"])<=int(post["total_max_new_tokens_ceiling"]),"post-probe token ceiling exceeded")
    ex=p["execution"]
    req(0<int(ex["hard_wall_seconds"])<=1500,"wall cap drifted")
    req(0<float(ex["max_hourly_usd"])<=2.25,"hourly cap drifted")
    req(0<float(ex["max_estimated_total_usd"])<=0.95,"total cost cap drifted")
    req(ex["network_volume_attached"] is False and ex["persistent_model_cache_allowed"] is False,"ephemeral boundary drifted")
    req(len(ex["gpu_type_ids"])==2 and all("RTX PRO 6000 Blackwell" in x for x in ex["gpu_type_ids"]),"cheap GPU allowlist drifted")
    req(not any("H200" in x or "B200" in x for x in ex["gpu_type_ids"]),"expensive fallback forbidden")
    obs=p["observability"]
    req(obs["write_training_step_immediately"] is True and obs["write_heartbeat_after_every_training_step"] is True,"training observability required")
    req(obs["write_post_adaptation_sample_immediately"] is True and obs["write_heartbeat_after_every_sample"] is True,"sample observability required")
    gate=p["advance_gate"]
    req(gate["on_pass"]=="eligible_for_full_recovery_consideration_only","advance semantics drifted")
    req(gate["on_nonpass"]=="do_not_start_full_recovery","failure semantics drifted")
    req(p["governance"]["paid_launch_allowed_now"] is False,"paid tiny adaptation must remain disabled now")
    return {"status":"valid","protocol_id":p["protocol_id"],"optimizer_steps":3,"training_examples":12,"post_probe_count":5,"post_max_new_tokens":sum(int(x["max_new_tokens"]) for x in s["probes"]),"gpu_type_ids":ex["gpu_type_ids"],"hard_wall_seconds":ex["hard_wall_seconds"],"max_estimated_total_usd":ex["max_estimated_total_usd"],"paid_launch_allowed_now":False}

def main()->int:
    print(json.dumps(validate(),indent=2,sort_keys=True)); return 0
if __name__=="__main__": raise SystemExit(main())