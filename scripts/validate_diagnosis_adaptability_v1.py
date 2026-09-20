#!/usr/bin/env python3
from __future__ import annotations
import hashlib,importlib.util,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/experiments/hephaestus_diagnosis_adaptability_v1.json"
def need(x,msg):
    if not x: raise RuntimeError(msg)
def canonical_sha(obj):
    return hashlib.sha256((json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode()).hexdigest()
def main():
    c=json.loads(CFG.read_text())
    need(c["protocol_id"]=="hephaestus_diagnosis_adaptability_v1","protocol drift")
    spec=importlib.util.spec_from_file_location("p",ROOT/c["pack"]["builder_path"]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);p=m.build_pack();m.validate(p)
    need(canonical_sha(p)==c["parent_zero_shot"]["pack_sha256"],"frozen pack hash mismatch")
    train=p["partitions"][c["pack"]["train_partition"]];held=p["partitions"][c["pack"]["held_out_partition"]]
    need(len(train)==72 and len(held)==48,"partition cardinality drift")
    need(not ({x["case_id"] for x in train}&{x["case_id"] for x in held}),"train/held-out leakage")
    need([x["candidate_id"] for x in c["candidates"]]==["granite42-3b","ministral3-3b-reasoning"],"finalist drift")
    tr=c["training"]
    need(tr["lora_rank"]==8 and tr["lora_alpha"]==16 and tr["lora_dropout"]==0,"LoRA hyperparameter drift")
    need(tr["optimizer_steps"]==32 and tr["gradient_accumulation_steps"]==4 and tr["micro_batch_size"]==1,"step budget drift")
    need(abs(float(tr["learning_rate"])-5e-5)<1e-12 and tr["seed"]==11,"optimizer/seed drift")
    expected_slots=tr["optimizer_steps"]*tr["gradient_accumulation_steps"]*tr["micro_batch_size"]*tr["max_sequence_length"]
    need(expected_slots==tr["padded_training_token_slots"]==98304,"training-token slot budget drift")
    for cand in c["candidates"]:
        re.compile(cand["lora_target_regex"])
        need("q_proj|k_proj|v_proj|o_proj" in cand["lora_target_regex"],"projection surface drift")
    mini=next(x for x in c["candidates"] if x["candidate_id"]=="ministral3-3b-reasoning")
    need("language_model" in mini["lora_target_regex"],"Ministral LoRA must target language model only")
    need("model.vision_tower." in mini["exclude_module_prefixes"],"Ministral vision exclusion missing")
    e=c["execution"]
    need(e["max_hourly_usd"]<=0.85 and e["max_estimated_total_usd"]<=0.95 and e["hard_wall_seconds"]<=4800,"cost bound drift")
    need(not any(x in " ".join(e["gpu_type_ids"]) for x in ["H100","H200","B200"]),"expensive GPU fallback forbidden")
    need(c["governance"]["paid_launch_allowed"] is True,"paid Stage-2 authorization missing")
    need(c["comparison"]["no_automatic_promotion"] is True,"automatic promotion forbidden")
    print(json.dumps({"status":"valid","train_cases":len(train),"held_out_cases":len(held),"padded_training_token_slots_per_candidate":expected_slots,"candidate_count":len(c["candidates"]),"max_estimated_total_usd":e["max_estimated_total_usd"]},sort_keys=True))
if __name__=="__main__": main()
