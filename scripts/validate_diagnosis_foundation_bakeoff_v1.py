#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/experiments/hephaestus_diagnosis_foundation_bakeoff_v1.json"
def need(x,msg):
    if not x: raise RuntimeError(msg)
def main():
    c=json.loads(CFG.read_text());need(c["protocol_id"]=="hephaestus_diagnosis_foundation_bakeoff_v1","protocol drift")
    spec=importlib.util.spec_from_file_location("p",ROOT/c["pack"]["builder_path"]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);p=m.build_pack();m.validate(p)
    need([x["candidate_id"] for x in c["candidates"]]==["granite42-3b","ministral3-3b-reasoning","causeanalyzer-7b-control","phi4-mini-reasoning-control"],"candidate order drift")
    need(sum(bool(x.get("adaptation_candidate")) for x in c["candidates"])==2,"exactly two foundations may advance to matched LoRA")
    need(c["candidates"][2]["base_model_id"]=="mistralai/Mistral-7B-Instruct-v0.1","CauseAnalyzer base drift")
    need(c["generation"]["max_new_tokens"]<=512,"default generation cap drift")
    vocab=c["contract_vocabulary"]
    expected_values={axis:{row["expected"][axis] for split in p["partitions"].values() for row in split} for axis in ("decision","action","primary_variable")}
    for axis in expected_values:
        need(expected_values[axis] <= set(vocab[axis]), f"contract vocabulary missing expected {axis} values")
    phi=next(x for x in c["candidates"] if x["candidate_id"]=="phi4-mini-reasoning-control")
    need(phi.get("max_new_tokens_override")==1536,"Phi compatibility budget drift")
    e=c["execution"];need(e["hard_wall_seconds"]<=2700 and e["max_estimated_total_usd"]<=0.65 and e["max_hourly_usd"]<=0.85,"cost bound drift")
    need(not any("H100" in x or "H200" in x or "B200" in x for x in e["gpu_type_ids"]),"expensive GPU fallback forbidden")
    need(c["stage2_plan"]["paid_launch_allowed_now"] is False,"stage2 may not launch before zero-shot evidence")
    print(json.dumps({"status":"valid","zero_shot_cases":len(p["partitions"]["zero_shot"]),"train_cases":len(p["partitions"]["micro_lora_train"]),"held_out_cases":len(p["partitions"]["held_out"]),"candidate_count":len(c["candidates"]),"max_estimated_total_usd":e["max_estimated_total_usd"]},sort_keys=True))
if __name__=="__main__":main()
