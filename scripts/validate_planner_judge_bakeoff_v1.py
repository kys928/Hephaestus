#!/usr/bin/env python3
from __future__ import annotations
import importlib.util,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CFG=ROOT/"configs/experiments/hephaestus_planner_judge_bakeoff_v1.json"
def need(v,msg):
    if not v: raise RuntimeError(msg)
def load_pack(c):
    p=ROOT/c["pack"]["builder_path"]
    s=importlib.util.spec_from_file_location("pjpack",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    pack=m.build_pack();m.validate(pack);return m,pack
def main():
    c=json.loads(CFG.read_text())
    need(c["protocol_id"]=="hephaestus_planner_judge_bakeoff_v1","protocol drift")
    m,p=load_pack(c)
    observed=m.canonical_sha256(p)
    expected=c["pack"]["canonical_sha256"]
    need(expected=="PENDING_STATIC_FREEZE" or expected==observed,"pack hash mismatch")
    need(len(c["role_rosters"]["planner"])==4 and len(c["role_rosters"]["judge"])==4,"role roster size drift")
    registry={x["candidate_id"]:x for x in c["candidate_registry"]}
    need(set(c["role_rosters"]["planner"])|set(c["role_rosters"]["judge"])==set(registry),"registry/roster mismatch")
    for cid,row in registry.items():
        need(row["parameter_count"]<10_000_000_000,f"{cid} exceeds sub-10B policy")
        need(re.fullmatch(r"[0-9a-f]{40}",row["revision"]) is not None,f"{cid} revision not immutable")
        re.compile(row["lora_target_regex"])
        need(row["observed_hf_license"] in {"apache-2.0","mit","other"},f"{cid} unexpected license metadata")
    tr=c["training"]
    need(tr["lora_rank"]==8 and tr["lora_alpha"]==16 and tr["optimizer_steps"]==32,"LoRA geometry drift")
    need(abs(tr["learning_rate"]-5e-5)<1e-12 and tr["seed"]==11,"optimizer/seed drift")
    slots=tr["optimizer_steps"]*tr["gradient_accumulation_steps"]*tr["micro_batch_size"]*tr["max_sequence_length"]
    need(slots==tr["padded_training_token_slots_per_pair"]==131072,"training token-slot budget drift")
    marker=json.loads((ROOT/"configs/experiments/planner_judge_bakeoff_v1.launch.json").read_text())
    paid=bool(c["governance"]["paid_launch_allowed"])
    authorized=bool(marker.get("authorized"))
    need(paid==authorized,"config paid_launch_allowed and launch marker authorized must agree")
    if authorized:
        need(marker.get("authorization_source")=="user_directive","authorized launch must record user directive")
        need(bool(str(marker.get("authorization_text") or "").strip()),"authorized launch must preserve authorization text")
    print(json.dumps({"status":"valid","pack_sha256":observed,"planner_cases":{k:len(v) for k,v in p["partitions"]["planner"].items()},"judge_cases":{k:len(v) for k,v in p["partitions"]["judge"].items()},"candidate_role_pairs":8,"paid_launch_allowed":paid,"authorized":authorized},sort_keys=True))
if __name__=="__main__": main()
