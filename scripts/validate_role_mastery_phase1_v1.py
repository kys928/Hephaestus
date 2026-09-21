#!/usr/bin/env python3
from __future__ import annotations
import json,re
from pathlib import Path
import role_mastery_common_v1 as common
ROOT=Path(__file__).resolve().parents[1]
def need(x,msg):
 if not x: raise RuntimeError(msg)
def main():
 c=common.load_cfg();m,p=common.load_pack(c);sha=m.canonical_sha256(p)
 need(sha==c["pack"]["canonical_sha256"],"pack hash mismatch")
 need(set(c["candidate_registry"])==set(p["partitions"]),"candidate-role mismatch")
 for role,cand in c["candidate_registry"].items():
  need(re.fullmatch(r"[0-9a-f]{40}",cand["revision"]) is not None,f"{role} revision not immutable")
  need(int(cand["parameter_count"])<10_000_000_000,f"{role} exceeds sub-10B policy")
  need(cand["license"] in {"apache-2.0","mit"},f"{role} license outside Phase I allowlist")
 tr=c["training"];need(tr["lora_rank"]==16 and tr["lora_alpha"]==32,"LoRA config drift")
 need(tr["epochs"]==[1,2] and tr["max_sequence_length"]==1024,"training geometry drift")
 need(c["governance"]["promotion_allowed"] is False and c["governance"]["lineage_mutation_allowed"] is False,"governance drift")
 print(json.dumps({"status":"valid","pack_sha256":sha,"roles":sorted(c["candidate_registry"]),"paid_launch_allowed":c["governance"]["paid_launch_allowed"]},sort_keys=True))
if __name__=="__main__":main()
