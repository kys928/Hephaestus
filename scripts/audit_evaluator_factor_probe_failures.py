#!/usr/bin/env python3
"""Read-only structural postmortem of 64 factor-only generations, no GPU."""
import json
from collections import Counter
from pathlib import Path
import run_interface_repair_v1 as storage
from evaluator_semantic_factor_probe_protocol import parse_factor_json, MAX_NEW_TOKENS

RUN="factor-only-D-38962439250"
PREFIX="hephaestus/scientific/v5/evaluator_factor_only_probe_v1/"+RUN+"/checkpoints/step-096/evaluation"
SIX={"gate_pass","complete","provenance_valid","effect","variance_stable","recheck_satisfied"}
SEVEN={"decision","action","primary_variable","confidence","evidence_refs","uncertainties","rationale"}

def main():
    client=storage.s3_client()
    stats=Counter();keys=Counter();lens=Counter();rows=[]
    for part in range(1,9):
        obj=client.get_object(Bucket=storage.bucket(),Key=PREFIX+f"/samples/shard-{part:04d}.json")
        try:shard=json.loads(obj["Body"].read())
        finally:obj["Body"].close()
        assert len(shard)==8
        rows+=shard
    assert len(rows)==64 and len({r["case_id"] for r in rows})==64
    for row in rows:
        raw=row["raw_factors"]
        txt=raw.strip()
        n=int(row["generated_tokens"])
        stats["strict_six_valid"]+=int(parse_factor_json(raw) is not None)
        stats["hit_token_limit"]+=int(n==MAX_NEW_TOKENS)
        stats["eos_or_early_stop"]+=int(n<MAX_NEW_TOKENS)
        stats["deadline_hit"]+=int(bool(row["deadline_hit"]))
        stats["stop_on_json"]+=int(bool(row["early_stop_valid_json"]))
        lens[f"{(n//16)*16}-{min(MAX_NEW_TOKENS,(n//16+1)*16-1)}"]+=1
        initial="object" if txt.startswith("{") else "angle_tag" if txt.startswith("<") else "code_fence" if txt.startswith(chr(96)) else "empty" if not txt else "other"
        stats["starts_"+initial]+=1
        stats["thinking_tag"]+=int("<think>" in txt or "</think>" in txt)
        try: parsed=json.loads(txt)
        except (TypeError,ValueError):parsed=None
        if isinstance(parsed,dict):
            stats["as_is_JSON_object"]+=1
            signature="|".join(sorted(parsed))
            keys[signature]+=1
            stats["seven_field_old_contract"]+=int(set(parsed)==SEVEN)
            stats["six_field_keys_but_invalid_values"]+=int(set(parsed)==SIX and parse_factor_json(txt) is None)
        embedded=storage.extract_json(txt)[0]
        if isinstance(embedded,dict):
            stats["embedded_JSON_object_detected"]+=1
            stats["embedded_six_field_keys"]+=int(set(embedded)==SIX)
            stats["embedded_seven_field_keys"]+=int(set(embedded)==SEVEN)
    result=dict(run_id=RUN,total=64,structural=dict(sorted(stats.items())),
        token_bins=dict(sorted(lens.items())),complete_key_signatures=dict(sorted(keys.items())),
        strict_factor_accuracy_assessable=False,additional_gpu_allocations=0,s3_writes=0,
        caveat="Exploratory JSON detection does not change the strict contract score; all data remain the original saved generations.")
    Path("evaluator-factor-probe-format-postmortem.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("EVALUATOR_FACTOR_FORMAT_POSTMORTEM "+json.dumps(result,sort_keys=True))
    return result

if __name__=="__main__":
    main()
