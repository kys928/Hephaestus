#!/usr/bin/env python3
"""Forensic report builder for completed post-training red-team V1 evidence."""
from __future__ import annotations
import json, os
from collections import defaultdict
from pathlib import Path
import sys

SCRIPTS=Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0,str(SCRIPTS))
import launch_first_bounded_scientific_training as storage

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/"configs/experiments/hephaestus_post_training_redteam_v1.json").read_text())

def read_json(client,key):
    return json.loads(storage.read_key(client,key))

def read_jsonl(client,key):
    raw=storage.read_key(client,key).decode("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]

def mean(xs):
    return sum(xs)/len(xs) if xs else 0.0

def failure_tags(row):
    s=row["score"];tags=[]
    if not s["behavior_exact"]: tags.append("behavior")
    if not s["strict_schema_compliant"]: tags.append("strict_schema")
    if s["components"]["evidence_precision"]<0.999999: tags.append("evidence_precision")
    if s["components"]["evidence_recall"]<0.999999: tags.append("evidence_recall")
    if s["components"]["confidence_calibration"]<0.999999: tags.append("confidence")
    if not s["uncertainty_obeyed"]: tags.append("uncertainty")
    if s["forbidden_claim_hits"]: tags.append("forbidden_claim")
    if s["hallucination_rate"]>0: tags.append("hallucinated_ref")
    if s["reasoning_or_wrapper_leak"]: tags.append("wrapper_or_reasoning_leak")
    return tags

def main():
    run_id=(os.environ.get("HEPHAESTUS_POST_TRAINING_REDTEAM_RUN_ID") or "").strip()
    if not run_id: raise RuntimeError("missing HEPHAESTUS_POST_TRAINING_REDTEAM_RUN_ID")
    client=storage.s3_client();base=f"{CFG['execution']['s3_prefix'].rstrip('/')}/{run_id}"
    aggregate=read_json(client,f"{base}/result.json")
    if aggregate.get("status")!="completed": raise RuntimeError("aggregate is not complete")
    report={"report_version":"hephaestus-post-training-redteam-forensics.v1","run_id":run_id,"roles":{}}
    global_regions=[]
    for role in CFG["candidates"]:
        result=read_json(client,f"{base}/roles/{role}/result.json")
        seeds=[int(x) for x in CFG["evaluation"]["adapted_seeds"]]
        rows_by_seed={seed:read_jsonl(client,f"{base}/roles/{role}/samples/adapted-seed-{seed}.jsonl") for seed in seeds}
        base_rows=read_jsonl(client,f"{base}/roles/{role}/samples/base-seed-{int(CFG['evaluation']['base_seeds'][0])}.jsonl")
        all_rows=[r for rows in rows_by_seed.values() for r in rows]
        by_case=defaultdict(list)
        for row in all_rows: by_case[row["case_id"]].append(row)
        case_forensics=[]
        for case_id,rows in by_case.items():
            q=mean([r["score"]["quality_100"] for r in rows])
            exact=mean([float(r["score"]["behavior_exact"]) for r in rows])
            tags=sorted({t for r in rows for t in failure_tags(r)})
            triplets=[]
            for r in rows:
                p=r["score"].get("parsed") or {}
                triplets.append((p.get("decision"),p.get("action"),p.get("primary_variable")))
            case_forensics.append({
              "case_id":case_id,"dimension":rows[0]["dimension"],"stressors":rows[0]["stressors"],
              "mean_quality_100":q,"behavior_exact_rate":exact,
              "seed_triplet_agreement":len(set(triplets))==1,"failure_tags":tags,
              "expected":rows[0]["expected"],
              "outputs":[{"seed":r["seed"],"raw_output":r["raw_output"],"score":r["score"]} for r in rows],
            })
        worst_cases=sorted(case_forensics,key=lambda x:(x["mean_quality_100"],x["behavior_exact_rate"],x["case_id"]))[:20]
        tag_counts=defaultdict(int)
        for c in case_forensics:
            for tag in c["failure_tags"]: tag_counts[tag]+=1
        dimension_failures=[]
        dims=result["adapted_aggregate"]["dimensions"]
        for name,data in dims.items():
            region={
              "role":role,"dimension":name,"quality_100":data["quality_100"],
              "exact_behavior_rate":data["exact_behavior_rate"],
              "strict_schema_rate":data["strict_schema_rate"],
              "evidence_precision":data["evidence_precision"],
              "evidence_recall":data["evidence_recall"],
              "hallucination_rate":data["hallucination_rate"],
            }
            dimension_failures.append(region);global_regions.append(region)
        comparison=result["base_vs_adapter"]
        report["roles"][role]={
          "model_id":result["model_id"],"revision":result["revision"],
          "original_certification_quality_100":CFG["candidates"][role]["certification_quality_100"],
          "redteam":result["adapted_aggregate"],
          "audit_flags":result["audit_flags"],
          "stochastic_consistency":result["stochastic_consistency"],
          "base_vs_adapter":comparison,
          "failure_tag_case_counts":dict(sorted(tag_counts.items())),
          "worst_cases":worst_cases,
          "weakest_dimensions":sorted(dimension_failures,key=lambda x:(x["quality_100"],x["dimension"]))[:15],
          "base_rows_summary":result["base_summary"],
        }
    report["weakest_regions_across_stack"]=sorted(global_regions,key=lambda x:(x["quality_100"],x["role"],x["dimension"]))[:40]
    raw=(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+"\n").encode()
    key=f"{base}/forensic_report.json"
    client.put_object(Bucket=storage.VOLUME_ID,Key=key,Body=raw)
    if storage.read_key(client,key)!=raw: raise RuntimeError("forensic report S3 readback mismatch")
    print("POST_TRAINING_REDTEAM_FORENSICS_JSON "+json.dumps({
      "run_id":run_id,"key":key,
      "weakest_regions":report["weakest_regions_across_stack"][:15],
      "roles":{r:{
        "quality_100":v["redteam"]["quality_100"],
        "audit_flags":v["audit_flags"],
        "failure_tag_case_counts":v["failure_tag_case_counts"],
      } for r,v in report["roles"].items()}
    },sort_keys=True))

if __name__=="__main__":
    main()
