#!/usr/bin/env python3
"""Read-only persisted V1/C/D semantic-factor audit; CPU and S3 GET only.

No pod allocation, inference, S3 mutation, dataset modification, certification
or production action. All per-case outputs remain on S3; only aggregate findings
and object-content hashes are emitted to a GitHub Actions artifact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import build_evaluator_semantic_invariance_repair_v1 as frozen
import run_interface_repair_v1 as storage
from evaluator_semantic_v2_screen import screen_cases, screen_hash
from hephaestus.policy.evaluator_semantic_oracle import scientific_state
from evaluator_semantic_factor_attribution import (
    analyze_observed, paired_comparison, reference_factors,
    decode_frozen_development_evidence, FIELDS,
)

FROZEN_DEVELOPMENT_SHA = "ae8b2bd702d9b0751f9b1b04958d5a4559188971fcaf1dd818be4da233cd6912"
D_SCREEN_SHA = "98c3346fde66cc92d355bae019de811cb9c7614746073d1e4f93e6a069dbe11c"
V1_ROOT = "hephaestus/scientific/v4/evaluator_semantic_invariance_repair/evaluator-semantic-repair-37005231044"
C_ROOT = "hephaestus/scientific/v5/evaluator_semantic_repair_v2/evaluator-semantic-v2-C-38935682810"
D_ROOT = "hephaestus/scientific/v5/evaluator_semantic_repair_v2/evaluator-semantic-v2-D-38945578442"
BASE_ROOT = "hephaestus/scientific/v4/evaluator_development_baseline/evaluator-v3-1-dev-baseline-36998587411"


def get_object(client, key, observations):
    obj = client.get_object(Bucket=storage.bucket(), Key=key)
    try:
        payload = obj["Body"].read()
    finally:
        obj["Body"].close()
    observations.append({"key":key, "bytes":len(payload), "sha256":hashlib.sha256(payload).hexdigest()})
    return payload.decode("utf-8")


def read_json(client,key,observations):
    return json.loads(get_object(client,key,observations))


def read_jsonl(client,key,observations):
    return [json.loads(line) for line in get_object(client,key,observations).splitlines() if line.strip()]


def main():
    pack = frozen.build_pack()
    frozen.validate(pack)
    cases = list(pack["partitions"]["evaluator"]["development_diagnostic"])
    if len(cases)!=256 or len({x["case_id"] for x in cases})!=256:
        raise RuntimeError("frozen development diagnostic identity/cardinality drift")
    serialized=json.dumps(cases,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    devhash=hashlib.sha256(serialized).hexdigest()
    if devhash!=FROZEN_DEVELOPMENT_SHA:
        raise RuntimeError("frozen development diagnostic hash changed")
    screen = screen_cases(cases)
    if screen_hash(screen)!=D_SCREEN_SHA:
        raise RuntimeError("predeclared D screen selection changed")
    for case in cases:
        labeled = reference_factors(case)
        decoded = decode_frozen_development_evidence(case)
        if decoded != labeled or scientific_state(decoded) != case["expected"]["decision"]:
            raise RuntimeError("independent frozen evidence text does not recover expected factor truth")
    client=storage.s3_client()
    client.head_bucket(Bucket=storage.bucket())
    reads=[]
    root_v1=V1_ROOT+"/checkpoints"
    v1_16=[r for i in range(1,9) for r in read_jsonl(client, f"{root_v1}/step-016/evaluation/samples/development_diagnostic/shard-{i:04d}.jsonl",reads)]
    v1_96=[r for i in range(1,9) for r in read_jsonl(client, f"{root_v1}/step-096/evaluation/samples/development_diagnostic/shard-{i:04d}.jsonl",reads)]
    baseline=[r for i in range(1,9) for r in read_jsonl(client, f"{BASE_ROOT}/samples/development_diagnostic/shard-{i:04d}.jsonl",reads)]
    c=[r for i in range(1,5) for r in read_json(client,f"{C_ROOT}/checkpoints/step-096/evaluation/samples/shard-{i:04d}.json",reads)]
    d=[r for i in range(1,3) for r in read_json(client,f"{D_ROOT}/checkpoints/step-096/evaluation/samples/shard-{i:04d}.json",reads)]
    if not (len(v1_16)==len(v1_96)==len(baseline)==256 and len(c)==128 and len(d)==64):
        raise RuntimeError("recorded sample counts do not match frozen evidence")
    expected_ids={r["case_id"] for r in cases}
    for label,rows in (("baseline",baseline),("V1_step16",v1_16),("V1_step96",v1_96)):
        if {r["case_id"] for r in rows}!=expected_ids:
            raise RuntimeError(f"{label} sample identity drift")
    if [r["case_id"] for r in c]!=[r["case_id"] for r in cases[:128]]:
        raise RuntimeError("C saved prefix identity/order drift")
    if [r["case_id"] for r in d]!=[r["case_id"] for r in screen]:
        raise RuntimeError("D fixed screen identity/order drift")
    for name,root,needed in (("D",D_ROOT,64),):
        recorded=read_json(client,root+"/checkpoints/step-096/evaluation/result.json",reads)
        if recorded.get("status")!="screen_rejected" or recorded.get("screen_evaluated_cases")!=needed:
            raise RuntimeError("D terminal scientific evidence mismatch")
    groups={
        "frozen_parent_full_256":baseline,
        "V1_step16_full_256":v1_16,
        "V1_step96_full_256":v1_96,
        "C_step96_ordered_prefix_128":c,
        "D_step96_balanced_screen_64":d,
    }
    results={name:analyze_observed(cases,rows,name) for name,rows in groups.items()}
    paired=[
        paired_comparison(v1_96,c,cases,"C_vs_V1_on_identical_128"),
        paired_comparison(v1_96,d,cases,"D_vs_V1_on_identical_balanced_64"),
        paired_comparison(v1_16,v1_96,cases,"V1_step96_vs_step16_all_256"),
    ]
    # This is a *different diagnostic contract* for future approved inference, not
    # something already observed in the current run. It must NOT be trained upon
    # nor treated as sealed independent confirmation.
    future_probe=dict(
        status="specification_only_no_inference",
        output_schema={"factors":{
            "gate_pass":"boolean","complete":"boolean","provenance_valid":"boolean",
            "effect":"positive|neutral|negative","variance_stable":"boolean",
            "recheck_satisfied":"boolean"},"decision":"one_existing_evaluator_state"},
        factor_names=list(FIELDS),
        discriminates_when_explicitly_reported=[
            "reported_factors_wrong_and_decision_agrees_with_reported_oracle: observable factor-report mismatch",
            "reported_factors_right_and_decision_disagrees_with_reported_oracle: observable precedence mismatch",
            "both_wrong_or_internally_inconsistent: compound error; no unique attribution"],
        limitations="Future explicitly requested factor reports change the output contract and do not reveal hidden chain of thought. CPU-only current analysis cannot measure model factor extraction.",
        full_development_pack_must_remain_frozen=True,
        new_paid_gpu_inference_authorized=False,
        no_training_data_generation_from_frozen_cases=True,
    )
    report={
        "version":"factor-attribution-observational-v1",
        "development_sha256":devhash,"screen_sha256":D_SCREEN_SHA,
        "frozen_reference_text_decodable_cases":len(cases),
        "frozen_reference_text_extraction_is_model_measurement":False,
        "runs":results,"same_case_comparisons":paired,
        "persisted_object_reads":reads,
        "future_explicit_factor_probe":future_probe,
        "identifiability_verdict":"The six scientific factor truths are available from frozen reference labels. Existing saved predictions expose only a final decision; extraction and precedence are not causally separable from them.",
        "paid_gpu_allocations":0,"s3_writes":0,"model_inference_performed":False,
        "scientific_gates_changed":False,"sealed_certification_performed":False,
    }
    out=Path("semantic-factor-observational-audit.json")
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    compact={name:{
        "count":record["observed_cases"],
        "correct":record["correct"],
        "predictions":record["predicted_state_counts"],
        "necessary_counterfactual_factors":record["factors_necessary_in_every_minimal_explanation"],
        "minimal_edits":record["minimal_counterfactual_factor_edits"],
        "state_accuracy_counts":record["state_counts"],
        } for name,record in results.items()}
    print("EVALUATOR_FACTOR_AUDIT_SUMMARY "+json.dumps({
        "observed":compact,"same_case_comparisons":paired,
        "identifiability_verdict":report["identifiability_verdict"],
        "reads":len(reads),"gpu_spend":0,
        },sort_keys=True))
    return report


if __name__=="__main__":
    main()
