#!/usr/bin/env python3
"""CPU-only, immutable S3 GET forensic attribution for completed D contrast inference.

Reads ONLY: completed manifest, result, seven persisted samples shards.
Writes ONLY: local GitHub Actions report files. No RunPod API, S3 writes,
training, inference, grading changes, promotion, or model modifications.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
from pathlib import Path

import run_interface_repair_v1 as s3
from evaluator_contrast_inference_protocol import (
    panel, manifest, inference_prompt, parse_decision, assess, PREFIX, sha
)
from evaluator_semantic_factor_probe_protocol import D_CHECKPOINT_SHA256


RUN_ID = "D-contrast-39993673479"
RUN_SHA = "41ca77598007c5620cd428d4f45e4d153f417ef6"
BASE = f"{PREFIX}/{RUN_ID}/evaluation"
OUTPUT = Path("D-contrast-56-forensic")
OUTPUT.mkdir(exist_ok=True)


def fetch(client, key):
    obj = client.get_object(Bucket=s3.bucket(), Key=key)
    try:
        raw = obj["Body"].read()
    finally:
        obj["Body"].close()
    return json.loads(raw), {"key": key, "bytes": len(raw),
                              "sha256": hashlib.sha256(raw).hexdigest()}


def independent_decision(observation):
    """Decode literal measurements and independently implement documented precedence."""
    assert observation["hard_gates"] in ("PASS", "FAIL")
    assert observation["required_records_present"] in ("40/40", "37/40")
    assert observation["artifact_attestation"] in ("VALID", "INVALID")
    assert observation["independent_confirmation"] in ("PASS", "NOT_PASSED")
    effect = re.fullmatch(r"([+-][0-9]+\.[0-9]+); threshold ±([0-9]+\.[0-9]+)",
                          observation["primary_quality_delta"])
    spread = re.fullmatch(r"([0-9]+\.[0-9]+); max ([0-9]+\.[0-9]+)",
                          observation["replicate_spread"])
    if not effect or not spread:
        raise ValueError("observation format has changed")
    delta = float(effect.group(1)); margin = float(effect.group(2))
    observed_spread = float(spread.group(1)); max_spread = float(spread.group(2))
    assert margin == 0.5 and max_spread == 0.1
    factors = dict(gate_pass=observation["hard_gates"] == "PASS",
                   complete=observation["required_records_present"] == "40/40",
                   provenance_valid=observation["artifact_attestation"] == "VALID",
                   effect="positive" if delta > margin else
                          "negative" if delta < -margin else "neutral",
                   variance_stable=observed_spread <= max_spread,
                   recheck_satisfied=observation["independent_confirmation"] == "PASS")
    if not factors["gate_pass"]: label = "scientific_rejection"
    elif not factors["complete"]: label = "incomplete_evidence"
    elif not factors["provenance_valid"]: label = "inconclusive"
    elif not factors["variance_stable"]: label = "recheck_required"
    elif factors["effect"] == "negative": label = "regressed"
    elif factors["effect"] == "neutral": label = "equivalent"
    elif factors["recheck_satisfied"]: label = "certification_ready"
    else: label = "improved"
    return label, factors


def count_dict(entries):
    return dict(sorted(collections.Counter(entries).items()))


def examples_where(rows, predicate, n=5):
    return [dict(id=r["id"], reference=r["reference"], predicted=r["predicted"],
                 factors=r["factors"], observation=r["observation"],
                 generated=r["raw"], usage=r["usage"],
                 prompt_sha256=r["prompt_sha256"]) for r in rows if predicate(r)][:n]


def read_training_contract():
    """Regenerate frozen training selection CPU-only; never load model or tokenizer."""
    from prepare_evaluator_semantic_repair_v2 import prepare
    from evaluator_semantic_v2_supervision import target_answer
    import run_interface_repair_v1 as runtime
    cfg, pack, selected, info = prepare("D")
    label_counts = count_dict(r["semantic_root"] for r in selected)
    answer = target_answer(selected[0])
    training_prompt = runtime.prompt(pack, selected[0], "evaluator")
    first_evidence = selected[0]["evidence"]
    # This describes the actual training distribution, not an intervention.
    return {
        "schedule_rows":len(selected),
        "schedule_counts_by_label":label_counts,
        "schedule_sha256":info["selected_schedule_sha256"],
        "training_answer_keys": sorted(json.loads(answer)),
        "training_prompt_chars":len(training_prompt),
        "training_prompt_excerpt":training_prompt[:1200],
        "training_evidence_example":first_evidence[:3],
        "training_prompt_has_raw_contrast_fields": {
            key:key in training_prompt for key in
            ("hard_gates", "required_records_present", "artifact_attestation",
             "primary_quality_delta", "replicate_spread", "independent_confirmation")},
    }


def main():
    unique,pairs = panel(); expected_manifest=manifest()
    assert len(unique)==56 and len(pairs)==60
    client=s3.s3_client()
    client.head_bucket(Bucket=s3.bucket())
    receipts=[]
    terminal,rec=fetch(client, BASE+"/result.json");receipts.append(rec)
    saved,rec=fetch(client, BASE+"/manifest.json");receipts.append(rec)
    if terminal.get("status")!="contrast_inference_complete" or terminal.get("run_id")!=RUN_ID:
        raise ValueError("not a completed, matching run")
    if terminal.get("repo_sha")!=RUN_SHA or saved.get("repo_sha")!=RUN_SHA:
        raise ValueError("pinned repository SHA changed")
    if terminal.get("checkpoint_sha256")!=D_CHECKPOINT_SHA256:
        raise ValueError("checkpoint identity mismatch")
    for k in ("protocol","D_checkpoint_sha256","prompts_sha256","pairs_sha256","paired_challenges"):
        if saved.get(k)!=expected_manifest[k]:
            raise ValueError("frozen scientific manifest drift: "+k)
    outputs=[]
    for i in range(1,8):
        shard,rec=fetch(client, BASE+f"/samples/shard-{i:04d}.json")
        receipts.append(rec);outputs.extend(shard)
    if len(outputs)!=len(unique) or len({r["prompt_sha256"] for r in outputs})!=56:
        raise ValueError("output shards incomplete or duplicated")
    if assess(unique,pairs,outputs)!=terminal["analysis"]:
        raise ValueError("independent original assessor mismatch")
    byoutput={r["prompt_sha256"]:r for r in outputs}
    usages=collections.Counter()
    for p in pairs:
        for side in ("baseline","modified"):
            usages[p[f"{side}_prompt_sha256"]]+=1
    details=[]
    for u in unique:
        row=byoutput[u["prompt_sha256"]]
        prompt=u["prompt"]
        assert sha(prompt)==u["prompt_sha256"]
        assert row["checkpoint_sha256"]==D_CHECKPOINT_SHA256
        observation=json.loads(prompt.split("OBSERVATIONS:\n",1)[1])
        independent,factors=independent_decision(observation)
        expected=inference_prompt(observation)
        assert expected==prompt
        parsed=parse_decision(row["raw"])
        assert parsed["strict"]==row["strict_state"] and parsed["exploratory"]==row["exploratory_state"]
        details.append(dict(
            id=u["prompt_sha256"][:12],prompt_sha256=u["prompt_sha256"],
            reference=independent,predicted=parsed["strict"],
            raw=row["raw"],factors=factors,observation=observation,
            prompt=prompt,usage=usages[u["prompt_sha256"]],
            generated_tokens=row.get("generated_tokens"),
            latency_seconds=row.get("latency_seconds")))
    detail_by_hash={r["prompt_sha256"]:r for r in details}
    if set(detail_by_hash)!=set(u["prompt_sha256"] for u in unique):
        raise ValueError("prompt identity mismatch")
    paired=[]
    disagreements=[]
    baseline_rows=[];modified_rows=[]
    for p in pairs:
        baseline=detail_by_hash[p["baseline_prompt_sha256"]]
        modified=detail_by_hash[p["modified_prompt_sha256"]]
        if baseline["reference"]!=p["baseline_state"] or modified["reference"]!=p["modified_state"]:
            disagreements.append(p["id"])
        baseline_rows.append(baseline);modified_rows.append(modified)
        paired.append(dict(id=p["id"],transition=p["transition"],
                           changed_factor=p["changed_factor"],
                           presentation=p["presentation"],
                           throughput_distractor=p["throughput_distractor"],
                           baseline_ref=baseline["reference"],modified_ref=modified["reference"],
                           baseline_pred=baseline["predicted"],modified_pred=modified["predicted"],
                           exact_both=baseline["predicted"]==baseline["reference"] and
                                      modified["predicted"]==modified["reference"],
                           change_expected=p["scientific_change_expected"],
                           predicted_change=baseline["predicted"]!=modified["predicted"]))
    if disagreements:
        raise ValueError("independent literal-field oracle disagrees with labels: "+str(disagreements[:5]))
    by_factor={}
    for f in sorted(set(p["changed_factor"] for p in paired)):
        gp=[p for p in paired if p["changed_factor"]==f]
        by_factor[f]=dict(total=len(gp),both_right=sum(x["exact_both"] for x in gp),
                          direction_consistent=sum(x["change_expected"]==x["predicted_change"] for x in gp))
    # Pair grouping across prompt render variants: every transition has 4 presentations.
    variant=[]
    for transition in sorted(set(p["transition"] for p in paired)):
        group=[p for p in paired if p["transition"]==transition]
        variant.append(dict(transition=transition,
                            baseline_predictions=count_dict(p["baseline_pred"] for p in group),
                            modified_predictions=count_dict(p["modified_pred"] for p in group)))
    baseline_unique={r["prompt_sha256"]:r for r in baseline_rows}
    baseline_ref_counts=count_dict(r["reference"] for r in baseline_rows)
    baseline_pred_counts=count_dict(r["predicted"] for r in baseline_rows)
    baseline_confusion=count_dict((r["reference"],r["predicted"]) for r in baseline_rows)
    unique_confusion=count_dict((r["reference"],r["predicted"]) for r in details)
    target_counts=count_dict(r["reference"] for r in details)
    pred_counts=count_dict(r["predicted"] for r in details)
    training=read_training_contract()
    report=dict(
        run_id=RUN_ID,repo_sha=RUN_SHA,checkpoint_sha256=D_CHECKPOINT_SHA256,
        CPU_only=True,S3_get_only=True,no_model_execution=True,
        output_count=56,pair_count=60,baseline_appearances=60,
        baseline_unique=len(baseline_unique),
        baseline_reference_counts=baseline_ref_counts,
        baseline_prediction_counts=baseline_pred_counts,
        baseline_confusion=[dict(reference=k[0],predicted=k[1],count=v) for k,v in sorted(baseline_confusion.items())],
        unique_reference_counts=target_counts,unique_predicted_counts=pred_counts,
        unique_confusion=[dict(reference=k[0],predicted=k[1],count=v) for k,v in sorted(unique_confusion.items())],
        strict_json_success=sum(r["predicted"] is not None for r in details),
        unique_correct=sum(r["reference"]==r["predicted"] for r in details),
        modified_correct_appearances=sum(r["reference"]==r["predicted"] for r in modified_rows),
        both_correct_pairs=sum(x["exact_both"] for x in paired),
        pair_direction_consistent=sum(x["change_expected"]==x["predicted_change"] for x in paired),
        by_changed_factor=by_factor,transition_variants=variant,
        independent_literal_oracle_agreement=60,
        training_contract=training,
        inference_contract={
            "keys":["decision"],"max_tokens":112,
            "sample_prompt_chars":len(unique[0]["prompt"]),
            "sample_prompt":unique[0]["prompt"],
            "reported_labels":list(target_counts),
        },
        examples=dict(
            baseline_equivalent_wrong=examples_where(baseline_rows,lambda x:x["reference"]=="equivalent" and x["predicted"]!="equivalent"),
            baseline_recheck_wrong=examples_where(baseline_rows,lambda x:x["reference"]=="recheck_required" and x["predicted"]!="recheck_required"),
            failed_gate=examples_where(details,lambda x:not x["factors"]["gate_pass"]),
            incorrect_but_stable=examples_where(details,lambda x:x["factors"]["variance_stable"] and x["reference"]!=x["predicted"]),
            unstable=examples_where(details,lambda x:not x["factors"]["variance_stable"])
        ),
        data_receipts=receipts,
        caveats=[
            "Predicted state is strict JSON parse of actual frozen checkpoint generations, not ground-truth latent intermediate factors.",
            "Labels independently reconstructed from exact numeric/policy observations; the prompt text omits full explicit precedence mapping.",
            "Identical prompts have duplicate pair appearances; unique-prompt and pair-weighted denominators differ.",
            "No causality can be assigned between prompt transfer and adapter learning without controlled matched inference.",
            "Training targets contain seven keys, while new contrast prompts demand a single key.",
        ],
    )
    (OUTPUT/"forensic-summary.json").write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
    (OUTPUT/"56-predictions-with-source-observations.json").write_text(json.dumps(details,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
    (OUTPUT/"60-matched-pairs.json").write_text(json.dumps(paired,indent=2,sort_keys=True,ensure_ascii=False)+"\n")
    # Avoid encoding tuple keys in log JSON.
    terse={k:v for k,v in report.items() if k in (
        "baseline_unique","baseline_reference_counts","baseline_prediction_counts",
        "unique_reference_counts","unique_predicted_counts","unique_correct",
        "modified_correct_appearances","strict_json_success",
        "both_correct_pairs","pair_direction_consistent",
        "independent_literal_oracle_agreement","by_changed_factor")}
    print("D_FORENSIC_56_SUMMARY "+json.dumps(terse,sort_keys=True),flush=True)
    print("D_FORENSIC_BASELINE_EXAMPLES "+json.dumps({
        k:[dict(reference=z["reference"],predicted=z["predicted"],
                observation=z["observation"],raw=z["generated"]) for z in report["examples"][k][:2]]
        for k in ("baseline_equivalent_wrong","baseline_recheck_wrong")
    },sort_keys=True),flush=True)
    print("D_FORENSIC_TRAIN_CONTRACT "+json.dumps({
        "schedule_rows":training["schedule_rows"],
        "schedule_counts_by_label":training["schedule_counts_by_label"],
        "training_answer_keys":training["training_answer_keys"],
        "training_prompt_has_raw_contrast_fields":training["training_prompt_has_raw_contrast_fields"]},
        sort_keys=True),flush=True)
    print("D_FORENSIC_OUTPUT_LOCAL_ARTIFACT "+str(OUTPUT),flush=True)

if __name__=="__main__":
    main()
