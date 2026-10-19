#!/usr/bin/env python3
"""Read-only sign/polarity audit using the exact saved 64 D factor reports.

No retraining, new inference, S3 mutation, or certification. Reference factors
come from independently decoded original evidence text, not model output.
"""
from collections import Counter,defaultdict
import hashlib,json
from pathlib import Path

import run_interface_repair_v1 as storage
from prepare_evaluator_semantic_repair_v2 import prepare
from evaluator_semantic_factor_probe_protocol import panel,parse_factor_json,D_CHECKPOINT_SHA256
from assess_evaluator_factor_probe_fenced import normalize_fenced_singleton_array
from evaluator_semantic_factor_attribution import decode_frozen_development_evidence

PROBE_RUN="factor-only-D-38962439250"
PROBE_ROOT="hephaestus/scientific/v5/evaluator_factor_only_probe_v1/"+PROBE_RUN+"/checkpoints/step-096/evaluation"
D_ROOT="hephaestus/scientific/v5/evaluator_semantic_repair_v2/evaluator-semantic-v2-D-38945578442/checkpoints/step-096/evaluation"
BLOCKED=("scientific_rejection","incomplete_evidence","inconclusive","recheck_required")

def read(client,key,receipts):
    rsp=client.get_object(Bucket=storage.bucket(),Key=key)
    try:blob=rsp["Body"].read()
    finally:rsp["Body"].close()
    receipts.append(dict(key=key,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
    return json.loads(blob)


def census_D_training():
    cfg,pack,selected,manifest=prepare("D")
    assert len(selected)==768 and manifest["arm"]=="D"
    family=Counter()
    interaction=Counter()
    secondary=Counter()
    expected=Counter()
    for row in selected:
        f=row.get("latent_factors")
        if not isinstance(f,dict):
            raise RuntimeError("D training factor metadata absent")
        state=row["expected"]["decision"]
        if row["semantic_root"]!=state:
            raise RuntimeError("D training row label mismatch")
        current=f["effect"]
        family[(row["curriculum_family"],current)]+=1
        expected[(state,current)]+=1
        has_secondary=any("secondary throughput delta=" in e["fact"].lower() for e in row["evidence"])
        secondary[(row["curriculum_family"],current,has_secondary)]+=1
        if has_secondary:
            # A negative secondary delta is expected *only for positive primary*.
            quality=[e["fact"] for e in row["evidence"] if "Primary quality delta=" in e["fact"]]
            if len(quality)!=1:
                raise ValueError("D training primary quality evidence shape changed")
            import re
            value=re.search(r"Primary quality delta=(-?\d+\.\d+)",quality[0])
            if value is None:raise ValueError("cannot parse D training primary delta")
            q=float(value.group(1))
            if (q>0)!=(current=="positive") and current!="neutral":
                raise ValueError("scientific polarity mismatch in D training generator")
            if (q<0)!=(current=="negative") and current!="neutral":
                raise ValueError("scientific sign conflict in D training generator")
            if current=="positive" and "secondary throughput delta=-" not in quality[0]:
                raise ValueError("negative throughput distractor missing")
    return {
        "selected_training_examples":len(selected),
        "scheduled_family_effect":{f"{k[0]} / {k[1]}":v for k,v in sorted(family.items())},
        "scheduled_state_effect":{f"{k[0]} / {k[1]}":v for k,v in sorted(expected.items())},
        "secondary_throughput_interactions":{f"{k[0]} / {k[1]} / secondary={k[2]}":v for k,v in sorted(secondary.items())},
        "structural_hypothesis":"Secondary throughput sign is deliberately opposite primary quality in D factorial examples; no causal training attribution without controlled model comparisons.",
    }


def audit():
    records=panel()
    reference={r["case_id"]:r for r in records}
    client=storage.s3_client()
    client.head_bucket(Bucket=storage.bucket())
    receipts=[]
    terminal=read(client,PROBE_ROOT+"/result.json",receipts)
    if terminal.get("status")!="factor_probe_complete" or terminal.get("completed")!=64:
        raise ValueError("factor-only probe not terminal")
    samples=[x for i in range(1,9) for x in read(client,PROBE_ROOT+f"/samples/shard-{i:04d}.json",receipts)]
    saved=[x for i in range(1,3) for x in read(client,D_ROOT+f"/samples/shard-{i:04d}.json",receipts)]
    indexed_decisions={x["case_id"]:x for x in saved}
    if len(samples)!=64 or len(indexed_decisions)!=64 or {r["case_id"] for r in samples}!=set(reference):
        raise ValueError("saved D sample identity mismatch")
    per_state=defaultdict(Counter)
    masked=defaultdict(Counter)
    by_advice=defaultdict(Counter)
    confusion=Counter()
    crossings=Counter()
    for entry in samples:
        ref=reference[entry["case_id"]]
        if entry["checkpoint_sha256"]!=D_CHECKPOINT_SHA256 or entry["prompt_sha256"]!=ref["prompt_sha256"]:
            raise ValueError("probe identity mismatch")
        text=normalize_fenced_singleton_array(entry["raw_factors"])
        fact=parse_factor_json(text)
        if fact is None:
            raise ValueError("saved output not exact-wrapped six-factor object")
        gold=ref["reference_factors"]
        state=ref["target_state"]
        primary_masked=state in BLOCKED
        expected_effect=gold.effect
        observed_effect=fact.effect
        right=(expected_effect==observed_effect)
        per_state[state][expected_effect+" -> "+observed_effect]+=1
        per_state[state]["total"]+=1
        per_state[state]["effect_correct"]+=int(right)
        group="masked" if primary_masked else "effect_decisive"
        masked[group]["total"]+=1
        masked[group]["effect_correct"]+=int(right)
        by_advice["advisory" if ref["has_advisory"] else "direct"]["total"]+=1
        by_advice["advisory" if ref["has_advisory"] else "direct"]["effect_correct"]+=int(right)
        confusion[expected_effect+" -> "+observed_effect]+=1
        earlier=(indexed_decisions[entry["case_id"]]["score"]["parsed"] or {}).get("decision")
        if expected_effect=="positive" and observed_effect=="negative":
            crossings["positive_to_negative_cases"]+=1
            crossings["positive_to_negative_with_precedence_masked"]+=int(primary_masked)
        if state=="equivalent" and expected_effect==observed_effect and earlier!="equivalent":
            crossings["equivalent_effect_read_correct_decision_wrong"]+=1
        if state=="recheck_required" and not fact.variance_stable and earlier!="recheck_required":
            crossings["recheck_variance_read_correct_decision_wrong"]+=1
    # Verify that the *frozen development* is purely linguistic, without
    # training-only negative throughput distractors, and has one effect fact.
    cfg,pack,_,_=prepare("D")
    dev=pack["partitions"]["evaluator"]["development_diagnostic"]
    assert len(dev)==256
    dev_summary=Counter()
    positive_training_bank=("advantage","gain","benefit","favorable")
    import build_evaluator_semantic_invariance_repair_v1 as original
    for c in dev:
        decoded=decode_frozen_development_evidence(c)
        effect_category="effect_"+decoded.effect
        vocab=set(original.DEV_PHRASES[effect_category])
        matches=[e["fact"] for e in c["evidence"] if e["fact"] in vocab]
        if len(matches)!=1:
            raise ValueError("frozen development effect sentence unrecognizable")
        if any("throughput" in e["fact"].lower() for e in c["evidence"]):
            raise ValueError("frozen development unexpectedly contains throughput distractor")
        if decoded.effect=="positive" and not any(word in matches[0].lower() for word in positive_training_bank):
            raise ValueError("positive development phrase lacks positive lexical evidence")
        dev_summary[effect_category]+=1
        dev_summary["throughput_distractor_cases"]+=any("throughput" in e["fact"].lower() for e in c["evidence"])
    output={
        "run":"D step96 separate factor reports vs frozen D decisions",
        "observed":64,
        "effect_confusion":dict(sorted(confusion.items())),
        "by_state":{k:dict(v) for k,v in sorted(per_state.items())},
        "precedence_masks_effect":{k:dict(v) for k,v in sorted(masked.items())},
        "by_advisory_presence":{k:dict(v) for k,v in sorted(by_advice.items())},
        "important_pairwise_observations":dict(sorted(crossings.items())),
        "frozen_development_template_check":dict(sorted(dev_summary.items())),
        "training_census":census_D_training(),
        "S3_read_receipts":receipts,
        "inference_performed":False,"new_spend":0,"scientific_certification":False,
        "limitations":"Correlational results under different prompts. Positive effect is non-decisive for blocked states; prevalence and negative secondary throughput in D training are candidate confounders, not proven causal mechanisms.",
    }
    Path("evaluator-effect-polarity-audit.json").write_text(json.dumps(output,indent=2,sort_keys=True)+"\n")
    summary={k:output[k] for k in ("effect_confusion","by_state","precedence_masks_effect",
             "by_advisory_presence","important_pairwise_observations","frozen_development_template_check","training_census")}
    print("EVALUATOR_EFFECT_POLARITY_AUDIT "+json.dumps(summary,sort_keys=True))
    return output


if __name__=="__main__":
    audit()
