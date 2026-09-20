#!/usr/bin/env python3
"""Deterministically construct the frozen Diagnosis Foundation Bakeoff V1 pack."""
from __future__ import annotations
import copy, hashlib, json
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"configs/eval_packs/hephaestus_cognitive_topology_v1.json"

EXPECTED={
"D1":{"decision":"inconclusive","action":"collect_more_evidence","primary_variable":"evidence","confidence_min":0.15,"confidence_max":0.65},
"D2":{"decision":"inconclusive","action":"collect_more_evidence","primary_variable":"evidence","confidence_min":0.15,"confidence_max":0.65},
"D3":{"decision":"tokenizer","action":"change_tokenizer","primary_variable":"tokenizer","confidence_min":0.85,"confidence_max":1.0},
"D4":{"decision":"tokenizer","action":"change_tokenizer","primary_variable":"tokenizer","confidence_min":0.85,"confidence_max":1.0},
"D5":{"decision":"inconclusive","action":"collect_more_evidence","primary_variable":"evidence","confidence_min":0.0,"confidence_max":0.45},
"D6":{"decision":"evaluation_integrity","action":"repair_evaluation","primary_variable":"evaluation_protocol","confidence_min":0.9,"confidence_max":1.0},
}
DATA_PROVEN={"decision":"data_quality","action":"repair_data","primary_variable":"data_quality_filter","confidence_min":0.85,"confidence_max":1.0}

def ev(ref:str,fact:str)->dict[str,str]: return {"ref":ref,"fact":fact}
def mk(case_id:str, root:str, split:str, idx:int, situation:str, evidence:list[dict[str,str]], allowed:list[str], expected:dict[str,Any], forbidden:list[str], condition:str)->dict[str,Any]:
    return {"case_id":case_id,"root_case_id":root,"split":split,"variant_index":idx,"role":"diagnosis","condition":condition,"pair_id":None,"situation":situation,"evidence":evidence,"allowed_evidence_refs":allowed,"forbidden_claims":forbidden,"expected":expected}

def variant(root:str, split:str, idx:int)->dict[str,Any]:
    tag=f"{split.upper()}-{idx:02d}"
    delta_a=round(0.78-(idx%5)*0.01,2); delta_b=round(0.58+(idx%4)*0.01,2)
    if root=="D1":
        if idx%4==3:
            refs=[f"E-{root}-{tag}-SCORE",f"E-{root}-{tag}-ABL"]
            return mk(f"{root}-{tag}",root,split,idx,
                "A candidate regressed after a data revision. A controlled rerun then changed only the dataset revision back to the prior immutable manifest and recovered the score in three repeated evaluations. Diagnose the most justified domain.",
                [ev(refs[0],f"Score changed from {delta_a} to {delta_b} after data revision."),ev(refs[1],"Three controlled A/B repeats held model, tokenizer, recipe, runtime, and eval pack fixed; reverting only the data manifest restored the prior score.")],
                refs,DATA_PROVEN,[],"causal_ablation")
        distract=["CUDA allocator warning","scheduler package update","storage latency alert","driver version change"][idx%4]
        refs=[f"E-{root}-{tag}-SCORE",f"E-{root}-{tag}-CHANGE",f"E-{root}-{tag}-NOABL",f"E-{root}-{tag}-DIST"]
        return mk(f"{root}-{tag}",root,split,idx,
            f"A candidate score fell after a dataset revision. A {distract} occurred nearby. No controlled ablation isolates any changed factor. Diagnose without pretending causality.",
            [ev(refs[0],f"Complete evaluation: score {delta_a} -> {delta_b}."),
             ev(refs[1],"Dataset manifest changed between baseline and candidate."),
             ev(refs[2],"No controlled ablation isolates dataset, runtime, or training changes."),
             ev(refs[3],f"Nearby observation: {distract}; execution completed without OOM or non-finite tensors.")],
            refs[:3],EXPECTED[root],["dataset caused the regression","CUDA caused the regression"],"causal_uncertainty")
    if root=="D2":
        if idx%5==4:
            refs=[f"E-{root}-{tag}-TOK",f"E-{root}-{tag}-GEN",f"E-{root}-{tag}-IRR"]
            return mk(f"{root}-{tag}",root,split,idx,
                "Generation fails before scientific samples because required special-token IDs exceed the runtime vocabulary. An unrelated historical note claims dataset changes often help instruction following.",
                [ev(refs[0],"Checkpoint requires special token id 100120; runtime tokenizer vocabulary ends at 99999."),ev(refs[1],"0 required scientific generations were produced."),ev(refs[2],"Historical note from another lineage says changing datasets improved instruction following.")],
                refs[:2],EXPECTED["D3"],["dataset change is required","historical note proves"],"irrelevant_evidence_with_strong_signal")
        refs=[f"E-{root}-{tag}-SEM",f"E-{root}-{tag}-CHANGE",f"E-{root}-{tag}-IRR"]
        return mk(f"{root}-{tag}",root,split,idx,
            "A complete evaluation regressed after several nearby changes, but no ablation isolates a cause. An unrelated successful lineage is described in the evidence. Diagnose this run only.",
            [ev(refs[0],f"All samples complete; score {delta_a} -> {delta_b}."),ev(refs[1],"A data revision and runtime package update both occurred; neither was isolated."),ev(refs[2],"An unrelated lineage improved after changing CUDA versions.")],
            refs[:2],EXPECTED[root],["unrelated lineage proves","CUDA caused the regression"],"irrelevant_evidence")
    if root=="D3":
        refs=[f"E-{root}-{tag}-TOK",f"E-{root}-{tag}-GEN",f"E-{root}-{tag}-DIST"]
        return mk(f"{root}-{tag}",root,split,idx,
            "A checkpoint and runtime tokenizer disagree on vocabulary/special-token identity and generation aborts before required samples. A benign environment warning is also present.",
            [ev(refs[0],f"Checkpoint tokenizer vocab={100352+idx}; runtime vocab={100000+idx}; required checkpoint token lies outside runtime range."),ev(refs[1],"0/18 required samples produced; abort occurs in tokenizer/model compatibility check."),ev(refs[2],"A non-fatal CUDA allocator warning was logged earlier.")],
            refs[:2],EXPECTED[root],["CUDA caused the scientific regression"],"tokenizer_runtime_failure")
    if root=="D4":
        if idx%3==2:
            refs=[f"E-{root}-{tag}-HASH",f"E-{root}-{tag}-SCORE",f"E-{root}-{tag}-DASH"]
            return mk(f"{root}-{tag}",root,split,idx,
                "A dashboard labels the run model_regression, but candidate and baseline were evaluated with different frozen-pack hashes. Decide what evidence dominates.",
                [ev(refs[0],"Baseline eval hash sha256:AAA; candidate eval hash sha256:BBB."),ev(refs[1],"Candidate aggregate score appears higher under the mismatched evaluator."),ev(refs[2],"Dashboard label says model_regression without sample-level evidence.")],
                refs[:2],EXPECTED["D6"],["dashboard proves model regression"],"contradictory_eval_integrity")
        refs=[f"E-{root}-{tag}-TOK",f"E-{root}-{tag}-GEN",f"E-{root}-{tag}-DASH"]
        return mk(f"{root}-{tag}",root,split,idx,
            "Generation aborts on an explicit tokenizer incompatibility while a dashboard calls the run semantic_regression. Choose the evidence-backed diagnosis.",
            [ev(refs[0],"Training/runtime tokenizer vocabularies and special-token maps are incompatible."),ev(refs[1],"0/18 scientific samples exist because generation aborts before evaluation."),ev(refs[2],"Dashboard says semantic_regression but links no samples.")],
            refs[:2],EXPECTED[root],["dashboard proves semantic regression"],"contradictory_evidence")
    if root=="D5":
        refs=[f"E-{root}-{tag}-OP",f"E-{root}-{tag}-MISS",f"E-{root}-{tag}-IRR"]
        missing=["deterministic scorecard and samples","baseline metrics and candidate samples","repeatability evidence and eval hash","all quantitative metrics"][idx%4]
        return mk(f"{root}-{tag}",root,split,idx,
            "An operator says the candidate looks worse, but material scientific evidence is missing. Decide whether a failure domain is justified.",
            [ev(refs[0],"Operator impression: candidate looks worse."),ev(refs[1],f"Missing: {missing}."),ev(refs[2],"A public benchmark says this model family is strong.")],
            [refs[1]],EXPECTED[root],["operator impression proves","benchmark proves"],"missing_evidence")
    if root=="D6":
        refs=[f"E-{root}-{tag}-HASH",f"E-{root}-{tag}-SCORE",f"E-{root}-{tag}-DIST"]
        mismatch=["eval-pack hash","judge rubric hash","deterministic scorer version","baseline/candidate decoding protocol"][idx%4]
        return mk(f"{root}-{tag}",root,split,idx,
            f"Candidate quality appears excellent, but the {mismatch} differs from the frozen baseline protocol. Classify the evidence before interpreting quality.",
            [ev(refs[0],f"Frozen baseline and candidate have different {mismatch} identities."),ev(refs[1],"Observed candidate score=0.94 and all candidate generations completed."),ev(refs[2],"A stakeholder asks to promote immediately because the score is high.")],
            refs[:2],EXPECTED[root],["high score proves improvement","promote immediately"],"evaluation_integrity")
    raise KeyError(root)

def build_pack()->dict[str,Any]:
    base=json.loads(BASE.read_text(encoding="utf-8"))
    originals=[copy.deepcopy(c) for c in base["cases"] if c["case_id"] in EXPECTED]
    for c in originals:
        c["root_case_id"]=c["case_id"]; c["split"]="zero_shot"; c["variant_index"]=0
    zero=list(originals)
    for root in EXPECTED:
        zero.extend(variant(root,"zero_shot",i) for i in range(1,5))
    train=[variant(root,"micro_lora_train",i) for root in EXPECTED for i in range(1,13)]
    held=[variant(root,"held_out",i) for root in EXPECTED for i in range(1,9)]
    pack={
        "protocol_id":"hephaestus_diagnosis_foundation_bakeoff_v1",
        "protocol_version":1,
        "purpose":"Compare small diagnosis foundations and their matched micro-LoRA adaptability without leaking held-out adversarial cases.",
        "response_schema":base["response_schema"],
        "scoring":base["scoring"],
        "role_rule":"Diagnose from evidence without pretending certainty. Missing or conflicting evidence must remain explicit; do not confuse temporal correlation with causality or runtime incompleteness with scientific regression.",
        "partitions":{"zero_shot":zero,"micro_lora_train":train,"held_out":held},
        "partition_policy":{"zero_shot_cases":30,"train_cases":72,"held_out_cases":48,"held_out_never_train":True},
    }
    return pack

def canonical_bytes(pack:dict[str,Any])->bytes:
    return (json.dumps(pack,sort_keys=True,separators=(",",":"),ensure_ascii=False)+"\n").encode()

def validate(pack:dict[str,Any])->None:
    parts=pack["partitions"]
    assert len(parts["zero_shot"])==30 and len(parts["micro_lora_train"])==72 and len(parts["held_out"])==48
    ids=[c["case_id"] for rows in parts.values() for c in rows]
    assert len(ids)==len(set(ids))
    assert not ({c["case_id"] for c in parts["micro_lora_train"]}&{c["case_id"] for c in parts["held_out"]})
    for split,rows in parts.items():
        for c in rows:
            assert c["split"]==split
            assert set(c["expected"])=={"decision","action","primary_variable","confidence_min","confidence_max"}
            assert 0<=c["expected"]["confidence_min"]<=c["expected"]["confidence_max"]<=1
            assert set(c["allowed_evidence_refs"])<={e["ref"] for e in c["evidence"]}
    assert any(c["expected"]["decision"]=="data_quality" for c in parts["zero_shot"])
    assert any(c["expected"]["decision"]=="inconclusive" for c in parts["zero_shot"])

if __name__=="__main__":
    p=build_pack(); validate(p); raw=canonical_bytes(p)
    print(json.dumps({"status":"valid","sha256":hashlib.sha256(raw).hexdigest(),"counts":{k:len(v) for k,v in p["partitions"].items()}},sort_keys=True))
