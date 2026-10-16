#!/usr/bin/env python3
"""Independently rescore D's immutable balanced screen without renting a GPU."""
import json
from collections import Counter
from pathlib import Path

import build_evaluator_semantic_repair_v2 as build
import run_interface_repair_v1 as s3tools
from evaluator_semantic_v2_metrics import score
from evaluator_semantic_v2_screen import screen_cases, verdict

D_RUN_ID = "evaluator-semantic-v2-D-38945578442"
D_SOURCE_SHA = "ffa0458d5850f9d572a06d5ee6660fd63fc8acc6"
D_PREFIX = "hephaestus/scientific/v5/evaluator_semantic_repair_v2/" + D_RUN_ID
V1_PREFIX = "hephaestus/scientific/v4/evaluator_semantic_invariance_repair/evaluator-semantic-repair-37005231044"


def read_json(client, key):
    obj = client.get_object(Bucket=s3tools.bucket(), Key=key)
    try:
        return json.loads(obj["Body"].read().decode("utf-8"))
    finally:
        obj["Body"].close()


def read_jsonl(client, key):
    obj = client.get_object(Bucket=s3tools.bucket(), Key=key)
    try:
        return [json.loads(x) for x in obj["Body"].read().decode("utf-8").splitlines() if x.strip()]
    finally:
        obj["Body"].close()


def assess():
    client = s3tools.s3_client()
    client.head_bucket(Bucket=s3tools.bucket())
    pack = build.build_pack("D")
    frozen = pack["partitions"]["evaluator"]["development_diagnostic"]
    selected = screen_cases(frozen)
    if len(selected) != 64:
        raise RuntimeError("balanced screen identity drift")
    prefix = D_PREFIX + "/checkpoints/step-096/evaluation"
    terminal = read_json(client, prefix + "/result.json")
    if terminal.get("status") not in ("screen_rejected", "evaluation_complete"):
        raise RuntimeError("D has no recognized terminal scientific evidence")
    if terminal.get("repo_sha") != D_SOURCE_SHA or terminal.get("run_id") != D_RUN_ID:
        raise RuntimeError("immutable D experiment identity mismatch")
    observed = read_json(client, prefix + "/samples/shard-0001.json") + read_json(client, prefix + "/samples/shard-0002.json")
    if len(observed) != 64 or {x["case_id"] for x in observed} != {x["case_id"] for x in selected}:
        raise RuntimeError("screen case identity/count mismatch")
    cases={x["case_id"]:x for x in selected}
    for row in observed:
        raw = row["generation"]["projected_output"]
        recalculated = score(pack, cases[row["case_id"]], raw)
        for field in ("decision_exact", "schema_compliant", "system_action_exact", "upstream_copy_violation"):
            if recalculated[field] != row["score"][field]:
                raise RuntimeError(f"persisted D score drift {row['case_id']}/{field}")
    expected = verdict(selected, observed)
    screen_result = read_json(client, prefix + "/screen/result.json")
    if expected != screen_result:
        raise RuntimeError("persisted D screen verdict drift")
    if terminal.get("screen") is not None and terminal["screen"] != expected:
        raise RuntimeError("terminal D screen verdict drift")
    reference={}
    for shard in range(1,9):
        for row in read_jsonl(client, f"{V1_PREFIX}/checkpoints/step-096/evaluation/samples/development_diagnostic/shard-{shard:04d}.jsonl"):
            reference[row["case_id"]]=row
    if len(reference)!=256:
        raise RuntimeError("V1 reference incomplete")
    paired=Counter()
    by_state={}
    for r in selected:
        cid=r["case_id"]
        state=r["expected"]["decision"]
        d=bool(next(x["score"]["decision_exact"] for x in observed if x["case_id"]==cid))
        v=bool(reference[cid]["score"]["decision_exact"])
        paired[(d,v)]+=1
        by_state.setdefault(state,Counter())["D_correct"]+=int(d)
        by_state[state]["V1_correct"]+=int(v)
        by_state[state]["total"]+=1
    report=dict(
       verdict=expected["status"],d_run_id=D_RUN_ID,source_sha=D_SOURCE_SHA,
       D_screen_correct=sum(bool(x["score"]["decision_exact"]) for x in observed),
       V1_same_screen_correct=sum(int(reference[c["case_id"]]["score"]["decision_exact"]) for c in selected),
       screen_total=64,
       paired_correctness={f"D_{d}_V1_{v}":n for (d,v),n in sorted(paired.items())},
       by_state={k:dict(v) for k,v in sorted(by_state.items())},
       screening=expected,
       terminal_status=terminal["status"],
       full_development_evaluated=(terminal["status"]=="evaluation_complete"),
       parent_adapter_mutated=False,certification_performed=False,
       experimental_limitations="Screen is a non-independent subset of the frozen 256-case development diagnostic. Only exact same-case V1 comparison and mathematical rejection are supported; no sealed certification or causal attribution to factorial design.")
    Path("semantic-v2-D-independent-assessment.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("EVALUATOR_D_INDEPENDENT_ASSESSMENT "+json.dumps(report,sort_keys=True))
    return report


if __name__=="__main__":
    assess()
