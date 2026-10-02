#!/usr/bin/env python3
"""Conditional breakdown for the burned sealed Evaluator autopsy.

Separates wording-template effects from direct/handoff and upstream-advice effects.
Read-only; no inference or training.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

import build_evaluator_certification_sealed_v1 as builder
import build_interface_repair_v3_evaluator as old
import run_interface_repair_v1 as v1

RUN_ID = "evaluator-sealed-cert-36929796355"
PREFIX = f"hephaestus/scientific/v4/evaluator_certification/{RUN_ID}"
PARTITIONS = ("certification", "regression")


def get_jsonl(client: Any, key: str) -> list[dict[str, Any]]:
    response = client.get_object(Bucket=v1.bucket(), Key=key)
    try:
        text = response["Body"].read().decode()
    finally:
        response["Body"].close()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def template_index(case: dict[str, Any]) -> int:
    observed = {str(x["fact"]) for x in case["evidence"]}
    for i, template in enumerate(builder.SEALED_FACT_SETS[str(case["semantic_root"])]):
        if observed == set(template):
            return i
    return -1


def rate(rows: list[dict[str, Any]]) -> float | None:
    return None if not rows else sum(x["exact"] for x in rows) / len(rows)


def main() -> int:
    client = v1.s3_client()
    pack = builder.build_pack()
    cases = {
        p: {str(x["case_id"]): x for x in pack["partitions"][builder.ROLE][p]}
        for p in PARTITIONS
    }
    items: list[dict[str, Any]] = []
    for p in PARTITIONS:
        rows: list[dict[str, Any]] = []
        for shard in range(1, 5):
            rows += get_jsonl(client, f"{PREFIX}/samples/{p}/shard-{shard:04d}.jsonl")
        for row in rows:
            case = cases[p][str(row["case_id"])]
            score = row["score"]
            parsed = score.get("parsed") if isinstance(score.get("parsed"), dict) else {}
            upstream = case.get("upstream_output") if isinstance(case.get("upstream_output"), dict) else None
            expected_action = str(row["expected"]["action"])
            items.append({
                "state": str(row["expected"]["decision"]),
                "pred": str(parsed.get("decision", "<invalid>")),
                "exact": bool(score.get("decision_exact")),
                "kind": str(case["kind"]),
                "template": template_index(case),
                "upstream_correct": None if upstream is None else str(upstream.get("action")) == expected_action,
                "upstream_action": None if upstream is None else str(upstream.get("action")),
                "copy": bool(score.get("upstream_copy_violation")),
            })
    report: dict[str, Any] = {}
    for state in sorted(old.VOCAB["decision"]):
        sr = [x for x in items if x["state"] == state]
        direct = [x for x in sr if x["kind"] == "rehearsal"]
        interface = [x for x in sr if x["kind"] != "rehearsal"]
        wrong = [x for x in interface if x["upstream_correct"] is False]
        correct = [x for x in interface if x["upstream_correct"] is True]
        templates: dict[str, Any] = {}
        for ti in range(4):
            tr = [x for x in sr if x["template"] == ti]
            td = [x for x in tr if x["kind"] == "rehearsal"]
            th = [x for x in tr if x["kind"] != "rehearsal"]
            templates[str(ti)] = {
                "n": len(tr), "accuracy": rate(tr),
                "direct_n": len(td), "direct_accuracy": rate(td),
                "handoff_n": len(th), "handoff_accuracy": rate(th),
                "predictions": dict(sorted(Counter(x["pred"] for x in tr).items())),
            }
        report[state] = {
            "n": len(sr), "accuracy": rate(sr),
            "direct_n": len(direct), "direct_accuracy": rate(direct),
            "handoff_n": len(interface), "handoff_accuracy": rate(interface),
            "wrong_advice_n": len(wrong), "wrong_advice_accuracy": rate(wrong),
            "correct_advice_n": len(correct), "correct_advice_accuracy": rate(correct),
            "upstream_copy_n": sum(x["copy"] for x in sr),
            "templates": templates,
        }
    payload = {
        "analysis_version": "hephaestus-evaluator-sealed-v1-conditionals.v1",
        "burned_diagnostic_only": True,
        "sample_count": len(items),
        "by_state": report,
    }
    key = f"{PREFIX}/burned-autopsy-conditionals-v1.json"
    v1.put_json(client, key, payload)
    print("EVALUATOR_SEALED_CONDITIONALS_JSON " + json.dumps({**payload, "result_key": key}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
