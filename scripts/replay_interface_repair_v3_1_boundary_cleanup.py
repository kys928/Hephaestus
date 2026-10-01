#!/usr/bin/env python3
"""Replay burned V3.1 neutral outputs through the approved certification boundary.

This performs no model inference, training, adapter mutation, certification claim, or
production promotion. It rescoring the already-persisted 128 neutral-holdout outputs
using deterministic primary-variable projection and strict evidence-reference
canonicalization to quantify the effect of the approved architecture changes.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

import build_interface_repair_v3_1_neutral_holdout as builder
import run_interface_repair_v1 as v1
import run_interface_repair_v3_evaluator as v3

RUN_ID = "v3-neutral-holdout-36766350227"
PREFIX = f"hephaestus/scientific/v3/neutral_holdout/{RUN_ID}"
RESULT_KEY = f"{PREFIX}/boundary-cleanup-replay-v1.json"
EXPECTED_PACK_SHA = "0bae520eda2b59f9317baacfd9d77a24f51f0b7c4a2b5e7e057e0a8cee2d7492"
EXPECTED_ADAPTER_SHA = "913797ddb8d9d95f83d09a244e8efe430d7bfc4383e589499c2fcc2d943487ed"


def _get_json(client: Any, key: str) -> dict[str, Any]:
    response = client.get_object(Bucket=v1.bucket(), Key=key)
    try:
        value = json.loads(response["Body"].read())
    finally:
        response["Body"].close()
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object at {key}")
    return value


def _get_jsonl(client: Any, key: str) -> list[dict[str, Any]]:
    response = client.get_object(Bucket=v1.bucket(), Key=key)
    try:
        text = response["Body"].read().decode("utf-8")
    finally:
        response["Body"].close()
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"expected object row at {key}")
            rows.append(value)
    return rows


def main() -> int:
    client = v1.s3_client()
    client.head_bucket(Bucket=v1.bucket())

    source_result = _get_json(client, f"{PREFIX}/result.json")
    if source_result.get("pack_sha256") != EXPECTED_PACK_SHA:
        raise RuntimeError("source neutral pack SHA drift")
    if (source_result.get("source_v31_adapter") or {}).get("sha256") != EXPECTED_ADAPTER_SHA:
        raise RuntimeError("source V3.1 adapter SHA drift")

    pack = builder.build_pack()
    builder.validate(pack)
    if builder.canonical_sha256(pack) != EXPECTED_PACK_SHA:
        raise RuntimeError("local neutral pack SHA drift")
    cases = {
        str(case["case_id"]): case
        for case in pack["partitions"][builder.ROLE][builder.NEUTRAL_PARTITION]
    }

    original_rows: list[dict[str, Any]] = []
    for index in range(1, 5):
        original_rows.extend(_get_jsonl(client, f"{PREFIX}/samples/shard-{index:04d}.jsonl"))
    if len(original_rows) != 128:
        raise RuntimeError(f"expected 128 burned rows, got {len(original_rows)}")

    replay_rows: list[dict[str, Any]] = []
    wrong_decisions: Counter[str] = Counter()
    for original in original_rows:
        case_id = str(original["case_id"])
        case = cases[case_id]
        generation = original.get("generation") or {}
        raw = str(generation.get("projected_output", ""))
        if not raw:
            raise RuntimeError(f"missing persisted projected output for {case_id}")
        score = v3.score_case_v3(pack, case, "evaluator", raw)
        replay_rows.append({
            "case_id": case_id,
            "source_case_id": original.get("source_case_id"),
            "expected": case["expected"],
            "score": score,
        })
        if not score["decision_exact"]:
            parsed = score.get("parsed") or {}
            wrong_decisions[f"{case['expected']['decision']} -> {parsed.get('decision', '<invalid>')}"] += 1

    summary = v3.summarize_v3(replay_rows)
    residual = {
        "wrong_decision_count": sum(not row["score"]["decision_exact"] for row in replay_rows),
        "decision_confusion_errors": dict(sorted(wrong_decisions.items())),
        "model_primary_variable_error_count": sum(
            not row["score"]["model_primary_variable_exact"] for row in replay_rows
        ),
        "authoritative_primary_variable_error_count": sum(
            not row["score"]["primary_variable_exact"] for row in replay_rows
        ),
        "evidence_ref_normalization_case_count": sum(
            bool(row["score"]["normalized_evidence_ref_pairs"]) for row in replay_rows
        ),
        "evidence_ref_normalization_count": sum(
            len(row["score"]["normalized_evidence_ref_pairs"]) for row in replay_rows
        ),
        "unresolved_evidence_case_count": sum(
            bool(row["score"]["hallucinated_evidence_refs"]) for row in replay_rows
        ),
        "system_action_error_count": sum(
            not row["score"]["system_action_exact"] for row in replay_rows
        ),
        "system_semantic_escalation_count": sum(
            row["score"]["system_semantic_escalation"] for row in replay_rows
        ),
    }
    payload = {
        "replay_version": "hephaestus-v3.1-boundary-cleanup-replay.v1",
        "source_run_id": RUN_ID,
        "source_is_burned_diagnostic_only": True,
        "pack_sha256": EXPECTED_PACK_SHA,
        "adapter_sha256": EXPECTED_ADAPTER_SHA,
        "new_inference_performed": False,
        "training_performed": False,
        "adapter_mutated": False,
        "certification_claim_performed": False,
        "production_promotion_performed": False,
        "approved_cleanup": {
            "authoritative_primary_variable": "deterministic_from_scientific_state",
            "evidence_reference_canonicalization": "exact_optional_E_prefix_unique_match_fail_closed",
        },
        "source_summary": source_result.get("summary"),
        "replayed_summary": summary,
        "residual": residual,
    }
    v1.put_json(client, RESULT_KEY, payload)
    print("V31_BOUNDARY_CLEANUP_REPLAY_JSON " + json.dumps({
        "result_key": RESULT_KEY,
        "summary": summary,
        "residual": residual,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
