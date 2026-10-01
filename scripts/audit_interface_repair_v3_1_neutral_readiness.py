#!/usr/bin/env python3
"""Read-only readiness audit over the burned V3.1 neutral diagnostic rows.

This script performs no inference, training, adapter mutation, certification, or
promotion. It reads the already-persisted neutral-holdout result/shards, reconstructs
the paired cases locally, and classifies every residual failure before a fresh sealed
certification attempt is designed.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

import build_interface_repair_v3_1_neutral_holdout as builder
import run_interface_repair_v1 as v1
import run_interface_repair_v2 as v2
from hephaestus.control.evaluator_boundary import project_evaluator_action

RUN_ID = "v3-neutral-holdout-36766350227"
PREFIX = f"hephaestus/scientific/v3/neutral_holdout/{RUN_ID}"
EXPECTED_PACK_SHA = "0bae520eda2b59f9317baacfd9d77a24f51f0b7c4a2b5e7e057e0a8cee2d7492"
EXPECTED_ADAPTER_SHA = "913797ddb8d9d95f83d09a244e8efe430d7bfc4383e589499c2fcc2d943487ed"


def _get_json(client: Any, key: str) -> dict[str, Any]:
    response = client.get_object(Bucket=v1.bucket(), Key=key)
    try:
        raw = response["Body"].read()
    finally:
        response["Body"].close()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object at {key}")
    return value


def _get_jsonl(client: Any, key: str) -> list[dict[str, Any]]:
    response = client.get_object(Bucket=v1.bucket(), Key=key)
    try:
        text = response["Body"].read().decode("utf-8")
    finally:
        response["Body"].close()
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"expected JSON object row at {key}")
            out.append(value)
    return out


def _compact_case(row: dict[str, Any], neutral: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    score = dict(row["score"])
    parsed = score.get("parsed") if isinstance(score.get("parsed"), dict) else {}
    expected = dict(row["expected"])
    facts = dict(neutral.get("verified_facts") or {})
    expected_projection = project_evaluator_action(
        str(expected["decision"]), facts, model_proposed_action=str(expected["action"])
    )

    hallucinated = [str(x) for x in score.get("hallucinated_evidence_refs", [])]
    source_refs = {str(x) for x in source.get("allowed_evidence_refs", [])}
    neutral_refs = {str(x) for x in neutral.get("allowed_evidence_refs", [])}
    hallucinated_source_matches = sorted(set(hallucinated) & source_refs)

    classes: list[str] = []
    if not bool(score.get("decision_exact")):
        classes.append("model_scientific_state_inference_failure")
    if not bool(score.get("primary_variable_exact")):
        classes.append("primary_variable_output_failure")
    if not bool(score.get("confidence_in_band")):
        classes.append("confidence_band_failure")
    if not bool(score.get("schema_compliant")):
        classes.append("schema_contract_failure")
    if hallucinated:
        classes.append("evidence_reference_grounding_failure")
        if hallucinated_source_matches:
            classes.append("stale_source_identifier_recall")
    if bool(score.get("upstream_copy_violation")):
        classes.append("upstream_copy_failure")
    if not bool(score.get("system_action_exact")):
        if expected_projection.effective_action != str(expected["action"]):
            classes.append("fixture_boundary_contract_mismatch")
        elif bool(score.get("decision_exact")):
            classes.append("deterministic_boundary_logic_failure")
        else:
            classes.append("model_state_error_not_contained_by_boundary")

    generation = row.get("generation") if isinstance(row.get("generation"), dict) else {}
    return {
        "case_id": row.get("case_id"),
        "source_case_id": row.get("source_case_id"),
        "kind": row.get("kind"),
        "semantic_root": row.get("semantic_root"),
        "focused_boundary": neutral.get("focused_boundary"),
        "classes": classes,
        "expected": expected,
        "predicted": {
            "decision": parsed.get("decision"),
            "action": parsed.get("action"),
            "primary_variable": parsed.get("primary_variable"),
            "confidence": parsed.get("confidence"),
            "evidence_refs": parsed.get("evidence_refs"),
            "uncertainties": parsed.get("uncertainties"),
            "rationale": parsed.get("rationale"),
        },
        "score": {
            "decision_exact": score.get("decision_exact"),
            "primary_variable_exact": score.get("primary_variable_exact"),
            "confidence_in_band": score.get("confidence_in_band"),
            "schema_compliant": score.get("schema_compliant"),
            "evidence_grounded": score.get("evidence_grounded"),
            "hallucinated_evidence_refs": hallucinated,
            "upstream_copy_violation": score.get("upstream_copy_violation"),
            "boundary_projected": score.get("boundary_projected"),
            "system_action_exact": score.get("system_action_exact"),
            "system_effective_action": score.get("system_effective_action"),
            "boundary_machine_override": score.get("boundary_machine_override"),
            "boundary_reasons": score.get("boundary_reasons"),
        },
        "verified_facts": facts,
        "upstream_output": neutral.get("upstream_output"),
        "evidence": neutral.get("evidence"),
        "neutral_allowed_evidence_refs": sorted(neutral_refs),
        "source_allowed_evidence_refs": sorted(source_refs),
        "hallucinated_source_identifier_matches": hallucinated_source_matches,
        "expected_boundary_projection": {
            "effective_action": expected_projection.effective_action,
            "machine_override": expected_projection.machine_override,
            "reasons": list(expected_projection.reasons),
        },
        "format_retry_attempted": generation.get("format_retry_attempted"),
        "format_retry_succeeded": generation.get("format_retry_succeeded"),
    }


def main() -> int:
    client = v1.s3_client()
    client.head_bucket(Bucket=v1.bucket())
    result = _get_json(client, f"{PREFIX}/result.json")
    if result.get("pack_sha256") != EXPECTED_PACK_SHA:
        raise RuntimeError("neutral result pack SHA drift")
    source = result.get("source_v31_adapter") or {}
    if source.get("sha256") != EXPECTED_ADAPTER_SHA:
        raise RuntimeError("neutral result adapter SHA drift")

    rows: list[dict[str, Any]] = []
    for index in range(1, 5):
        rows.extend(_get_jsonl(client, f"{PREFIX}/samples/shard-{index:04d}.jsonl"))
    if len(rows) != 128:
        raise RuntimeError(f"expected 128 neutral rows, got {len(rows)}")

    pack = builder.build_pack()
    builder.validate(pack)
    if builder.canonical_sha256(pack) != EXPECTED_PACK_SHA:
        raise RuntimeError("local paired pack SHA drift")
    partitions = pack["partitions"][builder.ROLE]
    neutral_by_id = {str(x["case_id"]): x for x in partitions[builder.NEUTRAL_PARTITION]}
    source_by_id = {str(x["case_id"]): x for x in partitions[builder.SOURCE_PARTITION]}

    audited: list[dict[str, Any]] = []
    for row in rows:
        case_id = str(row["case_id"])
        source_id = str(row["source_case_id"])
        neutral = neutral_by_id[case_id]
        source_case = source_by_id[source_id]
        item = _compact_case(row, neutral, source_case)
        if item["classes"]:
            audited.append(item)

    class_counts = Counter(cls for item in audited for cls in item["classes"])
    overlap = {
        "state_and_hallucination": sum(
            "model_scientific_state_inference_failure" in x["classes"]
            and "evidence_reference_grounding_failure" in x["classes"] for x in audited
        ),
        "state_and_system_action": sum(
            "model_scientific_state_inference_failure" in x["classes"]
            and "model_state_error_not_contained_by_boundary" in x["classes"] for x in audited
        ),
        "primary_only": sum(
            "primary_variable_output_failure" in x["classes"]
            and "model_scientific_state_inference_failure" not in x["classes"] for x in audited
        ),
    }

    audit = {
        "audit_version": "hephaestus-v3.1-burned-readiness-audit.v1",
        "source_run_id": RUN_ID,
        "source_is_burned_diagnostic_only": True,
        "training_performed": False,
        "new_inference_performed": False,
        "certification_claim_performed": False,
        "production_promotion_performed": False,
        "sample_count": len(rows),
        "residual_case_count": len(audited),
        "class_counts": dict(sorted(class_counts.items())),
        "overlap": overlap,
        "cases": audited,
    }
    key = f"{PREFIX}/readiness-audit-v1.json"
    v2.put_json(client, key, audit)
    print("V31_READINESS_AUDIT_JSON " + json.dumps({
        "result_key": key,
        "sample_count": len(rows),
        "residual_case_count": len(audited),
        "class_counts": dict(sorted(class_counts.items())),
        "overlap": overlap,
        "cases": audited,
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
