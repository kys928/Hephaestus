#!/usr/bin/env python3
"""Deep read-only autopsy of the burned Evaluator sealed certification v1 evidence.

This script performs no inference, training, adapter mutation, certification, or
promotion. The 256 sealed certification/regression cases are already burned by the
scientific rejection, so they may now be inspected strictly for diagnosis and
curriculum design. Exact failed prompts are not emitted as a future training set.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import build_evaluator_certification_sealed_v1 as builder
import build_interface_repair_v3_evaluator as v3builder
import run_interface_repair_v1 as v1

RUN_ID = "evaluator-sealed-cert-36929796355"
PREFIX = f"hephaestus/scientific/v4/evaluator_certification/{RUN_ID}"
EXPECTED_PACK_SHA = "1314d28c978538ca8bb5527c24e4200d5463f7ad90b084aaae1e656d0b530d25"
EXPECTED_ADAPTER_SHA = "913797ddb8d9d95f83d09a244e8efe430d7bfc4383e589499c2fcc2d943487ed"
PARTITIONS = ("certification", "regression")
STATE_SET = set(v3builder.VOCAB["decision"])
ACTION_SET = set(v3builder.VOCAB["action"])

BOUNDARIES = [
    ("improved", "equivalent"),
    ("improved", "certification_ready"),
    ("equivalent", "certification_ready"),
    ("inconclusive", "recheck_required"),
    ("inconclusive", "incomplete_evidence"),
    ("recheck_required", "scientific_rejection"),
    ("scientific_rejection", "regressed"),
]


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
                raise RuntimeError(f"expected object row at {key}")
            out.append(value)
    return out


def _template_index(case: dict[str, Any]) -> int:
    state = str(case["semantic_root"])
    observed = {str(item["fact"]) for item in case["evidence"]}
    for index, template in enumerate(builder.SEALED_FACT_SETS[state]):
        if observed == set(template):
            return index
    return -1


def _rate(correct: int, total: int) -> float | None:
    return None if total == 0 else correct / total


def _group_accuracy(items: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[str(item[field])].append(item)
    return {
        key: {
            "n": len(rows),
            "decision_exact": _rate(sum(bool(x["decision_exact"]) for x in rows), len(rows)),
            "schema_compliant": _rate(sum(bool(x["schema_compliant"]) for x in rows), len(rows)),
            "system_action_exact": _rate(sum(bool(x["system_action_exact"]) for x in rows), len(rows)),
            "upstream_copy_violation": _rate(sum(bool(x["upstream_copy_violation"]) for x in rows), len(rows)),
        }
        for key, rows in sorted(groups.items())
    }


def _compact(row: dict[str, Any], case: dict[str, Any], partition: str) -> dict[str, Any]:
    score = dict(row["score"])
    parsed = score.get("parsed") if isinstance(score.get("parsed"), dict) else {}
    expected = dict(row["expected"])
    predicted_decision = str(parsed.get("decision", "<invalid>"))
    upstream = case.get("upstream_output") if isinstance(case.get("upstream_output"), dict) else None
    upstream_action = None if upstream is None else str(upstream.get("action"))
    expected_action = str(expected["action"])
    hallucinated = [str(x) for x in score.get("hallucinated_evidence_refs", [])]
    decision_kind = (
        "state" if predicted_decision in STATE_SET else
        "action_token" if predicted_decision in ACTION_SET else
        "other_invalid"
    )
    facts = dict(case.get("verified_facts") or {})
    return {
        "partition": partition,
        "case_id": str(row["case_id"]),
        "expected_state": str(expected["decision"]),
        "predicted_state": predicted_decision,
        "predicted_decision_kind": decision_kind,
        "kind": str(case.get("kind")),
        "template_index": _template_index(case),
        "has_upstream": upstream is not None,
        "upstream_action": upstream_action,
        "upstream_action_is_correct": upstream_action == expected_action if upstream_action is not None else None,
        "decision_equals_upstream_action": predicted_decision == upstream_action if upstream_action is not None else False,
        "decision_exact": bool(score.get("decision_exact")),
        "schema_compliant": bool(score.get("schema_compliant")),
        "evidence_grounded": bool(score.get("evidence_grounded")),
        "hallucination": bool(hallucinated),
        "hallucinated_evidence_ref_count": len(hallucinated),
        "primary_variable_projection": bool(score.get("primary_variable_projected", score.get("primary_variable_projection", score.get("primary_variable_exact")))),
        "boundary_projected": bool(score.get("boundary_projected")),
        "system_action_exact": bool(score.get("system_action_exact")),
        "upstream_copy_violation": bool(score.get("upstream_copy_violation")),
        "model_action_agreement": bool(score.get("model_action_agreement")),
        "confidence_in_band": bool(score.get("confidence_in_band")),
        "machine_facts": facts,
        "predicted_action": parsed.get("action"),
        "predicted_primary_variable": parsed.get("primary_variable"),
        "predicted_confidence": parsed.get("confidence"),
        "predicted_evidence_refs": parsed.get("evidence_refs"),
        "predicted_uncertainties": parsed.get("uncertainties"),
        "predicted_rationale": parsed.get("rationale"),
        "expected_action": expected_action,
        "expected_primary_variable": str(expected["primary_variable"]),
    }


def _state_confusions(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(
        f"{x['expected_state']} -> {x['predicted_state']}"
        for x in items if not x["decision_exact"]
    )
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _boundary_report(items: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for left, right in BOUNDARIES:
        rows = [x for x in items if x["expected_state"] in {left, right}]
        left_rows = [x for x in rows if x["expected_state"] == left]
        right_rows = [x for x in rows if x["expected_state"] == right]
        direct_cross = sum(
            (x["expected_state"] == left and x["predicted_state"] == right)
            or (x["expected_state"] == right and x["predicted_state"] == left)
            for x in rows
        )
        report[f"{left}<->{right}"] = {
            "n": len(rows),
            "left_accuracy": _rate(sum(x["decision_exact"] for x in left_rows), len(left_rows)),
            "right_accuracy": _rate(sum(x["decision_exact"] for x in right_rows), len(right_rows)),
            "direct_cross_confusions": direct_cross,
            "all_errors": sum(not x["decision_exact"] for x in rows),
        }
    return report


def _template_report(items: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for state in sorted(STATE_SET):
        state_rows = [x for x in items if x["expected_state"] == state]
        templates: dict[str, Any] = {}
        for ti in range(4):
            rows = [x for x in state_rows if x["template_index"] == ti]
            templates[str(ti)] = {
                "n": len(rows),
                "decision_exact": _rate(sum(x["decision_exact"] for x in rows), len(rows)),
                "errors": sum(not x["decision_exact"] for x in rows),
            }
        out[state] = templates
    return out


def main() -> int:
    client = v1.s3_client()
    client.head_bucket(Bucket=v1.bucket())
    result = _get_json(client, f"{PREFIX}/result.json")
    if result.get("pack_sha256") != EXPECTED_PACK_SHA:
        raise RuntimeError("sealed result pack SHA drift")
    if result.get("candidate_adapter_sha256") != EXPECTED_ADAPTER_SHA:
        raise RuntimeError("sealed result adapter SHA drift")
    if result.get("status") != "scientifically_rejected":
        raise RuntimeError("autopsy is only valid for the burned scientific rejection")

    pack = builder.build_pack()
    builder.validate(pack)
    if builder.canonical_sha256(pack) != EXPECTED_PACK_SHA:
        raise RuntimeError("local sealed pack SHA drift")

    all_items: list[dict[str, Any]] = []
    partition_items: dict[str, list[dict[str, Any]]] = {}
    partition_case_maps = {
        p: {str(x["case_id"]): x for x in pack["partitions"][builder.ROLE][p]}
        for p in PARTITIONS
    }
    for partition in PARTITIONS:
        rows: list[dict[str, Any]] = []
        for index in range(1, 5):
            rows.extend(_get_jsonl(client, f"{PREFIX}/samples/{partition}/shard-{index:04d}.jsonl"))
        if len(rows) != 128:
            raise RuntimeError(f"expected 128 {partition} rows, got {len(rows)}")
        items = [_compact(row, partition_case_maps[partition][str(row["case_id"])], partition) for row in rows]
        partition_items[partition] = items
        all_items.extend(items)

    if len(all_items) != 256:
        raise RuntimeError(f"expected 256 burned rows, got {len(all_items)}")

    state_accuracy = _group_accuracy(all_items, "expected_state")
    kind_accuracy = _group_accuracy(all_items, "kind")
    partition_accuracy = _group_accuracy(all_items, "partition")

    interface = [x for x in all_items if x["has_upstream"]]
    rehearsal = [x for x in all_items if not x["has_upstream"]]
    bad_advice = [x for x in interface if x["upstream_action_is_correct"] is False]
    correct_advice = [x for x in interface if x["upstream_action_is_correct"] is True]

    invalid_decisions = [x for x in all_items if x["predicted_decision_kind"] != "state"]
    action_token_decisions = [x for x in invalid_decisions if x["predicted_decision_kind"] == "action_token"]
    hallucinations = [x for x in all_items if x["hallucination"]]
    schema_failures = [x for x in all_items if not x["schema_compliant"]]
    upstream_copy = [x for x in all_items if x["upstream_copy_violation"]]
    system_action_errors = [x for x in all_items if not x["system_action_exact"]]

    advisory = {
        "interface_n": len(interface),
        "rehearsal_n": len(rehearsal),
        "interface_decision_exact": _rate(sum(x["decision_exact"] for x in interface), len(interface)),
        "rehearsal_decision_exact": _rate(sum(x["decision_exact"] for x in rehearsal), len(rehearsal)),
        "wrong_upstream_advice_n": len(bad_advice),
        "wrong_upstream_advice_decision_exact": _rate(sum(x["decision_exact"] for x in bad_advice), len(bad_advice)),
        "correct_upstream_advice_n": len(correct_advice),
        "correct_upstream_advice_decision_exact": _rate(sum(x["decision_exact"] for x in correct_advice), len(correct_advice)),
        "decision_equals_upstream_action_n": sum(x["decision_equals_upstream_action"] for x in interface),
        "upstream_copy_violation_n": len(upstream_copy),
    }

    structural = {
        "invalid_decision_n": len(invalid_decisions),
        "action_token_in_decision_n": len(action_token_decisions),
        "action_token_in_decision_by_expected_state": dict(sorted(Counter(x["expected_state"] for x in action_token_decisions).items())),
        "schema_failure_n": len(schema_failures),
        "schema_failure_by_expected_state": dict(sorted(Counter(x["expected_state"] for x in schema_failures).items())),
        "hallucination_case_n": len(hallucinations),
        "hallucinated_ref_n": sum(x["hallucinated_evidence_ref_count"] for x in hallucinations),
        "hallucination_by_expected_state": dict(sorted(Counter(x["expected_state"] for x in hallucinations).items())),
        "system_action_error_n": len(system_action_errors),
        "system_action_error_by_expected_state": dict(sorted(Counter(x["expected_state"] for x in system_action_errors).items())),
    }

    weak_states = [
        state for state, metrics in state_accuracy.items()
        if (metrics["decision_exact"] or 0.0) < 0.90
    ]
    robust_states = [
        state for state, metrics in state_accuracy.items()
        if (metrics["decision_exact"] or 0.0) >= 0.95
    ]

    autopsy = {
        "autopsy_version": "hephaestus-evaluator-sealed-v1-burned-autopsy.v1",
        "source_run_id": RUN_ID,
        "source_is_burned_diagnostic_only": True,
        "source_pack_sha256": EXPECTED_PACK_SHA,
        "source_adapter_sha256": EXPECTED_ADAPTER_SHA,
        "new_inference_performed": False,
        "training_performed": False,
        "certification_claim_performed": False,
        "sample_count": 256,
        "partition_accuracy": partition_accuracy,
        "state_accuracy": state_accuracy,
        "kind_accuracy": kind_accuracy,
        "advisory_analysis": advisory,
        "structural_failures": structural,
        "state_confusions": _state_confusions(all_items),
        "conceptual_boundaries": _boundary_report(all_items),
        "template_sensitivity": _template_report(all_items),
        "weak_states_below_90pct": weak_states,
        "robust_states_at_or_above_95pct": robust_states,
        "diagnostic_case_index": [
            {
                "partition": x["partition"],
                "case_id": x["case_id"],
                "expected_state": x["expected_state"],
                "predicted_state": x["predicted_state"],
                "kind": x["kind"],
                "template_index": x["template_index"],
                "has_upstream": x["has_upstream"],
                "upstream_action": x["upstream_action"],
                "decision_exact": x["decision_exact"],
                "schema_compliant": x["schema_compliant"],
                "evidence_grounded": x["evidence_grounded"],
                "hallucination": x["hallucination"],
                "system_action_exact": x["system_action_exact"],
                "upstream_copy_violation": x["upstream_copy_violation"],
            }
            for x in all_items
            if not x["decision_exact"] or not x["schema_compliant"] or x["hallucination"] or not x["system_action_exact"] or x["upstream_copy_violation"]
        ],
    }

    key = f"{PREFIX}/burned-autopsy-v1.json"
    v1.put_json(client, key, autopsy)
    public = {k: v for k, v in autopsy.items() if k != "diagnostic_case_index"}
    public["result_key"] = key
    print("EVALUATOR_SEALED_AUTOPSY_JSON " + json.dumps(public, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
