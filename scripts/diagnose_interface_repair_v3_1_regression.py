#!/usr/bin/env python3
"""Diagnose persisted V3.1 Evaluator post-regression misses.

This script is read-only with respect to scientific evidence. It reconstructs the
immutable V3.1 regression pack, joins it to already-persisted sample rows, and
reports exact expected->predicted state confusions.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import build_interface_repair_v3_1_evaluator as builder


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    pack = builder.build_pack()
    builder.validate(pack)
    regression = list(pack["partitions"]["evaluator"]["regression"])
    case_by_id = {str(case["case_id"]): case for case in regression}
    if len(case_by_id) != 128:
        raise RuntimeError(f"unexpected V3.1 regression case count: {len(case_by_id)}")

    paths = sorted(Path(args.input_dir).glob("*.jsonl"))
    if not paths:
        raise RuntimeError(f"no JSONL shards found in {args.input_dir}")
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    if len(rows) != 128:
        raise RuntimeError(f"unexpected persisted post-regression row count: {len(rows)}")

    decision_confusions: Counter[tuple[str, str]] = Counter()
    primary_confusions: Counter[tuple[str, str]] = Counter()
    expected_totals: Counter[str] = Counter()
    expected_correct: Counter[str] = Counter()
    contexts_with_misses: Counter[str] = Counter()
    failure_components: Counter[str] = Counter()
    misses: list[dict[str, Any]] = []

    for row in rows:
        case_id = str(row["case_id"])
        case = case_by_id[case_id]
        score = dict(row["score"])
        parsed = score.get("parsed") if isinstance(score.get("parsed"), dict) else {}
        expected = dict(case["expected"])
        expected_decision = str(expected["decision"])
        predicted_decision = str(parsed.get("decision", "<missing>"))
        expected_primary = str(expected["primary_variable"])
        predicted_primary = str(parsed.get("primary_variable", "<missing>"))

        expected_totals[expected_decision] += 1
        decision_confusions[(expected_decision, predicted_decision)] += 1
        primary_confusions[(expected_primary, predicted_primary)] += 1
        if bool(score.get("decision_exact")):
            expected_correct[expected_decision] += 1

        semantic_miss = not bool(score.get("semantic_exact"))
        system_miss = not bool(score.get("system_action_exact"))
        if not (semantic_miss or system_miss):
            continue

        if not bool(score.get("decision_exact")):
            failure_components["decision"] += 1
        if not bool(score.get("primary_variable_exact")):
            failure_components["primary_variable"] += 1
        if not bool(score.get("confidence_in_band")):
            failure_components["confidence"] += 1
        if not bool(score.get("system_action_exact")):
            failure_components["system_action"] += 1
        if bool(score.get("upstream_copy_violation")):
            failure_components["upstream_copy"] += 1

        contexts_with_misses[str(case["context"])] += 1
        misses.append({
            "case_id": case_id,
            "context": case["context"],
            "kind": case["kind"],
            "semantic_root": case["semantic_root"],
            "situation": case["situation"],
            "evidence": case["evidence"],
            "upstream_output": case.get("upstream_output"),
            "verified_facts": case["verified_facts"],
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
                "semantic_quality_100": score.get("semantic_quality_100"),
                "semantic_exact": score.get("semantic_exact"),
                "decision_exact": score.get("decision_exact"),
                "primary_variable_exact": score.get("primary_variable_exact"),
                "confidence_in_band": score.get("confidence_in_band"),
                "model_action_agreement": score.get("model_action_agreement"),
                "model_semantic_escalation": score.get("model_semantic_escalation"),
                "upstream_copy_violation": score.get("upstream_copy_violation"),
                "boundary_projected": score.get("boundary_projected"),
                "system_effective_action": score.get("system_effective_action"),
                "system_action_exact": score.get("system_action_exact"),
                "system_semantic_escalation": score.get("system_semantic_escalation"),
                "boundary_machine_override": score.get("boundary_machine_override"),
                "boundary_reasons": score.get("boundary_reasons"),
            },
        })

    decision_matrix: dict[str, dict[str, int]] = defaultdict(dict)
    for (expected, predicted), count in sorted(decision_confusions.items()):
        decision_matrix[expected][predicted] = count
    primary_matrix: dict[str, dict[str, int]] = defaultdict(dict)
    for (expected, predicted), count in sorted(primary_confusions.items()):
        primary_matrix[expected][predicted] = count

    per_root_accuracy = {
        root: {
            "correct": expected_correct[root],
            "total": expected_totals[root],
            "decision_accuracy": expected_correct[root] / expected_totals[root],
        }
        for root in sorted(expected_totals)
    }

    report = {
        "persisted_rows": len(rows),
        "miss_union_count": len(misses),
        "semantic_miss_count": sum(not bool(row["score"].get("semantic_exact")) for row in rows),
        "decision_miss_count": sum(not bool(row["score"].get("decision_exact")) for row in rows),
        "primary_variable_miss_count": sum(not bool(row["score"].get("primary_variable_exact")) for row in rows),
        "system_action_miss_count": sum(not bool(row["score"].get("system_action_exact")) for row in rows),
        "failure_components": dict(sorted(failure_components.items())),
        "decision_confusion_matrix": dict(decision_matrix),
        "primary_variable_confusion_matrix": dict(primary_matrix),
        "per_expected_root_decision_accuracy": per_root_accuracy,
        "contexts_with_misses": dict(sorted(contexts_with_misses.items())),
        "misses": misses,
    }

    raw = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        Path(args.output).write_text(raw, encoding="utf-8")
    print(raw, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
