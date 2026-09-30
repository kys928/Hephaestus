#!/usr/bin/env python3
"""Read-only inspection of one Interface Repair V2 run on the RunPod S3 volume."""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config

RUN_ID = (os.environ.get("HEPHAESTUS_INTERFACE_REPAIR_RUN_ID") or "").strip()
if not RUN_ID:
    raise SystemExit("HEPHAESTUS_INTERFACE_REPAIR_RUN_ID is required")

bucket = os.environ["RUNPOD_NETWORK_VOLUME_ID"]
endpoint = os.environ["RUNPOD_S3_ENDPOINT_URL"].rstrip("/")
region = os.environ["RUNPOD_DATACENTER_ID"]
client = boto3.client(
    "s3",
    endpoint_url=endpoint,
    region_name=region,
    aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
    config=Config(retries={"mode": "standard", "max_attempts": 10}),
)

prefix = f"hephaestus/scientific/v2/interface_repair/{RUN_ID}/"
roles = ["planner", "evaluator", "judge", "controller"]


def get_json(key: str) -> dict[str, Any] | None:
    try:
        r = client.get_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001
        data = getattr(exc, "response", {})
        code = str(data.get("Error", {}).get("Code", "")) if isinstance(data, dict) else ""
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise
    try:
        value = json.loads(r["Body"].read().decode("utf-8"))
    finally:
        r["Body"].close()
    return value if isinstance(value, dict) else None


def get_text(key: str) -> str:
    r = client.get_object(Bucket=bucket, Key=key)
    try:
        return r["Body"].read().decode("utf-8")
    finally:
        r["Body"].close()


def compact_result(value: dict[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    cert = value.get("certification") if isinstance(value.get("certification"), dict) else {}
    adapter = value.get("adapter") if isinstance(value.get("adapter"), dict) else {}
    def summary(name: str) -> dict[str, Any] | None:
        item = value.get(name)
        return dict(item) if isinstance(item, dict) else None
    return {
        "status": value.get("status"),
        "role": value.get("role"),
        "repo_sha": value.get("repo_sha"),
        "completed_at_unix": value.get("completed_at_unix"),
        "error_type": value.get("error_type"),
        "error": value.get("error"),
        "certification": dict(cert),
        "pre_interface": summary("pre_interface"),
        "post_interface": summary("post_interface"),
        "pre_regression": summary("pre_regression"),
        "post_regression": summary("post_regression"),
        "interface_quality_gain_100": value.get("interface_quality_gain_100"),
        "training": summary("training"),
        "adapter_sha256": adapter.get("sha256"),
        "adapter_s3_key": adapter.get("s3_key"),
    }


def parse_contract(text: str) -> dict[str, Any] | None:
    text = str(text or "").strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except Exception:
            return None
    return None


def analyze_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    false_counts: Counter[str] = Counter()
    true_counts: Counter[str] = Counter()
    decision_confusion: Counter[str] = Counter()
    action_confusion: Counter[str] = Counter()
    variable_confusion: Counter[str] = Counter()
    wrong_examples: list[dict[str, Any]] = []
    score_value_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        score = row.get("score") if isinstance(row.get("score"), dict) else {}
        for key, value in score.items():
            if isinstance(value, bool):
                (true_counts if value else false_counts)[key] += 1
            elif isinstance(value, (str, int, float)) or value is None:
                score_value_counts[key][str(value)] += 1

        expected = row.get("expected") if isinstance(row.get("expected"), dict) else {}
        generation = row.get("generation") if isinstance(row.get("generation"), dict) else {}
        observed = parse_contract(str(generation.get("projected_output", ""))) or {}
        for field, counter in (("decision", decision_confusion), ("action", action_confusion), ("primary_variable", variable_confusion)):
            e, o = str(expected.get(field)), str(observed.get(field))
            if e != o:
                counter[f"{e} -> {o}"] += 1

        failed = [k for k, v in score.items() if isinstance(v, bool) and not v]
        if failed and len(wrong_examples) < 20:
            wrong_examples.append({
                "case_id": row.get("case_id"),
                "kind": row.get("kind"),
                "failed_boolean_scores": failed,
                "score": score,
                "expected": {k: expected.get(k) for k in ("decision", "action", "primary_variable", "confidence_min", "confidence_max")},
                "observed": {k: observed.get(k) for k in ("decision", "action", "primary_variable", "confidence", "evidence_refs")},
            })

    return {
        "rows": len(rows),
        "false_boolean_scores": dict(false_counts.most_common()),
        "true_boolean_scores": dict(true_counts.most_common()),
        "decision_confusion": dict(decision_confusion.most_common()),
        "action_confusion": dict(action_confusion.most_common()),
        "primary_variable_confusion": dict(variable_confusion.most_common()),
        "scalar_score_distributions": {k: dict(v.most_common(20)) for k, v in score_value_counts.items()},
        "representative_failures": wrong_examples,
    }


out: dict[str, Any] = {"run_id": RUN_ID, "prefix": prefix, "roles": {}}
for role in roles:
    base = f"{prefix}roles/{role}"
    out["roles"][role] = {
        "result": compact_result(get_json(f"{base}/result.json")),
        "progress": get_json(f"{base}/progress.json"),
    }

for candidate in ("live/result.json", "result.json"):
    value = get_json(prefix + candidate)
    if value is not None:
        out[candidate] = value

objects: list[dict[str, Any]] = []
token = None
while True:
    kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
    if token:
        kwargs["ContinuationToken"] = token
    page = client.list_objects_v2(**kwargs)
    for obj in page.get("Contents", []):
        key = str(obj.get("Key", ""))
        objects.append({"key": key, "bytes": int(obj.get("Size", 0))})
    if not page.get("IsTruncated"):
        break
    token = page.get("NextContinuationToken")

out["object_count"] = len(objects)
out["object_bytes"] = sum(x["bytes"] for x in objects)
out["notable_objects"] = [x for x in objects if x["key"].endswith(("result.json", "progress.json", "selected-adapter.tar.gz"))]

evaluator_analysis: dict[str, Any] = {}
for phase in ("post_interface", "post_regression"):
    marker = f"{prefix}roles/evaluator/samples/{phase}/"
    shard_keys = sorted(x["key"] for x in objects if x["key"].startswith(marker) and x["key"].endswith(".jsonl"))
    rows: list[dict[str, Any]] = []
    for key in shard_keys:
        for line in get_text(key).splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    evaluator_analysis[phase] = {"shards": shard_keys, **analyze_rows(rows)}
out["evaluator_failure_analysis"] = evaluator_analysis

Path("interface_repair_v2_recovery_inspect.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(out, sort_keys=True))
