"""Re-score ONLY original certified Granite Evaluator's exposed development samples.

This is read-only forensic verification of the *selected parent* at step 250,
not a new inference, a fresh validation set, or an assessment of a candidate.
The original 300-case certification partition must never be read by this tool.
No training, GPU, S3 write, active model routing, or promotion.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Mapping

from hephaestus.diagnosis.pinned_historical import PinnedEvidenceError, _parse_json
from hephaestus.evaluation.granite_dev_scoring import (
    PRODUCTION_DECISIONS, PRODUCTION_ONTOLOGY, _parse_generated,
)

ORIGINAL_SOURCE_SHA = "c831bbfc0e2054d54fccd1ef374fd84722612c79"
ORIGINAL_PACK_SHA = "6f4bb3b8c8312a4189136ea961e8fa32629c02fa5459e18fa960670badb203b3"
ORIGINAL_MODEL_ID = "ibm-granite/granite-3.3-8b-instruct"
ORIGINAL_MODEL_REVISION = "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b"
ORIGINAL_ADAPTER_SHA = "e60497854ea64655fc18be7962dec29f13091c75ea91e576726b0fa5c6eb2325"
RUN_ID = "role-mastery-v1-35596791201"
BUCKET = "cviwpryzao"
ROOT = "hephaestus/scientific/v1/role_mastery/" + RUN_ID + "/roles/evaluator/"
DEV_KEY = ROOT + "dev/step-0250.jsonl"
RESULT_KEY = ROOT + "result.json"
ADMITTED_KEYS = frozenset({DEV_KEY, RESULT_KEY})
MAX_DEV_BYTES = 8 * 1024 * 1024
MAX_RESULT_BYTES = 2 * 1024 * 1024
EXPECTED_DEV_COUNT = 200
REQUIRED_MANIFEST_KEYS = frozenset({
    "schema", "source_repo_sha", "training_pack_sha256",
    "ontology", "model_id", "model_revision", "selected_adapter_sha256",
    "selected_dev_step", "development_cases",
})
REQUIRED_CASE_KEYS = frozenset({
    "case_id", "root_case_id", "skill",
    "expected_decision", "expected_action", "expected_primary_variable",
})


def _s3_get(client, key: str) -> tuple[bytes, dict[str, object]]:
    if key not in ADMITTED_KEYS:
        raise PermissionError("original certification/raw holdout S3 access forbidden")
    limit = MAX_DEV_BYTES if key == DEV_KEY else MAX_RESULT_BYTES
    response = client.get_object(Bucket=BUCKET, Key=key)
    body = response["Body"]
    try:
        raw = body.read(limit + 1)
    finally:
        body.close()
    if not isinstance(raw, bytes) or not raw or len(raw) > limit:
        raise PinnedEvidenceError("original development evidence absent or over size limit")
    return raw, {
        "s3_key": key, "bytes": len(raw),
        "observed_sha256": hashlib.sha256(raw).hexdigest(),
        "sha_precommitted_independently": False,
    }


def _jsonl(raw: bytes) -> list[dict]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PinnedEvidenceError("non-UTF8 original development rows") from exc
    records = [_parse_json(line) for line in text.splitlines() if line.strip()]
    if len(records) != EXPECTED_DEV_COUNT or any(not isinstance(x, dict) for x in records):
        raise PinnedEvidenceError("original selected development row count/shape invalid")
    return records


def _references(manifest: object) -> dict[str, dict]:
    if not isinstance(manifest, dict) or set(manifest) != REQUIRED_MANIFEST_KEYS:
        raise PinnedEvidenceError("unrecognized frozen original development manifest")
    expected = {
        "schema": "granite-original-parent-exposed-development.v1",
        "source_repo_sha": ORIGINAL_SOURCE_SHA,
        "training_pack_sha256": ORIGINAL_PACK_SHA,
        "ontology": PRODUCTION_ONTOLOGY,
        "model_id": ORIGINAL_MODEL_ID,
        "model_revision": ORIGINAL_MODEL_REVISION,
        "selected_adapter_sha256": ORIGINAL_ADAPTER_SHA,
        "selected_dev_step": 250,
    }
    for key, value in expected.items():
        if type(manifest[key]) is not type(value) or manifest[key] != value:
            raise PinnedEvidenceError("original parent identity, pack, or ontology mismatch: " + key)
    cases = manifest["development_cases"]
    if not isinstance(cases, list) or len(cases) != EXPECTED_DEV_COUNT:
        raise PinnedEvidenceError("original parent development case count mismatch")
    result: dict[str, dict] = {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != REQUIRED_CASE_KEYS:
            raise PinnedEvidenceError("malformed original development reference")
        identity = case["case_id"]
        if not isinstance(identity, str) or not identity.startswith("EVALUATOR-DEV-") or identity in result:
            raise PinnedEvidenceError("duplicated or incompatible original case ID")
        if any(not isinstance(case[k], str) or not case[k] for k in REQUIRED_CASE_KEYS):
            raise PinnedEvidenceError("non-string original reference metadata")
        if case["expected_decision"] not in PRODUCTION_DECISIONS:
            raise PinnedEvidenceError("original development expected state outside production ontology")
        result[identity] = case
    return result


def _original_result(value: object) -> tuple[dict, dict]:
    if not isinstance(value, dict) or value.get("status") != "completed":
        raise PinnedEvidenceError("original role mastery result is not completed")
    fields = {
        "role": "evaluator", "run_id": RUN_ID, "repo_sha": ORIGINAL_SOURCE_SHA,
        "pack_sha256": ORIGINAL_PACK_SHA, "model_id": ORIGINAL_MODEL_ID,
        "revision": ORIGINAL_MODEL_REVISION, "selected_dev_step": 250,
    }
    for key, expected in fields.items():
        if value.get(key) != expected:
            raise PinnedEvidenceError("original frozen role result identity mismatch: " + key)
    adapter = value.get("selected_adapter")
    if not isinstance(adapter, dict) or (
        adapter.get("sha256") != ORIGINAL_ADAPTER_SHA
        or adapter.get("s3_key") != ROOT + "selected-adapter.tar.gz"
        or adapter.get("bytes") != 61503144
    ):
        raise PinnedEvidenceError("selected parent adapter does not match independently certified registry")
    checkpoints = value.get("dev_checkpoints")
    if not isinstance(checkpoints, list) or [c.get("step") for c in checkpoints] != [125, 250, 375]:
        raise PinnedEvidenceError("original development checkpoint inventory drift")
    selected = checkpoints[1].get("summary")
    if not isinstance(selected, dict) or selected.get("sample_count") != EXPECTED_DEV_COUNT:
        raise PinnedEvidenceError("original selected development scorecard missing")
    return value, selected


def _finite_number(value: object) -> float:
    if type(value) not in (float, int) or not math.isfinite(value):
        raise PinnedEvidenceError("original saved score not a finite number")
    return float(value)


def rescore_parent_development(client, manifest: object) -> dict[str, object]:
    """Recalculate exact six-state decisions; never consult certification cases."""
    references = _references(manifest)
    raw_result, result_receipt = _s3_get(client, RESULT_KEY)
    raw_dev, dev_receipt = _s3_get(client, DEV_KEY)
    try:
        result_json = _parse_json(raw_result.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise PinnedEvidenceError("original role result is not UTF-8") from exc
    _result, selected_summary = _original_result(result_json)
    observed = _jsonl(raw_dev)
    if len({r.get("case_id") for r in observed}) != EXPECTED_DEV_COUNT:
        raise PinnedEvidenceError("duplicate original development predictions")
    if {r["case_id"] for r in observed} != set(references):
        raise PinnedEvidenceError("saved original predictions do not match frozen development cases")
    confusion: dict[str, Counter[str]] = {k: Counter() for k in sorted(PRODUCTION_DECISIONS)}
    by_skill: dict[str, Counter[str]] = {}
    invalid: Counter[str] = Counter()
    original_quality: list[float] = []
    recorded_decision_drift = 0
    normalized_strict_correct = 0
    raw_strict_correct = 0
    saved_schema_count = 0
    saved_grounding: list[float] = []
    for entry in observed:
        gold = references[entry["case_id"]]
        if (entry.get("skill") != gold["skill"] or entry.get("root_case_id") != gold["root_case_id"]
                or entry.get("phase") != "dev_step_250"):
            raise PinnedEvidenceError("original saved row case/skill/phase identity mismatch")
        if not isinstance(entry.get("raw_output"), str) or not isinstance(entry.get("output"), str):
            raise PinnedEvidenceError("original raw/projected output missing")
        score = entry.get("score")
        if not isinstance(score, dict) or not isinstance(score.get("components"), dict):
            raise PinnedEvidenceError("original scored row incomplete")
        stored_decision = _finite_number(score["components"].get("decision"))
        if stored_decision not in {0.0, 1.0}:
            raise PinnedEvidenceError("original decision score outside Boolean domain")
        original_quality.append(_finite_number(score.get("quality_100")))
        if type(score.get("schema_compliant")) is not bool:
            raise PinnedEvidenceError("invalid persisted schema-compliance flag")
        saved_schema_count += int(score["schema_compliant"])
        saved_grounding.append(_finite_number(score["components"].get("evidence_grounding")))
        expected = gold["expected_decision"]
        projected, error = _parse_generated(entry["output"])
        raw, raw_error = _parse_generated(entry["raw_output"])
        candidate = projected["decision"] if projected is not None and error is None else "<invalid>"
        if candidate not in PRODUCTION_DECISIONS:
            candidate = "<invalid>"
            invalid[error or "decision_outside_ontology"] += 1
        is_correct = candidate == expected
        normalized_strict_correct += int(is_correct)
        raw_strict_correct += int(raw_error is None and raw is not None and raw["decision"] == expected)
        recorded_decision_drift += int(stored_decision != float(is_correct))
        confusion[expected][candidate] += 1
        by_skill.setdefault(gold["skill"], Counter())["total"] += 1
        by_skill[gold["skill"]]["correct"] += int(is_correct)

    recomputed_quality = sum(original_quality) / EXPECTED_DEV_COUNT
    recomputed_schema = saved_schema_count / EXPECTED_DEV_COUNT
    recomputed_grounding = sum(saved_grounding) / EXPECTED_DEV_COUNT
    recorded = {
        "quality_100": _finite_number(selected_summary.get("quality_100")),
        "schema_compliance": _finite_number(selected_summary.get("schema_compliance")),
        "evidence_grounding": _finite_number(selected_summary.get("evidence_grounding")),
    }
    comparison = {
        "quality_100": recomputed_quality,
        "schema_compliance": recomputed_schema,
        "evidence_grounding": recomputed_grounding,
    }
    summary_matches = all(
        math.isclose(comparison[key], recorded[key], rel_tol=1e-10, abs_tol=1e-8)
        for key in comparison
    )
    if not summary_matches or recorded_decision_drift:
        raise PinnedEvidenceError("original individual scores differ from saved selected-development summary")
    total = EXPECTED_DEV_COUNT
    return {
        "schema": "prometheus-original-certified-granite-dev-rescore.v1",
        "source_run_id": RUN_ID,
        "original_training_source_sha": ORIGINAL_SOURCE_SHA,
        "original_training_pack_sha256": ORIGINAL_PACK_SHA,
        "original_model_id": ORIGINAL_MODEL_ID,
        "original_model_revision": ORIGINAL_MODEL_REVISION,
        "original_selected_adapter_sha256": ORIGINAL_ADAPTER_SHA,
        "original_selected_dev_step": 250,
        "ontology": PRODUCTION_ONTOLOGY,
        "partition": "previously_exposed_model_selection_development",
        "count": total,
        "strict_projected_decision_correct": normalized_strict_correct,
        "strict_projected_decision_accuracy": normalized_strict_correct / total,
        "strict_raw_decision_correct": raw_strict_correct,
        "per_decision": {
            label: {
                "count": sum(confusion[label].values()),
                "correct": confusion[label][label],
                "confusion": dict(sorted(confusion[label].items())),
            } for label in sorted(PRODUCTION_DECISIONS)
        },
        "per_skill": {
            name: dict(sorted(counts.items())) for name, counts in sorted(by_skill.items())
        },
        "invalid_projected_reasons": dict(sorted(invalid.items())),
        "original_saved_scoring": {
            "mean_quality_100": recomputed_quality,
            "schema_compliance": recomputed_schema,
            "evidence_grounding": recomputed_grounding,
            "per_case_decision_score_disagreements": recorded_decision_drift,
            "matches_original_selected_development_summary": summary_matches,
        },
        "source_receipts": [result_receipt, dev_receipt],
        "source_bytes_not_previously_independently_hashed": True,
        "source_model_adapter_claim_matched_archived_certification": True,
        "model_weights_newly_rehashed": False,
        "original_300_case_certification_not_read": True,
        "research_development_not_independent_final": True,
        "original_parent_not_historical_eight_state_D": True,
        "new_model_inference": False,
        "training_updates": 0,
        "certification": False,
        "promotion": False,
        "causal_repair_hypothesis_established": False,
    }


def main() -> int:
    import boto3
    from botocore.config import Config

    if any(not os.environ.get(k) for k in (
        "RUNPOD_S3_ACCESS_KEY_ID", "RUNPOD_S3_SECRET_ACCESS_KEY"
    )):
        raise PermissionError("existing read-only S3 credentials unavailable")
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.references.read_bytes()
    if len(source) > 128 * 1024:
        raise PinnedEvidenceError("original reference manifest too large")
    try:
        manifest = _parse_json(source.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise PinnedEvidenceError("original reference manifest invalid UTF-8") from exc
    client = boto3.client(
        "s3", endpoint_url="https://s3api-eu-cz-1.runpod.io/",
        region_name="EU-CZ-1",
        aws_access_key_id=os.environ["RUNPOD_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["RUNPOD_S3_SECRET_ACCESS_KEY"],
        config=Config(retries={"mode": "standard", "max_attempts": 3}),
    )
    report = rescore_parent_development(client, manifest)
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print("PROMETHEUS_ORIGINAL_EVALUATOR_DEV_RESCORING", json.dumps({
        "cases": report["count"],
        "projected_correct": report["strict_projected_decision_correct"],
        "raw_correct": report["strict_raw_decision_correct"],
        "stored_score_drift": report["original_saved_scoring"]["per_case_decision_score_disagreements"],
        "matches_original_summary": report["original_saved_scoring"]["matches_original_selected_development_summary"],
        "certification_cases_read": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
