"""Offline fake-S3 gates for the original certified six-state Granite parent audit."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

from hephaestus.diagnosis.pinned_historical import PinnedEvidenceError
from hephaestus.evaluation.granite_dev_scoring import PRODUCTION_ONTOLOGY

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_granite_original_parent_dev_readonly.py"
SPEC = importlib.util.spec_from_file_location("granite_original_parent_dev", SCRIPT)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class FakeS3:
    def __init__(self, objects):
        self.objects = objects
        self.reads = []

    def get_object(self, *, Bucket, Key):
        assert Bucket == audit.BUCKET
        self.reads.append(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, **kwargs):
        raise AssertionError("only S3 GET is allowed")

    def list_objects_v2(self, **kwargs):
        raise AssertionError("arbitrary S3 listing is forbidden")


def _make():
    states = sorted(audit.PRODUCTION_DECISIONS)
    cases = [
        {
            "case_id": f"EVALUATOR-DEV-{i:05d}",
            "root_case_id": f"E{i%10:02d}",
            "skill": f"skill_{i%10:02d}",
            "expected_decision": states[i % len(states)],
            "expected_action": "retain_baseline",
            "expected_primary_variable": "candidate_quality",
        }
        for i in range(200)
    ]
    manifest = {
        "schema": "granite-original-parent-exposed-development.v1",
        "source_repo_sha": audit.ORIGINAL_SOURCE_SHA,
        "training_pack_sha256": audit.ORIGINAL_PACK_SHA,
        "ontology": PRODUCTION_ONTOLOGY,
        "model_id": audit.ORIGINAL_MODEL_ID,
        "model_revision": audit.ORIGINAL_MODEL_REVISION,
        "selected_adapter_sha256": audit.ORIGINAL_ADAPTER_SHA,
        "selected_dev_step": 250,
        "development_cases": cases,
    }
    rows = []
    for case in cases:
        answer = json.dumps({
            "decision": case["expected_decision"], "action": case["expected_action"],
            "primary_variable": case["expected_primary_variable"],
            "confidence": 0.8, "evidence_refs": ["E:valid"],
            "uncertainties": [], "rationale": "The evidence supports the specified state.",
        })
        rows.append({
            "case_id": case["case_id"], "root_case_id": case["root_case_id"],
            "skill": case["skill"], "phase": "dev_step_250",
            "raw_output": answer, "output": answer, "generation": {"finish_reason": "eos"},
            "score": {
                "quality_100": 100.0, "schema_compliant": True,
                "components": {"decision": 1.0, "evidence_grounding": 1.0},
            },
        })
    summary = {"sample_count": 200, "quality_100": 100.0,
               "schema_compliance": 1.0, "evidence_grounding": 1.0}
    result = {
        "status": "completed", "role": "evaluator", "run_id": audit.RUN_ID,
        "repo_sha": audit.ORIGINAL_SOURCE_SHA, "pack_sha256": audit.ORIGINAL_PACK_SHA,
        "model_id": audit.ORIGINAL_MODEL_ID, "revision": audit.ORIGINAL_MODEL_REVISION,
        "selected_dev_step": 250, "selected_adapter": {
            "s3_key": audit.ROOT + "selected-adapter.tar.gz",
            "sha256": audit.ORIGINAL_ADAPTER_SHA, "bytes": 61503144,
        },
        "dev_checkpoints": [
            {"step": 125, "summary": summary},
            {"step": 250, "summary": summary},
            {"step": 375, "summary": summary},
        ],
    }
    def bundle(rows, result):
        return FakeS3({
            audit.RESULT_KEY: json.dumps(result).encode("utf-8"),
            audit.DEV_KEY: "".join(json.dumps(x) + "\n" for x in rows).encode("utf-8"),
        })
    return bundle, manifest, rows, result


def test_genuine_structure_recomputes_original_six_state_development():
    bundle, manifest, rows, result = _make()
    client = bundle(rows, result)
    report = audit.rescore_parent_development(client, manifest)
    assert client.reads == [audit.RESULT_KEY, audit.DEV_KEY]
    assert report["count"] == 200
    assert report["strict_projected_decision_correct"] == 200
    assert report["strict_raw_decision_correct"] == 200
    assert report["original_saved_scoring"]["per_case_decision_score_disagreements"] == 0
    assert report["original_saved_scoring"]["matches_original_selected_development_summary"]
    assert report["ontology"] == PRODUCTION_ONTOLOGY
    assert report["original_300_case_certification_not_read"]
    assert report["research_development_not_independent_final"]
    assert report["source_bytes_not_previously_independently_hashed"]
    assert not report["model_weights_newly_rehashed"]
    assert report["training_updates"] == 0
    assert report["certification"] is False
    assert "The evidence supports" not in json.dumps(report)


def test_cannot_read_certification_or_any_nondev_object():
    bundle, manifest, rows, result = _make()
    client = bundle(rows, result)
    with pytest.raises(PermissionError):
        audit._s3_get(client, audit.ROOT + "certification.jsonl")
    assert client.reads == []


@pytest.mark.parametrize("key,value", [
    ("selected_dev_step", 375),
    ("model_revision", "different-frozen-revision"),
    ("selected_adapter_sha256", "0" * 64),
    ("ontology", "historical-semantic-invariance-evaluator.v1"),
    ("training_pack_sha256", "0" * 64),
])
def test_manifest_substitution_denied_before_get(key, value):
    bundle, manifest, rows, result = _make()
    manifest[key] = value
    client = bundle(rows, result)
    with pytest.raises(PinnedEvidenceError):
        audit.rescore_parent_development(client, manifest)
    assert client.reads == []


def test_original_result_wrong_selected_adapter_denied():
    bundle, manifest, rows, result = _make()
    result["selected_adapter"]["sha256"] = "0" * 64
    with pytest.raises(PinnedEvidenceError, match="selected parent adapter"):
        audit.rescore_parent_development(bundle(rows, result), manifest)


def test_missing_or_duplicate_development_predictions_denied():
    bundle, manifest, rows, result = _make()
    rows[0]["case_id"] = rows[1]["case_id"]
    with pytest.raises(PinnedEvidenceError):
        audit.rescore_parent_development(bundle(rows, result), manifest)


def test_wrong_saved_evaluation_score_denied():
    bundle, manifest, rows, result = _make()
    rows[0]["score"]["components"]["decision"] = 0
    with pytest.raises(PinnedEvidenceError, match="scores differ"):
        audit.rescore_parent_development(bundle(rows, result), manifest)


def test_mismatched_aggregate_summary_denied():
    bundle, manifest, rows, result = _make()
    result["dev_checkpoints"][1]["summary"]["quality_100"] = 50
    with pytest.raises(PinnedEvidenceError, match="summary"):
        audit.rescore_parent_development(bundle(rows, result), manifest)


def test_wrong_case_root_denied():
    bundle, manifest, rows, result = _make()
    rows[0]["root_case_id"] = "attacker"
    with pytest.raises(PinnedEvidenceError, match="case/skill/phase"):
        audit.rescore_parent_development(bundle(rows, result), manifest)


def test_duplicate_reference_denied():
    bundle, manifest, rows, result = _make()
    manifest["development_cases"][1]["case_id"] = manifest["development_cases"][0]["case_id"]
    with pytest.raises(PinnedEvidenceError, match="duplicated"):
        audit.rescore_parent_development(bundle(rows, result), manifest)


def test_report_does_not_include_original_generated_text():
    bundle, manifest, rows, result = _make()
    rows[0]["output"] = "unique-secret-generation-string"
    client = bundle(rows, result)
    with pytest.raises(PinnedEvidenceError):
        audit.rescore_parent_development(client, manifest)
