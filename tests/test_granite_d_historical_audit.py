"""No-network tests for the Prometheus original D read-only audit."""
import hashlib
import io
import json

import pytest

from hephaestus.evaluation.granite_dev_scoring import RESEARCH_ONTOLOGY
from hephaestus.diagnosis.pinned_historical import PinnedEvidenceError
import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_granite_d_historical_readonly.py"
_SPEC = importlib.util.spec_from_file_location("prometheus_historical_audit", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


class _FakeS3:
    def __init__(self, values):
        self.values = values
        self.reads = []

    def get_object(self, *, Bucket, Key):
        assert Bucket == audit.BUCKET
        self.reads.append(Key)
        return {"Body": io.BytesIO(self.values[Key])}

    def put_object(self, **kwargs):
        raise AssertionError("S3 mutation forbidden")

    def list_objects_v2(self, **kwargs):
        raise AssertionError("unscoped S3 listing forbidden")


def _fixture(monkeypatch):
    labels = list(audit.ARCHIVED_STATE_CORRECT)
    refs = [{"case_id": f"case-{i:02d}", "expected_decision": labels[i % 8]}
            for i in range(64)]
    screen = hashlib.sha256(json.dumps([x["case_id"] for x in refs],
                                        separators=(",", ":")).encode()).hexdigest()
    monkeypatch.setattr(audit, "SCREEN_ORDER_SHA", screen)
    archive_state = {label: 8 for label in labels}
    monkeypatch.setattr(audit, "ARCHIVED_STATE_CORRECT", archive_state)
    monkeypatch.setattr(audit, "ARCHIVED_CORRECT", 64)
    manifest = {
        "schema": "granite-d-exposed-screen-references.v1",
        "source_repo_sha": audit.D_SOURCE_SHA,
        "development_sha256": audit.DEVELOPMENT_SHA,
        "screen_case_order_sha256": screen,
        "ontology": RESEARCH_ONTOLOGY,
        "references": refs,
    }
    rows = []
    for r in refs:
        raw = json.dumps({
            "decision": r["expected_decision"], "action": "retain_baseline",
            "primary_variable": "candidate_quality", "confidence": 0.75,
            "evidence_refs": ["dev:case"], "uncertainties": [],
            "rationale": "The exposed diagnostic observations support this state.",
        })
        rows.append({"case_id": r["case_id"], "generation": {"projected_output": raw},
                     "score": {"decision_exact": True}})
    objects = {}
    shards = []
    for i in range(2):
        key = audit.BASE + f"shard-{i+1:04d}.json"
        blob = json.dumps(rows[i*32:(i+1)*32]).encode()
        objects[key] = blob
        shards.append((key, len(blob), hashlib.sha256(blob).hexdigest()))
    monkeypatch.setattr(audit, "SHARDS", tuple(shards))
    return _FakeS3(objects), manifest, shards, rows


def test_cold_saved_generations_rescore_and_no_raw_leak(monkeypatch):
    fake, manifest, shards, _ = _fixture(monkeypatch)
    result = audit.rescore_readonly(fake, manifest)
    assert len(fake.reads) == 2
    assert result["independent_strict_scoring"]["count"] == 64
    assert result["independent_strict_scoring"]["decision_correct"] == 64
    assert result["recorded_score_comparison"]["strict_per_case_score_disagreements"] == 0
    assert result["recorded_score_comparison"]["by_state_matches_archive"] is True
    assert result["no_raw_generations_published"] is True
    assert result["scientific_certification"] is False
    assert result["model_weights_independently_verified"] is False
    assert "dev:case" not in json.dumps(result)
    assert "diagnostic observations support" not in json.dumps(result)


def test_sha_tampering_is_blocked(monkeypatch):
    fake, manifest, shards, _ = _fixture(monkeypatch)
    fake.values[shards[0][0]] = fake.values[shards[0][0]].replace(b"case-00", b"case-XX")
    with pytest.raises(PinnedEvidenceError, match="byte identity"):
        audit.rescore_readonly(fake, manifest)


def test_missing_and_duplicate_cases_are_blocked(monkeypatch):
    fake, manifest, shards, rows = _fixture(monkeypatch)
    rows[1]["case_id"] = rows[0]["case_id"]
    blob = json.dumps(rows[:32]).encode()
    fake.values[shards[0][0]] = blob
    new_shards = ((shards[0][0], len(blob), hashlib.sha256(blob).hexdigest()), shards[1])
    with pytest.raises(PinnedEvidenceError, match="incomplete or duplicated"):
        audit.rescore_readonly(fake, manifest, shards=new_shards)


def test_invalid_reference_ontology_denied_before_s3_access(monkeypatch):
    fake, manifest, _, _ = _fixture(monkeypatch)
    manifest["ontology"] = "frozen-role-mastery-evaluator.v1"
    with pytest.raises(PinnedEvidenceError, match="source identity"):
        audit.rescore_readonly(fake, manifest)
    assert not fake.reads


def test_invalid_reference_order_denied(monkeypatch):
    fake, manifest, _, _ = _fixture(monkeypatch)
    manifest["references"] = list(reversed(manifest["references"]))
    with pytest.raises(PinnedEvidenceError, match="ordering"):
        audit.rescore_readonly(fake, manifest)
    assert not fake.reads


def test_s3_scope_is_not_caller_controlled(monkeypatch):
    fake, manifest, _, _ = _fixture(monkeypatch)
    with pytest.raises(PermissionError):
        audit.rescore_readonly(fake, manifest, shards=((
            "hephaestus/private/final-holdout.json", 10, "0" * 64,
        ),))


def test_historical_scores_can_disagree_with_stricter_parser(monkeypatch):
    fake, manifest, shards, rows = _fixture(monkeypatch)
    rows[0]["generation"]["projected_output"] = "NOT JSON"
    blob = json.dumps(rows[:32]).encode()
    fake.values[shards[0][0]] = blob
    new_shards = ((shards[0][0], len(blob), hashlib.sha256(blob).hexdigest()), shards[1])
    result = audit.rescore_readonly(fake, manifest, shards=new_shards)
    assert result["independent_strict_scoring"]["decision_correct"] == 63
    assert result["recorded_score_comparison"]["strict_per_case_score_disagreements"] == 1
    assert result["recorded_score_comparison"]["by_state_matches_archive"] is True
