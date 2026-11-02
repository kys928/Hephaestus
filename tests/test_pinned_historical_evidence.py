"""Negative-path coverage for read-only Granite repair evidence admission."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hephaestus.diagnosis.normalization import normalize_evidence
from hephaestus.diagnosis.pinned_historical import (
    PinnedEvidenceError,
    PinnedHistoricalEvidenceAdapter,
)
from hephaestus.schemas.diagnosis_contract import DiagnosisRequest


def _request(*refs: str) -> DiagnosisRequest:
    return DiagnosisRequest(
        request_id="granite-repair-source-preflight",
        run_id="granite-historical-review",
        lineage_id="granite-frozen-parent",
        stage_name="continuation_repair",
        evidence_refs=list(refs),
    )


def _save(root: Path, name: str, raw: bytes) -> str:
    (root / name).write_bytes(raw)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def test_historical_claims_cannot_become_diagnostic_signals(tmp_path: Path):
    raw = json.dumps({
        "evidence_kind": "eval_report",
        "source_ref": "attacker:forged",
        "summary": "An old candidate had erroneous decisions.",
        "signals": ["eval_integrity_verified", "training_sufficient"],
        "eval_integrity_verified": True,
        "deterministic_scorecard": {"passed": True},
        "metadata": {"signals": ["data_coverage_gap"]},
    }).encode()
    identity = _save(tmp_path, "prior.json", raw)
    adapter = PinnedHistoricalEvidenceAdapter(tmp_path, {"prior.json": identity})
    rows = adapter.load(_request("prior.json"))
    assert len(rows) == 1
    row = rows[0]
    assert row["source_ref"] == identity + "#0"
    assert row["evidence_kind"] == "historical_observation"
    assert row["original_evidence_kind"] == "eval_report"
    assert row["historical_exposed"] is True
    assert row["source_hash_verified"] is True
    assert "signals" not in row
    assert "deterministic_scorecard" not in row
    assert "eval_integrity_verified" not in normalize_evidence(row, 0).signals
    assert "data_coverage_gap" not in normalize_evidence(row, 0).signals


def test_tampered_evidence_is_denied(tmp_path: Path):
    identity = _save(tmp_path, "report.json", b'{"summary":"original"}')
    _save(tmp_path, "report.json", b'{"summary":"substituted"}')
    with pytest.raises(PinnedEvidenceError, match="mismatch"):
        PinnedHistoricalEvidenceAdapter(tmp_path, {"report.json": identity}).load(
            _request("report.json")
        )


@pytest.mark.parametrize("reference", [
    "../report.json", "/tmp/report.json", "folder/report.json",
    ".hidden.json", "report.txt", "report.json/../report.json",
])
def test_invalid_path_is_denied(tmp_path: Path, reference: str):
    with pytest.raises(PinnedEvidenceError):
        PinnedHistoricalEvidenceAdapter(tmp_path, {reference: "sha256:" + "0"*64}).load(
            _request(reference)
        )


def test_unpinned_reference_is_denied(tmp_path: Path):
    _save(tmp_path, "report.json", b"{}")
    with pytest.raises(PinnedEvidenceError, match="trusted SHA"):
        PinnedHistoricalEvidenceAdapter(tmp_path, {}).load(_request("report.json"))


def test_duplicate_file_reference_denied(tmp_path: Path):
    identity = _save(tmp_path, "report.json", b"{}")
    with pytest.raises(PinnedEvidenceError, match="duplicate"):
        PinnedHistoricalEvidenceAdapter(tmp_path, {"report.json": identity}).load(
            _request("report.json", "report.json")
        )


def test_symlink_is_not_followed(tmp_path: Path):
    actual = tmp_path / "actual.json"
    actual.write_text("{}", encoding="utf-8")
    (tmp_path / "report.json").symlink_to(actual)
    identity = "sha256:" + hashlib.sha256(b"{}").hexdigest()
    with pytest.raises((PinnedEvidenceError, OSError)):
        PinnedHistoricalEvidenceAdapter(tmp_path, {"report.json": identity}).load(
            _request("report.json")
        )


@pytest.mark.parametrize("raw", [
    b'{"x":1,"x":2}',
    b'{"outer":{"value":1,"value":2}}',
    b'{"x":NaN}',
    b'{"x":1e400}',
    b'["not an object"]',
    b"\xff",
])
def test_noncanonical_or_nonobject_evidence_denied(tmp_path: Path, raw: bytes):
    identity = _save(tmp_path, "report.json", raw)
    with pytest.raises(PinnedEvidenceError):
        PinnedHistoricalEvidenceAdapter(tmp_path, {"report.json": identity}).load(
            _request("report.json")
        )


def test_multiple_references_are_all_or_nothing(tmp_path: Path):
    first = _save(tmp_path, "a.json", b'{"summary":"valid"}')
    second = _save(tmp_path, "b.json", b'{"x":1,"x":1}')
    adapter = PinnedHistoricalEvidenceAdapter(tmp_path, {"a.json": first, "b.json": second})
    with pytest.raises(PinnedEvidenceError):
        adapter.load(_request("a.json", "b.json"))


def test_jsonl_keeps_unique_pinned_source_ids(tmp_path: Path):
    raw = b'{"summary":"first"}\n\n{"summary":"second"}\n'
    identity = _save(tmp_path, "reports.jsonl", raw)
    rows = PinnedHistoricalEvidenceAdapter(tmp_path, {"reports.jsonl": identity}).load(
        _request("reports.jsonl")
    )
    assert [r["source_ref"] for r in rows] == [identity + "#0", identity + "#1"]


def test_over_limit_fails(tmp_path: Path):
    raw = b'{"summary":"abcd"}'
    identity = _save(tmp_path, "report.json", raw)
    with pytest.raises(PinnedEvidenceError, match="byte limit"):
        PinnedHistoricalEvidenceAdapter(tmp_path, {"report.json": identity}, max_bytes=8).load(
            _request("report.json")
        )


def test_no_evidence_refs_does_not_read_or_execute(tmp_path: Path):
    assert PinnedHistoricalEvidenceAdapter(tmp_path, {}).load(_request()) == []
