"""Research archive custody checks; no GitHub mutation in these tests."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("hf_findings_freeze",
    Path(__file__).resolve().parents[1] / "scripts/freeze_governed_hf_findings.py")
freeze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze)


def _evidence(root, **changes):
    report = {key: False for key in ("new_full_campaign_execution", "corpus_access_authorized",
        "corpus_acquired", "training_executed", "scientific_certificate_issued")}
    report.update(candidate_reviews=[{}, {}, {}], workflow_state={"steps": {
        "data_acquisition_audit.dataset_discovery": {"completion_marker": True},
        "data_acquisition_audit.dataset_selection": {"completion_marker": False}}})
    report.update(changes)
    data = json.dumps(report).encode()
    (root / "report.json").write_bytes(data)
    (root / "sha256-index.json").write_text(json.dumps({"report.json": hashlib.sha256(data).hexdigest()}))


def test_valid_scoped_research_findings_can_be_frozen(tmp_path):
    _evidence(tmp_path)
    assert freeze.verified_findings(tmp_path)["corpus_acquired"] is False


def test_changed_evidence_and_incomplete_index_are_rejected(tmp_path):
    _evidence(tmp_path)
    (tmp_path / "report.json").write_text('{}')
    with pytest.raises(ValueError, match="custody index"):
        freeze.verified_findings(tmp_path)
    _evidence(tmp_path)
    (tmp_path / "unindexed.json").write_text('{}')
    with pytest.raises(ValueError, match="exactly"):
        freeze.verified_findings(tmp_path)


def test_research_freeze_cannot_claim_training_or_certification(tmp_path):
    _evidence(tmp_path, scientific_certificate_issued=True)
    with pytest.raises(ValueError, match="certification"):
        freeze.verified_findings(tmp_path)
