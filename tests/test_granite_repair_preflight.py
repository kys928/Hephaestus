"""CPU-free checks that the Granite repair preflight cannot claim success."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hephaestus.development.granite_repair_preflight import preflight
from hephaestus.diagnosis.pinned_historical import PinnedEvidenceError


def _write(path: Path, raw: bytes) -> str:
    path.write_bytes(raw)
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _inputs(tmp_path: Path):
    root = tmp_path / "historical"
    root.mkdir()
    data = b'{"evidence_kind":"eval_report","summary":"Old candidate failed", "signals":["eval_integrity_verified","undertraining_detected"]}'
    data_sha = _write(root / "old.json", data)
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"roles": {"evaluator": {
        "model_id": "ibm-granite/granite-3.3-8b-instruct",
        "revision": "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
        "adapter": {"sha256": "e60497854ea64655fc18be7962dec29f13091c75ea91e576726b0fa5c6eb2325"},
    }}}))
    index = tmp_path / "index.json"
    identity = {"schema": "granite-repair-historical-index.v1",
                "historical_exposed": True,
                "model_id": "ibm-granite/granite-3.3-8b-instruct",
                "model_revision": "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b",
                "adapter_sha256": "e60497854ea64655fc18be7962dec29f13091c75ea91e576726b0fa5c6eb2325",
                "files": {"old.json": data_sha}}
    index_sha = _write(index, json.dumps(identity).encode())
    return {"registry": reg, "index": index, "index_sha256": index_sha,
            "evidence_root": root, "run_id": "read-only-review",
            "lineage_id": "frozen-parent"}


def test_preflight_is_candid_about_no_model_and_no_training(tmp_path: Path):
    args = _inputs(tmp_path)
    result = preflight(**args)
    assert result["execution_kind"] == "read_only_deterministic_preflight"
    assert result["model_inference_performed"] is False
    assert result["training_updates"] == 0
    assert result["independent_certification"] is False
    assert result["scientific_repair_demonstrated"] is False
    assert result["diagnosis"]["status"] == "inconclusive"
    assert result["admissible_diagnostic_proposals"]
    assert all(p["intervention_kind"] == "collect_more_evidence"
               for p in result["admissible_diagnostic_proposals"])
    assert result["next_gate"] == "independently_verify_raw_evaluation_and_scoring_before_training"


def test_preflight_denies_rewritten_source(tmp_path: Path):
    args = _inputs(tmp_path)
    (args["evidence_root"] / "old.json").write_text("{}")
    with pytest.raises(PinnedEvidenceError, match="mismatch"):
        preflight(**args)


def test_preflight_denies_rewritten_index(tmp_path: Path):
    args = _inputs(tmp_path)
    args["index"].write_text("{}")
    with pytest.raises(PinnedEvidenceError, match="mismatch"):
        preflight(**args)


def test_preflight_denies_wrong_frozen_parent(tmp_path: Path):
    args = _inputs(tmp_path)
    old = json.loads(args["index"].read_text())
    old["model_revision"] = "substituted"
    args["index_sha256"] = _write(args["index"], json.dumps(old).encode())
    with pytest.raises(PinnedEvidenceError, match="frozen Evaluator identity"):
        preflight(**args)


def test_preflight_denies_index_lacking_exposure_status(tmp_path: Path):
    args = _inputs(tmp_path)
    old = json.loads(args["index"].read_text())
    old["historical_exposed"] = False
    args["index_sha256"] = _write(args["index"], json.dumps(old).encode())
    with pytest.raises(PinnedEvidenceError, match="exposed historical"):
        preflight(**args)


def test_preflight_denies_missing_or_unpinned_evidence(tmp_path: Path):
    args = _inputs(tmp_path)
    old = json.loads(args["index"].read_text())
    old["files"] = {"missing.json": "sha256:" + "0"*64}
    args["index_sha256"] = _write(args["index"], json.dumps(old).encode())
    with pytest.raises((PinnedEvidenceError, FileNotFoundError)):
        preflight(**args)


def test_preflight_denies_empty_historical_index(tmp_path: Path):
    args = _inputs(tmp_path)
    old = json.loads(args["index"].read_text())
    old["files"] = {}
    args["index_sha256"] = _write(args["index"], json.dumps(old).encode())
    with pytest.raises(PinnedEvidenceError, match="no pinned source"):
        preflight(**args)


def test_preflight_denies_joint_registry_and_index_substitution(tmp_path: Path):
    args = _inputs(tmp_path)
    catalog = json.loads(args["registry"].read_text())
    catalog["roles"]["evaluator"]["revision"] = "fake-revision"
    args["registry"].write_text(json.dumps(catalog))
    index = json.loads(args["index"].read_text())
    index["model_revision"] = "fake-revision"
    args["index_sha256"] = _write(args["index"], json.dumps(index).encode())
    with pytest.raises(PinnedEvidenceError, match="frozen Evaluator identity"):
        preflight(**args)
