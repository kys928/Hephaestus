from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hephaestus.llm.role_probe import checked_cases, main
from hephaestus.providers.models.role_stack import REQUIRED_ROLES


def _case(tmp_path):
    evidence = tmp_path / "immutable-result.json"
    evidence.write_text('{"status":"completed","samples":56}', encoding="utf-8")
    case = tmp_path / "cases.json"
    case.write_text(json.dumps({
        "run_id": "D-run", "lineage_id": "evaluator", "stage_name": "smoke_test",
        "situations": {r: "Review these outputs with uncertainty." for r in REQUIRED_ROLES},
        "evidence": [{
            "ref": "artifact://completed-56", "source_file": str(evidence),
            "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
            "fact": "The saved manifest reports 56 examples.",
        }],
    }), encoding="utf-8")
    return case, evidence


def test_gpu_probe_preflights_all_five_roles_and_local_source_hashes(tmp_path):
    case, evidence = _case(tmp_path)
    run, lineage, stage, situations, facts, registry = checked_cases(case)
    assert (run, lineage, stage) == ("D-run", "evaluator", "smoke_test")
    assert set(situations) == set(REQUIRED_ROLES)
    assert facts[0].ref == "artifact://completed-56"
    assert registry.validate(["artifact://completed-56"],
                             run_id="D-run", lineage_id="evaluator").passed is True


def test_gpu_probe_refuses_modified_evidence_before_importing_gpu_and_s3(tmp_path):
    case, evidence = _case(tmp_path)
    evidence.write_text('{"status":"tampered"}', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        main([
            "--case-file", str(case),
            "--s3-endpoint", "https://s3api-eu-cz-1.runpod.io/",
            "--s3-bucket", "read-only",
            "--adapter-cache", str(tmp_path / "cache"),
            "--hf-cache", str(tmp_path / "hf"),
        ])


def test_gpu_probe_rejects_missing_role_situation(tmp_path):
    case, evidence = _case(tmp_path)
    payload = json.loads(case.read_text())
    payload["situations"].pop("judge")
    case.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="all five"):
        checked_cases(case)
