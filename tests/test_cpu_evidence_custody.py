import hashlib
import json
import sqlite3
import pytest
import importlib.util
from pathlib import Path
_module_spec = importlib.util.spec_from_file_location("cpu_cold_custody", Path(__file__).resolve().parents[1] / "scripts" / "retain_cpu_campaign_evidence.py")
_module = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(_module)
verify_original_evidence, retain = _module.verify_original_evidence, _module.retain


def make_workspace(root):
    root.mkdir()
    (root/"artifacts").mkdir()
    sqlite3.connect(root/"state.sqlite").execute("CREATE TABLE IF NOT EXISTS proof (k text)").connection.close()
    (root/"artifacts"/"receipt.json").write_text('{"real":true}')
    (root/"cpu_campaign_report.json").write_text(json.dumps({
        "workflow_id": "wf", "status": "completed", "completed_operations": 45,
        "evidence": {"verified_operations": 45, "verified_domain_operations": 40,
                     "claim_scope": "internal_prospective_generated_task_research",
                     "independent_scientific_certification": False,
                     "five_frozen_cuda_roles_executed": False}}))
    files = {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob("*")) if p.is_file()}
    (root/"archive_sha256.json").write_text(json.dumps(files, sort_keys=True))
    return root


def test_copy_preserves_original_verified_cold_evidence(tmp_path):
    original = make_workspace(tmp_path/"original")
    target = tmp_path/"retained"
    receipt = retain(original, target)
    assert receipt == verify_original_evidence(target)
    assert receipt["member_count"] == 3
    assert receipt["resumable_from_copied_path"] is False
    with pytest.raises(ValueError):
        retain(original, target)


def test_source_tampering_and_extra_members_fail_closed(tmp_path):
    original = make_workspace(tmp_path/"original")
    (original/"artifacts"/"receipt.json").write_text('{"forged":true}')
    with pytest.raises(PermissionError, match="SHA256"):
        verify_original_evidence(original)
    original = make_workspace(tmp_path/"other")
    (original/"artifacts"/"not-in-manifest.txt").write_text("extra")
    with pytest.raises(PermissionError, match="inventory"):
        verify_original_evidence(original)


def test_incomplete_or_certified_claim_does_not_pass(tmp_path):
    original = make_workspace(tmp_path/"original")
    report = json.loads((original/"cpu_campaign_report.json").read_text())
    report["completed_operations"] = 44
    (original/"cpu_campaign_report.json").write_text(json.dumps(report))
    with pytest.raises(PermissionError, match="authentic complete"):
        verify_original_evidence(original)


def test_symlink_in_original_custody_fails(tmp_path):
    original = make_workspace(tmp_path/"original")
    (original/"artifacts"/"linked").symlink_to(original/"artifacts"/"receipt.json")
    with pytest.raises(PermissionError, match="symlinks"):
        verify_original_evidence(original)
