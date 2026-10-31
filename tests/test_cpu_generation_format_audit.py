"""CPU generation forensics must preserve archive integrity and scientific boundaries."""
import hashlib
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile
import pytest

path = Path(__file__).resolve().parents[1] / "scripts" / "audit_cpu_generation_format.py"
spec = importlib.util.spec_from_file_location("cpu_format_audit", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_zip(path, *, tamper=False, candidate_text="```python\npass"):
    def generation(checkpoint, output):
        return {"schema": "cpu-prospective-real-generation.v1",
                "checkpoint_ref": checkpoint, "repeat_index": 0,
                "protocol_ref": "sha256:" + "a"*64,
                "generation_settings_id": "greedy-max16-exact-prompt-no-truncation.v1",
                "samples": [{"sample_id": "sample-1", "output": output}],
                "json_valid": 0, "correct": 0, "count": 1,
                "external_certification": False, "mean_teacher_forced_target_nll": 3.0}
    members = {"cpu_campaign_report.json": json.dumps(
        {"status":"completed", "completed_operations": 45, "workflow_id": path.stem}).encode()}
    for obj in (generation(None, "```python\nprint(42)"),
                generation("step-8", candidate_text)):
        raw = json.dumps(obj, sort_keys=True).encode()
        digest = hashlib.sha256(raw).hexdigest()
        members[f"artifacts/objects/sha256/{digest[:2]}/{digest}"] = raw
    manifest = {k: hashlib.sha256(v).hexdigest() for k, v in members.items()}
    if tamper:
        members["cpu_campaign_report.json"] = b"{}"
    members["archive_sha256.json"] = json.dumps(manifest).encode()
    with ZipFile(path, "w") as archive:
        for name, body in members.items():
            archive.writestr(name, body)
    return path


def test_original_code_fence_is_not_a_claim_of_causal_diagnosis(tmp_path):
    report = module.audit([make_zip(tmp_path / "first.zip")])
    assert report["format_totals"]["baseline"] == {"python_code_fence": 1}
    assert report["format_totals"]["candidate"] == {"python_code_fence": 1}
    assert report["causal_diagnosis_established"] is False
    assert report["external_certification"] is False


def test_original_hash_tampering_blocks_report(tmp_path):
    with pytest.raises(ValueError, match="SHA mismatch"):
        module.audit([make_zip(tmp_path / "first.zip", tamper=True)])


def test_duplicate_run_is_not_independent_replication(tmp_path):
    src = make_zip(tmp_path / "first.zip")
    with pytest.raises(ValueError, match="duplicate"):
        module.audit([src, src])


def test_json_opening_does_not_overwrite_original_correctness(tmp_path):
    report = module.audit([make_zip(tmp_path / "first.zip", candidate_text='{"result":42}')])
    assert report["format_totals"]["candidate"] == {"json_opening": 1}
    assert report["campaigns"][0]["candidate"]["correct"] == 0
