import importlib.util
import json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("original_gpu_review",
    ROOT/"scripts/verify_cpu45_for_frozen_gpu.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_original_five_cuda_roles_remain_exactly_pinned():
    roles=module.verify_original_roles(ROOT/"docs/evidence/role-model-stack-v1/certification.json")
    assert len(roles["original_cuda_roles"])==5
    assert all(len(x["adapter_sha256"])==64 for x in roles["original_cuda_roles"].values())


def test_frozen_gpu_proposal_never_self_certifies(tmp_path):
    file=tmp_path/"altered.json"
    source=json.loads((ROOT/"docs/evidence/role-model-stack-v1/certification.json").read_text())
    source["roles"]["controller"]["adapter_sha256"]="not-a-valid-sha"
    file.write_text(json.dumps(source))
    with pytest.raises(PermissionError):
        module.verify_original_roles(file)


def test_missing_or_modified_cpu_evidence_fails_without_allocation(tmp_path):
    from zipfile import ZipFile
    file=tmp_path/"forged.zip"
    with ZipFile(file,"w") as z:
        z.writestr("cpu_campaign_report.json",'{"status":"completed","completed_operations":45}')
    with pytest.raises(PermissionError):
        module.verify_campaign_archive(file)
