"""Offline contract checks for owner-delegated original GPU 45 controller.

Mocks MUST NOT be presented as GPU science, spending proof, or certification.
"""
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from importlib.util import module_from_spec,spec_from_file_location
import pytest
pytest.importorskip("boto3", reason="RunPod budget controller requires optional s3 package")

ROOT=Path(__file__).resolve().parents[1]
spec=spec_from_file_location("original_budget_controller",
    ROOT/"scripts/launch_user_delegated_full_cuda45.py")
controller=module_from_spec(spec)
spec.loader.exec_module(controller)


@dataclass
class Account:
    credit_balance_usd:Decimal
    already_spending_usd_per_hour:Decimal
    autopay_enabled:bool=False


@dataclass
class Quote:
    price_usd_per_hour:Decimal


def test_entire_verified_prepaid_balance_delegation_reserves_evidence_and_other_spend():
    seconds=controller.decide_budget(
        Account(Decimal("2.83"),Decimal("0.07")),Quote(Decimal("0.50")))
    assert 3*3600 < seconds <= 4*3600
    assert seconds*(Decimal("0.57")/Decimal(3600))<=Decimal("2.83")-controller.RESERVE_USD


@pytest.mark.parametrize("balance,existing,price,autopay",[
    ("0.20","0.07","0.50",False),
    ("0.95","0.07","0.90",False),
    ("2.83","0.07","0.50",True),
])
def test_not_enough_prepaid_or_autopay_enabled_refuses_original_paid_gpu(
        balance,existing,price,autopay):
    with pytest.raises(PermissionError):
        controller.decide_budget(Account(Decimal(balance),Decimal(existing),autopay),
                                 Quote(Decimal(price)))


def test_frozen_full_45_startup_uses_no_sensitive_runpod_s3_credentials():
    cmd,env=controller.pod_command("a"*40,"b"*64,"sha256:"+"c"*64,3600,
        "external-original-cuda-grant-1234.json",run_tag="1234",code_bundle_sha256="d"*64)
    assert cmd[:2]==["bash","-lc"]
    script=cmd[2]
    assert "run_original_tier_a_cuda_worker.py --resume" in script
    assert "configs/research/tier_a_original_cuda_45_v1.json" in script
    assert "freeze_original_tier_a_cuda_archive.py" in script
    assert "worker_exit_status-1234.txt" in script
    assert "worker_stdout-1234.log" in script
    assert "git-"+"a"*40+".tar.gz" in script
    assert "sha256sum" in script and "tar -xzf" in script
    assert "git clone" not in script
    assert "--ignore-installed" in script and "cryptography>=44,<47" in script
    assert "cuda-original-role-cache" in script
    assert "RUNPOD_API_KEY" not in script+str(env)
    assert "RUNPOD_S3_SECRET_ACCESS_KEY" not in script+str(env)
    assert env["HEPHAESTUS_ENABLE_PINNED_MODEL_DOWNLOAD"]=="1"
    assert env["HF_HUB_ENABLE_HF_TRANSFER"]=="0"  # RunPod base image lacks this optional package


def test_never_allow_untagged_or_unpinned_gpu_science_commit():
    for sha in ("main","a"*39,"g"*40):
        with pytest.raises(PermissionError):
            controller.pod_command(sha,"b"*64,"sha256:"+"c"*64,3600,
                "external-original-cuda-grant-1234.json",run_tag="1234",code_bundle_sha256="d"*64)


def test_generated_pod_full_science_command_is_real_bash_syntax():
    import subprocess
    command,env=controller.pod_command("a"*40,"b"*64,"sha256:"+"c"*64,3600,
        "external-original-cuda-grant-1234.json",run_tag="1234",code_bundle_sha256="d"*64)
    subprocess.run(["bash","-n","-c",command[2]],check=True,timeout=5)
