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
    assert "configs/research/tier_a_original_cuda_45_fresh_20261030_r9.json" in script
    assert "cuda-json-45-original-five-20261030-r9" in script
    assert "cuda-json-45-original-five-20261028" not in script
    assert "original_frozen_45_archive.zip" in script
    assert controller.WORKFLOW=="cuda-json-45-original-five-20261030-r9"
    assert "freeze_original_tier_a_cuda_archive.py" in script
    assert "worker_exit_status-1234.txt" in script
    assert "worker_stdout-1234.log" in script
    assert "exec >> " in script and " >> " in script
    assert "ORIGINAL_CUDA_BOOTSTRAP_ATTEMPT" in script
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


def _fake_native_state_bucket(tmp_path, state):
    """Test-only S3 readback of actual SQLite bytes, never provider IO."""
    import io,json,sqlite3
    db=tmp_path/"state.sqlite"
    db.unlink(missing_ok=True)
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE state_records (sequence INTEGER PRIMARY KEY,"
                    "collection TEXT NOT NULL,payload_json TEXT NOT NULL)")
        if state is not None:
            con.execute("INSERT INTO state_records (collection,payload_json)"
                        " VALUES (?,?)",("staged_workflow_states",json.dumps(state)))
    class Missing(Exception):
        response={"Error":{"Code":"404"}}
    class S3:
        def __init__(self):
            self.reads=[]
        def get_object(self,*,Bucket,Key):
            self.reads.append((Bucket,Key))
            if Key==controller.PREFIX+"/state.sqlite":
                return {"Body":io.BytesIO(db.read_bytes())}
            raise Missing()
    return S3()


def test_paid_gpu_preflight_blocks_existing_terminal_first_stage(tmp_path):
    s3=_fake_native_state_bucket(tmp_path,{
        "workflow_id":controller.WORKFLOW,"run_id":controller.RUN,
        "status":"terminal_failure","resumable":False,"completion_marker":False,
        "current_phase":"judge_entry","current_substep":"judge_entry_decision"})
    with pytest.raises(PermissionError,match="paid GPU retry forbidden"):
        controller.refuse_terminal_native_state(s3,"original-volume")
    assert s3.reads


def test_paid_gpu_preflight_accepts_clean_or_resumable_native_state_only(tmp_path):
    s3=_fake_native_state_bucket(tmp_path,None)
    controller.refuse_terminal_native_state(s3,"original-volume")
    s3=_fake_native_state_bucket(tmp_path,{
        "workflow_id":controller.WORKFLOW,"run_id":controller.RUN,
        "status":"interrupted","resumable":True,"completion_marker":False})
    controller.refuse_terminal_native_state(s3,"original-volume")


def test_paid_gpu_preflight_fails_closed_on_edited_identity(tmp_path):
    s3=_fake_native_state_bucket(tmp_path,{
        "workflow_id":"other","run_id":controller.RUN,
        "status":"interrupted","resumable":True,"completion_marker":False})
    with pytest.raises(PermissionError,match="identity"):
        controller.refuse_terminal_native_state(s3,"original-volume")


def test_fresh_prepaid_controller_rejects_reused_historical_state(tmp_path):
    """Old terminal SQLite can never be treated as new r9 science."""
    original=_fake_native_state_bucket(tmp_path,{
        "workflow_id":"cuda-json-45-original-five-20261028",
        "run_id":"cuda-json-45-new-1117",
        "status":"terminal_failure","resumable":False,"completion_marker":False})
    with pytest.raises(PermissionError,match="identity mismatched"):
        controller.refuse_terminal_native_state(original,"cviwpryzao")


def test_fresh_source_admission_must_match_exact_new_signed_scope():
    import io,json
    from hephaestus.production.signed_cuda_grants import expected_gpu_operation_map
    req={"schema":"tier-a-original-cuda-compute-authorization.v1",
         "status":"requested",
         "workflow_id":controller.WORKFLOW,"run_id":controller.RUN,
         "source_policy_ref":controller.SOURCE_POLICY,
         "reviewed_trainable_ref":controller.TRAINABLE,
         "operations":expected_gpu_operation_map(controller.WORKFLOW)}
    seal={"schema":"fresh-original-cuda45-prepared.v1",
          "workflow_id":controller.WORKFLOW,"run_id":controller.RUN,
          "source_policy_ref":controller.SOURCE_POLICY,
          "reviewed_trainable_ref":controller.TRAINABLE,
          "native_operations_completed":0,"gpu_optimizer_steps_completed":0,
          "operator_signature_present":False,
          "original_terminal_state_retained":True,
          "source_files_sha256":{"artifacts/objects/sha256/"+
              str(k).zfill(64):"a"*64 for k in range(21)}}
    class FakeS3:
        def get_object(self,*,Bucket,Key):
            if Key.endswith("cuda-grant-unsigned-request.json"):
                return {"Body":io.BytesIO(json.dumps(req).encode())}
            if Key.endswith("_fresh_admission_manifest.json"):
                return {"Body":io.BytesIO(json.dumps(seal).encode())}
            raise AssertionError("unexpected read")
        def head_object(self,**kwargs):
            raise AssertionError("invalid admission reached state checks")
    s3=FakeS3()
    req["workflow_id"]="cuda-json-45-original-five-20261028"
    with pytest.raises(PermissionError,match="source scope"):
        controller.require_source(s3,"cviwpryzao")
    req["workflow_id"]=controller.WORKFLOW
    seal["operator_signature_present"]=True
    with pytest.raises(PermissionError,match="admission seal"):
        controller.require_source(s3,"cviwpryzao")


def test_new_r3_operator_authorization_is_required_before_paid_provider_call(monkeypatch):
    """No implicit extension of the prior r2 authorization to a new identity."""
    import sys
    monkeypatch.delenv("ORIGINAL_CUDA45_R9_OPERATOR_APPROVAL",raising=False)
    monkeypatch.setattr(sys,"argv",[
        "launch_user_delegated_full_cuda45.py",
        "--commit-sha","a"*40,"--run-tag","1234"])
    with pytest.raises(PermissionError,match="explicit new operator approval"):
        controller.main()


def test_alternative_gpu_identity_and_price_limits_are_explicit_per_campaign():
    assert controller.GPU=="NVIDIA RTX PRO 6000 Blackwell Server Edition"
    assert controller.MIN_GPU_VRAM_GB==48
    assert controller.MAX_GPU_HOURLY_USD==Decimal("2.49")
    assert controller.MAX_ALLOCATION_ATTEMPTS==6
    assert controller.MAX_SECONDS==4*3600


def test_blackwell_fallback_obeys_total_live_prepaid_ceiling_not_hardcoded_four_hours():
    maximum=controller.decide_budget(
        Account(Decimal("9.50"),Decimal("0.10")), Quote(Decimal("2.49")))
    assert 3600<=maximum<14400
    assert Decimal(maximum)*Decimal("2.59")/Decimal(3600)<=Decimal("9.50")-controller.RESERVE_USD
    with pytest.raises(PermissionError):
        controller.decide_budget(
            Account(Decimal("1.50"),Decimal("0.20")),Quote(Decimal("2.49")))


def test_fallback_gpu_ephemeral_disk_is_large_enough_for_all_original_pinned_model_weights():
    import inspect
    assert controller.GPU=="NVIDIA RTX PRO 6000 Blackwell Server Edition"
    assert controller.CONTAINER_DISK_GB>=100
    assert controller.MAX_CONTAINER_STORAGE_USD_PER_HOUR>=Decimal("0.02")
    source=inspect.getsource(controller.main)
    assert "container_disk_in_gb=CONTAINER_DISK_GB" in source


def test_prepaid_budget_includes_actual_gpu_plus_other_running_spend_plus_disk_reserve():
    account=Account(Decimal("10.00"),Decimal("0.10"))
    quote=Quote(Decimal("2.49"))
    seconds=controller.decide_budget(account,quote)
    assert seconds>=3600 and seconds<14400
    all_in_per_hour=(quote.price_usd_per_hour+account.already_spending_usd_per_hour
                     +controller.MAX_CONTAINER_STORAGE_USD_PER_HOUR)
    assert Decimal(seconds)*all_in_per_hour/3600 <= (
        account.credit_balance_usd-controller.RESERVE_USD)
