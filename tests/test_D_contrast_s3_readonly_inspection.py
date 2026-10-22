"""Read-only S3 forensics contract tests. No real remote requests."""
import io
from datetime import datetime,timezone
import inspect_D_contrast_run_s3_readonly as probe
import pytest

class NoSuchKey(Exception):
    def __init__(self):
        self.response={"Error":{"Code":"NoSuchKey"}}

class FakeS3:
    def __init__(self, data):
        self.data={k:v.encode() for k,v in data.items()}
        self.ops=[]
    def list_objects_v2(self,**kwargs):
        self.ops.append("list_objects_v2")
        return {"Contents":[{"Key":key,"Size":len(value),
             "LastModified":datetime(2026,10,19,18,12,tzinfo=timezone.utc)}
             for key,value in self.data.items() if key.startswith(kwargs["Prefix"])],
                "IsTruncated":False}
    def head_object(self,**kwargs):
        self.ops.append("head_object")
        if kwargs["Key"] not in self.data:raise NoSuchKey()
        return {"ContentLength":len(self.data[kwargs["Key"]]),
            "LastModified":datetime(2026,10,19,18,12,tzinfo=timezone.utc)}
    def get_object(self,**kwargs):
        self.ops.append("get_object")
        return {"Body":io.BytesIO(self.data[kwargs["Key"]])}
    def __getattr__(self,op):
        raise AssertionError("forbidden S3 operation: "+op)

def test_missing_evidence_reports_uncertainty_not_inference_failure():
    s3=FakeS3({})
    r=probe.inspect(s3,"cviwpryzao")
    assert r["read_only_s3"] and r["gpu_allocation"]==0
    assert r["persisted_unique_generations"]==0
    assert r["inference_readiness"]=="no_model_loaded_manifest_seen; model init not proven"
    assert r["fully_observed_pairs"]==0
    assert set(s3.ops)<=set(r["s3_operations"])
    assert r["timeline"]["manifest_write_utc"] is None

def test_bootstrap_readable_without_model_manifest():
    key=probe.ROOT+"/bootstrap/progress.json"
    s3=FakeS3({key:'{"stage":"environment_install_started","timestamp_unix":1792433200,"venv_ready":false,"nvidia_smi":"SHOULD_NOT_LOG"}'})
    r=probe.inspect(s3,"cviwpryzao")
    assert r["bootstrap"]["stage"]=="environment_install_started"
    assert "nvidia_smi" not in str(r["bootstrap"])
    assert r["manifest"] is None

def test_saved_shard_requires_known_case_identity_and_frozen_checkpoint():
    key=probe.ROOT+"/samples/shard-0001.json"
    s3=FakeS3({key:'[{"prompt_sha256":"unknown","checkpoint_sha256":"bogus","raw":"{}"}]'})
    with pytest.raises(ValueError,match="unknown prompt identity"):
        probe.inspect(s3,"cviwpryzao")

def test_byte_receipt_and_result_stage():
    key=probe.ROOT+"/result.json"
    s3=FakeS3({key:'{"status":"failed","error_type":"RuntimeError","error":"one-shot timeout"}'})
    r=probe.inspect(s3,"cviwpryzao")
    assert r["terminal_result"]["status"]=="failed"
    assert len(next(x for x in r["read_receipts"] if x["key"]==key)["sha256"])==64

def test_utc_wall_elapsed_from_verified_job():
    assert 938 < probe.elapsed(probe.OBSERVED_POD_CREATE_UTC,probe.OBSERVED_POD_DELETE_UTC) < 940
