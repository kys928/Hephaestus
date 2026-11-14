"""Offline safety/forensic gates for supervised Global Prometheus execution.

No RunPod API or paid GPU is invoked in this test module.
"""
from datetime import datetime, timedelta, timezone
import io
import json

import pytest

from hephaestus.development import prometheus_global_supervision as s
from hephaestus.development.prometheus_global_launcher import (
    _pod_command, _verify_worker_science,
)


class FakeBody(io.BytesIO):
    pass


class FakeS3:
    def __init__(self):
        self.items = {}
    def get_object(self, *, Bucket, Key):
        if Key not in self.items:
            error = RuntimeError("missing")
            error.response = {"Error": {"Code": "NoSuchKey"}}
            raise error
        return {"Body": FakeBody(self.items[Key])}
    def put_object(self, *, Bucket, Key, Body):
        self.items[Key] = Body


def _prefix():
    return s.campaign_prefix("prometheus-unbound-global-20261114-01", "a" * 40)


def test_prefix_and_pod_name_are_bounded():
    prefix = _prefix()
    assert prefix.endswith("/" + "a" * 40)
    assert s.pod_name(prefix).startswith("prometheus-global-")
    with pytest.raises(ValueError):
        s.campaign_prefix("../other", "a" * 40)
    with pytest.raises(ValueError):
        s.campaign_prefix("prometheus-unbound-global-20261114-01", "main")


def test_original_report_hash_rejects_substituted_input():
    with pytest.raises(PermissionError, match="SHA"):
        s.authenticate_sources(b"made-up-development", b"historical-D")
        

def test_immutable_s3_put_has_readback_and_does_not_overwrite():
    fake = FakeS3()
    key = _prefix() + "/source/test"
    digest = s.put_once(fake, key, b"original")
    assert digest == s.sha(b"original")
    with pytest.raises(FileExistsError):
        s.put_once(fake, key, b"tampered")
    assert fake.items[key] == b"original"


def test_watchdog_heartbeat_must_be_current_and_source_scoped():
    fake = FakeS3()
    now = datetime.now(timezone.utc)
    prefix = _prefix()
    fake.items[prefix + "/status/watchdog-ready.json"] = s.canonical({
        "schema": "prometheus-global-watchdog.v1", "prefix": prefix,
        "pod_name": s.pod_name(prefix),
        "hard_deadline": (now + timedelta(seconds=6000)).isoformat(),
    })
    fake.items[prefix + "/status/watchdog-heartbeat.json"] = s.canonical({
        "schema": "prometheus-global-heartbeat.v1", "prefix": prefix,
        "observed_at": now.isoformat(),
    })
    assert s.check_heartbeat(fake, prefix, now=now)["prefix"] == prefix
    with pytest.raises(PermissionError, match="stale"):
        s.check_heartbeat(fake, prefix, now=now + timedelta(seconds=95))
    fake.items[prefix + "/status/watchdog-heartbeat.json"] = s.canonical({
        "schema": "prometheus-global-heartbeat.v1", "prefix": "other",
        "observed_at": now.isoformat(),
    })
    with pytest.raises(PermissionError, match="authenticated"):
        s.check_heartbeat(fake, prefix, now=now)


def test_exact_global_pod_identity_refuses_wrong_mount_or_model():
    prefix = _prefix()
    exact = {"name": s.pod_name(prefix), "id": "abcdef1234",
             "cloud": "SECURE", "image": s.IMAGE, "dataCenterId": "EU-RO-1",
             "gpu": {"id": s.GPU_ID, "count": 1},
             "mounts": {"global": [{"volumeId": s.GLOBAL_ID, "path": "/workspace-global"}]}}
    s.verify_pod(exact, prefix)
    wrong = {**exact, "mounts": {"network": [{"volumeId": s.S3_BUCKET, "path": "/workspace"}]}}
    with pytest.raises(PermissionError):
        s.verify_pod(wrong, prefix)
    with pytest.raises(PermissionError):
        s.verify_pod({**exact, "gpu": {"id": "different", "count": 1}}, prefix)


def test_verifier_detects_tampered_durable_model_receipt():
    prefix = _prefix()
    fake = FakeS3()
    raw = b"private role receipt"
    path = "output/cycle/decision/inference-receipts/receipt.json"
    fake.items[prefix + "/result/files/" + path] = raw
    manifest = {"schema": "prometheus-global-evidence-manifest.v1",
                "prefix": prefix,
                "files": [{"path": path, "bytes": len(raw), "sha256": s.sha(raw)}]}
    fake.items[prefix + "/result/manifest.json"] = s.canonical(manifest)
    assert s.verify_manifest(fake, prefix)["prefix"] == prefix
    fake.items[prefix + "/result/files/" + path] = b"changed"
    with pytest.raises(PermissionError, match="readback"):
        s.verify_manifest(fake, prefix)


def test_controller_refuses_false_success_without_real_decision():
    fake = FakeS3()
    prefix = _prefix()
    manifest = {"schema": "prometheus-global-evidence-manifest.v1",
                "prefix": prefix, "worker_exit_code": 0,
                "source_parent_sha256": s.SOURCE_PARENT,
                "source_D_sha256": s.SOURCE_D,
                "global_volume_id": s.GLOBAL_ID,
                "files": [{"path": "output/cycle/campaign-result.json"}]}
    with pytest.raises(PermissionError, match="files missing"):
        _verify_worker_science(fake, prefix, manifest)


def test_boot_does_not_download_unpinned_code_or_use_regional_mount():
    cmd = _pod_command(3600)
    joined = " ".join(cmd)
    assert "PROMETHEUS_CODE_SHA256" in joined
    assert "sha256(raw)" in joined
    assert "source/code.tar.gz" in joined
    assert "/tmp/prometheus-extracted" in joined
    assert "--signal=TERM" in joined
    assert "mounts.network" not in joined
    with pytest.raises(ValueError):
        _pod_command(600)
