"""Offline tests: no provider credentials, allocation or network I/O."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal

import pytest

from hephaestus.providers.runpod.global_volume import (
    FrozenArtifactAdmission, GlobalArtifactLibrary, GlobalVolumeConfig,
    GlobalVolumeIntegrityError,
)
from hephaestus.providers.runpod.global_placement import (
    ApprovedGpuLease, build_global_gpu_request, verify_global_pod_mount,
)


@dataclass
class _BackupRecord:
    artifact_ref: str


class _IndependentBackup:
    def __init__(self):
        self.object = None
        self.fail_readback = False

    def put_file(self, source, *, expected_hash=None, media_type=None):
        data = source.read_bytes()
        self.object = data
        return _BackupRecord("sha256:" + hashlib.sha256(data).hexdigest())

    def verify(self, ref):
        return not self.fail_readback and self.object is not None and (
            ref == "sha256:" + hashlib.sha256(self.object).hexdigest()
        )


def _fixture(tmp_path):
    global_dir = tmp_path / "global"
    global_dir.mkdir()
    cfg = GlobalVolumeConfig("volume456", global_dir)
    lib = GlobalArtifactLibrary(cfg, require_mount=False, admission_verifier=lambda _: True)
    src = tmp_path / "immutable.safetensors"
    src.write_bytes(b"frozen scientific adapter bytes")
    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    admission = FrozenArtifactAdmission("sha256:"+digest, "sha256:"+("a"*64), "adapter", src.stat().st_size)
    return lib, src, admission


def test_config_does_not_require_or_override_regional_volume():
    cfg = GlobalVolumeConfig.from_env({"RUNPOD_GLOBAL_VOLUME_ID":"volume456", "RUNPOD_GLOBAL_MOUNT_PATH":"/workspace"})
    assert cfg.mounts() == {"global": [{"volumeId": "volume456", "path": "/workspace"}]}
    with pytest.raises(ValueError):
        GlobalVolumeConfig.from_env({})
    with pytest.raises(ValueError):
        GlobalVolumeConfig("volume456", "/")


def test_receipt_after_verified_independent_backup_and_readback(tmp_path):
    lib, src, approval = _fixture(tmp_path)
    backup = _IndependentBackup()
    receipt = lib.publish(src, approval, backup_store=backup,
                          backup_target="s3://independent-backup")
    assert lib.resolve(receipt.receipt_ref, expected_kind="adapter") == receipt.path
    assert lib.publish(src, approval, backup_store=backup,
                       backup_target="s3://independent-backup").receipt_ref == receipt.receipt_ref
    assert len(list((lib.config.mount_path / "hephaestus" / "objects").rglob(approval.source_artifact_ref[7:]))) == 1


def test_unapproved_or_unbacked_artifacts_never_gain_receipts(tmp_path):
    lib, src, approval = _fixture(tmp_path)
    backup = _IndependentBackup()
    lib.admission_verifier = lambda _: False
    with pytest.raises(PermissionError):
        lib.publish(src, approval, backup_store=backup, backup_target="s3://external")
    lib.admission_verifier = lambda _: True
    with pytest.raises(PermissionError):
        lib.publish(src, approval, backup_store=backup,
                    backup_target=lib.config.volume_id)
    backup.fail_readback = True
    with pytest.raises(GlobalVolumeIntegrityError):
        lib.publish(src, approval, backup_store=backup,
                    backup_target="s3://external")
    assert not list((lib.config.mount_path / "hephaestus" / "receipts").rglob("*.json"))


def test_modification_prevents_original_or_resolved_artifact_use(tmp_path):
    lib, src, approval = _fixture(tmp_path)
    backup = _IndependentBackup()
    receipt = lib.publish(src, approval, backup_store=backup,
                          backup_target="s3://external")
    receipt.path.write_bytes(b"tampered bytes")
    with pytest.raises(GlobalVolumeIntegrityError):
        lib.resolve(receipt.receipt_ref, expected_kind="adapter")
    with pytest.raises(GlobalVolumeIntegrityError):
        lib.publish(src, approval, backup_store=backup, backup_target="s3://external")


def test_source_requires_regular_file_and_mount_required(tmp_path):
    lib, src, approval = _fixture(tmp_path)
    symlink = tmp_path / "alias"
    symlink.symlink_to(src)
    with pytest.raises(ValueError):
        lib.publish(symlink, approval,
                    backup_store=_IndependentBackup(), backup_target="s3://external")
    prod = GlobalArtifactLibrary(lib.config)
    with pytest.raises(RuntimeError):
        prod.resolve("sha256:" + ("a"*64))


def test_worker_placement_fails_closed_on_wrong_dc_price_vram_credit():
    volume = GlobalVolumeConfig("volume456")
    lease = ApprovedGpuLease("sha256:"+("a"*64), "US-TX-3", "NVIDIA B200", 128,
                             Decimal("4.00"), 1800)
    kw = dict(
        observed_datacenter_id="US-TX-3", observed_gpu_id="NVIDIA B200",
        observed_vram_gb=180, observed_hourly_usd=Decimal("3.50"),
        authenticated_balance_usd=Decimal("20"), currently_spending_usd_hour=Decimal("1"),
        worker_image="reviewed-image:sha256-test", command=("python", "worker.py"),
        receipt_ref="sha256:"+("b"*64), approval_verified=True,
        independent_watchdog_ready=True, name="frozen-smoke",
    )
    body = build_global_gpu_request(volume, lease, **kw)
    assert body["mounts"] == volume.mounts()
    assert body["dataCenterIds"] == ["US-TX-3"]
    assert "network" not in body["mounts"]
    pod = dict(body, status="RUNNING", dataCenterId="US-TX-3")
    assert verify_global_pod_mount(pod, body) is True
    pod["dataCenterId"] = "EU-CZ-1"
    assert not verify_global_pod_mount(pod, body)
    for change in [dict(observed_vram_gb=24), dict(observed_hourly_usd=Decimal("9")),
                   dict(authenticated_balance_usd=Decimal("0")),
                   dict(independent_watchdog_ready=False), dict(approval_verified=False),
                   dict(observed_datacenter_id="EU-CZ-1")]:
        with pytest.raises((PermissionError, ValueError)):
            build_global_gpu_request(volume, lease, **(kw|change))
