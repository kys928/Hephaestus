"""Offline negative-path tests for opt-in Global Volume Prometheus role archives."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from hephaestus.development.prometheus_global_adapter_source import (
    ROLES, VerifiedGlobalRoleArchiveSource,
)


class FakeStack:
    def __init__(self, payloads):
        self.payloads = payloads

    def resolve(self, role):
        raw = self.payloads[role]
        digest = hashlib.sha256(raw).hexdigest()
        return SimpleNamespace(role=role, adapter=SimpleNamespace(
            sha256=digest, bytes=len(raw), s3_key=f"certified/{role}.tar.gz"))


def mounted_fixture(tmp_path: Path):
    payloads = {role: ("archive-for-" + role).encode() for role in ROLES}
    stack = FakeStack(payloads)
    client = VerifiedGlobalRoleArchiveSource(tmp_path, "global_fixture_01",
                                             stack, require_mount=False)
    for role in ROLES:
        spec = stack.resolve(role)
        target = (tmp_path / "hephaestus" / "runtime-models" / "role-mastery-v1" /
                  role / f"selected-adapter-{spec.adapter.sha256[:24]}.tar.gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payloads[role])
    return client, stack, payloads


def test_selected_archive_exact_bytes_and_key(tmp_path):
    client, stack, payloads = mounted_fixture(tmp_path)
    for role in ROLES:
        result = client.get_object(
            Bucket="global_fixture_01", Key=stack.resolve(role).adapter.s3_key)
        with result["Body"] as stream:
            assert stream.read() == payloads[role]


def test_wrong_bucket_and_unregistered_key_fail_closed(tmp_path):
    client, stack, _ = mounted_fixture(tmp_path)
    with pytest.raises(PermissionError):
        client.get_object(Bucket="cviwpryzao",
                          Key=stack.resolve("planner").adapter.s3_key)
    with pytest.raises(PermissionError):
        client.get_object(Bucket="global_fixture_01", Key="../../private")


def test_tampered_archive_refused(tmp_path):
    client, stack, _ = mounted_fixture(tmp_path)
    spec = stack.resolve("diagnosis")
    target = (tmp_path / "hephaestus" / "runtime-models" / "role-mastery-v1" /
              "diagnosis" / f"selected-adapter-{spec.adapter.sha256[:24]}.tar.gz")
    target.write_bytes(b"modified wrong length")
    with pytest.raises(FileNotFoundError):
        client.get_object(Bucket="global_fixture_01", Key=spec.adapter.s3_key)
    target.write_bytes(b"x" * spec.adapter.bytes)
    with pytest.raises(ValueError, match="checksum"):
        client.get_object(Bucket="global_fixture_01", Key=spec.adapter.s3_key)


def test_missing_mount_or_symlink_refused(tmp_path):
    with pytest.raises(RuntimeError, match="mount"):
        VerifiedGlobalRoleArchiveSource(tmp_path, "global_fixture_01", FakeStack({}),
                                        require_mount=True)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        VerifiedGlobalRoleArchiveSource(link, "global_fixture_01", FakeStack({}),
                                        require_mount=False)
