"""Original GPU mounted adapter access and fresh signed execution lease boundaries."""
from pathlib import Path
from types import SimpleNamespace
import pytest

from hephaestus.production.mounted_original_role_adapter import MountedReadOnlyOriginalS3Client
from hephaestus.production.authorized_cuda_context import AuthorizedCudaResearchContext


def test_genuine_mounted_volume_read_stream_has_no_s3_credentials(tmp_path):
    root=tmp_path/"runpod-network-volume"
    adapter=root/"hephaestus"/"scientific"/"role_mastery"/"original.tar.gz"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"test-only-signed-original-archive-bytes")
    client=MountedReadOnlyOriginalS3Client(root,"verified-original-volume")
    with client.get_object(Bucket="verified-original-volume",
        Key="hephaestus/scientific/role_mastery/original.tar.gz")["Body"] as stream:
        assert stream.read()==b"test-only-signed-original-archive-bytes"
    with pytest.raises(PermissionError):
        client.get_object(Bucket="other-volume",
            Key="hephaestus/scientific/role_mastery/original.tar.gz")
    for bad in ("../outside.tar.gz","/etc/passwd",r"..\\etc\\passwd"):
        with pytest.raises(PermissionError):
            client.get_object(Bucket="verified-original-volume",Key=bad)


def test_mount_cannot_follow_symlink_outside_original_volume(tmp_path):
    root=tmp_path/"volume"
    root.mkdir()
    outside=tmp_path/"foreign-adapter"
    outside.write_bytes(b"not originally certified")
    (root/"escape").symlink_to(outside)
    with pytest.raises(PermissionError):
        MountedReadOnlyOriginalS3Client(root,"original").get_object(
            Bucket="original",Key="escape")


def test_gpu_preparation_cannot_reset_cpu_deadline_without_signed_external_grant():
    with pytest.raises(PermissionError):
        AuthorizedCudaResearchContext.from_signed_approved(
            SimpleNamespace(),grants=SimpleNamespace(authorized={}),
            scope=SimpleNamespace(workflow_id="a",run_id="b",maximum_gpu_seconds=3600))
