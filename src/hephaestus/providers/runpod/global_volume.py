"""Read-only Global Volume transport for certified role adapter archives."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping

from hephaestus.providers.models.role_stack import CertifiedRoleModelStack, REQUIRED_ROLES
from hephaestus.storage.mounted_read import MountedReadOnlyObjectSource

_SHA_REF = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class GlobalVolumeConfig:
    volume_id: str
    mount_path: Path = Path("/workspace-global")

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{6,100}", self.volume_id):
            raise ValueError("Global Volume requires its provider resource ID")
        if (
            not self.mount_path.is_absolute()
            or str(self.mount_path) in {"/", "/tmp", "/root"}
            or ".." in self.mount_path.parts
        ):
            raise ValueError("Global Volume requires an explicit absolute mount path")

    def mounts(self) -> dict[str, list[dict[str, str]]]:
        return {"global": [{"volumeId": self.volume_id, "path": str(self.mount_path)}]}


def global_role_archive_source(
    config: GlobalVolumeConfig,
    stack: CertifiedRoleModelStack,
    *,
    receipts: Mapping[str, str] | None = None,
) -> MountedReadOnlyObjectSource | GlobalReceiptRoleArchiveSource:
    """Use the admitted registry to map source keys to exact Global locations.

    Receipt references, when supplied, must come from independent trusted
    configuration. The certified materializer checks bytes and SHA-256 for
    either layout before it exposes a local adapter.
    """
    archives = {role: stack.resolve(role).adapter for role in REQUIRED_ROLES}
    if len({archive.s3_key for archive in archives.values()}) != len(REQUIRED_ROLES):
        raise ValueError("certified adapter registry contains duplicate source keys")
    if any(not re.fullmatch(r"[0-9a-f]{64}", archive.sha256) for archive in archives.values()):
        raise ValueError("invalid certified adapter digest")
    if receipts is not None:
        if set(receipts) != set(REQUIRED_ROLES) or any(
            not _SHA_REF.fullmatch(ref) for ref in receipts.values()
        ):
            raise ValueError("Global receipt configuration must pin all five role receipts")
        return GlobalReceiptRoleArchiveSource(
            mount=MountedReadOnlyObjectSource(config.mount_path, config.volume_id, require_mount=True),
            objects={
                archive.s3_key: (receipts[role], archive.sha256, archive.bytes)
                for role, archive in archives.items()
            },
        )
    return MountedReadOnlyObjectSource(
        config.mount_path, config.volume_id, require_mount=True,
        object_paths={
            archive.s3_key: f"hephaestus/runtime-models/role-mastery-v1/{role}/selected-adapter-{archive.sha256[:24]}.tar.gz"
            for role, archive in archives.items()
        },
    )


@dataclass(frozen=True, slots=True)
class GlobalReceiptRoleArchiveSource:
    mount: MountedReadOnlyObjectSource
    objects: Mapping[str, tuple[str, str, int]]

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, BinaryIO]:
        if Bucket != self.mount.expected_bucket or Key not in self.objects:
            raise PermissionError("unregistered Global adapter source")
        receipt, artifact_digest, size = self.objects[Key]
        receipt_digest = receipt.removeprefix("sha256:")
        receipt_path = f"hephaestus/receipts/sha256/{receipt_digest[:2]}/{receipt_digest}.json"
        with self.mount.open_file(receipt_path) as stream:
            raw = stream.read(8193)
        if len(raw) > 8192 or hashlib.sha256(raw).hexdigest() != receipt_digest:
            raise ValueError("Global adapter receipt has invalid length or digest")
        doc = json.loads(raw)
        if (
            not isinstance(doc, dict)
            or doc.get("schema") != "hephaestus.global-artifact.v1"
            or doc.get("volume_id") != Bucket
            or doc.get("kind") != "adapter"
            or doc.get("artifact_ref") != f"sha256:{artifact_digest}"
            or type(doc.get("byte_size")) is not int
            or doc["byte_size"] != size
            or not _SHA_REF.fullmatch(str(doc.get("admission_ref", "")))
            or doc.get("backup_verified") is not True
            or not isinstance(doc.get("backup_target"), str)
            or not doc["backup_target"].strip()
            or doc["backup_target"] == Bucket
        ):
            raise ValueError("Global adapter receipt differs from certified source or custody")
        object_path = f"hephaestus/objects/sha256/{artifact_digest[:2]}/{artifact_digest}"
        return {"Body": self.mount.open_file(object_path)}
