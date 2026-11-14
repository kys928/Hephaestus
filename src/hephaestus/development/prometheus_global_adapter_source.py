"""Read-only S3-get-compatible adapter archive source backed by a mounted Global Volume.

An opt-in compatibility adapter for Prometheus scientific inference. It never
modifies model weights or the Global Volume. The certified registry, not a
caller-controlled S3 key, determines each allowed file path and checksum.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import os
import re

ROLES = ("controller", "diagnosis", "planner", "evaluator", "judge")


@dataclass(slots=True)
class VerifiedGlobalRoleArchiveSource:
    volume_root: Path
    volume_id: str
    certified_stack: object
    require_mount: bool = True

    def __post_init__(self) -> None:
        root = self.volume_root
        if not root.is_absolute() or root.is_symlink() or not root.is_dir():
            raise ValueError("Global Volume must be an existing absolute directory, not a symlink")
        if self.require_mount and not os.path.ismount(root):
            raise RuntimeError("Global Volume mount not present")
        if not re.fullmatch(r"[A-Za-z0-9_-]{6,100}", self.volume_id):
            raise ValueError("Global Volume ID not admissible")

    def _expected(self) -> dict[str, tuple[Path, str, int]]:
        mapping = {}
        for role in ROLES:
            spec = self.certified_stack.resolve(role)
            archive = spec.adapter
            if not re.fullmatch(r"[0-9a-f]{64}", archive.sha256):
                raise ValueError("certified adapter digest invalid")
            path = (self.volume_root / "hephaestus" / "runtime-models" /
                    "role-mastery-v1" / role /
                    ("selected-adapter-" + archive.sha256[:24] + ".tar.gz"))
            mapping[archive.s3_key] = (path, archive.sha256, archive.bytes)
        if len(mapping) != len(ROLES):
            raise ValueError("certified adapter registry contains duplicate source keys")
        return mapping

    def get_object(self, *, Bucket: str, Key: str):
        if Bucket != self.volume_id:
            raise PermissionError("Global Volume identity differs from admitted source")
        if self.require_mount and not os.path.ismount(self.volume_root):
            raise RuntimeError("Global Volume unmounted")
        match = self._expected().get(Key)
        if match is None:
            raise PermissionError("unregistered adapter key cannot be read from Global storage")
        path, expected_sha, expected_bytes = match
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected_bytes:
            raise FileNotFoundError("certified Global adapter missing or wrong byte size")
        digest = sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != expected_sha:
            raise ValueError("certified Global adapter checksum mismatch")
        # The downstream VerifiedRoleAdapters independently rehashes and safely
        # extracts this stream to local POSIX cache; never extract on Global.
        return {"Body": path.open("rb")}
