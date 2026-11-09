"""Opt-in receipt-gated RunPod Global Volume artifact publication and resolution.

Global Volumes are object-backed: no atomic rename, file locks, or POSIX
atomic creation. Caller must serialize publication outside this module. An
incomplete object without a verified receipt is never returned to consumers.
No mutable training state, checkpoint writes or model activation occurs here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol

_SHA = re.compile(r"^[0-9a-f]{64}$")
_KINDS = frozenset({"model", "adapter", "tokenizer", "dataset", "sealed_checkpoint", "scientific_evidence"})
_BUF_SIZE = 1024 * 1024


class GlobalVolumeIntegrityError(ValueError):
    """An immutable reference or its underlying bytes failed verification."""


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(_BUF_SIZE):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _canonical(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash_ref(ref: str) -> str:
    if not isinstance(ref, str) or not ref.startswith("sha256:") or not _SHA.fullmatch(ref[7:]):
        raise GlobalVolumeIntegrityError("invalid sha256 reference")
    return ref[7:]


@dataclass(frozen=True, slots=True)
class GlobalVolumeConfig:
    """Distinct from RunPodConfig: no regional S3 endpoint or DC affinity."""

    volume_id: str
    mount_path: Path = Path("/workspace-global")

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{6,100}", self.volume_id):
            raise ValueError("global volume ID is required and must be a provider resource ID")
        path = Path(self.mount_path)
        if not path.is_absolute() or str(path) in {"/", "/tmp", "/root"} or ".." in path.parts:
            raise ValueError("global mount must be an explicit safe absolute path")
        object.__setattr__(self, "mount_path", path)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "GlobalVolumeConfig":
        values = env if env is not None else os.environ
        name = values.get("RUNPOD_GLOBAL_VOLUME_ID", "").strip()
        if not name:
            raise ValueError("RUNPOD_GLOBAL_VOLUME_ID is required for global mode")
        return cls(name, Path(values.get("RUNPOD_GLOBAL_MOUNT_PATH", "/workspace-global")))

    def mounts(self) -> dict[str, list[dict[str, str]]]:
        """REST v2 Pod mounts fragment; combine with regional mounts only explicitly."""
        return {"global": [{"volumeId": self.volume_id, "path": str(self.mount_path)}]}


@dataclass(frozen=True, slots=True)
class FrozenArtifactAdmission:
    """Identity supplied by upstream deterministic approval/lineage owner.

    This structure is NOT an approval authority. Publisher requires an injected
    admission verifier, not a model-generated approval string.
    """

    source_artifact_ref: str
    admission_ref: str
    kind: str
    expected_bytes: int

    def __post_init__(self) -> None:
        _hash_ref(self.source_artifact_ref)
        _hash_ref(self.admission_ref)
        if self.kind not in _KINDS:
            raise ValueError("only admitted read-mostly artifact kinds are publishable")
        if type(self.expected_bytes) is not int or self.expected_bytes < 0:
            raise ValueError("expected artifact size must be nonnegative")


class IndependentBackupStore(Protocol):
    def put_file(self, source: Path, *, expected_hash: str | None = None, media_type: str | None = None): ...
    def verify(self, artifact_ref: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class GlobalArtifactReceipt:
    receipt_ref: str
    artifact_ref: str
    byte_size: int
    path: Path
    backup_target: str


@dataclass(slots=True)
class GlobalArtifactLibrary:
    config: GlobalVolumeConfig
    # In production the mount must be an actual mountpoint. False only for tests.
    require_mount: bool = True
    admission_verifier: Callable[[FrozenArtifactAdmission], bool] | None = None

    def _root(self) -> Path:
        root = self.config.mount_path
        if not root.is_dir() or (self.require_mount and not os.path.ismount(root)):
            raise RuntimeError("global volume mount not present; refusing publication or load")
        return root

    def _object_path(self, digest: str) -> Path:
        return self._root() / "hephaestus" / "objects" / "sha256" / digest[:2] / digest

    def _receipt_path(self, digest: str) -> Path:
        return self._root() / "hephaestus" / "receipts" / "sha256" / digest[:2] / (digest + ".json")

    @staticmethod
    def _checked_file(source: Path) -> Path:
        if not source.exists() or not stat.S_ISREG(source.lstat().st_mode):
            raise ValueError("source must be an existing regular file, not a symlink")
        return source.resolve(strict=True)

    @staticmethod
    def _verify_file(path: Path, digest: str, expected_bytes: int) -> None:
        if not path.is_file() or path.is_symlink():
            raise GlobalVolumeIntegrityError("artifact is absent or not a regular file")
        actual, size = _sha256(path)
        if actual != digest or size != expected_bytes:
            raise GlobalVolumeIntegrityError("artifact bytes or SHA-256 mismatch")

    @staticmethod
    def _create_once(path: Path, source: Path | bytes) -> None:
        """Single writer only. No atomic visibility claim on Global Volumes."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as out:
            if isinstance(source, bytes):
                out.write(source)
            else:
                with source.open("rb") as src:
                    while block := src.read(_BUF_SIZE):
                        out.write(block)
            out.flush()
        # Never rely on fsync/rename/locks on object-backed global storage.

    def publish(
        self,
        source: Path,
        admission: FrozenArtifactAdmission,
        *,
        backup_store: IndependentBackupStore,
        backup_target: str,
    ) -> GlobalArtifactReceipt:
        """Publish only after independent policy and backup verification.

        admission_verifier MUST be wired to a deterministic policy/lineage
        authority by the application composition layer, never a model role.
        No independent backup = no visible receipt.
        """
        if self.admission_verifier is None or self.admission_verifier(admission) is not True:
            raise PermissionError("upstream deterministic admission is required")
        if not backup_target or backup_target == self.config.volume_id or backup_store is None:
            raise PermissionError("an independent backup store/target is required")
        root = self._root()
        src = self._checked_file(Path(source))
        if src == root or root in src.parents:
            raise ValueError("cannot publish from the Global Volume into itself")
        digest = _hash_ref(admission.source_artifact_ref)
        self._verify_file(src, digest, admission.expected_bytes)
        destination = self._object_path(digest)
        if destination.exists():
            self._verify_file(destination, digest, admission.expected_bytes)
        else:
            self._create_once(destination, src)
            self._verify_file(destination, digest, admission.expected_bytes)
        record = backup_store.put_file(src, expected_hash=admission.source_artifact_ref)
        if getattr(record, "artifact_ref", None) != admission.source_artifact_ref:
            raise GlobalVolumeIntegrityError("backup returned a mismatched artifact identity")
        if not backup_store.verify(admission.source_artifact_ref):
            raise GlobalVolumeIntegrityError("independent backup read-back verification failed")
        self._verify_file(src, digest, admission.expected_bytes)
        self._verify_file(destination, digest, admission.expected_bytes)
        receipt: dict[str, object] = {
            "schema": "hephaestus.global-artifact.v1",
            "volume_id": self.config.volume_id,
            "artifact_ref": admission.source_artifact_ref,
            "admission_ref": admission.admission_ref,
            "kind": admission.kind,
            "byte_size": admission.expected_bytes,
            "backup_target": backup_target,
            "backup_verified": True,
        }
        encoded = _canonical(receipt)
        receipt_sha = hashlib.sha256(encoded).hexdigest()
        receipt_path = self._receipt_path(receipt_sha)
        if receipt_path.exists():
            if receipt_path.read_bytes() != encoded:
                raise GlobalVolumeIntegrityError("receipt identity collision or corruption")
        else:
            self._create_once(receipt_path, encoded)
            if receipt_path.read_bytes() != encoded:
                raise GlobalVolumeIntegrityError("receipt read-back mismatch")
        return GlobalArtifactReceipt(
            receipt_ref="sha256:" + receipt_sha,
            artifact_ref=admission.source_artifact_ref,
            byte_size=admission.expected_bytes,
            path=destination,
            backup_target=backup_target,
        )

    def resolve(self, receipt_ref: str, *, expected_kind: str | None = None) -> Path:
        """Verify receipt and FULL content before exposing an artifact path.

        Caller must also check trusted off-volume admission identity and routing.
        The receipt is a content hash, not a signature or model certificate.
        """
        digest = _hash_ref(receipt_ref)
        receipt_path = self._receipt_path(digest)
        if not receipt_path.is_file() or receipt_path.is_symlink():
            raise GlobalVolumeIntegrityError("immutable receipt missing")
        if receipt_path.stat().st_size > 8192:
            raise GlobalVolumeIntegrityError("oversized receipt")
        raw = receipt_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise GlobalVolumeIntegrityError("receipt hash mismatch")
        try:
            obj = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        except (ValueError, TypeError) as exc:
            raise GlobalVolumeIntegrityError("invalid receipt JSON") from exc
        if (not isinstance(obj, dict) or obj.get("schema") != "hephaestus.global-artifact.v1"
                or obj.get("volume_id") != self.config.volume_id or obj.get("backup_verified") is not True
                or not isinstance(obj.get("backup_target"), str) or not obj["backup_target"]
                or obj["backup_target"] == self.config.volume_id
                or obj.get("kind") not in _KINDS
                or (expected_kind is not None and obj.get("kind") != expected_kind)
                or type(obj.get("byte_size")) is not int or obj["byte_size"] < 0):
            raise GlobalVolumeIntegrityError("receipt fails identity/provenance checks")
        _hash_ref(obj.get("admission_ref"))
        object_hash = _hash_ref(obj.get("artifact_ref"))
        path = self._object_path(object_hash)
        self._verify_file(path, object_hash, obj["byte_size"])
        return path
