"""Explicit local deployment scope; admission is not scientific authorization."""
from dataclasses import dataclass
from pathlib import Path
from ._base import JsonSchema


@dataclass(slots=True)
class LocalStorageProfile(JsonSchema):
    root: str = "/tmp"
    purpose: str = "bounded_rehearsal"
    profile_id: str = "linux-local-fsync.v1"

    def __post_init__(self):
        if type(self.root) is not str or not Path(self.root).is_absolute():
            raise ValueError("storage profile root must be absolute")
        if self.profile_id != "linux-local-fsync.v1" or self.purpose not in {
                "bounded_rehearsal", "persistent_research"}:
            raise ValueError("unsupported storage profile")


@dataclass(slots=True)
class StorageAdmission(JsonSchema):
    profile_id: str
    purpose: str
    path: str
    root: str
    filesystem: str
    mountpoint: str
    device: str
    retention_scope: str
    cross_host_supported: bool = False
    external_rollback_protection: bool = False
