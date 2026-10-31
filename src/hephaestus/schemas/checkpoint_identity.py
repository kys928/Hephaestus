"""A producer/custodian commitment kept outside the checkpoint directory."""
from dataclasses import dataclass
import re

from ._base import JsonSchema


@dataclass(slots=True)
class CheckpointIdentity(JsonSchema):
    lineage_id: str
    checkpoint_ref: str
    manifest_hash: str
    evidence_ref: str

    def __post_init__(self):
        if (not self.lineage_id or not self.checkpoint_ref or not self.evidence_ref
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", self.manifest_hash)):
            raise ValueError("checkpoint identity requires lineage, reference, digest and provenance")
