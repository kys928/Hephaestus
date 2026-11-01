"""Append-only trusted checkpoint commitments; never trust-on-first-use.

Only deterministic producer/custodian adapters may retain a commitment. Legacy
imports must supply a digest from previously retained evidence, not compute a
new historical identity from whatever bytes happen to occupy an old path.
"""
from dataclasses import dataclass

from hephaestus.schemas.checkpoint_identity import CheckpointIdentity
from hephaestus.storage.base import StateRepository

CHECKPOINT_IDENTITIES = "checkpoint_identities"


@dataclass(slots=True)
class CheckpointIdentityStore:
    repository: StateRepository

    def for_lineage(self, lineage_id):
        return [CheckpointIdentity(**r) for r in self.repository.all(CHECKPOINT_IDENTITIES)
                if r.get("lineage_id") == lineage_id]

    def get(self, lineage_id: str, checkpoint_ref: str) -> CheckpointIdentity | None:
        rows = [CheckpointIdentity(**r) for r in self.repository.all(CHECKPOINT_IDENTITIES)
                if r.get("lineage_id") == lineage_id and r.get("checkpoint_ref") == checkpoint_ref]
        if not rows:
            return None
        if len({r.manifest_hash for r in rows}) != 1:
            raise PermissionError("conflicting retained checkpoint identity")
        return rows[0]

    def retain(self, identity: CheckpointIdentity) -> None:
        previous = self.get(identity.lineage_id, identity.checkpoint_ref)
        if previous is not None:
            if previous.manifest_hash != identity.manifest_hash:
                raise PermissionError("cannot replace retained checkpoint identity")
            return
        self.repository.append(CHECKPOINT_IDENTITIES, identity.to_dict())

    def verify(self, lineage_id: str, checkpoint_ref: str, observed: str) -> str:
        retained = self.get(lineage_id, checkpoint_ref)
        if retained is None:
            raise PermissionError("checkpoint identity missing; explicit trusted migration required")
        if retained.manifest_hash != observed:
            raise PermissionError("checkpoint identity differs from retained producer commitment")
        return retained.manifest_hash
