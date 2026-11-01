"""Technical campaign custody; these records confer no scientific authority."""
from dataclasses import dataclass
from ._base import JsonSchema


@dataclass(slots=True)
class StorageIdentity(JsonSchema):
    database_id: str
    commit_id: str


@dataclass(slots=True)
class StateDirectoryBinding(JsonSchema):
    database: str
    storage_profile: dict
    schema: str = 'campaign-state-directory.v1'


@dataclass(slots=True)
class CampaignWitnessRecord(JsonSchema):
    sequence: int
    previous: str
    kind: str
    before: dict | None
    after: dict | None
    schema: str = "campaign-commit-witness.v1"


@dataclass(slots=True)
class CampaignSnapshotManifest(JsonSchema):
    logical_root: str
    database_relative_path: str
    storage_identity: dict
    witness: dict
    files: dict
    schema: str = "campaign-snapshot.v1"
    relocatable: bool = False
    independent_exposure_authority: bool = False
