from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from hephaestus.schemas.discovery_contract import DatasetCandidate, DatasetSearchRequest
from hephaestus.utils.hashing import hash_file, hash_json


@dataclass(frozen=True, slots=True)
class LocalFixtureDescriptor:
    path: Path
    dataset_id: str
    revision: str | None = None
    task_types: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()
    license: str | None = None
    trust_level: str = "local_fixture"
    split: str = "train"
    synthetic: bool = False
    hard_negative: bool = False
    support_set: bool = False
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class LocalFixtureDatasetProvider:
    """Discovers explicitly configured local files without executing code."""

    fixtures: tuple[LocalFixtureDescriptor, ...]
    provider_id: str = "local_fixture"

    def search(self, request: DatasetSearchRequest) -> tuple[DatasetCandidate, ...]:
        del request
        candidates: list[DatasetCandidate] = []
        for fixture in sorted(self.fixtures, key=lambda item: (item.dataset_id, str(item.path))):
            path = fixture.path.expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"configured fixture does not exist: {path}")
            content_hash = hash_file(path)
            revision = fixture.revision or f"sha256:{content_hash}"
            suffix = path.suffix.lower()
            record_format = {".jsonl": "jsonl", ".json": "json", ".csv": "csv"}.get(suffix, "unknown")
            candidate_seed = {
                "provider_id": self.provider_id,
                "dataset_id": fixture.dataset_id,
                "revision": revision,
            }
            row_estimate = None
            if record_format == "jsonl":
                with path.open("rb") as handle:
                    row_estimate = sum(1 for line in handle if line.strip())
            candidates.append(
                DatasetCandidate(
                    candidate_id=f"dataset-{hash_json(candidate_seed)[:16]}",
                    provider_id=self.provider_id,
                    dataset_id=fixture.dataset_id,
                    revision=revision,
                    splits=[fixture.split],
                    task_types=list(fixture.task_types),
                    languages=list(fixture.languages),
                    domains=list(fixture.domains),
                    format_profile={"record_format": record_format, "path_suffix": suffix},
                    estimated_rows=row_estimate,
                    estimated_bytes=path.stat().st_size,
                    license=fixture.license,
                    provenance={
                        "kind": "local_fixture",
                        "path": str(path),
                        "content_hash": f"sha256:{content_hash}",
                    },
                    trust_level=fixture.trust_level,
                    compatibility={"local_readable": True, "remote_code_required": False},
                    artifact_ref=str(path),
                    evidence_refs=[str(path)],
                    metadata={
                        **fixture.metadata,
                        "synthetic": fixture.synthetic,
                        "hard_negative": fixture.hard_negative,
                        "support_set": fixture.support_set,
                    },
                )
            )
        return tuple(candidates)
