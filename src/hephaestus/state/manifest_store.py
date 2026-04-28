from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hephaestus.schemas.dataset_manifest import normalize_dataset_manifest
from hephaestus.state._json_store import JsonStore


@dataclass(slots=True)
class ManifestStore:
    root: Path

    def append(self, record: dict[str, object]) -> None:
        JsonStore(self.root, "manifests.jsonl").append(normalize_dataset_manifest(record))

    def all(self) -> list[dict[str, object]]:
        return [normalize_dataset_manifest(row) for row in JsonStore(self.root, "manifests.jsonl").all()]

    def get(self, manifest_id: str) -> dict[str, object] | None:
        row = JsonStore(self.root, "manifests.jsonl").get_latest("manifest_id", manifest_id)
        return normalize_dataset_manifest(row) if row else None

    def list_for_run(self, run_id: str) -> list[dict[str, object]]:
        return [row for row in self.all() if str(row.get("run_id")) == run_id]

    def list_for_lineage(self, lineage_id: str) -> list[dict[str, object]]:
        return [row for row in self.all() if str(row.get("lineage_id")) == lineage_id]
