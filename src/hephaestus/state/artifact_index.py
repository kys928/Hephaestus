from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hephaestus.state._json_store import JsonStore


@dataclass(slots=True)
class ArtifactIndex:
    root: Path

    def append(self, record: dict[str, object]) -> None:
        JsonStore(self.root, "artifact_index.jsonl").append(record)

    def all(self) -> list[dict[str, object]]:
        return JsonStore(self.root, "artifact_index.jsonl").all()
