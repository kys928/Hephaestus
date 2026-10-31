"""Append-only source screening records; all reports remain content-addressed."""
from dataclasses import dataclass
from pathlib import Path
from hephaestus.state._json_store import JsonStore


@dataclass(slots=True)
class ScreeningStore:
    root: Path

    def append(self, record: dict) -> None:
        JsonStore(self.root, "quarantine_screening.jsonl").append(record)
