from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class PreprocessedDataset:
    processed_dataset_ref: str
    operations: list[str] = field(default_factory=list)
    dropped_examples: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "processed_dataset_ref": self.processed_dataset_ref,
            "operations": list(self.operations),
            "dropped_examples": self.dropped_examples,
            **self.metadata,
        }


def normalize_preprocessing_output(payload: dict[str, Any]) -> dict[str, object]:
    operations = [str(op) for op in payload.get("operations", [])] if isinstance(payload.get("operations"), list) else []
    if not operations:
        operations = ["identity"]
    return PreprocessedDataset(
        processed_dataset_ref=str(payload.get("processed_dataset_ref") or payload.get("artifact_ref") or ""),
        operations=operations,
        dropped_examples=int(payload.get("dropped_examples", 0) or 0),
        metadata={k: v for k, v in payload.items() if k not in {"processed_dataset_ref", "artifact_ref", "operations", "dropped_examples"}},
    ).to_dict()
