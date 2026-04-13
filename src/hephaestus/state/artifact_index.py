"""State store module: ArtifactIndex."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ArtifactIndex:
    """Append-only / version-aware store skeleton.

    Heavy artifacts must be stored in artifacts/ and referenced by path only.
    """

    root: Path

    def append(self, record: dict[str, Any]) -> None:
        """TODO: persist record with schema validation and immutable history."""
        _ = record

    def get(self, key: str) -> dict[str, Any] | None:
        """TODO: read latest record version for key."""
        _ = key
        return None
