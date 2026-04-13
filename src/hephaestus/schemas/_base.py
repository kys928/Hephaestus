"""Shared schema helpers for JSON-serializable dataclass models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(slots=True)
class JsonSchema:
    """Base dataclass with explicit JSON-friendly export."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
