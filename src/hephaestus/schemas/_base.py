"""Shared schema helpers for JSON-serializable dataclass models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, TypeVar


T = TypeVar("T", bound="JsonSchema")


@dataclass(slots=True)
class JsonSchema:
    """Base dataclass with explicit JSON-friendly export and import."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls: type[T], payload: dict[str, Any]) -> T:
        return cls(**payload)
