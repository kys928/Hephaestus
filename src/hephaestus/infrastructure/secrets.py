"""Secret-reference contracts. Persist references; resolve values only at runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping, Protocol


class SecretResolutionError(LookupError):
    pass


@dataclass(frozen=True, slots=True)
class SecretReference:
    provider: str
    key: str

    def to_dict(self) -> dict[str, str]:
        return {"provider": self.provider, "key": self.key}


class SecretsProvider(Protocol):
    provider_id: str

    def resolve(self, reference: SecretReference) -> str: ...


@dataclass(slots=True)
class EnvironmentSecretsProvider:
    provider_id: str = "environment"
    environ: Mapping[str, str] | None = field(default=None, repr=False)

    def resolve(self, reference: SecretReference) -> str:
        if reference.provider != self.provider_id:
            raise SecretResolutionError(
                f"provider {reference.provider!r} is not supported by {self.provider_id!r}"
            )
        values = self.environ if self.environ is not None else os.environ
        try:
            return values[reference.key]
        except KeyError as exc:
            raise SecretResolutionError(f"secret reference is unavailable: {reference.key}") from exc
