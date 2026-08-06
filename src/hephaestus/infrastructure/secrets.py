"""Secret-reference contracts. Persist references; resolve values only at runtime."""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol

from .capabilities import OptionalCapabilityError


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


_SECRET_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")


@dataclass(slots=True)
class FileMountedSecretsProvider:
    """Resolve orchestrator-mounted secret files with strict ownership and modes."""

    root: Path
    provider_id: str = "file"
    require_owner: bool = True

    def resolve(self, reference: SecretReference) -> str:
        if reference.provider != self.provider_id:
            raise SecretResolutionError(
                f"provider {reference.provider!r} is not supported by {self.provider_id!r}"
            )
        if not _SECRET_KEY.fullmatch(reference.key) or ".." in Path(reference.key).parts:
            raise SecretResolutionError("secret reference has an unsafe file key")
        root = self.root.resolve()
        candidate = root / reference.key
        path = candidate.resolve()
        if root != path and root not in path.parents:
            raise SecretResolutionError("secret reference escapes the mounted secret root")
        try:
            metadata = candidate.lstat()
        except OSError as exc:
            raise SecretResolutionError(f"secret reference is unavailable: {reference.key}") from exc
        if candidate.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise SecretResolutionError("mounted secret must be a regular non-symlink file")
        if metadata.st_mode & 0o077:
            raise SecretResolutionError("mounted secret permissions must not grant group/other access")
        if self.require_owner and hasattr(os, "getuid") and metadata.st_uid != os.getuid():
            raise SecretResolutionError("mounted secret must be owned by the worker user")
        try:
            return path.read_text(encoding="utf-8").rstrip("\r\n")
        except OSError as exc:
            raise SecretResolutionError(f"secret reference is unreadable: {reference.key}") from exc


@dataclass(slots=True)
class InjectedSecretsProvider:
    """Cloud-neutral production boundary around an injected secret getter."""

    provider_id: str
    getter: Callable[[str], str] = field(repr=False)

    def resolve(self, reference: SecretReference) -> str:
        if reference.provider != self.provider_id:
            raise SecretResolutionError(
                f"provider {reference.provider!r} is not supported by {self.provider_id!r}"
            )
        if not _SECRET_KEY.fullmatch(reference.key):
            raise SecretResolutionError("secret reference has an unsafe key")
        try:
            value = self.getter(reference.key)
        except Exception as exc:
            raise SecretResolutionError(f"secret reference is unavailable: {reference.key}") from exc
        if not isinstance(value, str) or not value:
            raise SecretResolutionError(f"secret reference returned no string value: {reference.key}")
        return value


@dataclass(slots=True)
class AwsSecretsManagerProvider:
    """Optional AWS Secrets Manager adapter using an injected boto3-compatible client."""

    client: object = field(repr=False)
    provider_id: str = "aws-secrets-manager"

    @classmethod
    def from_boto3(cls, **client_kwargs: object) -> "AwsSecretsManagerProvider":
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise OptionalCapabilityError(
                "AWS secret support requires the 'aws-secrets' optional dependencies"
            ) from exc
        return cls(boto3.client("secretsmanager", **client_kwargs))

    def resolve(self, reference: SecretReference) -> str:
        if reference.provider != self.provider_id:
            raise SecretResolutionError(
                f"provider {reference.provider!r} is not supported by {self.provider_id!r}"
            )
        if not _SECRET_KEY.fullmatch(reference.key):
            raise SecretResolutionError("secret reference has an unsafe key")
        try:
            response = self.client.get_secret_value(SecretId=reference.key)
        except Exception as exc:
            raise SecretResolutionError(f"secret reference is unavailable: {reference.key}") from exc
        value = response.get("SecretString")
        if value is None:
            binary = response.get("SecretBinary")
            value = binary.decode("utf-8") if isinstance(binary, bytes) else None
        if not isinstance(value, str) or not value:
            raise SecretResolutionError(f"secret reference returned no string value: {reference.key}")
        return value
