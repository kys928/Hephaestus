"""RunPod provider configuration using secret references, never persisted values."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping
from urllib.parse import urlparse

from hephaestus.infrastructure.secrets import SecretReference


class RunPodConfigurationError(ValueError):
    """Raised when the RunPod provider environment contract is incomplete or invalid."""


def _environment_secret(name: str) -> SecretReference:
    return SecretReference("environment", name)


def _required(values: Mapping[str, str], name: str) -> str:
    value = values.get(name, "").strip()
    if not value:
        raise RunPodConfigurationError(f"required RunPod variable is missing: {name}")
    return value


def _https_url(value: str, name: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise RunPodConfigurationError(f"{name} must be an https URL")
    return value.rstrip("/")


@dataclass(frozen=True, slots=True)
class RunPodConfig:
    """Non-secret RunPod settings plus references to runtime-only credentials."""

    datacenter_id: str
    network_volume_id: str
    s3_endpoint_url: str
    api_base_url: str
    api_key_ref: SecretReference = field(
        default_factory=lambda: _environment_secret("RUNPOD_API_KEY")
    )
    s3_access_key_id_ref: SecretReference = field(
        default_factory=lambda: _environment_secret("RUNPOD_S3_ACCESS_KEY_ID")
    )
    s3_secret_access_key_ref: SecretReference = field(
        default_factory=lambda: _environment_secret("RUNPOD_S3_SECRET_ACCESS_KEY")
    )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "RunPodConfig":
        values = environ if environ is not None else os.environ
        datacenter_id = _required(values, "RUNPOD_DATACENTER_ID")
        network_volume_id = _required(values, "RUNPOD_NETWORK_VOLUME_ID")
        s3_endpoint_url = _https_url(
            _required(values, "RUNPOD_S3_ENDPOINT_URL"), "RUNPOD_S3_ENDPOINT_URL"
        )
        api_base_url = _https_url(
            _required(values, "RUNPOD_API_BASE_URL"), "RUNPOD_API_BASE_URL"
        )
        return cls(
            datacenter_id=datacenter_id,
            network_volume_id=network_volume_id,
            s3_endpoint_url=s3_endpoint_url,
            api_base_url=api_base_url,
        )
