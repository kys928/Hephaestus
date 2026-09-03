"""Direct S3-compatible access to a RunPod Network Volume."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from hephaestus.infrastructure.capabilities import OptionalCapabilityError
from hephaestus.infrastructure.secrets import SecretsProvider

from .config import RunPodConfig


@dataclass(frozen=True, slots=True)
class NetworkVolumeObject:
    key: str
    size: int
    last_modified: datetime | None = None


@dataclass(frozen=True, slots=True)
class NetworkVolumeListing:
    volume_id: str
    objects: tuple[NetworkVolumeObject, ...]
    is_truncated: bool


@dataclass(slots=True)
class RunPodNetworkVolumeStorage:
    """Narrow arbitrary-file adapter for one configured RunPod network volume."""

    client: Any
    config: RunPodConfig

    @classmethod
    def from_boto3(
        cls, config: RunPodConfig, secrets: SecretsProvider
    ) -> "RunPodNetworkVolumeStorage":
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise OptionalCapabilityError(
                "RunPod network-volume support requires the 's3' optional dependencies"
            ) from exc

        access_key = secrets.resolve(config.s3_access_key_id_ref)
        secret_key = secrets.resolve(config.s3_secret_access_key_ref)
        client = boto3.client(
            "s3",
            endpoint_url=config.s3_endpoint_url,
            region_name=config.datacenter_id,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(retries={"mode": "standard", "max_attempts": 10}),
        )
        return cls(client=client, config=config)

    def head_volume(self) -> None:
        self.client.head_bucket(Bucket=self.config.network_volume_id)

    def list_objects(self, *, prefix: str = "", max_items: int = 100) -> NetworkVolumeListing:
        if max_items <= 0 or max_items > 1000:
            raise ValueError("max_items must be between 1 and 1000")
        response = self.client.list_objects_v2(
            Bucket=self.config.network_volume_id,
            Prefix=prefix,
            MaxKeys=max_items,
        )
        objects = tuple(
            NetworkVolumeObject(
                key=str(item.get("Key", "")),
                size=int(item.get("Size", 0)),
                last_modified=item.get("LastModified"),
            )
            for item in (response.get("Contents") or [])
        )
        return NetworkVolumeListing(
            volume_id=self.config.network_volume_id,
            objects=objects,
            is_truncated=bool(response.get("IsTruncated", False)),
        )
