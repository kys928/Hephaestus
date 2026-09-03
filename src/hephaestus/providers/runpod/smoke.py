"""Opt-in live smoke test for RunPod credentials, storage, and Pod lifecycle."""

from __future__ import annotations

import argparse
import json
import uuid

from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider

from .config import RunPodConfig
from .execution import RunPodExecutionAdapter
from .storage import RunPodNetworkVolumeStorage


def _listing_payload(listing: object) -> dict[str, object]:
    return {
        "volume_id": listing.volume_id,
        "object_count_returned": len(listing.objects),
        "is_truncated": listing.is_truncated,
        "objects": [
            {"key": item.key, "size": item.size}
            for item in listing.objects
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-items", type=int, default=100)
    args = parser.parse_args()

    config = RunPodConfig.from_env()
    secrets = EnvironmentSecretsProvider()

    storage = RunPodNetworkVolumeStorage.from_boto3(config, secrets)
    storage.head_volume()
    listing = storage.list_objects(max_items=args.max_items)
    print(json.dumps({"storage": _listing_payload(listing)}, default=str, sort_keys=True))

    execution = RunPodExecutionAdapter(config=config, secrets=secrets)
    pod_id: str | None = None
    try:
        pod = execution.create_disposable_cpu_pod(
            name=f"hephaestus-smoke-{uuid.uuid4().hex[:8]}"
        )
        pod_id = str(pod["id"])
        attached = pod.get("networkVolume")
        attached_id = attached.get("id") if isinstance(attached, dict) else None
        if attached_id is not None and attached_id != config.network_volume_id:
            raise RuntimeError(
                "RunPod created the smoke Pod with an unexpected network volume"
            )
        print(
            json.dumps(
                {
                    "pod_created": {
                        "id": pod_id,
                        "desired_status": pod.get("desiredStatus"),
                        "network_volume_id": attached_id or config.network_volume_id,
                        "cost_per_hr": pod.get("costPerHr"),
                    }
                },
                sort_keys=True,
            )
        )
    finally:
        if pod_id is not None:
            execution.delete_pod(pod_id)
            print(json.dumps({"pod_deleted": {"id": pod_id}}, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
