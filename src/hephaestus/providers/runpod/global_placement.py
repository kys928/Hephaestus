"""Fail-closed plan for cross-datacenter RunPod Global Volume GPU workers.

Pure request construction only: does not allocate or bill for a Pod.
An independent durable operation journal / approval-bound controller must
submit and reconcile the exact request and ensure off-Pod termination.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .global_volume import GlobalVolumeConfig, _hash_ref


@dataclass(frozen=True, slots=True)
class ApprovedGpuLease:
    approval_ref: str
    datacenter_id: str
    gpu_id: str
    minimum_vram_gb: int
    maximum_hourly_usd: Decimal
    maximum_wall_seconds: int

    def __post_init__(self) -> None:
        _hash_ref(self.approval_ref)
        if not self.datacenter_id.strip() or not self.gpu_id.strip():
            raise ValueError("exact approved datacenter and GPU SKU required")
        if type(self.minimum_vram_gb) is not int or self.minimum_vram_gb < 1:
            raise ValueError("positive minimum GPU VRAM required")
        if not isinstance(self.maximum_hourly_usd, Decimal) or not self.maximum_hourly_usd.is_finite() or self.maximum_hourly_usd <= 0:
            raise ValueError("finite maximum hourly price required")
        if type(self.maximum_wall_seconds) is not int or not (60 <= self.maximum_wall_seconds <= 24 * 3600):
            raise ValueError("bounded wall-clock lease required")


def build_global_gpu_request(
    volume: GlobalVolumeConfig,
    lease: ApprovedGpuLease,
    *,
    observed_datacenter_id: str,
    observed_gpu_id: str,
    observed_vram_gb: int,
    observed_hourly_usd: Decimal,
    authenticated_balance_usd: Decimal,
    currently_spending_usd_hour: Decimal,
    worker_image: str,
    command: tuple[str, ...],
    receipt_ref: str,
    approval_verified: bool,
    independent_watchdog_ready: bool,
    name: str,
    container_disk_gb: int = 20,
) -> dict[str, object]:
    """Produce a single-DC/one-SKU Pod request from live corroborated observations.

    Approval and watchdog booleans must be produced by independent deterministic
    authorities, not model output. Always rerun price/stock checks at allocation.
    """
    _hash_ref(receipt_ref)
    if approval_verified is not True or independent_watchdog_ready is not True:
        raise PermissionError("validated human/policy approval and off-Pod watchdog are required")
    if observed_datacenter_id != lease.datacenter_id or observed_gpu_id != lease.gpu_id:
        raise ValueError("provider GPU/datacenter observation does not match approval")
    if type(observed_vram_gb) is not int or observed_vram_gb < lease.minimum_vram_gb:
        raise ValueError("offered GPU VRAM below approved scientific minimum")
    if not isinstance(observed_hourly_usd, Decimal) or not observed_hourly_usd.is_finite() or not (0 < observed_hourly_usd <= lease.maximum_hourly_usd):
        raise ValueError("provider price missing or exceeds approved ceiling")
    if not all(isinstance(v, Decimal) and v.is_finite() and v >= 0 for v in (authenticated_balance_usd, currently_spending_usd_hour)):
        raise ValueError("authenticated current provider balance/spend required")
    estimated_cost = (observed_hourly_usd + currently_spending_usd_hour) * Decimal(lease.maximum_wall_seconds) / Decimal(3600)
    if authenticated_balance_usd <= estimated_cost:
        raise ValueError("insufficient prepaid credit for bounded lease plus existing spend")
    if not name.strip() or not worker_image.strip() or not command or any(not isinstance(x, str) or not x.strip() for x in command):
        raise ValueError("nonempty Pod name, image and immutable command required")
    if type(container_disk_gb) is not int or container_disk_gb < 10:
        raise ValueError("container disk too small")
    return {
        "name": name,
        "gpu": {"id": lease.gpu_id, "count": 1},
        "cloud": "SECURE",
        "dataCenterIds": [lease.datacenter_id],
        "image": worker_image,
        "disk": container_disk_gb,
        "mounts": volume.mounts(),
        "entrypoint": [],
        "cmd": list(command),
        "env": {
            "HEPHAESTUS_GLOBAL_VOLUME_ID": volume.volume_id,
            "HEPHAESTUS_GLOBAL_MOUNT_PATH": str(volume.mount_path),
            "HEPHAESTUS_IMMUTABLE_RECEIPT_REF": receipt_ref,
            "HEPHAESTUS_APPROVED_LEASE_REF": lease.approval_ref,
        },
    }


def verify_global_pod_mount(pod: dict, planned: dict) -> bool:
    """Refuse activation if provider Pod contradicts approved spec."""
    mounts = pod.get("mounts") if isinstance(pod, dict) else None
    gpu = pod.get("gpu") if isinstance(pod, dict) else None
    return bool(
        isinstance(mounts, dict) and mounts == planned["mounts"]
        and isinstance(gpu, dict) and gpu == planned["gpu"]
        and pod.get("dataCenterId") in planned["dataCenterIds"]
        and pod.get("cloud") == "SECURE"
        and pod.get("image") == planned["image"]
        and pod.get("cmd") == planned["cmd"]
        and pod.get("status") == "RUNNING"
    )
