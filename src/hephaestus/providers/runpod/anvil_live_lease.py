"""Exact one-Pod lifetime fence for the approved first Anvil GPU pilot.

No allocation operation. Both controllers may use cleanup after checking
ownership on authenticated provider readback.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import re
from time import sleep

from hephaestus.providers.runpod.cost_intelligence import usd
from hephaestus.providers.runpod.execution import RunPodApiError
from hephaestus.providers.runpod.global_volume import GlobalVolumeConfig
from hephaestus.storage.verified import canonical_bytes

GLOBAL = "cmw51pjsk000007l9a0mv1jga"
REGIONAL = "cviwpryzao"  # operation journal namespace, not mounted
REGION = "EU-SE-1"
GPU = "NVIDIA A40"
IMAGE = "runpod/pytorch@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851"
MAX_RATE = Decimal("0.65")
MAX_DISK_GB = 20
MAX_SECONDS = 3600
CLEANUP_RESERVE_SECONDS = 300
BASE_NAME = "hephaestus-anvil-first-live"


@dataclass(frozen=True, slots=True)
class AnvilPodIdentity:
    marker: str

    def __post_init__(self):
        if not re.fullmatch(r"[0-9a-f]{40}", self.marker):
            raise ValueError("exact activation commit required")

    @property
    def operation_id(self):
        return sha256(canonical_bytes(["https://api.runpod.io/v2", REGIONAL, self.marker])).hexdigest()

    @property
    def name(self):
        return BASE_NAME + "-" + self.operation_id[:32]

    def owned(self, pod):
        env = pod.get("env")
        if (pod.get("name") != self.name or not isinstance(env, dict)
                or env.get("ANVIL_LIVE_ACTIVATION") != self.marker
                or env.get("HEPHAESTUS_OPERATION_ID") != self.operation_id
                or not re.fullmatch(r"[0-9a-f]{64}", str(env.get("HEPHAESTUS_REQUEST_SHA256", "")))
                or not re.fullmatch(r"[A-Za-z0-9]{4,40}", str(pod.get("id", "")))):
            raise PermissionError("refuse cleanup of unowned provider resource")
        return pod["id"]

    def qualified(self, pod):
        self.owned(pod)
        gpu = pod.get("gpu") or {}
        return (pod.get("cloud") == "SECURE" and pod.get("dataCenterId") == REGION
                and gpu.get("id") == GPU and gpu.get("count") == 1
                and pod.get("image") == IMAGE and pod.get("mounts") == GlobalVolumeConfig(GLOBAL).mounts()
                and type(pod.get("disk")) is int and 10 <= pod["disk"] <= MAX_DISK_GB
                and 0 < usd(pod.get("costPerHr", pod.get("cost"))) <= MAX_RATE)

    def discover(self, provider):
        rows = [row for row in provider.list_pods() if row.get("name") == self.name]
        result = []
        for row in rows:
            try:
                live = provider.get_pod(row["id"])
            except RunPodApiError as exc:
                if exc.status_code == 404:
                    continue
                raise
            self.owned(live)
            result.append(live)
        return result

    def cleanup(self, provider, *, pause=sleep):
        for attempt in range(4):
            rows = self.discover(provider)
            if not rows:
                return True
            for row in rows:
                identifier = self.owned(row)
                provider.delete_pod(identifier)
                try:
                    provider.get_pod(identifier)
                except RunPodApiError as exc:
                    if exc.status_code != 404:
                        raise
            if attempt < 3:
                pause(3)
        if self.discover(provider):
            raise RuntimeError("exact GPU Pod remains after bounded cleanup")
        return True

    def monitor(self, provider, *, deadline, now=lambda: datetime.now(timezone.utc), pause=sleep):
        if deadline.tzinfo is None or not 0 < (deadline - now()).total_seconds() <= MAX_SECONDS:
            raise ValueError("watchdog requires an absolute approved deadline")
        seen = False
        try:
            while (deadline - now()).total_seconds() > CLEANUP_RESERVE_SECONDS:
                rows = self.discover(provider)
                if rows:
                    seen = True
                    if len(rows) != 1 or not all(self.qualified(row) for row in rows):
                        raise PermissionError("unexpected own GPU allocation; terminate it")
                elif seen:
                    return "confirmed_absent"
                pause(10)
            return "deadline"
        finally:
            self.cleanup(provider, pause=pause)
