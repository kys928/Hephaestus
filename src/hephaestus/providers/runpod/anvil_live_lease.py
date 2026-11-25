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
from urllib.error import URLError

from hephaestus.providers.runpod.cost_intelligence import usd
from hephaestus.providers.runpod.execution import RunPodApiError
from hephaestus.providers.runpod.global_volume import GlobalVolumeConfig
from hephaestus.storage.verified import canonical_bytes
from hephaestus.providers.runpod.anvil_hardware import GPU_NAMES, valid_region

GLOBAL = "cmw51pjsk000007l9a0mv1jga"
REGIONAL = "cviwpryzao"  # operation journal namespace, not mounted
REGION = "EU-SE-1"
GPU = "NVIDIA A40"
IMAGE = "runpod/pytorch@sha256:4d1721e62b56d345c83b4fd6090664be6daf9312caab5b2e76f23d8231941851"
MAX_POD_COST_USD = Decimal("0.65")  # Historical grant: total mission cost, not hourly price.
# RunPod publishes USD 0.10/GB/month for running container disks.
# USD 0.01/hour conservatively covers 20 GB even for a 28-day month.
DISK_ALLOWANCE = Decimal("0.01")
MAX_DISK_GB = 20
MAX_SECONDS = 3600
LIVE_LEASE_SECONDS = 2400  # Shorter than the approved maximum; includes cleanup.
MAX_RATE = MAX_POD_COST_USD * Decimal(3600) / Decimal(LIVE_LEASE_SECONDS)
FUNDING_RESERVE_USD = Decimal('0.05')


def mission_cost(hourly_total, seconds=LIVE_LEASE_SECONDS):
    if type(seconds) is not int or not 0 < seconds <= MAX_SECONDS:
        raise ValueError('cost duration outside approved lease')
    return hourly_total * Decimal(seconds) / Decimal(3600)


def account_draw(account, hourly_total, seconds=LIVE_LEASE_SECONDS):
    if type(seconds) is not int or not 0 < seconds <= MAX_SECONDS:
        raise ValueError('funding duration outside approved lease')
    return (hourly_total + account.already_spending_usd_per_hour) * Decimal(seconds) / Decimal(3600)


def fundable(account, hourly_total, seconds=LIVE_LEASE_SECONDS):
    return (not account.autopay_enabled and 0 < hourly_total
            and mission_cost(hourly_total,seconds)<=MAX_POD_COST_USD
            and account.credit_balance_usd >= account_draw(account,hourly_total,seconds)+FUNDING_RESERVE_USD)
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

    def qualified(self, pod, *, region=REGION, sku=GPU):
        self.owned(pod)
        gpu = pod.get("gpu") or {}
        return (pod.get("cloud") == "SECURE" and valid_region(region) and pod.get("dataCenterId") == region
                and sku in GPU_NAMES and gpu.get("id") == sku and gpu.get("count") == 1
                and pod.get("image") == IMAGE and pod.get("mounts") == GlobalVolumeConfig(GLOBAL).mounts()
                and type(pod.get("disk")) is int and 10 <= pod["disk"] <= MAX_DISK_GB
                and 0 < usd(pod.get("costPerHr", pod.get("cost"))) <= MAX_RATE - DISK_ALLOWANCE)

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

    def _known_live(self, provider, known):
        # Never use a list omission as proof that a known resource is gone.
        for row in self.discover(provider):
            known.add(self.owned(row))
        live = []
        for pod_id in sorted(known):
            try:
                pod = provider.get_pod(pod_id)
            except RunPodApiError as exc:
                if exc.status_code == 404:
                    continue
                raise
            if self.owned(pod) != pod_id:
                raise PermissionError("known resource identity differs")
            live.append(pod)
        return live

    def cleanup(self, provider, *, known_ids=(), pause=sleep):
        known = set(known_ids)
        for attempt in range(8):
            try:
                rows = self._known_live(provider, known)
                if not rows:
                    # Two authenticated observations protect against a lost
                    # allocation acknowledgement and transient list omissions.
                    pause(3)
                    if not self._known_live(provider, known):
                        return True
                    continue
                for row in rows:
                    provider.delete_pod(self.owned(row))
            except (RunPodApiError, URLError, TimeoutError, OSError):
                # Unknown provider state is not successful deletion.
                pass
            if attempt < 7:
                pause(3)
        if self._known_live(provider, known):
            raise RuntimeError("exact GPU Pod remains after bounded cleanup")
        return True

    def monitor(self, provider, *, deadline, placement=None, now=lambda: datetime.now(timezone.utc), pause=sleep):
        if deadline.tzinfo is None or not 0 < (deadline - now()).total_seconds() <= MAX_SECONDS:
            raise ValueError("watchdog requires an absolute approved deadline")
        known = set()
        errors = 0
        try:
            while (deadline - now()).total_seconds() > CLEANUP_RESERVE_SECONDS:
                try:
                    rows = self._known_live(provider, known)
                    errors = 0
                except (RunPodApiError, URLError, TimeoutError, OSError):
                    errors += 1
                    if errors >= 3:
                        raise RuntimeError("provider visibility lost; attempt exact cleanup")
                    pause(5)
                    continue
                if rows:
                    if len(rows) != 1 or not all(self.qualified(row, **(placement() if placement else {})) for row in rows):
                        raise PermissionError("unexpected own GPU allocation; terminate it")
                elif known:
                    return "confirmed_absent"
                pause(10)
            return "deadline"
        finally:
            self.cleanup(provider, known_ids=known, pause=pause)
