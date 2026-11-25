from datetime import datetime, timedelta, timezone

import pytest

from hephaestus.providers.runpod.anvil_live_lease import AnvilPodIdentity, GPU, GLOBAL, IMAGE, REGION
from hephaestus.providers.runpod.execution import RunPodApiError
from hephaestus.providers.runpod.global_volume import GlobalVolumeConfig


class Provider:
    def __init__(self, pods):
        self.pods = {row["id"]: row for row in pods}
        self.deleted = []

    def list_pods(self):
        return list(self.pods.values())

    def get_pod(self, pod_id):
        if pod_id not in self.pods:
            raise RunPodApiError("absent", status_code=404)
        return self.pods[pod_id]

    def delete_pod(self, pod_id):
        self.deleted.append(pod_id)
        self.pods.pop(pod_id)


def row(identity, **overrides):
    return dict({"id": "owned1234", "name": identity.name,
                 "env": {"ANVIL_LIVE_ACTIVATION": identity.marker,
                         "HEPHAESTUS_OPERATION_ID": identity.operation_id,
                         "HEPHAESTUS_REQUEST_SHA256": "f" * 64},
                 "cloud": "SECURE", "dataCenterId": REGION, "gpu": {"id": GPU, "count": 1},
                 "disk": 20, "image": IMAGE, "mounts": GlobalVolumeConfig(GLOBAL).mounts(),
                 "costPerHr": "0.59"}, **overrides)


def test_gpu_watchdog_deadline_and_unrelated_resources():
    identity = AnvilPodIdentity("a" * 40)
    provider = Provider([row(identity), {"id": "unrelated", "name": "other"}])
    start = datetime.now(timezone.utc)
    current = [start]
    def advance(seconds):
        current[0] += timedelta(seconds=seconds)
    assert identity.monitor(provider, deadline=start + timedelta(seconds=310),
                            now=lambda: current[0], pause=advance) == "deadline"
    assert provider.deleted == ["owned1234"]
    assert "unrelated" in provider.pods


def test_gpu_watchdog_terminates_own_wrong_sku_or_price():
    identity = AnvilPodIdentity("b" * 40)
    for pod in [row(identity, costPerHr="0.66"),
                row(identity, gpu={"id": "different", "count": 1})]:
        provider = Provider([pod])
        start = datetime.now(timezone.utc)
        with pytest.raises(PermissionError, match="unexpected"):
            identity.monitor(provider, deadline=start + timedelta(minutes=10),
                             now=lambda: start, pause=lambda _: None)
        assert provider.deleted == ["owned1234"]


def test_gpu_watchdog_refuses_name_only_ownership():
    identity = AnvilPodIdentity("c" * 40)
    provider = Provider([row(identity, env={})])
    with pytest.raises(PermissionError, match="unowned"):
        identity.cleanup(provider, pause=lambda _: None)
    assert provider.deleted == []


def test_known_id_cannot_disappear_only_from_listing():
    identity = AnvilPodIdentity("d" * 40)
    provider = Provider([row(identity)])
    provider.list_pods = lambda: []
    assert identity.cleanup(provider, known_ids={"owned1234"}, pause=lambda _: None)
    assert provider.deleted == ["owned1234"]


def test_cleanup_retries_transient_read_errors():
    identity = AnvilPodIdentity("e" * 40)
    provider = Provider([row(identity)])
    get = provider.get_pod
    attempts = [0]
    def transient(pod_id):
        attempts[0] += 1
        if attempts[0] == 1:
            raise RunPodApiError("transient",status_code=503)
        return get(pod_id)
    provider.get_pod = transient
    assert identity.cleanup(provider, known_ids={"owned1234"}, pause=lambda _: None)
    assert provider.deleted == ["owned1234"]
