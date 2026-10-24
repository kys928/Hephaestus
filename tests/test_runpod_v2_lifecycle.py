"""Documented REST v2 shapes and fault injection. No network or paid resources."""
from copy import deepcopy
import hashlib
from dataclasses import replace
from urllib.parse import parse_qs, urlsplit

import pytest

from hephaestus.providers.runpod.config import RunPodConfig, RunPodConfigurationError
from hephaestus.providers.runpod.execution import AmbiguousRunPodOperation
from hephaestus.schemas.external_operation import ExternalOperation
from hephaestus.state.external_operation_store import ExternalOperationStore
from hephaestus.storage.sqlite import SQLiteStateRepository
from hephaestus.storage.verified import canonical_bytes
from test_external_allocation_recovery import AcceptingProvider, ControllerCrash, _adapter, _create
from test_runpod_provider import _config


def test_v1_configuration_is_rejected_before_credentials_or_http():
    with pytest.raises(RunPodConfigurationError, match="v1 Pod contract"):
        replace(_config(), api_base_url="https://rest.runpod.io/v1")


def test_nested_gpu_request_matches_documented_v2_contract(tmp_path):
    provider = AcceptingProvider()
    pod = _create(_adapter(tmp_path, provider))
    # The v2 OpenAPI CreatePodRequest replaces all v1 placement/storage fields.
    assert set(pod) == {"name", "image", "gpu", "cloud", "dataCenterIds", "disk", "mounts",
                        "entrypoint", "cmd", "env", "id", "status", "dataCenterId"}
    assert pod["mounts"] == {"network": [{"volumeId": "cviwpryzao", "path": "/workspace"}]}
    assert pod["gpu"] == {"id": "test-gpu", "count": 1}
    assert not _adapter(tmp_path, provider).is_ready(pod)
    assert _adapter(tmp_path, provider).is_ready({**pod, "status": "RUNNING"})
    for status in ("STARTING", "EXITED", "ERROR", "TERMINATED", None):
        assert not _adapter(tmp_path, provider).is_ready({"status": status, "desiredStatus": "RUNNING"})


@pytest.mark.parametrize("duplicate", [False, True])
def test_reconciliation_scans_all_cursor_pages_before_accepting(tmp_path, duplicate):
    class Paginated(AcceptingProvider):
        def request(self, method, url, **kwargs):
            if method == "GET" and urlsplit(url).path.endswith("/pods"):
                cursor = parse_qs(urlsplit(url).query).get("cursor")
                if not cursor:
                    return 200, {"pods": deepcopy(self.pods) if duplicate else [],
                                 "pagination": {"hasNextPage": True, "nextCursor": "opaque +/="}}
                assert cursor == ["opaque +/="]
                return 200, {"pods": deepcopy(self.pods), "pagination": {"hasNextPage": False, "nextCursor": None}}
            return super().request(method, url, **kwargs)
    provider = Paginated(crash_after_accept=True)
    with pytest.raises(ControllerCrash):
        _create(_adapter(tmp_path, provider))
    if duplicate:
        with pytest.raises(AmbiguousRunPodOperation):
            _create(_adapter(tmp_path, provider))
    else:
        assert _create(_adapter(tmp_path, provider))["id"] == "pod-1"
    assert provider.posts == 1


def test_gpu_catalog_selection_is_pinned_and_not_repeated_on_retry(tmp_path):
    class Catalog(AcceptingProvider):
        def request(self, method, url, **kwargs):
            if "/catalog/" in url:
                assert "/EU-CZ-1?include=GPU_AVAILABILITY" in url
                return 200, {"id": "EU-CZ-1", "gpuAvailability": [{"id": "fallback", "availability": "LOW"}]}
            return super().request(method, url, **kwargs)
    provider = Catalog(crash_after_accept=True)
    with pytest.raises(ControllerCrash):
        _create(_adapter(tmp_path, provider), gpu_type_ids=["unavailable", "fallback"])
    recovered = _create(_adapter(tmp_path, provider), gpu_type_ids=["unavailable", "fallback"])
    assert recovered["gpu"]["id"] == "fallback" and provider.posts == 1


def test_unsupported_interruptible_request_has_no_external_effects(tmp_path):
    provider = AcceptingProvider()
    with pytest.raises(ValueError, match="interruptible"):
        _create(_adapter(tmp_path, provider), interruptible=True)
    assert not provider.posts and not provider.gets


def test_lost_termination_response_retries_delete_and_records_absence(tmp_path):
    class Deleting(AcceptingProvider):
        deletes = 0
        def request(self, method, url, **kwargs):
            if method == "DELETE":
                self.deletes += 1
                if self.deletes == 1:
                    self.pods.clear()
                    raise TimeoutError("response lost after deletion")
                return 404, {"title": "Not Found", "status": 404, "detail": "pod not found"}
            return super().request(method, url, **kwargs)
    provider = Deleting()
    _create(_adapter(tmp_path, provider))
    with pytest.raises(TimeoutError):
        _adapter(tmp_path, provider).delete_pod("pod-1")
    store = ExternalOperationStore(SQLiteStateRepository(tmp_path / "runpod-operations.sqlite3"))
    assert store.for_resource("pod-1").status == "terminating"
    _adapter(tmp_path, provider).delete_pod("pod-1")
    assert store.for_resource("pod-1").status == "terminated"
    _adapter(tmp_path, provider).delete_pod("pod-1")
    assert provider.deletes == 2
    with pytest.raises(AmbiguousRunPodOperation, match="termination"):
        _create(_adapter(tmp_path, provider))
    assert provider.posts == 1


def test_v1_intent_cannot_be_ignored_by_new_version(tmp_path):
    key = hashlib.sha256(canonical_bytes(["https://rest.runpod.io/v1", _config().network_volume_id,
                                          "stable-staged-operation"])).hexdigest()
    store = ExternalOperationStore(SQLiteStateRepository(tmp_path / "runpod-operations.sqlite3"))
    store.record(ExternalOperation("runpod-"+key, "runpod", "old-hash", "old-resource", "submitting"))
    provider = AcceptingProvider()
    with pytest.raises(AmbiguousRunPodOperation, match="legacy v1 intent"):
        _create(_adapter(tmp_path, provider))
    assert provider.posts == 0
