"""Deterministic provider acceptance/crash simulations; never contacts RunPod."""
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
import pytest

from hephaestus.providers.runpod.execution import RunPodExecutionAdapter, AmbiguousRunPodOperation
from hephaestus.state.external_operation_store import ExternalOperationStore, EXTERNAL_OPERATIONS
from hephaestus.storage.sqlite import SQLiteStateRepository
from test_runpod_provider import _config, _secrets


class ControllerCrash(BaseException):
    pass


@dataclass
class AcceptingProvider:
    crash_after_accept: bool = False
    timeout_after_accept: bool = False
    hide_resources: bool = False
    duplicate_match: bool = False
    posts: int = 0
    gets: int = 0
    pods: list[dict] = field(default_factory=list)

    def request(self, method, url, *, headers, json_body=None):
        if method == "POST":
            self.posts += 1
            pod = {**deepcopy(json_body), "id":f"pod-{self.posts}", "status":"PROVISIONING", "dataCenterId":"EU-CZ-1"}
            self.pods.append(pod)
            if self.crash_after_accept:
                self.crash_after_accept = False
                raise ControllerCrash("provider accepted; controller never recorded ID")
            if self.timeout_after_accept:
                self.timeout_after_accept = False
                raise TimeoutError("response lost after remote acceptance")
            return 201, deepcopy(pod)
        if method == "GET":
            self.gets += 1
            if urlsplit(url).path.endswith("/pods"):
                return 200, {"pods": [] if self.hide_resources else deepcopy(self.pods + (self.pods if self.duplicate_match else [])),
                    "pagination": {"nextCursor": None, "hasNextPage": False}}
            pod_id = urlsplit(url).path.rsplit("/",1)[-1]
            return 200, deepcopy(next(p for p in self.pods if p["id"] == pod_id))
        raise AssertionError(method)


def _adapter(root, provider):
    return RunPodExecutionAdapter(_config(), _secrets(), provider, state_root=root)


def _create(adapter, **changes):
    args = dict(name="bounded-recovery", image_name="test-image:immutable", gpu_type_ids=["test-gpu"],
        docker_start_cmd=["python", "worker.py"], env={"RUN_ID":"R", "PRIVATE_TEST_VALUE":"do-not-persist"},
        operation_id="stable-staged-operation")
    return adapter.create_bounded_gpu_pod(**{**args, **changes})


@pytest.mark.parametrize("failure", ["crash", "timeout"])
def test_remote_acceptance_without_local_id_is_reconciled_without_duplicate_post(tmp_path, failure):
    provider = AcceptingProvider(crash_after_accept=failure=="crash", timeout_after_accept=failure=="timeout")
    first = _adapter(tmp_path, provider)
    with pytest.raises(ControllerCrash if failure == "crash" else AmbiguousRunPodOperation):
        _create(first)
    # New controller and repository instances; only the durable intent survives.
    recovered = _create(_adapter(tmp_path, provider))
    assert recovered["id"] == "pod-1"
    assert provider.posts == 1 and provider.gets == 1
    assert _create(_adapter(tmp_path, provider))["id"] == "pod-1"
    rows = SQLiteStateRepository(tmp_path / "runpod-operations.sqlite3").all(EXTERNAL_OPERATIONS)
    assert rows[0]["status"] == "submitting"
    assert rows[-1]["status"] == "accepted"
    assert "do-not-persist" not in str(rows)
    assert "api-test" not in str(rows)


@pytest.mark.parametrize("fault", ["not_found", "duplicate", "markers", "volume", "image"])
def test_ambiguous_or_mismatched_remote_resources_never_trigger_reallocation(tmp_path, fault):
    provider = AcceptingProvider(crash_after_accept=True)
    with pytest.raises(ControllerCrash): _create(_adapter(tmp_path, provider))
    if fault == "not_found": provider.hide_resources = True
    elif fault == "duplicate": provider.duplicate_match = True
    elif fault == "markers": provider.pods[0]["env"]["HEPHAESTUS_REQUEST_SHA256"] = "wrong"
    elif fault == "volume": provider.pods[0]["mounts"]["network"][0]["volumeId"] = "unrelated-volume"
    else: provider.pods[0]["image"] = "unrelated-image"
    for _ in range(5):
        with pytest.raises(AmbiguousRunPodOperation): _create(_adapter(tmp_path, provider))
    assert provider.posts == 1
    assert provider.gets <= 3


def test_stable_operation_cannot_be_reused_with_a_changed_request(tmp_path):
    provider = AcceptingProvider()
    _create(_adapter(tmp_path, provider))
    with pytest.raises(ValueError, match="changed its request"):
        _create(_adapter(tmp_path, provider), docker_start_cmd=["python", "different-experiment.py"])
    assert provider.posts == 1


def test_staged_controller_recovers_acceptance_window_with_same_operation_identity(tmp_path):
    from test_staged_autonomous_orchestration import FakeOperationRouter, _orchestrator
    provider = AcceptingProvider(crash_after_accept=True)
    class LaunchRouter(FakeOperationRouter):
        def execute(self, request):
            if request.substep == "lifecycle_launch":
                _create(_adapter(tmp_path / "provider-state", provider), operation_id=request.operation_id)
            return super().execute(request)
        def reconcile_operation(self, request):
            # Explicit adapter: the actual RunPod _create path reconciles its
            # durable intent/provider identity before considering allocation.
            return self.execute(request)
    first, _ = _orchestrator(tmp_path / "workflow", LaunchRouter())
    with pytest.raises(ControllerCrash): first.run()
    resumed, _ = _orchestrator(tmp_path / "workflow", LaunchRouter())
    result = resumed.run()
    if result.status == "running":
        result = resumed.run()
    assert result.status == "completed"
    assert provider.posts == 1 and provider.gets == 1


def test_staged_transport_timeout_waits_and_reconciles_instead_of_failing_or_resubmitting(tmp_path):
    from test_staged_autonomous_orchestration import FakeOperationRouter, _orchestrator
    provider = AcceptingProvider(timeout_after_accept=True)
    class LaunchRouter(FakeOperationRouter):
        def execute(self, request):
            if request.substep == "lifecycle_launch":
                _create(_adapter(tmp_path / "provider-state", provider), operation_id=request.operation_id)
            return super().execute(request)
    first, _ = _orchestrator(tmp_path / "workflow", LaunchRouter())
    waiting = first.run()
    assert waiting.status == "interrupted" and waiting.resumable is True
    assert provider.posts == 1
    resumed, _ = _orchestrator(tmp_path / "workflow", LaunchRouter())
    assert resumed.run().status == "completed"
    assert provider.posts == 1 and provider.gets == 1


def test_receipt_write_failure_after_acceptance_reconciles_without_duplicate_post(tmp_path, monkeypatch):
    provider = AcceptingProvider()
    original = ExternalOperationStore.record
    failed = False
    def storage_failure(self, operation):
        nonlocal failed
        if operation.status == "accepted" and not failed:
            failed = True
            raise OSError("injected durable receipt write failure")
        return original(self, operation)
    monkeypatch.setattr(ExternalOperationStore, "record", storage_failure)
    with pytest.raises(AmbiguousRunPodOperation, match="could not be persisted"):
        _create(_adapter(tmp_path, provider))
    assert _create(_adapter(tmp_path, provider))["id"] == "pod-1"
    assert provider.posts == 1
