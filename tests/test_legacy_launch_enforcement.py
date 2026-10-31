"""F11: legacy denial is enforced and approval-looking strings grant nothing."""
import pytest

from hephaestus.backends.local_process_backend import LocalProcessBackend
from hephaestus.control.orchestrator import build_orchestrator
from hephaestus.config_loader import ConfigError
from hephaestus.policy.runtime_policy import RuntimePolicy
from hephaestus.roles.runtime_monitor import RuntimeMonitorRole
from hephaestus.safety.launch_guard import evaluate_launch_guard


def test_legacy_denial_stops_before_backend_launch(tmp_path):
    class Sentinel(LocalProcessBackend):
        calls = 0
        def launch_training(self, prepared_job):
            self.calls += 1
            raise RuntimeError("unexpected launch")
    backend = Sentinel()
    with pytest.raises((ConfigError, PermissionError)):
        build_orchestrator(tmp_path, "legacy-denied", backend=backend).run("legacy-denied")
    assert backend.calls == 0


def test_caller_approval_string_does_not_authorize_legacy_launch():
    result = evaluate_launch_guard({"dry_run": False, "approval_request_id": "caller-invented"})
    assert not result.passed


def test_runtime_monitor_also_enforces_denial_before_prepare():
    class Sentinel:
        def prepare_training_job(self, **kwargs):
            raise AssertionError("denial must happen before preparation")
    with pytest.raises(PermissionError):
        RuntimeMonitorRole(Sentinel(), RuntimePolicy()).run(run_id="R",
            experiment_plan={}, training_plan={}, data_contract={},
            launch_config={"dry_run": False, "approval_request_id": "fake"})


def test_fixture_mode_cannot_authorize_an_arbitrary_configured_runner(tmp_path):
    class Sentinel(LocalProcessBackend):
        def launch_training(self, prepared_job):
            pytest.fail("unapproved runner reached launch")
    with pytest.raises(PermissionError, match="configurable trainer"):
        RuntimeMonitorRole(Sentinel(), RuntimePolicy()).run(run_id="R",
            experiment_plan={}, training_plan={"run_id": "R", "max_steps": 1},
            data_contract={"processed_dataset_ref": "synthetic:input"}, legacy_fixture_mode=True,
            launch_config={"dry_run": False, "artifact_root": str(tmp_path),
                           "parameters": {"trainer_script": str(tmp_path / "arbitrary.py")}})
