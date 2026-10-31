"""F10: waiting is observation; only real failures consume retry allowance."""
import pytest

from hephaestus.production.composition import ProductionCompositionRoot, ProductionCompositionSettings
from hephaestus.production.loop import ProductionCycleResult, ProductionLoopRunner
from hephaestus.production.recovery import InfrastructureRecoveryController, RecoverableInfrastructureError
from hephaestus.control.spine import SPINE_ORDER
from hephaestus.storage.sqlite import SQLiteStateRepository


def test_more_than_five_healthy_polls_then_completion_across_restarts(tmp_path):
    class PendingDriver:
        calls = 0
        def execute_cycle(self, **kwargs):
            self.calls += 1
            return ProductionCycleResult(cycle_id="p:cycle:1", run_id="R", experiment_id="E",
                status="pending" if self.calls <= 9 else "completed", judge_action="request_recheck",
                phase_evidence={p.value: ["synthetic:" + p.value] for p in SPINE_ORDER})
    driver = PendingDriver()
    settings = ProductionCompositionSettings(state_root=tmp_path / "state", artifact_root=tmp_path / "artifacts")
    for _ in range(9):
        runtime = ProductionCompositionRoot(settings).build()
        result = ProductionLoopRunner(runtime, driver, maximum_cycles=1).run(
            program_id="p", lineage_id="L", stage_name="smoke_test")
        assert result.status == "waiting" and result.recovery_attempts == 0
    runtime = ProductionCompositionRoot(settings).build()
    result = ProductionLoopRunner(runtime, driver, maximum_cycles=1).run(
        program_id="p", lineage_id="L", stage_name="smoke_test")
    assert result.cycle_index == 1 and driver.calls == 10


def test_exhaustion_is_explicit_and_persisted_without_an_extra_call(tmp_path):
    repository = SQLiteStateRepository(tmp_path / "state.sqlite3")
    calls = []
    def fails(attempt):
        calls.append(attempt)
        raise RecoverableInfrastructureError("capacity_unavailable", "no capacity")
    with pytest.raises(RecoverableInfrastructureError):
        InfrastructureRecoveryController(repository, maximum_attempts=3).run("op", fails)
    assert calls == [1, 2, 3]
    with pytest.raises(RecoverableInfrastructureError, match="exhausted"):
        InfrastructureRecoveryController(repository, maximum_attempts=3).run("op", fails)
    assert calls == [1, 2, 3]


def test_success_does_not_erase_prior_failure_count(tmp_path):
    repository = SQLiteStateRepository(tmp_path / "state.sqlite3")
    controller = InfrastructureRecoveryController(repository, maximum_attempts=2)
    attempts = []
    def one_failure(attempt):
        attempts.append(attempt)
        if attempt == 1:
            raise RecoverableInfrastructureError("capacity_unavailable", "retry")
        return "pending"
    assert controller.run("op", one_failure) == "pending"
    for _ in range(10):
        assert controller.run("op", lambda attempt: attempt) == 2
    assert attempts == [1, 2]
