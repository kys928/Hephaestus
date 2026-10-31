"""F03: concurrency and ambiguous crash outcomes must not repeat side effects."""
from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import patch

import pytest

from hephaestus.control.staged_autonomous import build_staged_autonomous_orchestrator
from hephaestus.control.staged_state import StagedAutonomousDependencies, StagedAutonomousServices, StagedOperationResult
from hephaestus.state.single_writer import WriterBusyError
from hephaestus.storage.sqlite import SQLiteStateRepository


def orchestrator(root, entry):
    return build_staged_autonomous_orchestrator(workflow_id="W", run_id="R", lineage_id="L",
        stage_name="smoke_test", dependencies=StagedAutonomousDependencies(
            services=StagedAutonomousServices(judge_entry=entry),
            state_repository=SQLiteStateRepository(root / "state.sqlite3")))


def test_contending_controller_cannot_invoke_same_operation(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []
    class Entry:
        def execute(self, request):
            calls.append(request.operation_id)
            entered.set()
            assert release.wait(5)
            return StagedOperationResult(status="blocked", resumable=False)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(orchestrator(tmp_path, Entry()).run)
        assert entered.wait(5)
        try:
            with pytest.raises(WriterBusyError):
                orchestrator(tmp_path, Entry()).run()
        finally:
            release.set()
        assert first.result().status == "blocked"
    orchestrator(tmp_path, Entry()).run()
    assert len(calls) == 1


def test_hard_stop_after_side_effect_is_not_blindly_reexecuted(tmp_path):
    calls = []
    class ControllerDeath(BaseException):
        pass
    class Entry:
        def execute(self, request):
            calls.append(request.operation_id)
            raise ControllerDeath()
    with pytest.raises(ControllerDeath):
        orchestrator(tmp_path, Entry()).run()
    result = orchestrator(tmp_path, Entry()).run()
    assert result.status == "blocked"
    assert "ambiguous_operation_outcome" in result.blocking_issues
    orchestrator(tmp_path, Entry()).run()
    assert len(calls) == 1


def test_returned_receipt_survives_crash_before_step_snapshot(tmp_path):
    calls = []
    class Entry:
        def execute(self, request):
            calls.append(request.operation_id)
            return StagedOperationResult(status="blocked", resumable=False)
    first = orchestrator(tmp_path, Entry())
    original = first._persist_state
    def crash(state):
        if state.steps:
            raise RuntimeError("crash-before-step-snapshot")
        return original(state)
    with patch.object(first, "_persist_state", side_effect=crash):
        with pytest.raises(RuntimeError, match="crash"):
            first.run()
    assert orchestrator(tmp_path, Entry()).run().status == "blocked"
    assert len(calls) == 1


def _own_until_killed(path, ready):
    from hephaestus.state.execution_ownership import execution_owner
    import time
    repository = SQLiteStateRepository(path)
    with execution_owner(repository, "workflow", "W"):
        ready.set()
        time.sleep(15)


def test_kernel_fence_prevents_process_overlap_and_releases_after_death(tmp_path):
    import multiprocessing
    from hephaestus.state.execution_ownership import execution_owner
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    path = tmp_path / "state.sqlite3"
    repository = SQLiteStateRepository(path)
    process = ctx.Process(target=_own_until_killed, args=(path, ready))
    process.start()
    try:
        assert ready.wait(8)
        with pytest.raises(WriterBusyError):
            with execution_owner(repository, "workflow", "W"):
                pytest.fail("overlapping owner")
    finally:
        process.kill()
        process.join(5)
    with execution_owner(repository, "workflow", "W"):
        pass
