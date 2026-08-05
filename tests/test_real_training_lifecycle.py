from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from hephaestus.interfaces import TrainingLifecycleService
from hephaestus.schemas.experiment_contract import (
    ExperimentProposal,
    TrainingControlRequest,
)
from hephaestus.training import (
    FakeTrainingLifecycleService,
    LocalTrainingLifecycleService,
)

_ACTIVE = {"preparing", "queued", "running", "interrupting", "resuming"}


def _proposal(tmp_path: Path, run_id: str, **overrides: object) -> ExperimentProposal:
    dataset = tmp_path / f"{run_id.replace('/', '_')}-train.txt"
    dataset.write_text("hephaestus bounded fixture language-model training corpus", encoding="utf-8")
    constraints: dict[str, object] = {
        "backend_id": "local_fixture",
        "model_id": "tiny-char-lm",
        "model_revision": "fixture-v1",
        "architecture_family": "tiny_linear_lm",
        "tokenizer_ref": "fixture://byte-tokenizer/v1",
        "training_recipe_ref": "fixture://recipes/smoke-v1",
        "data_contract_ref": str(dataset),
        "data_contract_hash": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "max_steps": 8,
        "learning_rate": 0.05,
    }
    constraints.update(overrides)
    return ExperimentProposal(
        experiment_id=f"experiment-{run_id}",
        run_id=run_id,
        lineage_id="lineage-fixture",
        stage_name="smoke_test",
        diagnosis_report_id="diagnosis-fixture",
        intervention_id="intervention-fixture",
        primary_variable="training_recipe",
        training_constraints=constraints,
        status="ready",
    )


def _wait(service: LocalTrainingLifecycleService, run_id: str, wanted: set[str], timeout: float = 8.0):
    deadline = time.monotonic() + timeout
    handle = service.status(run_id)
    while handle.status not in wanted and time.monotonic() < deadline:
        time.sleep(0.02)
        handle = service.status(run_id)
    assert handle.status in wanted, handle.to_dict()
    return handle


def test_tiny_fixture_runs_real_bounded_training_and_hashes_checkpoint(tmp_path: Path) -> None:
    service = LocalTrainingLifecycleService(tmp_path / "runs")
    assert isinstance(service, TrainingLifecycleService)
    launched = service.launch(_proposal(tmp_path, "real-success"))
    assert launched.status == "running"
    completed = _wait(service, "real-success", {"completed", "failed"})
    assert completed.status == "completed"
    assert Path(completed.metrics_ref or "").is_file()
    assert Path(completed.event_stream_ref or "").is_file()
    assert Path(str(completed.metadata["log_ref"])).is_file()
    assert len(completed.checkpoint_refs) == 1

    record = json.loads(Path(str(completed.metadata["checkpoint_record_ref"])).read_text(encoding="utf-8"))
    checkpoint = Path(record["checkpoint_ref"])
    assert record["integrity_level"] == "content_hash_verified"
    assert record["content_hash"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert record["resume_compatibility"]["architecture_family"] == "tiny_linear_lm"
    metrics = json.loads(Path(completed.metrics_ref or "").read_text(encoding="utf-8"))
    assert metrics["steps"] == 8
    assert metrics["finite"] is True


def test_interrupt_is_honest_and_compatible_resume_completes(tmp_path: Path) -> None:
    service = LocalTrainingLifecycleService(tmp_path / "runs")
    service.launch(_proposal(tmp_path, "interrupt-resume", max_steps=80, step_delay_seconds=0.01))
    metrics_stream = tmp_path / "runs" / "interrupt-resume" / "metrics.jsonl"
    deadline = time.monotonic() + 5
    while not metrics_stream.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    interrupted_request = TrainingControlRequest("control-1", "interrupt-resume", "interrupt", "test", "verify checkpointed interruption")
    assert service.control(interrupted_request).status == "interrupting"
    interrupted = _wait(service, "interrupt-resume", {"interrupted", "failed"})
    assert interrupted.status == "interrupted"
    assert Path(interrupted.resume_token_ref or "").is_file()

    resume_request = TrainingControlRequest("control-2", "interrupt-resume", "resume", "test", "continue compatible run")
    assert service.control(resume_request).status == "resuming"
    completed = _wait(service, "interrupt-resume", {"completed", "failed"})
    assert completed.status == "completed"


def test_resume_refuses_model_or_tokenizer_mismatch(tmp_path: Path) -> None:
    service = LocalTrainingLifecycleService(tmp_path / "runs")
    service.launch(_proposal(tmp_path, "resume-mismatch", max_steps=80, step_delay_seconds=0.01))
    metrics_stream = tmp_path / "runs" / "resume-mismatch" / "metrics.jsonl"
    deadline = time.monotonic() + 5
    while not metrics_stream.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    service.control(TrainingControlRequest("control-1", "resume-mismatch", "interrupt", "test", "pause"))
    interrupted = _wait(service, "resume-mismatch", {"interrupted", "failed"})
    assert interrupted.status == "interrupted"

    refused = service.control(TrainingControlRequest(
        "control-2",
        "resume-mismatch",
        "resume",
        "test",
        "attempt incompatible resume",
        metadata={"model_revision": "different-revision"},
    ))
    assert refused.status == "interrupted"
    assert refused.issues[-1].code == "resume_request_mismatch"


def test_cancel_reports_transition_then_terminal_cancellation(tmp_path: Path) -> None:
    service = LocalTrainingLifecycleService(tmp_path / "runs")
    service.launch(_proposal(tmp_path, "cancelled-run", max_steps=80, step_delay_seconds=0.01))
    metrics_stream = tmp_path / "runs" / "cancelled-run" / "metrics.jsonl"
    deadline = time.monotonic() + 5
    while not metrics_stream.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    cancelling = service.control(TrainingControlRequest("control-cancel", "cancelled-run", "cancel", "test", "stop bounded run"))
    assert cancelling.status == "interrupting"
    cancelled = _wait(service, "cancelled-run", {"cancelled", "failed"})
    assert cancelled.status == "cancelled"


def test_non_zero_exit_becomes_persisted_incident(tmp_path: Path) -> None:
    service = LocalTrainingLifecycleService(tmp_path / "runs")
    service.launch(_proposal(tmp_path, "non-zero", force_exit_code=7))
    failed = _wait(service, "non-zero", {"completed", "failed"})
    assert failed.status == "failed"
    assert any(issue.code == "non_zero_exit" for issue in failed.issues)
    incidents = Path(str(failed.metadata["incidents_ref"]))
    assert "non_zero_exit" in incidents.read_text(encoding="utf-8")


def test_missing_required_output_fails_the_run(tmp_path: Path) -> None:
    service = LocalTrainingLifecycleService(tmp_path / "runs")
    service.launch(_proposal(tmp_path, "missing-output", omit_artifacts=["metrics"]))
    failed = _wait(service, "missing-output", {"completed", "failed"})
    assert failed.status == "failed"
    assert any(issue.code == "missing_required_artifact" for issue in failed.issues)


def test_prepared_job_rejects_unverified_data_and_pending_proposal(tmp_path: Path) -> None:
    service = LocalTrainingLifecycleService(tmp_path / "runs")
    proposal = _proposal(tmp_path, "bad-input", data_contract_hash="not-the-real-hash")
    proposal.status = "pending"
    failed = service.launch(proposal)
    assert failed.status == "failed"
    codes = {issue.code for issue in failed.issues}
    assert {"experiment_not_ready", "data_contract_hash_mismatch"} <= codes
    assert not (tmp_path / "runs" / "bad-input" / "prepared_job.json").exists()


def test_unsafe_run_id_cannot_escape_artifact_root(tmp_path: Path) -> None:
    service = LocalTrainingLifecycleService(tmp_path / "runs")
    failed = service.launch(_proposal(tmp_path, "../escape"))
    assert failed.status == "failed"
    assert failed.issues[0].code == "invalid_run_id"
    assert not (tmp_path / "escape").exists()


def test_fake_lifecycle_is_deterministic_and_download_free(tmp_path: Path) -> None:
    service = FakeTrainingLifecycleService()
    proposal = _proposal(tmp_path, "fake")
    first = service.launch(proposal)
    second = service.launch(proposal)
    assert first.to_dict() == second.to_dict()
    assert first.status == "completed"
    assert first.metadata["deterministic_fake"] is True
