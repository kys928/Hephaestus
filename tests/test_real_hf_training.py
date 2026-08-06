from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hephaestus.backends.hf_causal_lm import (
    TransformersTrainingCapability,
    directory_content_identity,
)
from hephaestus.interfaces import TrainingLifecycleService
from hephaestus.schemas.experiment_contract import ExperimentProposal, TrainingControlRequest
from hephaestus.schemas.trainable_data_contract import TrainableDataContract
from hephaestus.training import (
    LocalTrainingLifecycleService,
    TransformersTrainingLifecycleService,
)
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest

HAS_ML_EXTRA = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "tokenizers"))
requires_ml_extra = pytest.mark.skipif(not HAS_ML_EXTRA, reason="optional Transformers training extra is not installed")


def _sha(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _data_evidence(tmp_path: Path, tokenizer_ref: str) -> dict[str, str]:
    processed = tmp_path / "trainable.jsonl"
    processed.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in (
                {"text": "hello world tiny model", "record_kind": "text", "token_count": 4},
                {"text": "tiny model learns hello", "record_kind": "text", "token_count": 4},
            )
        ),
        encoding="utf-8",
    )
    contract = TrainableDataContract(
        contract_id="trainable-data-test",
        run_id="data-run",
        manifest_id="manifest-test",
        processed_dataset_ref=str(processed),
        schema_version="trainable-data.v1",
        min_tokens=1,
    )
    contract_ref = tmp_path / "trainable_data_contract.json"
    contract_ref.write_text(json.dumps(contract.to_dict(), sort_keys=True), encoding="utf-8")
    evidence = {
        "processed_dataset_ref": str(processed),
        "processed_content_hash": _sha(processed),
        "wrapper": {
            "kind": "explicit_prompt_target_template",
            "template": "<|prompt|>\n{prompt}\n<|target|>\n{target}",
        },
        "prompt_target_boundary": {
            "status": "explicit",
            "prompt_marker": "<|prompt|>",
            "target_marker": "<|target|>",
        },
        "tokenizer_compatibility": {
            "status": "checked",
            "tokenizer_ref": tokenizer_ref,
            "checker_id": "test-tokenizer-checker-v1",
            "compatible": True,
        },
    }
    evidence_ref = tmp_path / "processing_evidence.json"
    evidence_ref.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
    return {
        "processed_ref": str(processed),
        "processed_hash": _sha(processed),
        "contract_ref": str(contract_ref),
        "contract_hash": _sha(contract_ref),
        "evidence_ref": str(evidence_ref),
        "evidence_hash": _sha(evidence_ref),
    }


def _fake_local_identity(tmp_path: Path) -> tuple[Path, str, int]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"gpt2"}', encoding="utf-8")
    return model_dir, directory_content_identity(model_dir), 100


def _proposal(
    tmp_path: Path,
    run_id: str,
    *,
    model_dir: Path | None = None,
    model_revision: str | None = None,
    parameter_count: int | None = None,
    **overrides: object,
) -> ExperimentProposal:
    if model_dir is None or model_revision is None or parameter_count is None:
        model_dir, model_revision, parameter_count = _fake_local_identity(tmp_path)
    data = _data_evidence(tmp_path, str(model_dir))
    constraints: dict[str, object] = {
        "backend_id": "transformers_causal_lm",
        "model_id": str(model_dir),
        "model_revision": model_revision,
        "architecture_family": "gpt2",
        "tokenizer_id": str(model_dir),
        "tokenizer_revision": model_revision,
        "tokenizer_ref": str(model_dir),
        "training_mode": "full_finetune",
        "trainable_data_contract_ref": data["contract_ref"],
        "trainable_data_contract_hash": data["contract_hash"],
        "processed_dataset_ref": data["processed_ref"],
        "processed_dataset_hash": data["processed_hash"],
        "processing_evidence_ref": data["evidence_ref"],
        "processing_evidence_hash": data["evidence_hash"],
        "optimizer": "adamw",
        "scheduler": "constant",
        "parameter_count": parameter_count,
        "hidden_size": 16,
        "vocabulary_size": 8,
        "special_token_ids": {"pad_token_id": 0, "eos_token_id": 1, "unk_token_id": 2},
        "context_length": 16,
        "device": "cpu",
        "dtype": "float32",
        "seed": 7,
        "learning_rate": 0.01,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_steps": 2,
        "checkpoint_every_steps": 2,
        "logging_every_steps": 1,
        "max_total_tokens": 256,
        "local_files_only": True,
    }
    constraints.update(overrides)
    return ExperimentProposal(
        experiment_id=f"experiment-{run_id}",
        run_id=run_id,
        lineage_id="lineage-hf-test",
        stage_name="smoke_test",
        diagnosis_report_id="diagnosis-test",
        intervention_id="intervention-test",
        primary_variable="training_recipe",
        training_constraints=constraints,
        status="ready",
    )


def _wait(
    service: TransformersTrainingLifecycleService,
    run_id: str,
    wanted: set[str],
    timeout: float = 30.0,
):
    deadline = time.monotonic() + timeout
    handle = service.status(run_id)
    while handle.status not in wanted and time.monotonic() < deadline:
        time.sleep(0.02)
        handle = service.status(run_id)
    assert handle.status in wanted, handle.to_dict()
    return handle


def test_core_import_does_not_import_optional_ml_frameworks() -> None:
    script = """
import sys
sys.modules['torch'] = None
sys.modules['transformers'] = None
sys.modules['tokenizers'] = None
from hephaestus.training import LocalTrainingLifecycleService, TransformersTrainingLifecycleService
assert LocalTrainingLifecycleService
assert TransformersTrainingLifecycleService
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env={"PYTHONPATH": str(Path(__file__).parents[1] / "src")},
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_missing_optional_dependencies_return_explicit_capability_issue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsupported = TransformersTrainingCapability(
        supported=False,
        missing_packages=("torch", "transformers", "tokenizers"),
        framework_versions={"torch": None, "transformers": None, "tokenizers": None},
    )
    monkeypatch.setattr("hephaestus.training.hf_lifecycle.transformers_training_capability", lambda: unsupported)
    service = TransformersTrainingLifecycleService(tmp_path / "runs")
    assert isinstance(service, TrainingLifecycleService)
    failed = service.launch(_proposal(tmp_path, "unsupported"))
    assert failed.status == "failed"
    assert {issue.code for issue in failed.issues} == {"transformers_training_unavailable"}
    assert failed.metadata["capability"]["missing_packages"] == ["torch", "transformers", "tokenizers"]
    assert isinstance(LocalTrainingLifecycleService(tmp_path / "fixture-runs"), TrainingLifecycleService)


def test_preparation_rejects_dataset_hash_mismatch(tmp_path: Path) -> None:
    proposal = _proposal(tmp_path, "bad-data", processed_dataset_hash="sha256:not-real")
    failed = TransformersTrainingLifecycleService(tmp_path / "runs").launch(proposal)
    codes = {issue.code for issue in failed.issues}
    assert "processed_dataset_hash_mismatch" in codes
    assert "processing_dataset_hash_mismatch" in codes


def test_preparation_rejects_tokenizer_compatibility_mismatch(tmp_path: Path) -> None:
    proposal = _proposal(tmp_path, "bad-tokenizer-evidence")
    evidence_ref = Path(str(proposal.training_constraints["processing_evidence_ref"]))
    evidence = json.loads(evidence_ref.read_text(encoding="utf-8"))
    evidence["tokenizer_compatibility"]["compatible"] = False
    evidence_ref.write_text(json.dumps(evidence, sort_keys=True), encoding="utf-8")
    proposal.training_constraints["processing_evidence_hash"] = _sha(evidence_ref)
    failed = TransformersTrainingLifecycleService(tmp_path / "runs").launch(proposal)
    assert "tokenizer_compatibility_unverified" in {issue.code for issue in failed.issues}


def test_preparation_rejects_invalid_special_tokens_and_remote_code(tmp_path: Path) -> None:
    proposal = _proposal(
        tmp_path,
        "bad-specials",
        special_token_ids={"eos_token_id": 1},
        trust_remote_code=True,
    )
    failed = TransformersTrainingLifecycleService(tmp_path / "runs").launch(proposal)
    codes = {issue.code for issue in failed.issues}
    assert {"invalid_special_token_configuration", "remote_code_forbidden"} <= codes


def test_remote_download_is_disabled_without_governance_evidence(tmp_path: Path) -> None:
    proposal = _proposal(tmp_path, "remote-disabled")
    proposal.training_constraints.update(
        {
            "model_id": "example/tiny-model",
            "model_revision": "a" * 40,
            "tokenizer_id": "example/tiny-tokenizer",
            "tokenizer_revision": "b" * 40,
        }
    )
    failed = TransformersTrainingLifecycleService(tmp_path / "runs").launch(proposal)
    codes = {issue.code for issue in failed.issues}
    assert {"external_download_disabled", "model_license_missing", "model_provenance_missing"} <= codes
    assert "model_download_approval_missing" in codes


def test_partial_checkpoint_is_never_accepted(tmp_path: Path) -> None:
    partial = tmp_path / "checkpoint.partial"
    partial.mkdir()
    valid, reason = validate_checkpoint_manifest(partial)
    assert valid is False
    assert "partial" in reason


def _tiny_transformers_fixture(tmp_path: Path) -> tuple[Path, str, int]:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    model_dir = tmp_path / "tiny-gpt2"
    model_dir.mkdir()
    vocabulary = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "<bos>": 3,
        "hello": 4,
        "world": 5,
        "tiny": 6,
        "model": 7,
        "learns": 8,
    }
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
        bos_token="<bos>",
    )
    config = transformers.GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=32,
        n_ctx=32,
        n_embd=16,
        n_layer=1,
        n_head=1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    torch.manual_seed(123)
    model = transformers.AutoModelForCausalLM.from_config(config)
    model.save_pretrained(model_dir, safe_serialization=False)
    tokenizer.save_pretrained(model_dir)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return model_dir, directory_content_identity(model_dir), parameters


def _real_proposal(tmp_path: Path, run_id: str, **overrides: object) -> ExperimentProposal:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_dir, revision, parameters = _tiny_transformers_fixture(tmp_path)
    return _proposal(
        tmp_path,
        run_id,
        model_dir=model_dir,
        model_revision=revision,
        parameter_count=parameters,
        vocabulary_size=9,
        special_token_ids={"pad_token_id": 0, "eos_token_id": 1, "unk_token_id": 2, "bos_token_id": 3},
        **overrides,
    )


@requires_ml_extra
def test_tiny_local_causal_lm_updates_parameters_and_persists_evidence(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    proposal = _real_proposal(tmp_path, "real-hf")
    original = transformers.AutoModelForCausalLM.from_pretrained(proposal.training_constraints["model_id"])
    original_weight = original.get_input_embeddings().weight.detach().clone()
    service = TransformersTrainingLifecycleService(tmp_path / "runs")
    assert service.launch(proposal).status == "running"
    completed = _wait(service, "real-hf", {"completed", "failed"})
    assert completed.status == "completed", completed.to_dict()
    checkpoint = Path(completed.checkpoint_refs[0])
    trained = transformers.AutoModelForCausalLM.from_pretrained(checkpoint / "model")
    assert not torch.equal(original_weight, trained.get_input_embeddings().weight.detach())
    valid, manifest_hash = validate_checkpoint_manifest(checkpoint)
    assert valid is True
    record = json.loads(Path(str(completed.metadata["checkpoint_record_ref"])).read_text())
    assert record["checkpoint_manifest_hash"] == manifest_hash
    assert record["partial_write"] is False
    summary = json.loads(Path(completed.metrics_ref or "").read_text())
    assert summary["finite"] is True
    assert summary["tokens_processed"] > 0
    assert "not semantic evaluation" in summary["limitation"]
    assert Path(str(completed.metadata["normalized_config_ref"])).is_file()
    assert Path(str(completed.metadata["resource_estimate_ref"])).is_file()
    assert Path(str(completed.metadata["final_result_ref"])).is_file()


@requires_ml_extra
def test_interrupt_resume_and_strict_compatibility(tmp_path: Path) -> None:
    proposal = _real_proposal(
        tmp_path,
        "hf-resume",
        max_steps=20,
        checkpoint_every_steps=20,
        step_delay_seconds=0.02,
    )
    service = TransformersTrainingLifecycleService(tmp_path / "runs")
    service.launch(proposal)
    metrics = tmp_path / "runs" / "hf-resume" / "metrics.jsonl"
    deadline = time.monotonic() + 20
    while not metrics.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    interrupt = TrainingControlRequest("stop-1", "hf-resume", "interrupt", "test", "checkpoint")
    assert service.control(interrupt).status == "interrupting"
    interrupted = _wait(service, "hf-resume", {"interrupted", "failed"})
    assert interrupted.status == "interrupted", interrupted.to_dict()
    fingerprint_refused = service.control(
        TrainingControlRequest(
            "resume-fingerprint-bad",
            "hf-resume",
            "resume",
            "test",
            "fingerprint mismatch",
            metadata={"config_fingerprint": "sha256:different"},
        )
    )
    assert fingerprint_refused.status == "interrupted"
    assert fingerprint_refused.issues[-1].code == "resume_request_mismatch"
    refused = service.control(
        TrainingControlRequest(
            "resume-bad",
            "hf-resume",
            "resume",
            "test",
            "mismatch",
            metadata={"tokenizer_revision": "sha256:different"},
        )
    )
    assert refused.status == "interrupted"
    assert refused.issues[-1].code == "resume_request_mismatch"
    resumed = service.control(
        TrainingControlRequest("resume-good", "hf-resume", "resume", "test", "strict resume")
    )
    assert resumed.status == "resuming"
    completed = _wait(service, "hf-resume", {"completed", "failed"})
    assert completed.status == "completed", completed.to_dict()


@requires_ml_extra
def test_cancel_and_oom_are_normalized_honestly(tmp_path: Path) -> None:
    cancel_proposal = _real_proposal(
        tmp_path / "cancel-input",
        "hf-cancel",
        max_steps=20,
        checkpoint_every_steps=20,
        step_delay_seconds=0.02,
    )
    cancel_service = TransformersTrainingLifecycleService(tmp_path / "cancel-runs")
    cancel_service.launch(cancel_proposal)
    deadline = time.monotonic() + 20
    metrics = tmp_path / "cancel-runs" / "hf-cancel" / "metrics.jsonl"
    while not metrics.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    cancelling = cancel_service.control(
        TrainingControlRequest("cancel-1", "hf-cancel", "cancel", "test", "cancel")
    )
    assert cancelling.status == "interrupting"
    cancelled = _wait(cancel_service, "hf-cancel", {"cancelled", "failed"})
    assert cancelled.status == "cancelled"

    oom_input = tmp_path / "oom-input"
    oom_input.mkdir()
    oom_proposal = _real_proposal(oom_input, "hf-oom", simulate_oom="cpu")
    oom_service = TransformersTrainingLifecycleService(tmp_path / "oom-runs")
    oom_service.launch(oom_proposal)
    failed = _wait(oom_service, "hf-oom", {"failed"})
    issue = next(issue for issue in failed.issues if issue.code == "out_of_memory")
    assert issue.metadata["device"] == "cpu"
    assert "reduce batch_size" in issue.metadata["suggested_reductions"]


@requires_ml_extra
def test_repeated_tiny_cpu_run_is_deterministic_within_recorded_limits(tmp_path: Path) -> None:
    model_dir, revision, parameters = _tiny_transformers_fixture(tmp_path)
    losses: list[float] = []
    for run_id in ("deterministic-a", "deterministic-b"):
        input_dir = tmp_path / run_id
        input_dir.mkdir()
        proposal = _proposal(
            input_dir,
            run_id,
            model_dir=model_dir,
            model_revision=revision,
            parameter_count=parameters,
            vocabulary_size=9,
            special_token_ids={"pad_token_id": 0, "eos_token_id": 1, "unk_token_id": 2, "bos_token_id": 3},
        )
        service = TransformersTrainingLifecycleService(tmp_path / f"runs-{run_id}")
        service.launch(proposal)
        completed = _wait(service, run_id, {"completed", "failed"})
        assert completed.status == "completed"
        summary = json.loads(Path(completed.metrics_ref or "").read_text())
        losses.append(summary["final_training_loss"])
        assert summary["determinism"]["perfect_determinism_claimed"] is False
    assert losses[0] == pytest.approx(losses[1], rel=0.0, abs=1e-12)


@requires_ml_extra
def test_gpu_smoke_is_optional_and_skips_without_cuda(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is optional")
    proposal = _real_proposal(tmp_path, "hf-gpu", device="cuda", dtype="float16", max_steps=1)
    service = TransformersTrainingLifecycleService(tmp_path / "runs")
    service.launch(proposal)
    assert _wait(service, "hf-gpu", {"completed", "failed"}).status == "completed"
