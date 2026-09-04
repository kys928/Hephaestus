#!/usr/bin/env python3
"""Run the first bounded scientific Transformers training job inside a RunPod GPU Pod."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hephaestus.backends.hf_causal_lm import directory_content_identity
from hephaestus.control.autonomous_experiment import (
    AutonomousExperimentCoordinator,
    GuardedTrainingLifecycleService,
    InMemoryIntegrationRecordSink,
)
from hephaestus.schemas.experiment_contract import ExperimentProposal, TrainingControlRequest
from hephaestus.training import TransformersTrainingLifecycleService
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest

VOLUME = Path(os.environ.get("HEPHAESTUS_VOLUME_ROOT", "/workspace"))
ROOT = VOLUME / "hephaestus/scientific/v1"
RUNS = ROOT / "runs"
BINDINGS = ROOT / "runtime_bindings"
RUN_ID = os.environ.get("HEPHAESTUS_RUN_ID", "first-bounded-scientific-training-001")
APPROVAL = os.environ.get(
    "HEPHAESTUS_OPERATOR_APPROVAL_REF",
    "approval://operator/explicit-request-2026-09-04-first-bounded-scientific-training",
)
REPO_SHA = os.environ.get("HEPHAESTUS_REPO_SHA", "unknown")
WALL_SECONDS = int(os.environ.get("HEPHAESTUS_MAX_WALL_SECONDS", "1200"))

BUNDLE = "sha256:6774e92d2b595353a18211ffa772fb82b362d462ee2e3c26144f705d26525436"
RAW = "sha256:e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"
PROCESSED = "sha256:f7c512199b6a34ce07fabcd4bdbd45a613aad650190c11bd32c0bbb979910b5c"
TOKENIZER = "sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce"
MODEL = "sha256:7dbbc38ae31de5075fbf06f1362f17b6ff3b46bc822e85fc9b5f2ea05c6dad39"
TRAINABLE_CONTRACT = "sha256:8682e2c8477ffb24edf489c49ee66e900f0bafe5e75e31667c9b64083f7e6d87"
PROCESSING_EVIDENCE = "sha256:01c494c3f3c62c1f20493b3295a1af357012a4d75d02cadecd05f1239546a35a"
EXPERIMENT_PROPOSAL = "sha256:a37c5c462983aca717f8660053ce743d4cbbe08248c46a62dd75d23813ae5127"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
PARAMETERS, HIDDEN, CONTEXT, VOCAB = 1_874_688, 128, 256, 8_192


def digest(ref: str) -> str:
    value = ref.removeprefix("sha256:").lower()
    if len(value) != 64:
        raise ValueError(f"invalid sha256 identity: {ref}")
    int(value, 16)
    return value


def object_path(ref: str) -> Path:
    d = digest(ref)
    return ROOT / "objects/sha256" / d[:2] / d


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def verified(ref: str) -> Path:
    path = object_path(ref)
    if not path.is_file() or hash_file(path) != ref:
        raise RuntimeError(f"verified bootstrap object is missing or changed: {ref}")
    return path


def verified_json(ref: str) -> dict[str, Any]:
    payload = json.loads(verified(ref).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"bootstrap JSON object has wrong shape: {ref}")
    return payload


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def copy_verified(source: Path, target: Path, expected: str) -> None:
    if target.exists():
        if not target.is_file() or hash_file(target) != expected:
            raise RuntimeError(f"runtime binding collision: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".partial")
    shutil.copyfile(source, tmp)
    if hash_file(tmp) != expected:
        raise RuntimeError("runtime dataset materialization changed content")
    os.replace(tmp, target)


def prepare_bindings() -> tuple[Path, Path, Path, Path, dict[str, int]]:
    verified(BUNDLE); verified(RAW)
    processed_source = verified(PROCESSED)
    contract = verified_json(TRAINABLE_CONTRACT)
    evidence = verified_json(PROCESSING_EVIDENCE)
    verified_json(EXPERIMENT_PROPOSAL)

    model_dir = ROOT / "materialized/models" / digest(MODEL)
    tokenizer_dir = ROOT / "materialized/tokenizers" / digest(TOKENIZER)
    if directory_content_identity(model_dir) != MODEL:
        raise RuntimeError("model directory identity drift")
    if directory_content_identity(tokenizer_dir) != TOKENIZER:
        raise RuntimeError("tokenizer directory identity drift")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False)
    if len(tok) != VOCAB or tok.eos_token_id is None or tok.pad_token_id is None:
        raise RuntimeError("verified tokenizer runtime contract changed")
    special = {
        name: int(value)
        for name, value in {
            "eos_token_id": tok.eos_token_id,
            "pad_token_id": tok.pad_token_id,
            "bos_token_id": tok.bos_token_id,
            "unk_token_id": tok.unk_token_id,
        }.items()
        if value is not None
    }

    binding = BINDINGS / RUN_ID
    dataset = binding / "dataset/trainable.jsonl"
    copy_verified(processed_source, dataset, PROCESSED)
    runtime_contract = dict(contract)
    runtime_contract["processed_dataset_ref"] = str(dataset)
    contract_ref = binding / "trainable_data_contract.json"
    write_json(contract_ref, runtime_contract)

    runtime_evidence = json.loads(json.dumps(evidence))
    runtime_evidence["processed_dataset_ref"] = str(dataset)
    compat = runtime_evidence.get("tokenizer_compatibility")
    if not isinstance(compat, dict):
        raise RuntimeError("tokenizer compatibility evidence missing")
    compat["tokenizer_ref"] = str(tokenizer_dir)
    runtime_evidence["runtime_binding"] = {
        "source_processing_evidence": PROCESSING_EVIDENCE,
        "source_trainable_data_contract": TRAINABLE_CONTRACT,
        "source_processed_dataset": PROCESSED,
        "source_tokenizer_identity": TOKENIZER,
        "source_model_identity": MODEL,
    }
    evidence_ref = binding / "processing_evidence.json"
    write_json(evidence_ref, runtime_evidence)
    write_json(binding / "runtime_binding.json", {
        "binding_version": "first-bounded-scientific-training.v1",
        "run_id": RUN_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scientific_content_invariant": {
            "processed_dataset": PROCESSED,
            "tokenizer": TOKENIZER,
            "model": MODEL,
            "bootstrap_bundle": BUNDLE,
        },
        "runtime_refs": {
            "dataset": str(dataset),
            "contract": str(contract_ref),
            "contract_hash": hash_file(contract_ref),
            "processing_evidence": str(evidence_ref),
            "processing_evidence_hash": hash_file(evidence_ref),
            "model": str(model_dir),
            "tokenizer": str(tokenizer_dir),
        },
    })
    return contract_ref, evidence_ref, model_dir, tokenizer_dir, special


def build_proposal(contract: Path, evidence: Path, model: Path, tokenizer: Path, special: dict[str, int]) -> ExperimentProposal:
    original = ExperimentProposal.from_dict(verified_json(EXPERIMENT_PROPOSAL))
    constraints = {
        **original.training_constraints,
        "backend_id": "transformers_causal_lm",
        "model_id": str(model), "model_revision": MODEL, "architecture_family": "gpt2",
        "tokenizer_id": str(tokenizer), "tokenizer_revision": TOKENIZER,
        "training_mode": "full_finetune", "optimizer": "adamw", "scheduler": "linear",
        "trainable_data_contract_ref": str(contract), "trainable_data_contract_hash": hash_file(contract),
        "processed_dataset_ref": str(BINDINGS / RUN_ID / "dataset/trainable.jsonl"), "processed_dataset_hash": PROCESSED,
        "processing_evidence_ref": str(evidence), "processing_evidence_hash": hash_file(evidence),
        "seed": 1729, "device": "cuda", "dtype": "float32",
        "parameter_count": PARAMETERS, "hidden_size": HIDDEN, "vocabulary_size": VOCAB,
        "special_token_ids": special, "context_length": CONTEXT,
        "batch_size": 8, "gradient_accumulation_steps": 1, "max_steps": 50,
        "learning_rate": 5e-4, "warmup_steps": 5, "gradient_clipping": 1.0,
        "weight_decay": 0.01, "checkpoint_every_steps": 25, "logging_every_steps": 5,
        "max_total_tokens": 15_000_000, "shuffle": False, "local_files_only": True,
        "trust_remote_code": False, "loader_settings": {"use_safetensors": True},
        "approval_refs": [APPROVAL], "training_recipe_ref": "first-bounded-scientific-training.v1",
    }
    return replace(original, run_id=RUN_ID, status="ready", training_constraints=constraints, metadata={
        **original.metadata,
        "bootstrap_only": False,
        "launch_authorized": True,
        "approval_evidence": {"model_selection_approval": APPROVAL},
        "operator_approval_ref": APPROVAL,
        "source_experiment_proposal_artifact_ref": EXPERIMENT_PROPOSAL,
        "source_bootstrap_bundle_artifact_ref": BUNDLE,
        "execution_semantics": "random-initialized causal-LM pretraining using lifecycle full_finetune mode",
    })


def launch(proposal: ExperimentProposal) -> tuple[Any, InMemoryIntegrationRecordSink]:
    service = GuardedTrainingLifecycleService(TransformersTrainingLifecycleService(
        artifact_root=RUNS, maximum_allowed_steps=1_000,
        maximum_dataset_bytes=64 * 1024 * 1024, maximum_rows=100_000,
    ))
    sink = InMemoryIntegrationRecordSink()
    coordinator = AutonomousExperimentCoordinator(
        diagnosis_service=None, planner=None, dataset_registry=None, dataset_selector=None,
        model_providers={}, model_selector=None, training_service=service,
        evaluation_service=None, record_sink=sink,
    )  # type: ignore[arg-type]
    handle = coordinator.launch_approved(proposal, {"model_selection_approval": APPROVAL})
    if handle.status == "failed":
        return handle, sink
    deadline = time.monotonic() + WALL_SECONDS
    while time.monotonic() < deadline:
        handle = service.status(RUN_ID)
        if handle.status in {"completed", "failed", "cancelled", "interrupted"}:
            return handle, sink
        time.sleep(2)
    service.control(TrainingControlRequest(
        request_id=f"control-{RUN_ID}-wall-timeout", run_id=RUN_ID, action="cancel",
        requested_by="first-bounded-scientific-training-driver",
        reason=f"wall-clock budget exceeded {WALL_SECONDS} seconds",
    ))
    for _ in range(30):
        handle = service.status(RUN_ID)
        if handle.status in {"completed", "failed", "cancelled", "interrupted"}:
            return handle, sink
        time.sleep(1)
    return service.status(RUN_ID), sink


def result_for(proposal: ExperimentProposal, handle: Any, sink: InMemoryIntegrationRecordSink) -> dict[str, object]:
    run_root = RUNS / RUN_ID
    result: dict[str, object] = {
        "result_version": "first-bounded-scientific-training-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID, "experiment_id": proposal.experiment_id,
        "lineage_id": proposal.lineage_id, "stage_name": proposal.stage_name,
        "status": handle.status, "operator_approval_ref": APPROVAL, "repo_sha": REPO_SHA,
        "verified_inputs": {
            "dataset_id": "Salesforce/wikitext", "dataset_revision": DATASET_REVISION,
            "raw_dataset_sha256": RAW, "processed_dataset_sha256": PROCESSED,
            "tokenizer_directory_identity": TOKENIZER, "model_directory_identity": MODEL,
            "bootstrap_bundle_artifact_ref": BUNDLE,
        },
        "bounded_recipe": {"max_steps": 50, "batch_size": 8, "context_length": CONTEXT,
            "learning_rate": 5e-4, "warmup_steps": 5, "optimizer": "adamw",
            "scheduler": "linear", "checkpoint_every_steps": 25, "max_wall_seconds": WALL_SECONDS},
        "runtime_environment": runtime_environment(),
        "training_run_handle": handle.to_dict(), "integration_records": sink.records,
        "run_root": str(run_root),
    }
    for name in ("prepared_job.json", "normalized_training_config.json", "resource_estimate.json",
                 "metrics_summary.json", "runtime_result.json", "final_result.json", "checkpoint_record.json"):
        path = run_root / name
        if path.is_file():
            result[name.removesuffix(".json")] = json.loads(path.read_text(encoding="utf-8"))
    if handle.status == "completed" and handle.checkpoint_refs:
        checkpoint = Path(handle.checkpoint_refs[-1])
        valid, manifest_hash = validate_checkpoint_manifest(checkpoint)
        result["checkpoint_verification"] = {"valid": valid, "checkpoint_ref": str(checkpoint),
                                              "checkpoint_manifest_hash": manifest_hash}
        if not valid:
            result["status"] = "failed_verification"
    return result


def runtime_environment() -> dict[str, object]:
    import torch, transformers, tokenizers
    value: dict[str, object] = {"python": sys.version.split()[0], "torch": torch.__version__,
        "transformers": transformers.__version__, "tokenizers": tokenizers.__version__,
        "hostname": socket.gethostname(), "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        value.update({"cuda_runtime": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0),
                      "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory})
    return value


def main() -> int:
    sentinel = ROOT / "executions" / RUN_ID / "driver_result.json"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    try:
        if (RUNS / RUN_ID).exists():
            raise RuntimeError("run evidence already exists; refusing implicit overwrite")
        if not VOLUME.is_dir():
            raise RuntimeError("RunPod Network Volume is not mounted")
        contract, evidence, model, tokenizer, special = prepare_bindings()
        proposal = build_proposal(contract, evidence, model, tokenizer, special)
        write_json(BINDINGS / RUN_ID / "approved_runtime_proposal.json", proposal.to_dict())
        handle, sink = launch(proposal)
        result = result_for(proposal, handle, sink)
        if (RUNS / RUN_ID).is_dir():
            write_json(RUNS / RUN_ID / "scientific_run_result.json", result)
        write_json(sentinel, result)
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("status") == "completed" else 1
    except BaseException as exc:  # noqa: BLE001
        failure = {"result_version": "first-bounded-scientific-training-result.v1",
            "created_at": datetime.now(timezone.utc).isoformat(), "run_id": RUN_ID,
            "status": "driver_failed", "error_type": type(exc).__name__, "error": str(exc),
            "repo_sha": REPO_SHA, "operator_approval_ref": APPROVAL,
            "input_identities": {"processed_dataset_sha256": PROCESSED,
                "tokenizer_directory_identity": TOKENIZER, "model_directory_identity": MODEL}}
        write_json(sentinel, failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
