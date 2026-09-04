#!/usr/bin/env python3
"""Run the Planner-approved second controlled training experiment on a mounted GPU volume.

Scientific contract:
- consume the exact prepared TrainableDataContract for planned-run-b8e558e54effac85;
- start from the same frozen random initialization as the first bounded run;
- keep tokenizer, seed, architecture, optimizer, scheduler, LR, warmup, batch,
  context, precision, checkpoint cadence, shuffle policy, and 100-step budget fixed;
- change only the Planner-selected primary variable: dataset_mixture;
- never evaluate, promote, or apply a Judge action in this driver.
"""
from __future__ import annotations

import copy
import json
import os
import socket
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from hephaestus.backends.hf_causal_lm import directory_content_identity
from hephaestus.control.autonomous_experiment import (
    AutonomousExperimentCoordinator,
    GuardedTrainingLifecycleService,
    InMemoryIntegrationRecordSink,
)
from hephaestus.schemas.dataset_manifest import DatasetManifest
from hephaestus.schemas.experiment_contract import ExperimentProposal, TrainingControlRequest
from hephaestus.schemas.trainable_data_contract import TrainableDataContract
from hephaestus.training import TransformersTrainingLifecycleService
from hephaestus.training.hf_lifecycle import validate_checkpoint_manifest

VOLUME = Path(os.environ.get("HEPHAESTUS_VOLUME_ROOT", "/workspace"))
ROOT = VOLUME / "hephaestus/scientific/v1"
RUNS = ROOT / "runs"
BINDINGS = ROOT / "runtime_bindings"
REPO_ROOT = Path(__file__).resolve().parents[1]

RUN_ID = "planned-run-b8e558e54effac85"
EXPERIMENT_ID = "experiment-d0e911d6bd1fb7ae"
LINEAGE_ID = "lineage-first-scientific"
STAGE_NAME = "smoke_test"
PRIMARY_VARIABLE = "dataset_mixture"
BASELINE_RUN_ID = "first-bounded-scientific-training-001-33866198758"
APPROVAL = os.environ.get(
    "HEPHAESTUS_OPERATOR_APPROVAL_REF",
    "approval://operator/chat-2026-09-04-second-controlled-experiment",
)
REPO_SHA = os.environ.get("HEPHAESTUS_REPO_SHA", "unknown")
WALL_SECONDS = int(os.environ.get("HEPHAESTUS_MAX_WALL_SECONDS", "1200"))

MODEL = "sha256:7dbbc38ae31de5075fbf06f1362f17b6ff3b46bc822e85fc9b5f2ea05c6dad39"
TOKENIZER = "sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce"
PROCESSED = "sha256:bac39c4c25394e32e86d0e73fe410123e38fcd0d67064e2e1b59a1e31e822fac"
PROCESSED_BYTES = 157_151_627
PROCESSED_RECORDS = 171_295
SOURCE_CONTRACT_HASH = "sha256:ef273fe913f582289ffad2cd05a431e9d541091a51db97b0a649eb47579f2a5a"
SOURCE_MANIFEST_HASH = "sha256:0495018a0cc7c70494d5a00bc51a471568e850d8e3fa11cb0696c9674c71cc76"
SOURCE_EVIDENCE_HASH = "sha256:d78c9aef9d9522fa0befb77f275ae0b025df1c11dc8b43845098747a96deb0f6"
SOURCE_APPROVAL_HASH = "sha256:9c377cec52d831412f1a716ac756695e21a28a4c59bb6ba55c673733bee7d48e"
SOURCE_RECEIPT_HASH = "sha256:c66bdc6f9d46c425e3ba88ab123de45be52061ea373963ca276c69ebfd2aed37"
EXPERIMENT_FILE_HASH = "sha256:530032d5bf2ab1b443592fbd8d3bb61616ca11e293a6ca30d327cc020ff73ffc"
DATASET_ID = "sail/symbolic-instruction-tuning"
DATASET_REVISION = "c0b1111933a7b87bef0e5b3221d8e5f76b5ac27c"
MANIFEST_ID = "manifest-planned-run-b8e558e54effac85"
CONTRACT_ID = "trainable-data-planned-run-b8e558e54effac85"
PARAMETERS, HIDDEN, CONTEXT, VOCAB = 1_874_688, 128, 256, 8_192

EXPERIMENT_FILE = REPO_ROOT / "docs/evidence/first-autonomous-dataset-discovery-001-33876486327/experiment.json"
PREPARED = BINDINGS / RUN_ID / "dataset"
TRAINING_BINDING = BINDINGS / RUN_ID / "training"


def hash_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def require_file(path: Path, expected_hash: str, *, expected_size: int | None = None) -> Path:
    if not path.is_file():
        raise RuntimeError(f"required durable input is missing: {path}")
    observed = hash_file(path)
    if observed != expected_hash:
        raise RuntimeError(f"durable input hash drift: {path}: {observed} != {expected_hash}")
    if expected_size is not None and path.stat().st_size != expected_size:
        raise RuntimeError(f"durable input size drift: {path}")
    return path


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def load_frozen_experiment() -> ExperimentProposal:
    require_file(EXPERIMENT_FILE, EXPERIMENT_FILE_HASH)
    payload = json.loads(EXPERIMENT_FILE.read_text(encoding="utf-8"))
    raw = payload.get("experiment_proposal")
    if not isinstance(raw, dict):
        raise RuntimeError("frozen follow-up ExperimentProposal is missing")
    proposal = ExperimentProposal.from_dict(raw)
    if proposal.experiment_id != EXPERIMENT_ID or proposal.run_id != RUN_ID:
        raise RuntimeError("frozen follow-up experiment identity drifted")
    if proposal.lineage_id != LINEAGE_ID or proposal.stage_name != STAGE_NAME:
        raise RuntimeError("frozen follow-up lineage/stage identity drifted")
    if proposal.primary_variable != PRIMARY_VARIABLE:
        raise RuntimeError("follow-up experiment no longer changes only dataset_mixture")
    if proposal.dataset_selection_id != "dataset-selection-fd8699f8cbd8b4957ca2":
        raise RuntimeError("follow-up dataset selection identity drifted")
    controlled = dict(proposal.controlled_variables)
    expected = {
        "architecture": MODEL,
        "tokenizer": TOKENIZER,
        "random_seed": 1729,
        "evaluation_reference": "sha256:ee4acffa6d6ac3dadd1705931d65fc02bc4206f2fbddacf71b25af4d1cb5e3ad",
        "training_recipe": "sha256:c1a82687a3a5d7651f10ee94cae06057691fa7e21f7f79311af230f3a45a4d97",
    }
    for key, value in expected.items():
        if controlled.get(key) != value:
            raise RuntimeError(f"controlled-variable identity drift for {key}")
    return proposal


def prepare_bindings() -> tuple[Path, Path, Path, Path, dict[str, int], DatasetManifest, TrainableDataContract]:
    processed = require_file(PREPARED / "trainable.jsonl", PROCESSED, expected_size=PROCESSED_BYTES)
    contract_path = require_file(PREPARED / "trainable_data_contract.json", SOURCE_CONTRACT_HASH)
    manifest_path = require_file(PREPARED / "dataset_manifest.json", SOURCE_MANIFEST_HASH)
    evidence_path = require_file(PREPARED / "processing_evidence.json", SOURCE_EVIDENCE_HASH)
    require_file(PREPARED / "dataset_approval.json", SOURCE_APPROVAL_HASH)
    require_file(PREPARED / "acquisition_receipt.json", SOURCE_RECEIPT_HASH)

    contract = TrainableDataContract.from_dict(json.loads(contract_path.read_text(encoding="utf-8")))
    manifest = DatasetManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if contract.contract_id != CONTRACT_ID or contract.run_id != RUN_ID or contract.manifest_id != MANIFEST_ID:
        raise RuntimeError("prepared TrainableDataContract identity drifted")
    if manifest.manifest_id != MANIFEST_ID or manifest.run_id != RUN_ID:
        raise RuntimeError("prepared DatasetManifest identity drifted")
    if manifest.processed_content_hash != PROCESSED:
        raise RuntimeError("DatasetManifest processed content identity drifted")
    if not isinstance(evidence, dict) or evidence.get("processed_content_hash") != PROCESSED:
        raise RuntimeError("processing evidence does not bind the prepared dataset")
    wrapper = evidence.get("wrapper")
    boundary = evidence.get("prompt_target_boundary")
    if not isinstance(wrapper, dict) or wrapper.get("template") != "<|prompt|>\n{prompt}\n<|target|>\n{target}":
        raise RuntimeError("prompt/target wrapper identity drifted")
    if not isinstance(boundary, dict) or boundary.get("status") != "explicit":
        raise RuntimeError("prompt/target boundary is no longer explicit")
    tokenizer_evidence = evidence.get("tokenizer_compatibility")
    if not isinstance(tokenizer_evidence, dict) or tokenizer_evidence.get("compatible") is not True:
        raise RuntimeError("prepared tokenizer compatibility is not positively verified")
    if tokenizer_evidence.get("tokenizer_ref") != TOKENIZER:
        raise RuntimeError("prepared tokenizer identity drifted")

    model_dir = ROOT / "materialized/models" / MODEL.removeprefix("sha256:")
    tokenizer_dir = ROOT / "materialized/tokenizers" / TOKENIZER.removeprefix("sha256:")
    if directory_content_identity(model_dir) != MODEL:
        raise RuntimeError("random-init model directory identity drift")
    if directory_content_identity(tokenizer_dir) != TOKENIZER:
        raise RuntimeError("frozen tokenizer directory identity drift")

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True, trust_remote_code=False)
    if len(tokenizer) != VOCAB or tokenizer.eos_token_id is None or tokenizer.pad_token_id is None:
        raise RuntimeError("frozen tokenizer runtime contract changed")
    special = {
        name: int(value)
        for name, value in {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "unk_token_id": tokenizer.unk_token_id,
        }.items()
        if value is not None
    }

    TRAINING_BINDING.mkdir(parents=True, exist_ok=True)
    runtime_contract = contract.to_dict()
    runtime_contract["processed_dataset_ref"] = str(processed)
    runtime_contract_path = TRAINING_BINDING / "trainable_data_contract.json"
    write_json(runtime_contract_path, runtime_contract)

    runtime_evidence = copy.deepcopy(evidence)
    runtime_evidence["processed_dataset_ref"] = str(processed)
    runtime_tokenizer = runtime_evidence.get("tokenizer_compatibility")
    if not isinstance(runtime_tokenizer, dict):
        raise RuntimeError("runtime tokenizer evidence missing")
    runtime_tokenizer["tokenizer_ref"] = str(tokenizer_dir)
    runtime_evidence["runtime_binding"] = {
        "source_trainable_data_contract": SOURCE_CONTRACT_HASH,
        "source_dataset_manifest": SOURCE_MANIFEST_HASH,
        "source_processing_evidence": SOURCE_EVIDENCE_HASH,
        "source_processed_dataset": PROCESSED,
        "source_model_identity": MODEL,
        "source_tokenizer_identity": TOKENIZER,
        "primary_variable": PRIMARY_VARIABLE,
        "controlled_baseline_run_id": BASELINE_RUN_ID,
    }
    runtime_evidence_path = TRAINING_BINDING / "processing_evidence.json"
    write_json(runtime_evidence_path, runtime_evidence)
    write_json(
        TRAINING_BINDING / "runtime_binding.json",
        {
            "binding_version": "second-controlled-training.v1",
            "run_id": RUN_ID,
            "experiment_id": EXPERIMENT_ID,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "primary_variable": PRIMARY_VARIABLE,
            "scientific_invariants": {
                "model": MODEL,
                "tokenizer": TOKENIZER,
                "seed": 1729,
                "batch_size": 8,
                "context_length": CONTEXT,
                "learning_rate": 5e-4,
                "warmup_steps": 10,
                "optimizer": "adamw",
                "scheduler": "linear",
                "max_steps": 100,
                "dtype": "float32",
                "shuffle": False,
            },
            "changed_input": {
                "kind": PRIMARY_VARIABLE,
                "dataset_id": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
                "manifest_id": MANIFEST_ID,
                "processed_dataset_sha256": PROCESSED,
            },
            "runtime_refs": {
                "processed_dataset": str(processed),
                "runtime_contract": str(runtime_contract_path),
                "runtime_contract_hash": hash_file(runtime_contract_path),
                "runtime_processing_evidence": str(runtime_evidence_path),
                "runtime_processing_evidence_hash": hash_file(runtime_evidence_path),
                "model": str(model_dir),
                "tokenizer": str(tokenizer_dir),
            },
        },
    )
    return runtime_contract_path, runtime_evidence_path, model_dir, tokenizer_dir, special, manifest, contract


def build_proposal(
    contract: Path,
    evidence: Path,
    model: Path,
    tokenizer: Path,
    special: dict[str, int],
) -> ExperimentProposal:
    frozen = load_frozen_experiment()
    constraints = {
        **frozen.training_constraints,
        "backend_id": "transformers_causal_lm",
        "model_id": str(model),
        "model_revision": MODEL,
        "architecture_family": "gpt2",
        "tokenizer_id": str(tokenizer),
        "tokenizer_revision": TOKENIZER,
        "training_mode": "full_finetune",
        "optimizer": "adamw",
        "scheduler": "linear",
        "trainable_data_contract_ref": str(contract),
        "trainable_data_contract_hash": hash_file(contract),
        "processed_dataset_ref": str(PREPARED / "trainable.jsonl"),
        "processed_dataset_hash": PROCESSED,
        "processing_evidence_ref": str(evidence),
        "processing_evidence_hash": hash_file(evidence),
        "seed": 1729,
        "device": "cuda",
        "dtype": "float32",
        "parameter_count": PARAMETERS,
        "hidden_size": HIDDEN,
        "vocabulary_size": VOCAB,
        "special_token_ids": special,
        "context_length": CONTEXT,
        "batch_size": 8,
        "gradient_accumulation_steps": 1,
        "max_steps": 100,
        "learning_rate": 5e-4,
        "warmup_steps": 10,
        "gradient_clipping": 1.0,
        "weight_decay": 0.01,
        "checkpoint_every_steps": 100,
        "logging_every_steps": 5,
        "max_total_tokens": 20_000_000,
        "shuffle": False,
        "local_files_only": True,
        "trust_remote_code": False,
        "loader_settings": {"use_safetensors": True},
        "approval_refs": [APPROVAL],
        "training_recipe_ref": "first-bounded-scientific-training.v2",
        "one_primary_variable": PRIMARY_VARIABLE,
    }
    return replace(
        frozen,
        status="ready",
        training_constraints=constraints,
        metadata={
            **frozen.metadata,
            "launch_authorized": True,
            "approval_evidence": {"dataset_selection_approval": APPROVAL},
            "operator_approval_ref": APPROVAL,
            "source_trainable_data_contract_hash": SOURCE_CONTRACT_HASH,
            "source_dataset_manifest_hash": SOURCE_MANIFEST_HASH,
            "source_processing_evidence_hash": SOURCE_EVIDENCE_HASH,
            "source_processed_dataset_hash": PROCESSED,
            "controlled_follow_up": True,
            "baseline_training_run_id": BASELINE_RUN_ID,
            "execution_semantics": "same random initialization and training recipe; dataset_mixture is the only primary variable",
        },
    )


def launch(proposal: ExperimentProposal) -> tuple[Any, InMemoryIntegrationRecordSink]:
    service = GuardedTrainingLifecycleService(
        TransformersTrainingLifecycleService(
            artifact_root=RUNS,
            maximum_allowed_steps=1_000,
            maximum_dataset_bytes=192 * 1024 * 1024,
            maximum_rows=200_000,
        )
    )
    sink = InMemoryIntegrationRecordSink()
    coordinator = AutonomousExperimentCoordinator(
        diagnosis_service=None,
        planner=None,
        dataset_registry=None,
        dataset_selector=None,
        model_providers={},
        model_selector=None,
        training_service=service,
        evaluation_service=None,
        record_sink=sink,
    )  # type: ignore[arg-type]
    handle = coordinator.launch_approved(proposal, {"dataset_selection_approval": APPROVAL})
    if handle.status == "failed":
        return handle, sink
    deadline = time.monotonic() + WALL_SECONDS
    while time.monotonic() < deadline:
        handle = service.status(RUN_ID)
        if handle.status in {"completed", "failed", "cancelled", "interrupted"}:
            return handle, sink
        time.sleep(2)
    service.control(
        TrainingControlRequest(
            request_id=f"control-{RUN_ID}-wall-timeout",
            run_id=RUN_ID,
            action="cancel",
            requested_by="second-controlled-training-driver",
            reason=f"wall-clock budget exceeded {WALL_SECONDS} seconds",
        )
    )
    for _ in range(30):
        handle = service.status(RUN_ID)
        if handle.status in {"completed", "failed", "cancelled", "interrupted"}:
            return handle, sink
        time.sleep(1)
    return service.status(RUN_ID), sink


def runtime_environment() -> dict[str, object]:
    import torch
    import tokenizers
    import transformers

    result: dict[str, object] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "hostname": socket.gethostname(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        result.update(
            {
                "cuda_runtime": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        )
    return result


def result_for(proposal: ExperimentProposal, handle: Any, sink: InMemoryIntegrationRecordSink) -> dict[str, object]:
    run_root = RUNS / RUN_ID
    result: dict[str, object] = {
        "result_version": "second-controlled-training-result.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "experiment_id": EXPERIMENT_ID,
        "lineage_id": LINEAGE_ID,
        "stage_name": STAGE_NAME,
        "status": handle.status,
        "operator_approval_ref": APPROVAL,
        "repo_sha": REPO_SHA,
        "primary_variable": PRIMARY_VARIABLE,
        "baseline_training_run_id": BASELINE_RUN_ID,
        "verified_inputs": {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "dataset_manifest_id": MANIFEST_ID,
            "dataset_manifest_sha256": SOURCE_MANIFEST_HASH,
            "trainable_data_contract_id": CONTRACT_ID,
            "trainable_data_contract_sha256": SOURCE_CONTRACT_HASH,
            "processing_evidence_sha256": SOURCE_EVIDENCE_HASH,
            "processed_dataset_sha256": PROCESSED,
            "processed_dataset_bytes": PROCESSED_BYTES,
            "processed_dataset_records": PROCESSED_RECORDS,
            "tokenizer_directory_identity": TOKENIZER,
            "model_directory_identity": MODEL,
        },
        "controlled_recipe": {
            "max_steps": 100,
            "batch_size": 8,
            "context_length": CONTEXT,
            "learning_rate": 5e-4,
            "warmup_steps": 10,
            "optimizer": "adamw",
            "scheduler": "linear",
            "weight_decay": 0.01,
            "gradient_clipping": 1.0,
            "checkpoint_every_steps": 100,
            "logging_every_steps": 5,
            "max_total_tokens": 20_000_000,
            "shuffle": False,
            "dtype": "float32",
            "seed": 1729,
            "recipe_version": "first-bounded-scientific-training.v2",
            "only_primary_variable_changed": PRIMARY_VARIABLE,
        },
        "runtime_environment": runtime_environment(),
        "training_run_handle": handle.to_dict(),
        "integration_records": sink.records,
        "run_root": str(run_root),
        "evaluation_performed": False,
        "judge_action_applied": False,
    }
    for name in (
        "prepared_job.json",
        "normalized_training_config.json",
        "resource_estimate.json",
        "metrics_summary.json",
        "runtime_result.json",
        "final_result.json",
        "checkpoint_record.json",
    ):
        path = run_root / name
        if path.is_file():
            result[name.removesuffix(".json")] = json.loads(path.read_text(encoding="utf-8"))
    if handle.status == "completed" and handle.checkpoint_refs:
        checkpoint = Path(handle.checkpoint_refs[-1])
        valid, manifest_hash = validate_checkpoint_manifest(checkpoint)
        result["checkpoint_verification"] = {
            "valid": valid,
            "checkpoint_ref": str(checkpoint),
            "checkpoint_manifest_hash": manifest_hash,
        }
        if not valid:
            result["status"] = "failed_verification"
    return result


def main() -> int:
    if os.environ.get("HEPHAESTUS_RUN_ID", RUN_ID) != RUN_ID:
        raise RuntimeError("second controlled experiment run ID must remain fixed")
    sentinel = ROOT / "executions" / RUN_ID / "driver_result.json"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    try:
        if (RUNS / RUN_ID).exists():
            raise RuntimeError("controlled run evidence already exists; refusing implicit overwrite")
        if not VOLUME.is_dir():
            raise RuntimeError("RunPod Network Volume is not mounted")
        contract, evidence, model, tokenizer, special, _manifest, _source_contract = prepare_bindings()
        proposal = build_proposal(contract, evidence, model, tokenizer, special)
        write_json(TRAINING_BINDING / "approved_runtime_proposal.json", proposal.to_dict())
        handle, sink = launch(proposal)
        result = result_for(proposal, handle, sink)
        if (RUNS / RUN_ID).is_dir():
            write_json(RUNS / RUN_ID / "scientific_run_result.json", result)
        write_json(sentinel, result)
        print(json.dumps(result, sort_keys=True))
        return 0 if result.get("status") == "completed" else 1
    except BaseException as exc:
        failure = {
            "result_version": "second-controlled-training-result.v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": RUN_ID,
            "experiment_id": EXPERIMENT_ID,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "repo_sha": REPO_SHA,
            "primary_variable": PRIMARY_VARIABLE,
            "evaluation_performed": False,
            "judge_action_applied": False,
        }
        write_json(sentinel, failure)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
