#!/usr/bin/env python3
"""Acquire and preprocess the first autonomously selected dataset intervention.

Boundary crossed by this driver:

    explicit operator approval
      -> immutable remote acquisition
      -> bounded preprocessing
      -> durable TrainableDataContract

The exact selected dataset decision and ExperimentProposal are frozen repo evidence.
This command does not launch training, mutate the model, or execute the experiment.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Sequence

import boto3
from botocore.config import Config
from tokenizers import Tokenizer

from hephaestus.data import (
    AutonomousDataPreprocessor,
    DataProcessingConfig,
    DatasetAcquisitionApproval,
    DatasetAcquisitionCache,
    RemoteAcquisitionLimits,
    RemoteDatasetAcquisitionService,
)
from hephaestus.providers.datasets import HuggingFaceDatasetProvider
from hephaestus.schemas.dataset_manifest import DatasetManifest
from hephaestus.schemas.discovery_contract import DatasetCandidate, DatasetSelectionDecision
from hephaestus.schemas.experiment_contract import ExperimentProposal
from hephaestus.schemas.trainable_data_contract import TrainableDataContract
from hephaestus.storage.base import ArtifactRecord
from hephaestus.utils.hashing import hash_file

ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = ROOT / "docs/evidence/first-autonomous-dataset-discovery-001-33876486327/selection.json"
EXPERIMENT_PATH = ROOT / "docs/evidence/first-autonomous-dataset-discovery-001-33876486327/experiment.json"
SELECTION_SHA = "f9b379a36106927d8b1d395baf475df072a837b9babbc4acdcd15b101a8d300b"
EXPERIMENT_SHA = "530032d5bf2ab1b443592fbd8d3bb61616ca11e293a6ca30d327cc020ff73ffc"
EXPECTED_SELECTION_ID = "dataset-selection-fd8699f8cbd8b4957ca2"
EXPECTED_CANDIDATE_ID = "dataset-fb91684d87fe5f28"
EXPECTED_DATASET_ID = "sail/symbolic-instruction-tuning"
EXPECTED_REVISION = "c0b1111933a7b87bef0e5b3221d8e5f76b5ac27c"
EXPECTED_EXPERIMENT_ID = "experiment-d0e911d6bd1fb7ae"
EXPECTED_RUN_ID = "planned-run-b8e558e54effac85"
TOKENIZER_REF = "sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce"
TOKENIZER_JSON_SHA = "3ebcc9816398d7a2afa341a9db07de5f0ac30d2625ffe63e4752c2eddce40f25"
TOKENIZER_KEY = "hephaestus/scientific/v1/runs/first-bounded-scientific-training-001-33866198758/checkpoint_step_100/tokenizer/tokenizer.json"
USER_APPROVAL_REF = "approval://operator/chat-2026-09-04-selected-dataset-preparation"
STORE_PREFIX = "hephaestus/scientific/v1"
MAX_SHARD_BYTES = 512 * 1024 * 1024
MAX_ROWS = 100_000
CHUNK_SIZE_TOKENS = 256
WORK_ROOT = Path(".first_selected_dataset_preparation_work")
OUTPUT = Path("first_selected_dataset_preparation.json")


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def s3_client() -> Any:
    return boto3.client(
        "s3",
        endpoint_url=required("RUNPOD_S3_ENDPOINT_URL").rstrip("/"),
        region_name=required("RUNPOD_DATACENTER_ID"),
        aws_access_key_id=required("RUNPOD_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=required("RUNPOD_S3_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4", retries={"max_attempts": 8, "mode": "standard"}),
    )


def stream_s3_hash(client: Any, bucket: str, key: str) -> tuple[str, int]:
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    hasher = hashlib.sha256()
    size = 0
    try:
        while True:
            chunk = body.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
            size += len(chunk)
    finally:
        body.close()
    return hasher.hexdigest(), size


@dataclass(slots=True)
class RunPodS3ArtifactStore:
    client: Any
    bucket: str
    prefix: str = STORE_PREFIX

    def _key(self, digest: str) -> str:
        return f"{self.prefix}/objects/sha256/{digest[:2]}/{digest}"

    def _digest(self, artifact_ref: str) -> str:
        digest = artifact_ref.removeprefix("sha256:").strip().lower()
        if len(digest) != 64:
            raise ValueError(f"invalid artifact ref: {artifact_ref}")
        int(digest, 16)
        return digest

    def _head(self, key: str) -> dict[str, object] | None:
        try:
            return self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 - S3 compatibility boundary
            response = getattr(exc, "response", {})
            code = str(response.get("Error", {}).get("Code", "")) if isinstance(response, dict) else ""
            if code in {"404", "NoSuchKey", "NotFound"}:
                return None
            raise

    def _verify_key(self, key: str, digest: str, size: int) -> bool:
        head = self._head(key)
        if head is None or int(head.get("ContentLength", -1)) != size:
            return False
        metadata = head.get("Metadata")
        if not isinstance(metadata, dict) or metadata.get("sha256") != digest:
            return False
        observed_hash, observed_size = stream_s3_hash(self.client, self.bucket, key)
        return observed_hash == digest and observed_size == size

    def put_bytes(self, data: bytes, *, expected_hash: str | None = None, media_type: str | None = None) -> ArtifactRecord:
        digest = hashlib.sha256(data).hexdigest()
        expected = str(expected_hash or "").removeprefix("sha256:")
        if expected and expected != digest:
            raise ValueError("expected content hash does not match bytes")
        key = self._key(digest)
        if self._head(key) is None:
            args: dict[str, object] = {
                "Bucket": self.bucket,
                "Key": key,
                "Body": data,
                "Metadata": {"sha256": digest},
            }
            if media_type:
                args["ContentType"] = media_type
            self.client.put_object(**args)
        if not self._verify_key(key, digest, len(data)):
            raise RuntimeError(f"RunPod object verification failed: {key}")
        return ArtifactRecord(
            artifact_ref=f"sha256:{digest}",
            content_hash=digest,
            hash_algorithm="sha256",
            byte_size=len(data),
            storage_path=f"s3://{self.bucket}/{key}",
            created_at=datetime.now(timezone.utc),
            media_type=media_type,
        )

    def put_file(self, source: Path, *, expected_hash: str | None = None, media_type: str | None = None) -> ArtifactRecord:
        source = Path(source)
        digest = hash_file(source)
        expected = str(expected_hash or "").removeprefix("sha256:")
        if expected and expected != digest:
            raise ValueError("expected content hash does not match file")
        size = source.stat().st_size
        key = self._key(digest)
        if self._head(key) is None:
            extra: dict[str, object] = {"Metadata": {"sha256": digest}}
            if media_type:
                extra["ContentType"] = media_type
            self.client.upload_file(str(source), self.bucket, key, ExtraArgs=extra)
        if not self._verify_key(key, digest, size):
            raise RuntimeError(f"RunPod file verification failed: {key}")
        return ArtifactRecord(
            artifact_ref=f"sha256:{digest}",
            content_hash=digest,
            hash_algorithm="sha256",
            byte_size=size,
            storage_path=f"s3://{self.bucket}/{key}",
            created_at=datetime.now(timezone.utc),
            media_type=media_type,
        )

    def open(self, artifact_ref: str) -> BinaryIO:
        digest = self._digest(artifact_ref)
        response = self.client.get_object(Bucket=self.bucket, Key=self._key(digest))
        body = response["Body"]
        try:
            data = body.read()
        finally:
            body.close()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("RunPod object hash mismatch")
        return io.BytesIO(data)

    def verify(self, artifact_ref: str) -> bool:
        digest = self._digest(artifact_ref)
        key = self._key(digest)
        head = self._head(key)
        if head is None:
            return False
        return self._verify_key(key, digest, int(head.get("ContentLength", 0)))

    def stage_json(self, payload: object) -> ArtifactRecord:
        return self.put_bytes(canonical_bytes(payload), media_type="application/json")

    def stage_named_file(self, *, key: str, source: Path, media_type: str | None = None) -> dict[str, object]:
        digest = hash_file(source)
        size = source.stat().st_size
        if self._head(key) is None:
            extra: dict[str, object] = {"Metadata": {"sha256": digest}}
            if media_type:
                extra["ContentType"] = media_type
            self.client.upload_file(str(source), self.bucket, key, ExtraArgs=extra)
        if not self._verify_key(key, digest, size):
            raise RuntimeError(f"named RunPod materialization verification failed: {key}")
        return {
            "storage_path": f"s3://{self.bucket}/{key}",
            "sha256": f"sha256:{digest}",
            "byte_size": size,
            "key": key,
        }

    def stage_named_json(self, *, key: str, payload: object) -> dict[str, object]:
        data = canonical_bytes(payload)
        digest = hashlib.sha256(data).hexdigest()
        if self._head(key) is None:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                Metadata={"sha256": digest},
                ContentType="application/json",
            )
        if not self._verify_key(key, digest, len(data)):
            raise RuntimeError(f"named RunPod JSON verification failed: {key}")
        return {
            "storage_path": f"s3://{self.bucket}/{key}",
            "sha256": f"sha256:{digest}",
            "byte_size": len(data),
            "key": key,
        }


@dataclass(slots=True)
class ExactTokenizerChecker:
    tokenizer: Tokenizer
    tokenizer_ref: str = TOKENIZER_REF
    checker_id: str = "exact-frozen-bpe-unknown-token-check-v1"

    def check(self, texts: Sequence[str]) -> tuple[bool, dict[str, object]]:
        unknown_id = self.tokenizer.token_to_id("<|unk|>")
        tokens = 0
        unknown = 0
        over_context = 0
        maximum = 0
        for text in texts:
            ids = self.tokenizer.encode(text).ids
            tokens += len(ids)
            maximum = max(maximum, len(ids))
            if len(ids) > 256:
                over_context += 1
            if unknown_id is not None:
                unknown += sum(token_id == unknown_id for token_id in ids)
        fraction = unknown / max(tokens, 1)
        return bool(texts) and fraction <= 0.01, {
            "records_checked": len(texts),
            "tokens_checked": tokens,
            "unknown_token_count": unknown,
            "unknown_token_fraction": fraction,
            "over_context_records": over_context,
            "maximum_tokenized_length": maximum,
            "criterion": "unknown-token fraction <= 0.01; over-context records remain explicit preprocessing evidence",
        }


def load_frozen_boundary() -> tuple[DatasetSelectionDecision, DatasetCandidate, ExperimentProposal]:
    if sha_file(SELECTION_PATH) != SELECTION_SHA or sha_file(EXPERIMENT_PATH) != EXPERIMENT_SHA:
        raise RuntimeError("frozen dataset selection/experiment evidence drifted")
    selection = DatasetSelectionDecision.from_dict(json.loads(SELECTION_PATH.read_text(encoding="utf-8")))
    experiment_payload = json.loads(EXPERIMENT_PATH.read_text(encoding="utf-8"))
    raw_experiment = experiment_payload.get("experiment_proposal")
    if not isinstance(raw_experiment, dict):
        raise RuntimeError("frozen experiment evidence is missing the ExperimentProposal")
    experiment = ExperimentProposal.from_dict(raw_experiment)
    if selection.decision_id != EXPECTED_SELECTION_ID or selection.status != "selected":
        raise RuntimeError("unexpected or non-selected dataset decision")
    if experiment.experiment_id != EXPECTED_EXPERIMENT_ID or experiment.run_id != EXPECTED_RUN_ID:
        raise RuntimeError("unexpected ExperimentProposal identity")
    if experiment.dataset_selection_id != selection.decision_id or experiment.primary_variable != "dataset_mixture":
        raise RuntimeError("experiment does not bind the selected one-variable dataset intervention")
    material = selection.metadata.get("material_candidates", [])
    candidates = [DatasetCandidate.from_dict(dict(item)) for item in material if isinstance(item, dict)]
    candidate = next((item for item in candidates if item.candidate_id == EXPECTED_CANDIDATE_ID), None)
    if candidate is None:
        raise RuntimeError("selected candidate is absent from frozen material candidate evidence")
    if (
        candidate.dataset_id != EXPECTED_DATASET_ID
        or candidate.revision != EXPECTED_REVISION
        or candidate.license != "mit"
        or not candidate.compatibility.get("tokenizer_compatible")
    ):
        raise RuntimeError("selected candidate identity, license, or tokenizer evidence drifted")
    return selection, candidate, experiment


def select_bounded_train_shard(provider: HuggingFaceDatasetProvider, candidate: DatasetCandidate) -> tuple[str, int, dict[str, object]]:
    snapshot = provider.resolve_revision(candidate.dataset_id, candidate.revision or "main")
    if snapshot.resolved_revision != EXPECTED_REVISION or snapshot.license != "mit" or snapshot.remote_code_required:
        raise RuntimeError("provider snapshot no longer matches the selected immutable MIT candidate")
    files = provider.enumerate_files(snapshot)
    eligible = []
    for item in files:
        path = PurePosixPath(item.relative_path)
        suffix = path.suffix.casefold()
        if not path.parts or path.parts[0].casefold() != "train":
            continue
        if suffix not in {".json", ".jsonl", ".csv", ".parquet"}:
            continue
        if item.size_bytes is None or item.size_bytes <= 0 or item.size_bytes > MAX_SHARD_BYTES:
            continue
        eligible.append(item)
    if not eligible:
        inventory = [
            {"path": item.relative_path, "size_bytes": item.size_bytes}
            for item in files
            if item.relative_path.casefold().startswith("train/")
        ]
        raise RuntimeError(f"no bounded immutable training shard <= {MAX_SHARD_BYTES} bytes; train inventory={inventory}")
    selected = sorted(eligible, key=lambda item: (int(item.size_bytes or 0), item.relative_path))[0]
    return selected.relative_path, int(selected.size_bytes or 0), {
        "selection_rule": "smallest positive-size immutable train data file within the 512 MiB bound",
        "eligible_train_files": [
            {"path": item.relative_path, "size_bytes": item.size_bytes}
            for item in sorted(eligible, key=lambda item: item.relative_path)
        ],
    }


def load_tokenizer(client: Any, bucket: str) -> Tokenizer:
    response = client.get_object(Bucket=bucket, Key=TOKENIZER_KEY)
    body = response["Body"]
    try:
        data = body.read()
    finally:
        body.close()
    if hashlib.sha256(data).hexdigest() != TOKENIZER_JSON_SHA:
        raise RuntimeError("frozen tokenizer JSON failed exact SHA-256 verification")
    path = WORK_ROOT / "tokenizer.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return Tokenizer.from_file(str(path))


def main() -> None:
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    selection, candidate, experiment = load_frozen_boundary()
    client = s3_client()
    bucket = required("RUNPOD_NETWORK_VOLUME_ID")
    client.head_bucket(Bucket=bucket)
    store = RunPodS3ArtifactStore(client, bucket)
    provider = HuggingFaceDatasetProvider(enable_network=True, max_results=10)

    shard_path, shard_bytes, shard_evidence = select_bounded_train_shard(provider, candidate)
    acquisition_candidate = DatasetCandidate.from_dict(candidate.to_dict())
    acquisition_candidate.metadata = {
        **dict(acquisition_candidate.metadata),
        "acquisition_files": [shard_path],
        "bounded_acquisition": True,
        "bounded_acquisition_max_bytes": MAX_SHARD_BYTES,
    }

    cache = DatasetAcquisitionCache(WORK_ROOT / "acquisition-cache")
    service = RemoteDatasetAcquisitionService(
        providers={"huggingface": provider},
        cache=cache,
        artifact_store=store,
    )
    planning = service.plan(
        candidate=acquisition_candidate,
        selection=selection,
        limits=RemoteAcquisitionLimits(
            max_bytes=MAX_SHARD_BYTES,
            max_files=1,
            chunk_size=1024 * 1024,
            timeout_seconds=60.0,
            disk_reserve_bytes=256 * 1024 * 1024,
            allowed_suffixes=(PurePosixPath(shard_path).suffix.casefold(),),
        ),
    )
    if planning.status != "ready" or planning.plan is None:
        raise RuntimeError(f"remote acquisition plan blocked: {[issue.to_dict() for issue in planning.issues]}")
    plan = planning.plan
    if len(plan.files) != 1 or plan.files[0].relative_path != shard_path:
        raise RuntimeError("acquisition plan escaped the bounded selected shard")

    approved_requirements = tuple(sorted(set(plan.required_approvals) | set(experiment.required_approvals)))
    approval = DatasetAcquisitionApproval(
        selection_decision_id=selection.decision_id,
        approved_candidate_ids=(candidate.candidate_id,),
        approval_refs=(USER_APPROVAL_REF,),
        approved_requirements=approved_requirements,
    )
    acquisition = service.acquire(plan, approval)
    receipt = acquisition.receipt
    if not acquisition.completed or len(receipt.acquired_files) != 1:
        raise RuntimeError(f"immutable acquisition failed: {receipt.to_dict()}")
    raw = receipt.acquired_files[0]
    if raw.relative_path != shard_path or raw.size_bytes != shard_bytes:
        raise RuntimeError("acquisition receipt does not match selected immutable shard")
    if not raw.artifact_ref or not store.verify(raw.artifact_ref):
        raise RuntimeError("acquired shard is not durably verified on the RunPod Network Volume")

    tokenizer = load_tokenizer(client, bucket)
    checker = ExactTokenizerChecker(tokenizer)
    processor = AutonomousDataPreprocessor(
        DataProcessingConfig(
            artifact_root=WORK_ROOT / "processed",
            max_input_bytes=MAX_SHARD_BYTES,
            max_rows=MAX_ROWS,
            chunk_size_tokens=CHUNK_SIZE_TOKENS,
            min_tokens=1,
            near_duplicate_threshold=None,
        ),
        tokenizer_checker=checker,
    )
    result = processor.process_remote_acquisition(
        run_id=experiment.run_id,
        lineage_id=experiment.lineage_id,
        stage_name=experiment.stage_name,
        candidate=acquisition_candidate,
        selection=selection,
        approval=approval,
        receipt=receipt,
        relative_path=shard_path,
        tokenizer_ref=TOKENIZER_REF,
    )

    processed_path = Path(result.trainable_data_contract.processed_dataset_ref)
    processed_record = store.put_file(
        processed_path,
        expected_hash=result.processed_content_hash,
        media_type="application/jsonl",
    )
    if not store.verify(processed_record.artifact_ref):
        raise RuntimeError("processed dataset failed durable RunPod verification")

    evidence_payload = copy.deepcopy(result.processing_evidence)
    evidence_payload["processed_dataset_ref"] = processed_record.storage_path
    evidence_payload["approval_ref"] = USER_APPROVAL_REF
    evidence_payload["acquisition_receipt_id"] = receipt.receipt_id
    evidence_payload["selected_shard"] = shard_path
    evidence_payload["selected_shard_bytes"] = shard_bytes
    evidence_payload["bounded_source_note"] = "one immutable training shard selected under an explicit byte bound; not a full-corpus claim"
    evidence_record = store.stage_json(evidence_payload)

    manifest_payload = result.manifest.to_dict()
    manifest_payload.setdefault("metadata", {})
    manifest_payload["metadata"] = {
        **dict(manifest_payload.get("metadata", {})),
        "source_dataset_selection_id": selection.decision_id,
        "source_candidate_id": candidate.candidate_id,
        "source_dataset_id": candidate.dataset_id,
        "source_revision": EXPECTED_REVISION,
        "source_license": "mit",
        "source_shard": shard_path,
        "source_shard_hash": raw.local_content_hash,
        "source_shard_bytes": raw.size_bytes,
        "acquisition_receipt_id": receipt.receipt_id,
        "bounded_max_rows": MAX_ROWS,
        "full_corpus_processed": False,
    }
    final_manifest = DatasetManifest.from_dict(manifest_payload)
    manifest_record = store.stage_json(final_manifest.to_dict())

    contract_payload = result.trainable_data_contract.to_dict()
    contract_payload["processed_dataset_ref"] = processed_record.storage_path
    final_contract = TrainableDataContract.from_dict(contract_payload)
    contract_record = store.stage_json(final_contract.to_dict())
    preprocessing_record = store.stage_json(result.preprocessing_report.to_dict())
    plan_record = store.stage_json(plan.to_dict())
    receipt_record = store.stage_json(receipt.to_dict())
    approval_payload = {
        "selection_decision_id": approval.selection_decision_id,
        "approved_candidate_ids": list(approval.approved_candidate_ids),
        "approval_refs": list(approval.approval_refs),
        "approved_requirements": list(approval.approved_requirements),
        "source": "explicit operator instruction in project conversation",
    }
    approval_record = store.stage_json(approval_payload)

    runtime_prefix = f"{STORE_PREFIX}/runtime_bindings/{experiment.run_id}/dataset"
    named_processed = store.stage_named_file(
        key=f"{runtime_prefix}/trainable.jsonl",
        source=processed_path,
        media_type="application/jsonl",
    )
    named_contract = store.stage_named_json(key=f"{runtime_prefix}/trainable_data_contract.json", payload=final_contract.to_dict())
    named_manifest = store.stage_named_json(key=f"{runtime_prefix}/dataset_manifest.json", payload=final_manifest.to_dict())
    named_preprocessing = store.stage_named_json(key=f"{runtime_prefix}/preprocessing_report.json", payload=result.preprocessing_report.to_dict())
    named_evidence = store.stage_named_json(key=f"{runtime_prefix}/processing_evidence.json", payload=evidence_payload)
    named_receipt = store.stage_named_json(key=f"{runtime_prefix}/acquisition_receipt.json", payload=receipt.to_dict())
    named_approval = store.stage_named_json(key=f"{runtime_prefix}/dataset_approval.json", payload=approval_payload)

    proof = {
        "status": "completed",
        "boundary": "approval -> immutable acquisition -> preprocessing -> TrainableDataContract",
        "experiment_id": experiment.experiment_id,
        "run_id": experiment.run_id,
        "lineage_id": experiment.lineage_id,
        "stage_name": experiment.stage_name,
        "primary_variable": experiment.primary_variable,
        "selection_decision_id": selection.decision_id,
        "candidate_id": candidate.candidate_id,
        "dataset_id": candidate.dataset_id,
        "revision": candidate.revision,
        "license": candidate.license,
        "approval": approval_payload,
        "acquisition_plan": plan.to_dict(),
        "acquisition_receipt": receipt.to_dict(),
        "selected_shard": {
            "relative_path": shard_path,
            "size_bytes": shard_bytes,
            **shard_evidence,
        },
        "preprocessing": {
            "max_rows": MAX_ROWS,
            "chunk_size_tokens": CHUNK_SIZE_TOKENS,
            "processed_content_hash": result.processed_content_hash,
            "dataset_identity": result.dataset_identity,
            "processing_evidence": evidence_payload,
        },
        "dataset_manifest": final_manifest.to_dict(),
        "trainable_data_contract": final_contract.to_dict(),
        "durable_records": {
            "raw_source_artifact_ref": raw.artifact_ref,
            "raw_source_storage_path": next((item.artifact_ref for item in receipt.acquired_files if item.relative_path == shard_path), raw.artifact_ref),
            "processed": processed_record.to_dict(),
            "processing_evidence": evidence_record.to_dict(),
            "dataset_manifest": manifest_record.to_dict(),
            "trainable_data_contract": contract_record.to_dict(),
            "preprocessing_report": preprocessing_record.to_dict(),
            "acquisition_plan": plan_record.to_dict(),
            "acquisition_receipt": receipt_record.to_dict(),
            "approval": approval_record.to_dict(),
        },
        "runtime_materialization": {
            "processed": named_processed,
            "trainable_data_contract": named_contract,
            "dataset_manifest": named_manifest,
            "preprocessing_report": named_preprocessing,
            "processing_evidence": named_evidence,
            "acquisition_receipt": named_receipt,
            "approval": named_approval,
        },
        "training_launched": False,
        "model_mutated": False,
        "experiment_executed": False,
    }
    atomic_json(OUTPUT, proof)
    atomic_json(Path("first_selected_dataset_approval.json"), approval_payload)
    atomic_json(Path("first_selected_dataset_acquisition_receipt.json"), receipt.to_dict())
    atomic_json(Path("first_selected_dataset_trainable_data_contract.json"), final_contract.to_dict())
    print(f"status={proof['status']}")
    print(f"dataset={candidate.dataset_id}@{candidate.revision}")
    print(f"shard={shard_path} bytes={shard_bytes}")
    print(f"receipt={receipt.receipt_id}")
    print(f"processed_hash={result.processed_content_hash}")
    print(f"contract_id={final_contract.contract_id}")
    print("training_launched=false")


if __name__ == "__main__":
    main()
