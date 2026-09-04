#!/usr/bin/env python3
"""Bootstrap the first real scientific data/model evidence on a RunPod Network Volume.

This command performs no paid training and creates no RunPod Pod. It:

1. resolves and selects a real immutable WikiText-2 raw training shard;
2. acquires it through the governed remote-data service;
3. stages verified bytes in content-addressed RunPod S3 storage;
4. preprocesses the real corpus and trains a deterministic tokenizer;
5. selects and instantiates a bounded random-initialized GPT-style model;
6. stages model/tokenizer directories under immutable directory identities; and
7. emits the typed DatasetManifest -> TrainableDataContract ->
   ModelSelectionDecision -> ExperimentProposal evidence chain.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable

import boto3
from botocore.config import Config

from hephaestus.backends.hf_causal_lm import directory_content_identity, directory_manifest
from hephaestus.data import (
    AutonomousDataPreprocessor,
    DataProcessingConfig,
    DatasetAcquisitionApproval,
    DatasetAcquisitionCache,
    DeterministicDatasetSelectionService,
    RemoteAcquisitionLimits,
    RemoteDatasetAcquisitionService,
)
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.planning.service import ClosedLoopExperimentPlanner
from hephaestus.providers.datasets import HuggingFaceDatasetProvider
from hephaestus.providers.models.selection import DeterministicModelSelectionService
from hephaestus.schemas.diagnosis_contract import (
    DiagnosisReport,
    DiagnosticHypothesis,
    EvidenceObservation,
)
from hephaestus.schemas.discovery_contract import (
    DatasetCandidate,
    DatasetSearchRequest,
    ModelCandidate,
)
from hephaestus.schemas.dataset_manifest import DatasetManifest
from hephaestus.schemas.trainable_data_contract import TrainableDataContract
from hephaestus.storage.base import ArtifactRecord
from hephaestus.utils.hashing import hash_file, hash_json


RUN_ID = "first-scientific-bootstrap-001"
LINEAGE_ID = "lineage-first-scientific"
STAGE_NAME = "smoke_test"
TARGET_DATASET_ID = "Salesforce/wikitext"
TARGET_DATASET_PATH = "wikitext-2-raw-v1/train-00000-of-00001.parquet"
DATASET_MAX_BYTES = 16 * 1024 * 1024
MODEL_MAX_PARAMETERS = 3_000_000
MODEL_CONTEXT_LENGTH = 256
TOKENIZER_TARGET_VOCAB = 8_192
RANDOM_SEED = 1729
PRIOR_VOLUME_INVENTORY_HASH = (
    "sha256:21cdd070f23e3973fca510c58f7139c747fb0fd79772d087aad6428f32c26b90"
)
USER_APPROVAL_REF = "approval://operator/explicit-request-2026-09-04-first-scientific-bootstrap"
STORE_PREFIX = "hephaestus/scientific/v1"
OUTPUT_ROOT = Path("first_scientific_bootstrap")
WORK_ROOT = Path(".first_scientific_bootstrap_work")

SPECIAL_TOKENS = (
    "<|unk|>",
    "<|bos|>",
    "<|eos|>",
    "<|pad|>",
    "<|prompt|>",
    "<|target|>",
)


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"missing environment variable: {name}")
    return value


def _canonical_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error")
    return isinstance(error, dict) and str(error.get("Code")) in {
        "404",
        "NoSuchKey",
        "NotFound",
    }


@dataclass(slots=True)
class RunPodContentAddressedStore:
    """Single-writer immutable store for one RunPod Network Volume.

    Content objects use SHA-256 keys. Named materializations are only written
    beneath a path that already contains the immutable directory/content identity.
    Existing objects are accepted only after metadata, size, and byte verification.
    """

    client: Any
    bucket: str
    prefix: str = STORE_PREFIX

    def _object_key(self, digest: str) -> str:
        return f"{self.prefix}/objects/sha256/{digest[:2]}/{digest}"

    @staticmethod
    def _digest_from_ref(artifact_ref: str) -> str:
        if not artifact_ref.startswith("sha256:"):
            raise ValueError(f"invalid artifact ref: {artifact_ref!r}")
        digest = artifact_ref.removeprefix("sha256:").lower()
        if len(digest) != 64:
            raise ValueError(f"invalid artifact ref: {artifact_ref!r}")
        int(digest, 16)
        return digest

    def _head(self, key: str) -> dict[str, object] | None:
        try:
            return self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 - provider boundary
            if _not_found(exc):
                return None
            raise

    def _verify_key(self, key: str, digest: str, byte_size: int) -> bool:
        head = self._head(key)
        if head is None:
            return False
        metadata = head.get("Metadata")
        if not isinstance(metadata, dict) or metadata.get("sha256") != digest:
            raise RuntimeError(f"immutable RunPod object has invalid sha256 metadata: {key}")
        if int(head.get("ContentLength", -1)) != byte_size:
            raise RuntimeError(f"immutable RunPod object has unexpected byte size: {key}")
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        body = response["Body"]
        hasher = hashlib.sha256()
        observed = 0
        try:
            while True:
                chunk = body.read(1024 * 1024)
                if not chunk:
                    break
                observed += len(chunk)
                hasher.update(chunk)
        finally:
            body.close()
        if observed != byte_size or hasher.hexdigest() != digest:
            raise RuntimeError(f"immutable RunPod object failed byte verification: {key}")
        return True

    def _put_verified_bytes(
        self,
        key: str,
        data: bytes,
        *,
        expected_hash: str,
        media_type: str | None = None,
    ) -> None:
        digest = expected_hash.removeprefix("sha256:")
        if _sha256_bytes(data) != digest:
            raise ValueError("named staging expected hash does not match supplied bytes")
        if self._head(key) is not None:
            self._verify_key(key, digest, len(data))
            return
        arguments: dict[str, object] = {
            "Bucket": self.bucket,
            "Key": key,
            "Body": data,
            "Metadata": {"sha256": digest},
        }
        if media_type:
            arguments["ContentType"] = media_type
        self.client.put_object(**arguments)
        self._verify_key(key, digest, len(data))

    def put_bytes(
        self,
        data: bytes,
        *,
        expected_hash: str | None = None,
        media_type: str | None = None,
    ) -> ArtifactRecord:
        digest = _sha256_bytes(data)
        expected = str(expected_hash or "").removeprefix("sha256:")
        if expected and expected != digest:
            raise ValueError(f"expected sha256 {expected}, computed {digest}")
        key = self._object_key(digest)
        self._put_verified_bytes(
            key,
            data,
            expected_hash=f"sha256:{digest}",
            media_type=media_type,
        )
        return ArtifactRecord(
            artifact_ref=f"sha256:{digest}",
            content_hash=digest,
            hash_algorithm="sha256",
            byte_size=len(data),
            storage_path=f"s3://{self.bucket}/{key}",
            created_at=datetime.now(timezone.utc),
            media_type=media_type,
        )

    def put_file(
        self,
        source: Path,
        *,
        expected_hash: str | None = None,
        media_type: str | None = None,
        **_: object,
    ) -> ArtifactRecord:
        source = Path(source)
        digest = hash_file(source)
        expected = str(expected_hash or "").removeprefix("sha256:")
        if expected and expected != digest:
            raise ValueError(f"expected sha256 {expected}, computed {digest}")
        data = source.read_bytes()
        return self.put_bytes(
            data,
            expected_hash=f"sha256:{digest}",
            media_type=media_type,
        )

    def open(self, artifact_ref: str) -> BinaryIO:
        digest = self._digest_from_ref(artifact_ref)
        response = self.client.get_object(Bucket=self.bucket, Key=self._object_key(digest))
        body = response["Body"]
        try:
            data = body.read()
        finally:
            body.close()
        if _sha256_bytes(data) != digest:
            raise RuntimeError("RunPod artifact failed SHA-256 verification")
        return io.BytesIO(data)

    def verify(self, artifact_ref: str) -> bool:
        digest = self._digest_from_ref(artifact_ref)
        head = self._head(self._object_key(digest))
        if head is None:
            return False
        return self._verify_key(
            self._object_key(digest), digest, int(head.get("ContentLength", 0))
        )

    def stage_json(self, payload: object) -> ArtifactRecord:
        return self.put_bytes(
            _canonical_bytes(payload), media_type="application/json"
        )

    def stage_named_file(
        self,
        *,
        key: str,
        source: Path,
        media_type: str | None = None,
    ) -> dict[str, object]:
        digest = hash_file(source)
        data = source.read_bytes()
        self._put_verified_bytes(
            key,
            data,
            expected_hash=f"sha256:{digest}",
            media_type=media_type,
        )
        return {
            "key": key,
            "sha256": f"sha256:{digest}",
            "byte_size": len(data),
            "storage_path": f"s3://{self.bucket}/{key}",
        }

    def stage_directory(
        self,
        *,
        kind: str,
        source: Path,
        directory_identity: str,
    ) -> tuple[ArtifactRecord, dict[str, object]]:
        digest = directory_identity.removeprefix("sha256:")
        component_hashes = directory_manifest(source)
        volume_prefix = f"{self.prefix}/materialized/{kind}/{digest}"
        components: dict[str, object] = {}
        for relative_path, component_digest in component_hashes.items():
            component_source = source / relative_path
            content_record = self.put_file(
                component_source,
                expected_hash=f"sha256:{component_digest}",
            )
            named = self.stage_named_file(
                key=f"{volume_prefix}/{relative_path}",
                source=component_source,
            )
            components[relative_path] = {
                "sha256": f"sha256:{component_digest}",
                "content_artifact_ref": content_record.artifact_ref,
                "content_storage_path": content_record.storage_path,
                "materialized_storage_path": named["storage_path"],
                "byte_size": named["byte_size"],
            }
        manifest_payload = {
            "manifest_version": "runpod-directory-materialization.v1",
            "kind": kind,
            "directory_content_identity": directory_identity,
            "volume_id": self.bucket,
            "volume_prefix": volume_prefix,
            "components": components,
        }
        manifest_record = self.stage_json(manifest_payload)
        return manifest_record, manifest_payload


def _record_ref(record: ArtifactRecord) -> dict[str, object]:
    return {
        "artifact_ref": record.artifact_ref,
        "content_hash": f"sha256:{record.content_hash}",
        "byte_size": record.byte_size,
        "storage_path": record.storage_path,
    }


def _build_s3() -> tuple[Any, RunPodContentAddressedStore, str]:
    bucket = _required("RUNPOD_NETWORK_VOLUME_ID")
    endpoint = _required("RUNPOD_S3_ENDPOINT_URL").rstrip("/")
    region = _required("RUNPOD_DATACENTER_ID")
    access_key = _required("RUNPOD_S3_ACCESS_KEY_ID")
    secret_key = _required("RUNPOD_S3_SECRET_ACCESS_KEY")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(retries={"mode": "standard", "max_attempts": 10}),
    )
    client.head_bucket(Bucket=bucket)
    return client, RunPodContentAddressedStore(client, bucket), region


def _target_train_file(files: Iterable[Any]) -> Any:
    files = tuple(files)
    exact = [item for item in files if item.relative_path == TARGET_DATASET_PATH]
    if len(exact) == 1:
        return exact[0]
    fallback = [
        item
        for item in files
        if "wikitext-2-raw-v1" in item.relative_path
        and PurePosixPath(item.relative_path).name.startswith("train")
        and item.relative_path.casefold().endswith(".parquet")
    ]
    if len(fallback) != 1:
        raise RuntimeError(
            "could not uniquely identify the WikiText-2 raw train Parquet at the resolved revision"
        )
    return fallback[0]


def _discover_dataset(
    provider: HuggingFaceDatasetProvider,
) -> tuple[DatasetSearchRequest, DatasetCandidate, Any, Any]:
    discovery_request = DatasetSearchRequest(
        request_id="dataset-discovery-first-scientific-001",
        diagnosis_report_id="first-scientific-bootstrap-prerequisite",
        problem_statement=TARGET_DATASET_ID,
        provider_allowlist=[provider.provider_id],
        evidence_refs=["operator-goal:first-scientific-bootstrap"],
    )
    discovered = tuple(provider.search(discovery_request))
    matches = [
        candidate
        for candidate in discovered
        if candidate.dataset_id.casefold() == TARGET_DATASET_ID.casefold()
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Hugging Face discovery did not return exactly one {TARGET_DATASET_ID!r} candidate"
        )
    discovered_candidate = matches[0]
    snapshot = provider.resolve_revision(
        discovered_candidate.dataset_id,
        discovered_candidate.revision or "main",
    )
    files = provider.enumerate_files(snapshot)
    train_file = _target_train_file(files)
    if train_file.size_bytes is None:
        raise RuntimeError("provider did not expose the selected train shard byte size")
    if train_file.size_bytes > DATASET_MAX_BYTES:
        raise RuntimeError(
            f"selected WikiText-2 raw train shard exceeds bootstrap budget: {train_file.size_bytes}"
        )
    allowed_licenses = {"cc-by-sa-3.0", "gfdl", "cc-by-sa-3.0,gfdl"}
    if snapshot.license not in allowed_licenses:
        raise RuntimeError(
            f"resolved WikiText license {snapshot.license!r} is outside the bootstrap allowlist"
        )

    candidate = DatasetCandidate.from_dict(discovered_candidate.to_dict())
    candidate.revision = snapshot.resolved_revision
    candidate.splits = ["train"]
    candidate.task_types = sorted(
        set(candidate.task_types) | {"language-modeling", "causal_lm"}
    )
    candidate.languages = sorted(set(candidate.languages) | {"en"})
    candidate.domains = ["general_text", "wikipedia"]
    candidate.format_profile = {
        "record_format": "parquet",
        "relative_path": train_file.relative_path,
        "variant": "wikitext-2-raw-v1",
    }
    candidate.estimated_bytes = int(train_file.size_bytes)
    candidate.license = snapshot.license
    candidate.provenance = {
        "kind": "huggingface_provider_commit_metadata",
        "hub_id": candidate.dataset_id,
        "sha": snapshot.resolved_revision,
        "requested_revision": snapshot.requested_revision,
        "dataset_card_ref": snapshot.dataset_card_ref,
        "dataset_card_revision": snapshot.dataset_card_revision,
        "selected_file": train_file.relative_path,
        "provider_hash": train_file.provider_hash,
        "provider_hash_algorithm": train_file.provider_hash_algorithm,
    }
    candidate.trust_level = "external_metadata"
    candidate.compatibility.update(
        {
            "remote_code_required": False,
            "immutable_revision_available": True,
            "model_compatible": True,
        }
    )
    candidate.metadata.update(
        {
            "description": "WikiText-2 raw English Wikipedia language-modeling corpus",
            "acquisition_files": [train_file.relative_path],
            "requires_normalization": True,
            "dataset_card_ref": snapshot.dataset_card_ref,
        }
    )
    candidate.evidence_refs = sorted(
        set(candidate.evidence_refs)
        | {
            snapshot.dataset_card_ref or "",
            train_file.source_url,
        }
        - {""}
    )
    candidate.missing_metadata = [
        item
        for item in candidate.missing_metadata
        if item not in {"revision", "license", "estimated_bytes"}
    ]

    selection_request = DatasetSearchRequest(
        request_id="dataset-search-first-scientific-001",
        diagnosis_report_id="first-scientific-bootstrap-prerequisite",
        problem_statement="Establish a real bounded English causal-language-modeling corpus from WikiText",
        capability_targets=["language-modeling"],
        required_languages=["en"],
        required_domains=["wikipedia"],
        required_formats=["parquet"],
        size_constraints={"max_bytes": DATASET_MAX_BYTES},
        license_allowlist=sorted(allowed_licenses),
        provider_allowlist=[provider.provider_id],
        evidence_refs=[
            snapshot.dataset_card_ref or f"provider:{provider.provider_id}:{candidate.dataset_id}"
        ],
        metadata={"max_selected_candidates": 1, "mixture_score_delta": 0.0},
    )
    return selection_request, candidate, snapshot, train_file


def _preprocess_and_build_tokenizer(
    *,
    candidate: DatasetCandidate,
    selection: Any,
    approval: DatasetAcquisitionApproval,
    receipt: Any,
) -> tuple[Any, Any, Path, str, dict[str, object]]:
    initial = AutonomousDataPreprocessor(
        DataProcessingConfig(
            artifact_root=WORK_ROOT / "initial_preprocessing",
            max_input_bytes=DATASET_MAX_BYTES,
            max_rows=100_000,
            chunk_size_tokens=64,
            min_tokens=2,
        )
    ).process_remote_acquisition(
        run_id=f"{RUN_ID}-tokenizer-source",
        lineage_id=LINEAGE_ID,
        stage_name=STAGE_NAME,
        candidate=candidate,
        selection=selection,
        approval=approval,
        receipt=receipt,
    )

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer_dir = WORK_ROOT / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS[0]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=TOKENIZER_TARGET_VOCAB,
        min_frequency=2,
        special_tokens=list(SPECIAL_TOKENS),
    )

    source_path = Path(initial.trainable_data_contract.processed_dataset_ref)

    def text_iterator() -> Iterable[str]:
        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                text = str(payload.get("text", "")).strip()
                if text:
                    yield text

    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=SPECIAL_TOKENS[0],
        bos_token=SPECIAL_TOKENS[1],
        eos_token=SPECIAL_TOKENS[2],
        pad_token=SPECIAL_TOKENS[3],
        additional_special_tokens=list(SPECIAL_TOKENS[4:]),
        model_max_length=MODEL_CONTEXT_LENGTH,
    )
    fast.save_pretrained(tokenizer_dir)
    tokenizer_identity = directory_content_identity(tokenizer_dir)

    @dataclass(slots=True)
    class _TokenizerChecker:
        tokenizer_ref: str
        checker_id: str = "pretrained-tokenizer-fast-context-check.v1"

        def check(self, texts: Any) -> tuple[bool, dict[str, object]]:
            max_length = 0
            checked = 0
            vocab_size = len(fast)
            for text in texts:
                encoded = fast.encode(str(text), add_special_tokens=False)
                if not encoded:
                    return False, {
                        "reason": "empty_encoding",
                        "checked_records": checked,
                        "vocab_size": vocab_size,
                    }
                max_length = max(max_length, len(encoded))
                if any(token_id < 0 or token_id >= vocab_size for token_id in encoded):
                    return False, {
                        "reason": "token_id_out_of_range",
                        "checked_records": checked,
                        "vocab_size": vocab_size,
                    }
                checked += 1
            return max_length <= MODEL_CONTEXT_LENGTH, {
                "checked_records": checked,
                "max_tokenized_length": max_length,
                "model_context_length": MODEL_CONTEXT_LENGTH,
                "vocab_size": vocab_size,
                "special_tokens": list(SPECIAL_TOKENS),
            }

    checker = _TokenizerChecker(tokenizer_identity)
    final = AutonomousDataPreprocessor(
        DataProcessingConfig(
            artifact_root=WORK_ROOT / "final_preprocessing",
            max_input_bytes=DATASET_MAX_BYTES,
            max_rows=100_000,
            chunk_size_tokens=48,
            min_tokens=2,
        ),
        tokenizer_checker=checker,
    ).process_remote_acquisition(
        run_id=RUN_ID,
        lineage_id=LINEAGE_ID,
        stage_name=STAGE_NAME,
        candidate=candidate,
        selection=selection,
        approval=approval,
        receipt=receipt,
        tokenizer_ref=tokenizer_identity,
    )
    tokenizer_evidence = {
        "directory_content_identity": tokenizer_identity,
        "vocab_size": len(fast),
        "model_context_length": MODEL_CONTEXT_LENGTH,
        "special_tokens": list(SPECIAL_TOKENS),
    }
    return initial, final, tokenizer_dir, tokenizer_identity, tokenizer_evidence


def _gpt2_parameter_estimate(vocab_size: int, n_embd: int, n_layer: int) -> int:
    # GPT-2 family with tied LM head: token embeddings + position embeddings +
    # n_layer * (attention/MLP/layernorm parameters) + final layer norm.
    return (
        vocab_size * n_embd
        + MODEL_CONTEXT_LENGTH * n_embd
        + n_layer * (12 * n_embd * n_embd + 13 * n_embd)
        + 2 * n_embd
    )


def _architecture_specs(vocab_size: int) -> list[dict[str, object]]:
    values = [
        ("gpt2-random-small-a", 128, 4, 4),
        ("gpt2-random-small-b", 192, 6, 6),
        ("gpt2-random-small-c", 256, 6, 8),
    ]
    specs: list[dict[str, object]] = []
    for model_id, n_embd, n_layer, n_head in values:
        specs.append(
            {
                "spec_version": "random-init-gpt2-spec.v1",
                "model_id": model_id,
                "model_type": "gpt2",
                "vocab_size": vocab_size,
                "n_positions": MODEL_CONTEXT_LENGTH,
                "n_ctx": MODEL_CONTEXT_LENGTH,
                "n_embd": n_embd,
                "n_layer": n_layer,
                "n_head": n_head,
                "estimated_parameters": _gpt2_parameter_estimate(
                    vocab_size, n_embd, n_layer
                ),
                "initialization": "transformers-default-random-init",
                "random_seed": RANDOM_SEED,
                "pretrained_weights": False,
            }
        )
    return specs


def _candidate_from_spec(
    *,
    spec: dict[str, object],
    tokenizer_identity: str,
    spec_record: ArtifactRecord,
    dataset_manifest_ref: str,
) -> ModelCandidate:
    return ModelCandidate(
        candidate_id=f"first-scientific:{spec['model_id']}@random-init-spec-v1",
        provider_id="first_scientific_catalog",
        model_id=str(spec["model_id"]),
        revision="random-init-spec-v1",
        architecture_family="gpt2",
        parameter_count=int(spec["estimated_parameters"]),
        context_length=MODEL_CONTEXT_LENGTH,
        tokenizer_ref=tokenizer_identity,
        license="internal-research-generated",
        capabilities=["causal_lm", "pretraining", "smoke_test"],
        runtime_requirements={
            "supported_backends": ["transformers_causal_lm"],
            "memory_gb": 2.0,
            "estimated_runtime_seconds": 900,
        },
        compatibility={
            "runtime": True,
            "checkpoint_integrity": "content_hash",
            "smoke_test": False,
        },
        risk_signals=["non_public_license_scope", "not_yet_instantiated"],
        artifact_ref=spec_record.artifact_ref,
        evidence_refs=[
            spec_record.storage_path,
            dataset_manifest_ref,
        ],
        metadata={
            "initialization": "random",
            "pretrained_weights": False,
            "random_seed": RANDOM_SEED,
            "architecture_spec": dict(spec),
        },
    )


def _build_diagnosis(
    *,
    dataset_manifest_record: ArtifactRecord,
    data_contract_record: ArtifactRecord,
    processed_record: ArtifactRecord,
    processing_evidence_record: ArtifactRecord,
    tokenizer_identity: str,
    tokenizer_manifest_record: ArtifactRecord,
) -> DiagnosisReport:
    baseline_ref = f"runpod-volume-inventory:{PRIOR_VOLUME_INVENTORY_HASH}"
    evidence_refs = [
        baseline_ref,
        dataset_manifest_record.storage_path,
        data_contract_record.storage_path,
        processed_record.storage_path,
        processing_evidence_record.storage_path,
        tokenizer_manifest_record.storage_path,
    ]
    hypothesis = DiagnosticHypothesis(
        hypothesis_id="hypothesis-establish-first-random-init-model",
        failure_domain="architecture",
        summary=(
            "No prior scientific model or checkpoint exists on the paid volume; "
            "establish the first bounded random-initialized causal-LM architecture "
            "while holding the newly verified real dataset and tokenizer fixed."
        ),
        supporting_evidence_refs=evidence_refs,
        required_tests=[
            "model_parameter_budget_check",
            "local_forward_smoke_check",
            "directory_content_identity_verification",
        ],
        recommended_intervention_kinds=["change_model"],
        confidence=0.99,
    )
    return DiagnosisReport(
        report_id="diagnosis-first-scientific-bootstrap-001",
        request_id="diagnosis-request-first-scientific-bootstrap-001",
        run_id=RUN_ID,
        lineage_id=LINEAGE_ID,
        stage_name=STAGE_NAME,
        status="completed",
        observations=[
            EvidenceObservation(
                observation_id="observation-paid-volume-no-real-model",
                evidence_kind="artifact_inventory",
                source_ref=baseline_ref,
                summary=(
                    "Verified paid-volume inventory contained no viable real model "
                    "weights or scientific checkpoints before this bootstrap."
                ),
                severity="info",
                confidence=1.0,
            ),
            EvidenceObservation(
                observation_id="observation-real-data-now-fixed",
                evidence_kind="dataset_manifest",
                source_ref=dataset_manifest_record.storage_path,
                summary=(
                    "A real immutable dataset, processed data contract, and tokenizer "
                    "are staged and can be held fixed while model architecture changes."
                ),
                severity="info",
                confidence=0.99,
            ),
        ],
        hypotheses=[hypothesis],
        leading_hypothesis_id=hypothesis.hypothesis_id,
        missing_evidence=[],
        confidence=0.99,
        metadata={
            "baseline_ref": None,
            "baseline_justification": (
                "This is the first scientific lineage: the verified pre-bootstrap "
                "RunPod volume contained no real scientific model/checkpoint baseline."
            ),
            "capability_targets": ["causal_lm", "pretraining"],
            "task_requirements": ["causal_lm"],
            "architecture_constraints": {
                "allowed_families": ["gpt2"],
                "max_parameters": MODEL_MAX_PARAMETERS,
                "min_context_length": MODEL_CONTEXT_LENGTH,
            },
            "tokenizer_constraints": {"tokenizer_ref": tokenizer_identity},
            "runtime_constraints": {
                "backend": "transformers_causal_lm",
                "min_context_length": MODEL_CONTEXT_LENGTH,
                "max_memory_gb": 8,
                "checkpoint_integrity": "content_hash",
                "smoke_test_required": False,
            },
            "budget": {
                "max_parameters": MODEL_MAX_PARAMETERS,
                "max_runtime_seconds": 3600,
            },
            "model_license_allowlist": ["internal-research-generated"],
            "model_provider_allowlist": ["first_scientific_catalog"],
            "lineage_trust_level": "verified",
            "training_constraints": {
                "backend_id": "transformers_causal_lm",
                "training_mode": "causal_lm_pretraining",
                "trainable_data_contract_ref": data_contract_record.storage_path,
                "trainable_data_contract_hash": f"sha256:{data_contract_record.content_hash}",
                "processed_dataset_ref": processed_record.storage_path,
                "processed_dataset_hash": f"sha256:{processed_record.content_hash}",
                "processing_evidence_ref": processing_evidence_record.storage_path,
                "dataset_manifest_ref": dataset_manifest_record.storage_path,
                "dataset_manifest_hash": f"sha256:{dataset_manifest_record.content_hash}",
                "tokenizer_directory_manifest_ref": tokenizer_manifest_record.storage_path,
                "tokenizer_revision": tokenizer_identity,
                "optimizer": "adamw_torch",
                "scheduler": "linear",
                "learning_rate": 0.0005,
                "weight_decay": 0.01,
                "per_device_train_batch_size": 8,
                "gradient_accumulation_steps": 1,
                "max_steps": 50,
                "save_steps": 25,
                "logging_steps": 5,
                "max_seq_length": MODEL_CONTEXT_LENGTH,
                "device": "cuda",
                "random_seed": RANDOM_SEED,
                "fixed_dataset": True,
                "fixed_tokenizer": True,
                "launch_authorized": False,
                "runpod_pod_created": False,
            },
        },
    )


def _instantiate_model(
    *,
    spec: dict[str, object],
    tokenizer_dir: Path,
    tokenizer_identity: str,
) -> tuple[Path, str, int, dict[str, object]]:
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        tokenizer_dir, local_files_only=True
    )
    torch.manual_seed(RANDOM_SEED)
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=MODEL_CONTEXT_LENGTH,
        n_ctx=MODEL_CONTEXT_LENGTH,
        n_embd=int(spec["n_embd"]),
        n_layer=int(spec["n_layer"]),
        n_head=int(spec["n_head"]),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = GPT2LMHeadModel(config)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters > MODEL_MAX_PARAMETERS:
        raise RuntimeError(
            f"selected random-init model exceeds parameter budget: {actual_parameters}"
        )
    sample = tokenizer(
        "The first Hephaestus scientific model is random initialized.",
        return_tensors="pt",
        truncation=True,
        max_length=MODEL_CONTEXT_LENGTH,
    )
    model.eval()
    with torch.no_grad():
        logits = model(**sample).logits
    finite = bool(torch.isfinite(logits).all().item())
    if not finite or logits.ndim != 3 or logits.shape[-1] != len(tokenizer):
        raise RuntimeError("random-init model failed the local forward smoke check")

    model_dir = WORK_ROOT / "model"
    model_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(model_dir, safe_serialization=True)
    model_identity = directory_content_identity(model_dir)
    smoke = {
        "status": "passed",
        "device": "cpu",
        "input_tokens": int(sample["input_ids"].numel()),
        "logits_shape": list(logits.shape),
        "finite_logits": finite,
        "parameter_count": actual_parameters,
        "random_seed": RANDOM_SEED,
        "tokenizer_revision": tokenizer_identity,
    }
    return model_dir, model_identity, actual_parameters, smoke


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    OUTPUT_ROOT.mkdir(parents=True)
    WORK_ROOT.mkdir(parents=True)

    _client, store, datacenter = _build_s3()
    provider = HuggingFaceDatasetProvider(enable_network=True, max_results=20)
    selection_request, candidate, snapshot, train_file = _discover_dataset(provider)
    dataset_selection = DeterministicDatasetSelectionService().select(
        selection_request, [candidate]
    )
    if dataset_selection.status != "selected" or dataset_selection.selected_candidate_ids != [candidate.candidate_id]:
        raise RuntimeError(
            "governed dataset selector did not select the resolved WikiText candidate: "
            + json.dumps(dataset_selection.to_dict(), sort_keys=True)
        )

    acquisition_service = RemoteDatasetAcquisitionService(
        providers={provider.provider_id: provider},
        cache=DatasetAcquisitionCache(WORK_ROOT / "dataset_cache"),
        artifact_store=store,
        secrets_provider=EnvironmentSecretsProvider(),
    )
    planning = acquisition_service.plan(
        candidate=candidate,
        selection=dataset_selection,
        limits=RemoteAcquisitionLimits(
            max_bytes=DATASET_MAX_BYTES,
            max_files=1,
            chunk_size=1024 * 1024,
            timeout_seconds=30.0,
            disk_reserve_bytes=64 * 1024 * 1024,
            allowed_suffixes=(".parquet",),
        ),
    )
    if planning.status != "ready" or planning.plan is None:
        raise RuntimeError(
            "remote acquisition planning was blocked: "
            + json.dumps(
                [issue.to_dict() for issue in planning.issues], sort_keys=True
            )
        )
    approval = DatasetAcquisitionApproval(
        selection_decision_id=dataset_selection.decision_id,
        approved_candidate_ids=(candidate.candidate_id,),
        approval_refs=(USER_APPROVAL_REF,),
        approved_requirements=planning.plan.required_approvals,
    )
    acquired = acquisition_service.acquire(planning.plan, approval)
    if not acquired.completed:
        raise RuntimeError(
            "remote dataset acquisition failed: "
            + json.dumps(acquired.receipt.to_dict(), sort_keys=True)
        )
    receipt = acquired.receipt
    if len(receipt.acquired_files) != 1:
        raise RuntimeError("bootstrap expected exactly one acquired dataset shard")
    raw_file = receipt.acquired_files[0]
    if not raw_file.artifact_ref or not store.verify(raw_file.artifact_ref):
        raise RuntimeError("staged raw dataset artifact failed RunPod verification")

    initial, final, tokenizer_dir, tokenizer_identity, tokenizer_evidence = (
        _preprocess_and_build_tokenizer(
            candidate=candidate,
            selection=dataset_selection,
            approval=approval,
            receipt=receipt,
        )
    )
    del initial
    tokenizer_manifest_record, tokenizer_materialization = store.stage_directory(
        kind="tokenizers",
        source=tokenizer_dir,
        directory_identity=tokenizer_identity,
    )

    processed_path = Path(final.trainable_data_contract.processed_dataset_ref)
    processed_record = store.put_file(
        processed_path,
        expected_hash=final.processed_content_hash,
        media_type="application/jsonl",
    )
    if not store.verify(processed_record.artifact_ref):
        raise RuntimeError("processed dataset failed RunPod verification")

    processing_evidence_payload = copy.deepcopy(final.processing_evidence)
    processing_evidence_payload["processed_dataset_ref"] = processed_record.storage_path
    processing_evidence_payload["processing_evidence_ref"] = "self"
    processing_evidence_payload["runpod_staging"] = {
        "volume_id": store.bucket,
        "datacenter_id": datacenter,
        "raw_source_artifact_ref": raw_file.artifact_ref,
        "raw_source_storage_path": raw_file.artifact_ref
        and next(
            (
                item.get("storage_path")
                for item in [
                    {
                        "storage_path": f"s3://{store.bucket}/{store._object_key(raw_file.artifact_ref.removeprefix('sha256:'))}"
                    }
                ]
            ),
            None,
        ),
        "processed_artifact_ref": processed_record.artifact_ref,
        "processed_storage_path": processed_record.storage_path,
    }
    processing_evidence_record = store.stage_json(processing_evidence_payload)

    manifest_payload = final.manifest.to_dict()
    manifest_payload["artifact_ref"] = processed_record.storage_path
    manifest_payload["datasets"][0]["source"] = (
        raw_file.artifact_ref or raw_file.local_content_hash
    )
    manifest_payload["datasets"][0]["version"] = receipt.resolved_revision
    manifest_payload["datasets"][0]["license"] = receipt.license
    manifest_payload["metadata"]["processing_evidence_ref"] = (
        processing_evidence_record.storage_path
    )
    manifest_payload["metadata"]["acquisition_receipt_id"] = receipt.receipt_id
    manifest_payload["metadata"]["raw_source_artifact_ref"] = raw_file.artifact_ref
    manifest_payload["metadata"]["raw_source_hash"] = raw_file.local_content_hash
    final_manifest = DatasetManifest.from_dict(manifest_payload)
    dataset_manifest_record = store.stage_json(final_manifest.to_dict())

    data_contract_payload = final.trainable_data_contract.to_dict()
    data_contract_payload["processed_dataset_ref"] = processed_record.storage_path
    final_contract = TrainableDataContract.from_dict(data_contract_payload)
    data_contract_record = store.stage_json(final_contract.to_dict())

    dataset_request_record = store.stage_json(selection_request.to_dict())
    dataset_candidate_record = store.stage_json(candidate.to_dict())
    dataset_selection_record = store.stage_json(dataset_selection.to_dict())
    acquisition_plan_record = store.stage_json(planning.plan.to_dict())
    acquisition_receipt_record = store.stage_json(receipt.to_dict())

    tokenizer_evidence.update(
        {
            "directory_manifest_artifact_ref": tokenizer_manifest_record.artifact_ref,
            "directory_manifest_storage_path": tokenizer_manifest_record.storage_path,
            "volume_prefix": tokenizer_materialization["volume_prefix"],
        }
    )

    diagnosis = _build_diagnosis(
        dataset_manifest_record=dataset_manifest_record,
        data_contract_record=data_contract_record,
        processed_record=processed_record,
        processing_evidence_record=processing_evidence_record,
        tokenizer_identity=tokenizer_identity,
        tokenizer_manifest_record=tokenizer_manifest_record,
    )
    planner = ClosedLoopExperimentPlanner()
    interventions = planner.propose_interventions(diagnosis)
    change_model = next(
        (item for item in interventions if item.intervention_kind == "change_model"),
        None,
    )
    if change_model is None:
        raise RuntimeError("closed-loop planner did not produce a change_model intervention")
    _dataset_request, model_request = planner.create_discovery_requests(
        diagnosis, change_model
    )
    if model_request is None:
        raise RuntimeError("closed-loop planner did not produce a model search request")

    from transformers import PreTrainedTokenizerFast

    fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        tokenizer_dir, local_files_only=True
    )
    specs = _architecture_specs(len(fast_tokenizer))
    spec_records: dict[str, ArtifactRecord] = {}
    spec_candidates: list[ModelCandidate] = []
    for spec in specs:
        spec_record = store.stage_json(spec)
        spec_records[str(spec["model_id"])] = spec_record
        spec_candidates.append(
            _candidate_from_spec(
                spec=spec,
                tokenizer_identity=tokenizer_identity,
                spec_record=spec_record,
                dataset_manifest_ref=dataset_manifest_record.storage_path,
            )
        )
    architecture_selection = DeterministicModelSelectionService().select(
        model_request, spec_candidates
    )
    if architecture_selection.status != "selected" or not architecture_selection.selected_candidate_id:
        raise RuntimeError(
            "model architecture selection was blocked: "
            + json.dumps(architecture_selection.to_dict(), sort_keys=True)
        )
    selected_spec_candidate = next(
        candidate_item
        for candidate_item in spec_candidates
        if candidate_item.candidate_id == architecture_selection.selected_candidate_id
    )
    selected_spec = next(
        spec
        for spec in specs
        if str(spec["model_id"]) == selected_spec_candidate.model_id
    )

    model_dir, model_identity, actual_parameters, smoke = _instantiate_model(
        spec=selected_spec,
        tokenizer_dir=tokenizer_dir,
        tokenizer_identity=tokenizer_identity,
    )
    model_manifest_record, model_materialization = store.stage_directory(
        kind="models",
        source=model_dir,
        directory_identity=model_identity,
    )
    actual_candidate = ModelCandidate(
        candidate_id=(
            f"first-scientific:{selected_spec['model_id']}@{model_identity.removeprefix('sha256:')[:16]}"
        ),
        provider_id="first_scientific_catalog",
        model_id=str(selected_spec["model_id"]),
        revision=model_identity,
        architecture_family="gpt2",
        parameter_count=actual_parameters,
        context_length=MODEL_CONTEXT_LENGTH,
        tokenizer_ref=tokenizer_identity,
        license="internal-research-generated",
        capabilities=["causal_lm", "pretraining", "smoke_test"],
        runtime_requirements={
            "supported_backends": ["transformers_causal_lm"],
            "memory_gb": 2.0,
            "estimated_runtime_seconds": 900,
        },
        compatibility={
            "runtime": True,
            "compatible": True,
            "checkpoint_integrity": "content_hash",
            "smoke_test": True,
        },
        risk_signals=["non_public_license_scope"],
        artifact_ref=model_manifest_record.artifact_ref,
        evidence_refs=[
            model_manifest_record.storage_path,
            tokenizer_manifest_record.storage_path,
            dataset_manifest_record.storage_path,
        ],
        metadata={
            "initialization": "random",
            "pretrained_weights": False,
            "random_seed": RANDOM_SEED,
            "architecture_spec": dict(selected_spec),
            "directory_content_identity": model_identity,
            "volume_prefix": model_materialization["volume_prefix"],
            "forward_smoke": smoke,
        },
    )
    canonical_candidates = [actual_candidate] + [
        item
        for item in spec_candidates
        if item.model_id != actual_candidate.model_id
    ]
    model_selection = DeterministicModelSelectionService().select(
        model_request, canonical_candidates
    )
    if (
        model_selection.status != "selected"
        or model_selection.selected_candidate_id != actual_candidate.candidate_id
    ):
        raise RuntimeError(
            "canonical model selection did not select the materialized random-init model: "
            + json.dumps(model_selection.to_dict(), sort_keys=True)
        )

    diagnosis.metadata["training_constraints"].update(
        {
            "model_id": model_materialization["volume_prefix"],
            "model_revision": model_identity,
            "architecture_family": "gpt2",
            "tokenizer_id": tokenizer_materialization["volume_prefix"],
            "tokenizer_revision": tokenizer_identity,
            "model_directory_manifest_ref": model_manifest_record.storage_path,
            "tokenizer_directory_manifest_ref": tokenizer_manifest_record.storage_path,
        }
    )
    # Regenerate the intervention after adding the exact materialized model/tokenizer
    # bindings so its resulting experiment carries the finalized fixed constraints.
    interventions = planner.propose_interventions(diagnosis)
    change_model = next(
        item for item in interventions if item.intervention_kind == "change_model"
    )
    _unused_dataset_request, final_model_request = planner.create_discovery_requests(
        diagnosis, change_model
    )
    if final_model_request is None:
        raise RuntimeError("final model search request is missing")
    # Request identity is derived from diagnosis/intervention IDs, so the materialized
    # path additions above do not alter the model-search constraints. Re-run selection
    # against the exact final request to keep the typed binding canonical.
    model_selection = DeterministicModelSelectionService().select(
        final_model_request, canonical_candidates
    )
    if model_selection.selected_candidate_id != actual_candidate.candidate_id:
        raise RuntimeError("final canonical model selection changed unexpectedly")
    experiment = planner.propose_experiment(
        diagnosis,
        change_model,
        dataset_selection=None,
        model_selection=model_selection,
    )
    experiment.metadata.update(
        {
            "launch_authorized": False,
            "runpod_pod_created": False,
            "bootstrap_only": True,
            "dataset_manifest_artifact_ref": dataset_manifest_record.artifact_ref,
            "trainable_data_contract_artifact_ref": data_contract_record.artifact_ref,
            "model_directory_manifest_artifact_ref": model_manifest_record.artifact_ref,
            "tokenizer_directory_manifest_artifact_ref": tokenizer_manifest_record.artifact_ref,
        }
    )

    diagnosis_record = store.stage_json(diagnosis.to_dict())
    intervention_record = store.stage_json(change_model.to_dict())
    model_request_record = store.stage_json(final_model_request.to_dict())
    architecture_candidates_record = store.stage_json(
        [item.to_dict() for item in spec_candidates]
    )
    architecture_selection_record = store.stage_json(architecture_selection.to_dict())
    model_candidate_record = store.stage_json(actual_candidate.to_dict())
    model_selection_record = store.stage_json(model_selection.to_dict())
    experiment_record = store.stage_json(experiment.to_dict())

    evidence = {
        "dataset_search_request": _record_ref(dataset_request_record),
        "dataset_candidate": _record_ref(dataset_candidate_record),
        "dataset_selection_decision": _record_ref(dataset_selection_record),
        "dataset_acquisition_plan": _record_ref(acquisition_plan_record),
        "dataset_acquisition_receipt": _record_ref(acquisition_receipt_record),
        "processing_evidence": _record_ref(processing_evidence_record),
        "dataset_manifest": _record_ref(dataset_manifest_record),
        "trainable_data_contract": _record_ref(data_contract_record),
        "processed_dataset": _record_ref(processed_record),
        "tokenizer_directory_manifest": _record_ref(tokenizer_manifest_record),
        "diagnosis_report": _record_ref(diagnosis_record),
        "intervention_proposal": _record_ref(intervention_record),
        "model_search_request": _record_ref(model_request_record),
        "model_architecture_candidates": _record_ref(architecture_candidates_record),
        "model_architecture_selection": _record_ref(architecture_selection_record),
        "model_candidate": _record_ref(model_candidate_record),
        "model_selection_decision": _record_ref(model_selection_record),
        "model_directory_manifest": _record_ref(model_manifest_record),
        "experiment_proposal": _record_ref(experiment_record),
    }
    bundle = {
        "manifest_version": "first-scientific-bootstrap.v1",
        "run_id": RUN_ID,
        "lineage_id": LINEAGE_ID,
        "stage_name": STAGE_NAME,
        "source_volume": {
            "provider": "runpod-network-volume",
            "volume_id": store.bucket,
            "datacenter_id": datacenter,
            "prior_verified_inventory_hash": PRIOR_VOLUME_INVENTORY_HASH,
        },
        "dataset": {
            "provider_id": candidate.provider_id,
            "dataset_id": candidate.dataset_id,
            "resolved_revision": receipt.resolved_revision,
            "selected_file": train_file.relative_path,
            "license": receipt.license,
            "raw_content_hash": raw_file.local_content_hash,
            "raw_artifact_ref": raw_file.artifact_ref,
            "processed_content_hash": f"sha256:{processed_record.content_hash}",
            "dataset_manifest_id": final_manifest.manifest_id,
            "trainable_data_contract_id": final_contract.contract_id,
        },
        "tokenizer": {
            **tokenizer_evidence,
            "materialized_volume_prefix": tokenizer_materialization["volume_prefix"],
        },
        "model": {
            "model_id": actual_candidate.model_id,
            "model_candidate_id": actual_candidate.candidate_id,
            "directory_content_identity": model_identity,
            "parameter_count": actual_parameters,
            "architecture_family": "gpt2",
            "context_length": MODEL_CONTEXT_LENGTH,
            "pretrained_weights": False,
            "random_seed": RANDOM_SEED,
            "materialized_volume_prefix": model_materialization["volume_prefix"],
            "forward_smoke": smoke,
        },
        "typed_chain": {
            "dataset_manifest_id": final_manifest.manifest_id,
            "trainable_data_contract_id": final_contract.contract_id,
            "model_selection_decision_id": model_selection.decision_id,
            "experiment_id": experiment.experiment_id,
        },
        "evidence": evidence,
        "launch_boundary": {
            "training_launched": False,
            "runpod_pod_created": False,
            "launch_authorized": False,
            "note": "Bootstrap stages evidence only; paid training requires a separate explicit launch action.",
        },
    }
    bundle_record = store.stage_json(bundle)

    # Persist user-readable evidence locally for the workflow artifact.
    outputs = {
        "dataset_search_request.json": selection_request.to_dict(),
        "dataset_candidate.json": candidate.to_dict(),
        "dataset_selection_decision.json": dataset_selection.to_dict(),
        "dataset_acquisition_plan.json": planning.plan.to_dict(),
        "dataset_acquisition_receipt.json": receipt.to_dict(),
        "processing_evidence.json": processing_evidence_payload,
        "dataset_manifest.json": final_manifest.to_dict(),
        "trainable_data_contract.json": final_contract.to_dict(),
        "tokenizer_directory_manifest.json": tokenizer_materialization,
        "diagnosis_report.json": diagnosis.to_dict(),
        "intervention_proposal.json": change_model.to_dict(),
        "model_search_request.json": final_model_request.to_dict(),
        "model_architecture_candidates.json": [item.to_dict() for item in spec_candidates],
        "model_architecture_selection.json": architecture_selection.to_dict(),
        "model_candidate.json": actual_candidate.to_dict(),
        "model_selection_decision.json": model_selection.to_dict(),
        "model_directory_manifest.json": model_materialization,
        "experiment_proposal.json": experiment.to_dict(),
        "first_scientific_manifest.json": bundle,
    }
    for filename, payload in outputs.items():
        _write_json(OUTPUT_ROOT / filename, payload)
    summary = {
        "status": "staged_not_launched",
        "volume_id": store.bucket,
        "datacenter_id": datacenter,
        "bundle_artifact_ref": bundle_record.artifact_ref,
        "bundle_storage_path": bundle_record.storage_path,
        "dataset_id": candidate.dataset_id,
        "dataset_revision": receipt.resolved_revision,
        "dataset_raw_sha256": raw_file.local_content_hash,
        "processed_dataset_sha256": f"sha256:{processed_record.content_hash}",
        "tokenizer_directory_identity": tokenizer_identity,
        "model_directory_identity": model_identity,
        "model_parameter_count": actual_parameters,
        "dataset_manifest_id": final_manifest.manifest_id,
        "trainable_data_contract_id": final_contract.contract_id,
        "model_selection_decision_id": model_selection.decision_id,
        "experiment_id": experiment.experiment_id,
        "required_approvals_for_experiment": list(experiment.required_approvals),
        "training_launched": False,
        "runpod_pod_created": False,
        "launch_authorized": False,
    }
    _write_json(OUTPUT_ROOT / "bootstrap_summary.json", summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
