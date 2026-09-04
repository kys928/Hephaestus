from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from hephaestus.schemas.dataset_manifest import DatasetManifest
from hephaestus.schemas.discovery_contract import DatasetCandidate, DatasetSelectionDecision
from hephaestus.schemas.preprocessing_report import PreprocessingReport
from hephaestus.schemas.trainable_data_contract import TrainableDataContract
from hephaestus.utils.hashing import hash_json, hash_text
from hephaestus.utils.io import write_json

from .acquisition import (
    DatasetAcquisitionApproval,
    LocalAcquisition,
    acquire_approved_local_candidate,
)
from .acquisition_models import AcquisitionReceipt
from .chunking import chunk_records
from .contract_builder import build_preprocessing_contracts
from .dedup import deduplicate_records
from .manifest_builder import build_dataset_manifest
from .materialization import validate_remote_acquisition_for_preprocessing
from .normalization import normalize_record


@dataclass(frozen=True, slots=True)
class PreprocessedDataset:
    processed_dataset_ref: str
    operations: list[str] = field(default_factory=list)
    dropped_examples: int = 0
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "processed_dataset_ref": self.processed_dataset_ref,
            "operations": list(self.operations),
            "dropped_examples": self.dropped_examples,
            **self.metadata,
        }


def normalize_preprocessing_output(payload: dict[str, Any]) -> dict[str, object]:
    operations = (
        [str(op) for op in payload.get("operations", [])]
        if isinstance(payload.get("operations"), list)
        else []
    )
    if not operations:
        operations = ["identity"]
    return PreprocessedDataset(
        processed_dataset_ref=str(
            payload.get("processed_dataset_ref") or payload.get("artifact_ref") or ""
        ),
        operations=operations,
        dropped_examples=int(payload.get("dropped_examples", 0) or 0),
        metadata={
            key: value
            for key, value in payload.items()
            if key
            not in {
                "processed_dataset_ref",
                "artifact_ref",
                "operations",
                "dropped_examples",
            }
        },
    ).to_dict()


class RecordFilter(Protocol):
    filter_id: str

    def keep(self, record: dict[str, object]) -> bool: ...


class ContaminationChecker(Protocol):
    reference_set_id: str

    def is_contaminated(self, record: dict[str, object]) -> bool: ...


class TokenizerCompatibilityChecker(Protocol):
    tokenizer_ref: str
    checker_id: str

    def check(self, texts: Sequence[str]) -> tuple[bool, dict[str, object]]: ...


@dataclass(frozen=True, slots=True)
class DataProcessingConfig:
    artifact_root: Path
    max_input_bytes: int = 64 * 1024 * 1024
    max_rows: int = 100_000
    chunk_size_tokens: int = 512
    min_tokens: int = 1
    near_duplicate_threshold: float | None = None
    near_duplicate_max_records: int = 5_000
    prompt_target_template: str = "<|prompt|>\n{prompt}\n<|target|>\n{target}"

    def __post_init__(self) -> None:
        if self.max_input_bytes <= 0 or self.max_rows <= 0:
            raise ValueError("max_input_bytes and max_rows must be positive")
        if self.chunk_size_tokens <= 0 or self.min_tokens < 0:
            raise ValueError(
                "chunk_size_tokens must be positive and min_tokens must be non-negative"
            )
        if self.near_duplicate_max_records <= 0:
            raise ValueError("near_duplicate_max_records must be positive")
        if (
            self.near_duplicate_threshold is not None
            and not 0.0 <= self.near_duplicate_threshold <= 1.0
        ):
            raise ValueError("near_duplicate_threshold must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class DataFactoryResult:
    candidate_id: str
    dataset_identity: str
    processed_content_hash: str
    artifact_dir: Path
    manifest: DatasetManifest
    preprocessing_report: PreprocessingReport
    trainable_data_contract: TrainableDataContract
    processing_evidence: dict[str, object]


@dataclass(frozen=True, slots=True)
class _LoadResult:
    records: tuple[dict[str, object], ...]
    rows_seen: int
    malformed_rows: int
    truncated: bool


def _load_parquet_records(path: Path, max_rows: int) -> tuple[list[object], bool]:
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            "Parquet preprocessing requires the optional 'pyarrow' dependency"
        ) from exc

    parquet_file = parquet.ParquetFile(path)
    total_rows = int(parquet_file.metadata.num_rows)
    raw_records: list[object] = []
    for batch in parquet_file.iter_batches(batch_size=min(2048, max_rows)):
        for row in batch.to_pylist():
            if len(raw_records) >= max_rows:
                return raw_records, True
            raw_records.append(row)
        if len(raw_records) >= max_rows:
            break
    return raw_records, total_rows > len(raw_records)


def _load_records(path: Path, record_format: str, max_rows: int) -> _LoadResult:
    raw_records: list[object]
    malformed = 0
    truncated = False
    if record_format == "jsonl":
        raw_records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if len(raw_records) >= max_rows:
                    truncated = True
                    break
                try:
                    raw_records.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed += 1
    elif record_format == "json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            payload = payload["data"]
        if not isinstance(payload, list):
            raise ValueError("JSON dataset must be a list or an object with a data list")
        truncated = len(payload) > max_rows
        raw_records = payload[:max_rows]
    elif record_format == "csv":
        raw_records = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if len(raw_records) >= max_rows:
                    truncated = True
                    break
                raw_records.append(dict(row))
    elif record_format == "parquet":
        raw_records, truncated = _load_parquet_records(path, max_rows)
    else:
        raise ValueError(f"unsupported record format: {record_format}")

    normalized: list[dict[str, object]] = []
    rows_seen = len(raw_records) + malformed
    for raw in raw_records:
        if not isinstance(raw, dict):
            malformed += 1
            continue
        record = normalize_record(raw)
        if record is None:
            malformed += 1
            continue
        normalized.append(record)
    return _LoadResult(tuple(normalized), rows_seen, malformed, truncated)


def _slug(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in value
    )
    return normalized.strip("-.") or "dataset"


@dataclass(slots=True)
class AutonomousDataPreprocessor:
    """Bounded preprocessing after a separately governed acquisition boundary."""

    config: DataProcessingConfig
    record_filter: RecordFilter | None = None
    contamination_checker: ContaminationChecker | None = None
    tokenizer_checker: TokenizerCompatibilityChecker | None = None

    def process(
        self,
        *,
        run_id: str,
        lineage_id: str,
        stage_name: str,
        candidate: DatasetCandidate,
        selection: DatasetSelectionDecision,
        approval: DatasetAcquisitionApproval,
        tokenizer_ref: str | None = None,
    ) -> DataFactoryResult:
        """Process an explicitly approved local candidate."""

        acquisition = acquire_approved_local_candidate(
            candidate,
            selection,
            approval,
            max_bytes=self.config.max_input_bytes,
        )
        return self._process_acquisition(
            run_id=run_id,
            lineage_id=lineage_id,
            stage_name=stage_name,
            candidate=candidate,
            selection=selection,
            acquisition=acquisition,
            tokenizer_ref=tokenizer_ref,
            source_acquisition={"kind": "approved_local_candidate"},
        )

    def process_remote_acquisition(
        self,
        *,
        run_id: str,
        lineage_id: str,
        stage_name: str,
        candidate: DatasetCandidate,
        selection: DatasetSelectionDecision,
        approval: DatasetAcquisitionApproval,
        receipt: AcquisitionReceipt,
        relative_path: str | None = None,
        tokenizer_ref: str | None = None,
    ) -> DataFactoryResult:
        """Process one file from a completed, byte-verified remote receipt.

        Acquisition remains a separate operation. This method verifies its receipt
        and immutable cache bytes before entering the existing transformation
        pipeline; it does not download, authorize, or relabel provider provenance.
        """

        acquisition = validate_remote_acquisition_for_preprocessing(
            candidate=candidate,
            selection=selection,
            approval=approval,
            receipt=receipt,
            max_bytes=self.config.max_input_bytes,
            relative_path=relative_path,
        )
        return self._process_acquisition(
            run_id=run_id,
            lineage_id=lineage_id,
            stage_name=stage_name,
            candidate=candidate,
            selection=selection,
            acquisition=acquisition,
            tokenizer_ref=tokenizer_ref,
            source_acquisition={
                "kind": "remote_acquisition_receipt",
                "receipt_id": receipt.receipt_id,
                "plan_id": receipt.plan_id,
                "provider_id": receipt.provider_id,
                "dataset_id": receipt.dataset_id,
                "requested_revision": receipt.requested_revision,
                "resolved_revision": receipt.resolved_revision,
                "artifact_refs": list(receipt.artifact_refs),
                "cache_status": receipt.cache_status,
            },
        )

    def _process_acquisition(
        self,
        *,
        run_id: str,
        lineage_id: str,
        stage_name: str,
        candidate: DatasetCandidate,
        selection: DatasetSelectionDecision,
        acquisition: LocalAcquisition,
        tokenizer_ref: str | None,
        source_acquisition: dict[str, object],
    ) -> DataFactoryResult:
        loaded = _load_records(
            acquisition.source_path,
            acquisition.record_format,
            self.config.max_rows,
        )
        audit_scope = "sampled_audit" if loaded.truncated else "full_scan"

        filtered: list[dict[str, object]] = []
        filtered_rows = 0
        contamination_rows = 0
        for record in loaded.records:
            if self.record_filter is not None and not self.record_filter.keep(record):
                filtered_rows += 1
                continue
            if (
                self.contamination_checker is not None
                and self.contamination_checker.is_contaminated(record)
            ):
                contamination_rows += 1
                continue
            filtered.append(record)

        if (
            self.config.near_duplicate_threshold is not None
            and len(filtered) > self.config.near_duplicate_max_records
        ):
            raise ValueError(
                "approximate deduplication record limit exceeded; reduce max_rows or disable the bounded check"
            )
        deduplicated = deduplicate_records(
            filtered,
            near_duplicate_threshold=self.config.near_duplicate_threshold,
        )
        chunked = chunk_records(
            deduplicated.records,
            chunk_size_tokens=self.config.chunk_size_tokens,
            min_tokens=self.config.min_tokens,
            prompt_target_template=self.config.prompt_target_template,
        )
        if not chunked.records:
            raise ValueError("preprocessing produced no trainable records")

        texts = [str(record["text"]) for record in chunked.records]
        if tokenizer_ref and self.tokenizer_checker is None:
            tokenizer_evidence: dict[str, object] = {
                "status": "declared_not_verified",
                "tokenizer_ref": tokenizer_ref,
                "compatible": None,
            }
        elif self.tokenizer_checker is not None:
            if tokenizer_ref and self.tokenizer_checker.tokenizer_ref != tokenizer_ref:
                raise ValueError(
                    "tokenizer checker reference does not match requested tokenizer"
                )
            compatible, details = self.tokenizer_checker.check(texts)
            tokenizer_evidence = {
                "status": "checked",
                "tokenizer_ref": self.tokenizer_checker.tokenizer_ref,
                "checker_id": self.tokenizer_checker.checker_id,
                "compatible": bool(compatible),
                "details": details,
            }
            if not compatible:
                raise ValueError(
                    "processed dataset is incompatible with the requested tokenizer"
                )
        else:
            tokenizer_evidence = {
                "status": "not_requested",
                "tokenizer_ref": None,
                "compatible": None,
            }

        serialized = "".join(
            json.dumps(
                record,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
            for record in chunked.records
        )
        output_digest = hash_text(serialized)
        processing_policy = {
            "record_filter_id": self.record_filter.filter_id
            if self.record_filter
            else None,
            "contamination_reference_set_id": (
                self.contamination_checker.reference_set_id
                if self.contamination_checker
                else None
            ),
            "tokenizer_checker_id": self.tokenizer_checker.checker_id
            if self.tokenizer_checker
            else None,
            "tokenizer_ref": tokenizer_ref,
            "chunk_size_tokens": self.config.chunk_size_tokens,
            "min_tokens": self.config.min_tokens,
            "near_duplicate_threshold": self.config.near_duplicate_threshold,
            "near_duplicate_max_records": self.config.near_duplicate_max_records,
            "prompt_target_template": self.config.prompt_target_template,
        }
        processing_policy_hash = f"sha256:{hash_json(processing_policy)}"
        revision_slug = _slug(acquisition.revision.removeprefix("sha256:")[:20])
        content_dir = (
            self.config.artifact_root.expanduser().resolve()
            / _slug(candidate.dataset_id)
            / revision_slug
            / processing_policy_hash.removeprefix("sha256:")[:20]
            / output_digest[:20]
        )
        artifact_dir = content_dir / "runs" / _slug(run_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        processed_path = content_dir / "trainable.jsonl"
        processed_path.write_text(serialized, encoding="utf-8")
        processed_hash = f"sha256:{output_digest}"
        dataset_identity = (
            f"{candidate.dataset_id}@{acquisition.revision}"
            f"+policy:{processing_policy_hash}+content:{processed_hash}"
        )

        contamination_status = (
            "not_checked"
            if self.contamination_checker is None
            else "partially_checked"
            if loaded.truncated
            else "checked_against_named_reference_set"
        )
        filtering_status = (
            "not_checked" if self.record_filter is None else "applied"
        )
        operations = [
            "schema_validation",
            "unicode_and_whitespace_normalization",
            "exact_deduplication",
        ]
        if acquisition.record_format == "parquet":
            operations.insert(0, "parquet_record_decode")
        if self.record_filter is not None:
            operations.append(f"record_filter:{self.record_filter.filter_id}")
        if self.contamination_checker is not None:
            operations.append(
                f"contamination_check:{self.contamination_checker.reference_set_id}"
            )
        if self.config.near_duplicate_threshold is not None:
            operations.append("approximate_deduplication")
        operations.extend(
            ["wrapper_construction", "token_chunking", "stable_artifact_hashing"]
        )
        evidence_path = artifact_dir / "processing_evidence.json"
        evidence: dict[str, object] = {
            "dataset_identity": dataset_identity,
            "candidate_id": candidate.candidate_id,
            "selection_decision_id": selection.decision_id,
            "approval_refs": list(acquisition.approval_refs),
            "source_content_hash": acquisition.source_content_hash,
            "source_acquisition": dict(source_acquisition),
            "processing_policy": processing_policy,
            "processing_policy_hash": processing_policy_hash,
            "processed_content_hash": processed_hash,
            "processed_dataset_ref": str(processed_path),
            "processing_evidence_ref": str(evidence_path),
            "audit_scope": audit_scope,
            "sample_validation": {
                "status": "bounded_complete"
                if not loaded.truncated
                else "bounded_sample",
                "rows_seen": loaded.rows_seen,
                "valid_rows": len(loaded.records),
                "malformed_rows": loaded.malformed_rows,
                "truncated_at_max_rows": loaded.truncated,
            },
            "filtering": {
                "status": filtering_status,
                "filter_id": self.record_filter.filter_id
                if self.record_filter
                else None,
                "rows_removed": filtered_rows,
                "pii_integrity_claim": "not_checked"
                if self.record_filter is None
                else "filter_applied_not_proven_complete",
            },
            "contamination": {
                "status": contamination_status,
                "reference_set_id": (
                    self.contamination_checker.reference_set_id
                    if self.contamination_checker
                    else None
                ),
                "rows_removed": contamination_rows,
            },
            "deduplication": {
                "exact_status": deduplicated.exact_status,
                "exact_duplicates_removed": deduplicated.exact_duplicates_removed,
                "approximate_status": deduplicated.approximate_status,
                "approximate_duplicates_removed": deduplicated.approximate_duplicates_removed,
            },
            "preprocessing": {
                "operations": operations,
                "malformed_rows_dropped": loaded.malformed_rows,
                "total_rows_dropped": (
                    loaded.malformed_rows
                    + filtered_rows
                    + contamination_rows
                    + deduplicated.exact_duplicates_removed
                    + deduplicated.approximate_duplicates_removed
                    + chunked.dropped_below_min_tokens
                ),
            },
            "wrapper": {
                "kind": "explicit_prompt_target_template",
                "template": self.config.prompt_target_template,
            },
            "prompt_target_boundary": {
                "status": "explicit",
                "prompt_marker": "<|prompt|>",
                "target_marker": "<|target|>",
            },
            "tokenizer_compatibility": tokenizer_evidence,
            "chunking": {
                "kind": "whitespace_token_chunks",
                "chunk_size_tokens": self.config.chunk_size_tokens,
                "min_tokens": self.config.min_tokens,
                "source_records": chunked.source_records,
                "output_records": len(chunked.records),
                "dropped_below_min_tokens": chunked.dropped_below_min_tokens,
            },
        }
        write_json(evidence_path, evidence)

        manifest = build_dataset_manifest(
            run_id=run_id,
            lineage_id=lineage_id,
            stage_name=stage_name,
            candidate=candidate,
            revision=acquisition.revision,
            processed_dataset_ref=str(processed_path),
            processed_content_hash=processed_hash,
            processed_bytes=len(serialized.encode("utf-8")),
            output_rows=len(chunked.records),
            mixture_weight=float(
                selection.mixture_weights.get(candidate.candidate_id, 1.0)
            ),
            evidence=evidence,
            tokenizer_ref=tokenizer_ref,
        )
        dropped_examples = int(dict(evidence["preprocessing"])["total_rows_dropped"])
        report, contract = build_preprocessing_contracts(
            run_id=run_id,
            manifest_id=manifest.manifest_id,
            processed_dataset_ref=str(processed_path),
            operations=operations,
            dropped_examples=dropped_examples,
            min_tokens=self.config.min_tokens,
        )
        write_json(artifact_dir / "dataset_manifest.json", manifest.to_dict())
        write_json(artifact_dir / "preprocessing_report.json", report.to_dict())
        write_json(
            artifact_dir / "trainable_data_contract.json",
            contract.to_dict(),
        )
        return DataFactoryResult(
            candidate_id=candidate.candidate_id,
            dataset_identity=dataset_identity,
            processed_content_hash=processed_hash,
            artifact_dir=artifact_dir,
            manifest=manifest,
            preprocessing_report=report,
            trainable_data_contract=contract,
            processing_evidence=evidence,
        )
