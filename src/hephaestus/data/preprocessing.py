"""Preprocessing boundary for the Data Preprocessor role."""

from __future__ import annotations

from hephaestus.backends.base import ExecutionBackend
from hephaestus.data.chunking import chunking_profile
from hephaestus.data.contract_builder import build_trainable_data_contract
from hephaestus.data.dedup import deduplication_profile
from hephaestus.data.normalization import normalization_profile, normalize_operations
from hephaestus.schemas.preprocessing_report import PreprocessingReport
from hephaestus.schemas.trainable_data_contract import TrainableDataContract


def preprocess_dataset(
    *,
    backend: ExecutionBackend,
    run_id: str,
    manifest_id: str,
) -> tuple[PreprocessingReport, TrainableDataContract]:
    """Run backend preprocessing and emit explicit schema-backed outputs."""
    processed = dict(backend.preprocess(run_id))
    processed_dataset_ref = str(processed.get("processed_dataset_ref", "")).strip()
    operations = normalize_operations(processed)
    metadata_operations = [
        f"normalization_profile:{normalization_profile(processed)}",
        f"deduplication_profile:{deduplication_profile(processed)}",
        f"chunking_profile:{chunking_profile(processed)}",
    ]
    report = PreprocessingReport(
        report_id=str(processed.get("report_id") or f"prep-{run_id}"),
        run_id=run_id,
        manifest_id=manifest_id,
        operations=[*operations, *metadata_operations],
        processed_dataset_ref=processed_dataset_ref,
        dropped_examples=int(processed.get("dropped_examples", 0) or 0),
    )
    contract = build_trainable_data_contract(
        run_id=run_id,
        manifest_id=manifest_id,
        processed=processed,
        processed_dataset_ref=report.processed_dataset_ref,
    )
    return report, contract
