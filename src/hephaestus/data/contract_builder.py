from __future__ import annotations

from hephaestus.schemas.preprocessing_report import PreprocessingReport
from hephaestus.schemas.trainable_data_contract import TrainableDataContract


def build_preprocessing_contracts(
    *,
    run_id: str,
    manifest_id: str,
    processed_dataset_ref: str,
    operations: list[str],
    dropped_examples: int,
    min_tokens: int,
) -> tuple[PreprocessingReport, TrainableDataContract]:
    report = PreprocessingReport(
        report_id=f"prep-{run_id}",
        run_id=run_id,
        manifest_id=manifest_id,
        operations=operations,
        processed_dataset_ref=processed_dataset_ref,
        dropped_examples=dropped_examples,
    )
    contract = TrainableDataContract(
        contract_id=f"trainable-data-{run_id}",
        run_id=run_id,
        manifest_id=manifest_id,
        processed_dataset_ref=processed_dataset_ref,
        schema_version="trainable-data.v1",
        min_tokens=min_tokens,
    )
    return report, contract
