from __future__ import annotations

from dataclasses import dataclass

from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.schemas.preprocessing_report import PreprocessingReport
from hephaestus.schemas.trainable_data_contract import TrainableDataContract


@dataclass(slots=True)
class DataPreprocessorRole:
    backend: DryRunBackend
    name: str = "data_preprocessor"

    def run(self, run_id: str, manifest_id: str) -> tuple[PreprocessingReport, TrainableDataContract]:
        processed = self.backend.preprocess(run_id)
        report = PreprocessingReport(
            report_id=f"prep-{run_id}",
            run_id=run_id,
            manifest_id=manifest_id,
            operations=[str(op) for op in processed["operations"]],
            processed_dataset_ref=str(processed["processed_dataset_ref"]),
            dropped_examples=int(processed["dropped_examples"]),
        )
        contract = TrainableDataContract(
            contract_id=f"contract-{run_id}",
            run_id=run_id,
            manifest_id=manifest_id,
            processed_dataset_ref=report.processed_dataset_ref,
            schema_version="v1",
            min_tokens=256,
        )
        return report, contract
