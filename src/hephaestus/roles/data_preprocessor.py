from __future__ import annotations

from dataclasses import dataclass

from hephaestus.backends.base import ExecutionBackend
from hephaestus.data.preprocessing import preprocess_dataset
from hephaestus.schemas.preprocessing_report import PreprocessingReport
from hephaestus.schemas.trainable_data_contract import TrainableDataContract


@dataclass(slots=True)
class DataPreprocessorRole:
    backend: ExecutionBackend
    name: str = "data_preprocessor"

    def run(self, run_id: str, manifest_id: str) -> tuple[PreprocessingReport, TrainableDataContract]:
        return preprocess_dataset(backend=self.backend, run_id=run_id, manifest_id=manifest_id)
