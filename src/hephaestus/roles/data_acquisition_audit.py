from __future__ import annotations

from dataclasses import dataclass

from hephaestus.backends.base import ExecutionBackend
from hephaestus.data.acquisition import acquire_and_audit_dataset
from hephaestus.schemas.dataset_manifest import DatasetManifest
from hephaestus.schemas.dataset_profile import DatasetProfile


@dataclass(slots=True)
class DataAcquisitionAuditRole:
    backend: ExecutionBackend
    name: str = "data_acquisition_audit"

    def run(self, run_id: str, lineage_id: str, stage_name: str | None = None) -> tuple[DatasetProfile, DatasetManifest]:
        return acquire_and_audit_dataset(
            backend=self.backend,
            run_id=run_id,
            lineage_id=lineage_id,
            stage_name=stage_name,
        )
