from __future__ import annotations

from dataclasses import dataclass

from hephaestus.backends.dry_run_backend import DryRunBackend
from hephaestus.schemas.dataset_manifest import DatasetManifest
from hephaestus.schemas.dataset_profile import DatasetProfile


@dataclass(slots=True)
class DataAcquisitionAuditRole:
    backend: DryRunBackend
    name: str = "data_acquisition_audit"

    def run(self, run_id: str, lineage_id: str) -> tuple[DatasetProfile, DatasetManifest]:
        acquired = self.backend.acquire_dataset(run_id)
        profile = DatasetProfile(
            dataset_id=str(acquired["dataset_id"]),
            source_identity=str(acquired["source_identity"]),
            license=str(acquired["license"]),
            quality_score=float(acquired["quality_score"]),
            risks=[str(risk) for risk in acquired["risks"]],
        )
        manifest = DatasetManifest(
            manifest_id=f"manifest-{run_id}",
            run_id=run_id,
            lineage_id=lineage_id,
            source_ids=[str(source) for source in acquired["source_ids"]],
            total_examples=int(acquired["total_examples"]),
            artifact_ref=str(acquired["artifact_ref"]),
        )
        return profile, manifest
