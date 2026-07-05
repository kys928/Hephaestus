from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from hephaestus.backends.base import ExecutionBackend
from hephaestus.data.acquisition import normalize_acquired_dataset
from hephaestus.schemas.dataset_manifest import DatasetManifest
from hephaestus.schemas.dataset_profile import DatasetProfile


@dataclass(slots=True)
class DataAcquisitionAuditRole:
    backend: ExecutionBackend
    name: str = "data_acquisition_audit"

    def run(self, run_id: str, lineage_id: str, stage_name: str | None = None) -> tuple[DatasetProfile, DatasetManifest]:
        acquired = normalize_acquired_dataset(self.backend.acquire_dataset(run_id))
        profile = DatasetProfile(
            dataset_id=str(acquired["dataset_id"]),
            source_identity=str(acquired["source_identity"]),
            license=str(acquired["license"]),
            quality_score=float(acquired["quality_score"]),
            risks=[str(risk) for risk in acquired["risks"]],
        )

        dataset_id = str(acquired.get("dataset_id", "")).strip()
        dataset_entry = {
            "dataset_id": dataset_id or None,
            "source": str(acquired.get("source_identity", "")).strip() or None,
            "version": str(acquired.get("version", "")).strip() or None,
            "split": str(acquired.get("split", "")).strip() or None,
            "row_count": int(acquired["total_examples"]) if acquired.get("total_examples") is not None else None,
            "byte_size": int(acquired["byte_size"]) if acquired.get("byte_size") is not None else None,
            "content_hash": str(acquired.get("content_hash", "")).strip() or None,
            "hash_type": str(acquired.get("hash_type", "")).strip() or None,
            "license": str(acquired.get("license", "")).strip() or None,
            "trust_level": str(acquired.get("trust_level", "")).strip() or None,
            "domain": str(acquired.get("domain", "")).strip() or None,
            "notes": str(acquired.get("notes", "")).strip() or None,
        }

        mixture_weights = dict(acquired.get("mixture_weights", {})) if isinstance(acquired.get("mixture_weights"), dict) else {}
        if dataset_id and dataset_id not in mixture_weights:
            mixture_weights[dataset_id] = 1.0

        manifest = DatasetManifest.from_dict(
            {
                "manifest_id": f"manifest-{run_id}",
                "run_id": run_id,
                "lineage_id": lineage_id,
                "stage_name": stage_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "artifact_ref": str(acquired.get("artifact_ref", "")).strip() or None,
                "datasets": [dataset_entry],
                "mixture_weights": mixture_weights,
                "sampling_policy": dict(acquired.get("sampling_policy", {})) if isinstance(acquired.get("sampling_policy"), dict) else {},
                "stage_data_policy_ref": str(acquired.get("stage_data_policy_ref", "")).strip() or None,
                "filtering_profile": dict(acquired.get("filtering_profile", {})) if isinstance(acquired.get("filtering_profile"), dict) else {},
                "preprocessing_profile": dict(acquired.get("preprocessing_profile", {})) if isinstance(acquired.get("preprocessing_profile"), dict) else {},
                "deduplication_profile": dict(acquired.get("deduplication_profile", {})) if isinstance(acquired.get("deduplication_profile"), dict) else {},
                "contamination_checks": dict(acquired.get("contamination_checks", {})) if isinstance(acquired.get("contamination_checks"), dict) else {},
                "chunking_policy": dict(acquired.get("chunking_policy", {})) if isinstance(acquired.get("chunking_policy"), dict) else {},
                "wrapper_policy": dict(acquired.get("wrapper_policy", {})) if isinstance(acquired.get("wrapper_policy"), dict) else {},
                "prompt_target_boundary_policy": dict(acquired.get("prompt_target_boundary_policy", {}))
                if isinstance(acquired.get("prompt_target_boundary_policy"), dict)
                else {},
                "tokenizer_ref": str(acquired.get("tokenizer_ref", "")).strip() or None,
                "tokenizer_compatibility": dict(acquired.get("tokenizer_compatibility", {}))
                if isinstance(acquired.get("tokenizer_compatibility"), dict)
                else {},
                "uses_synthetic_data": bool(acquired.get("uses_synthetic_data", False)),
                "synthetic_data_profile": dict(acquired.get("synthetic_data_profile", {})) if isinstance(acquired.get("synthetic_data_profile"), dict) else {},
                "uses_hard_negatives": bool(acquired.get("uses_hard_negatives", False)),
                "hard_negative_profile": dict(acquired.get("hard_negative_profile", {})) if isinstance(acquired.get("hard_negative_profile"), dict) else {},
                "uses_support_sets": bool(acquired.get("uses_support_sets", False)),
                "support_set_profile": dict(acquired.get("support_set_profile", {})) if isinstance(acquired.get("support_set_profile"), dict) else {},
                "metadata": {"source_ids": [str(source) for source in acquired.get("source_ids", [])]},
            }
        )
        return profile, manifest
