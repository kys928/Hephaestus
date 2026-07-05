"""Build schema-backed dataset manifests from acquisition metadata."""

from __future__ import annotations

from hephaestus.data.registry import as_dict, as_ref, as_str_list, utc_now_iso
from hephaestus.schemas.dataset_manifest import DatasetManifest


def build_dataset_entry(acquired: dict[str, object]) -> dict[str, object]:
    dataset_id = as_ref(acquired.get("dataset_id") or acquired.get("id") or acquired.get("name"))
    return {
        "dataset_id": dataset_id,
        "source": as_ref(acquired.get("source_identity") or acquired.get("source")),
        "version": as_ref(acquired.get("version")),
        "split": as_ref(acquired.get("split")),
        "row_count": acquired.get("total_examples") or acquired.get("row_count"),
        "byte_size": acquired.get("byte_size"),
        "content_hash": as_ref(acquired.get("content_hash") or acquired.get("hash")),
        "hash_type": as_ref(acquired.get("hash_type")),
        "license": as_ref(acquired.get("license")),
        "trust_level": as_ref(acquired.get("trust_level")),
        "domain": as_ref(acquired.get("domain")),
        "notes": as_ref(acquired.get("notes")),
    }


def build_dataset_manifest(
    *,
    run_id: str,
    lineage_id: str,
    acquired: dict[str, object],
    stage_name: str | None = None,
) -> DatasetManifest:
    dataset_entry = build_dataset_entry(acquired)
    dataset_id = as_ref(dataset_entry.get("dataset_id"))
    mixture_weights = as_dict(acquired.get("mixture_weights"))
    if dataset_id and dataset_id not in mixture_weights:
        mixture_weights[dataset_id] = 1.0

    return DatasetManifest.from_dict(
        {
            "manifest_id": as_ref(acquired.get("manifest_id")) or f"manifest-{run_id}",
            "run_id": run_id,
            "lineage_id": lineage_id,
            "stage_name": stage_name,
            "created_at": as_ref(acquired.get("created_at")) or utc_now_iso(),
            "artifact_ref": as_ref(acquired.get("artifact_ref")),
            "datasets": [dataset_entry],
            "mixture_weights": mixture_weights,
            "sampling_policy": as_dict(acquired.get("sampling_policy")),
            "stage_data_policy_ref": as_ref(acquired.get("stage_data_policy_ref") or acquired.get("data_policy_ref")),
            "filtering_profile": as_dict(acquired.get("filtering_profile")),
            "preprocessing_profile": as_dict(acquired.get("preprocessing_profile")),
            "deduplication_profile": as_dict(acquired.get("deduplication_profile")),
            "contamination_checks": as_dict(acquired.get("contamination_checks")),
            "chunking_policy": as_dict(acquired.get("chunking_policy")),
            "wrapper_policy": as_dict(acquired.get("wrapper_policy")),
            "prompt_target_boundary_policy": as_dict(acquired.get("prompt_target_boundary_policy")),
            "tokenizer_ref": as_ref(acquired.get("tokenizer_ref")),
            "tokenizer_compatibility": as_dict(acquired.get("tokenizer_compatibility")),
            "uses_synthetic_data": bool(acquired.get("uses_synthetic_data", False)),
            "synthetic_data_profile": as_dict(acquired.get("synthetic_data_profile")),
            "uses_hard_negatives": bool(acquired.get("uses_hard_negatives", False)),
            "hard_negative_profile": as_dict(acquired.get("hard_negative_profile")),
            "uses_support_sets": bool(acquired.get("uses_support_sets", False)),
            "support_set_profile": as_dict(acquired.get("support_set_profile")),
            "metadata": {"source_ids": as_str_list(acquired.get("source_ids"))},
        }
    )
