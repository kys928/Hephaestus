from __future__ import annotations

from hephaestus.schemas.dataset_manifest import DatasetManifest
from hephaestus.schemas.discovery_contract import DatasetCandidate


def build_dataset_manifest(
    *,
    run_id: str,
    lineage_id: str,
    stage_name: str,
    candidate: DatasetCandidate,
    revision: str,
    processed_dataset_ref: str,
    processed_content_hash: str,
    processed_bytes: int,
    output_rows: int,
    mixture_weight: float,
    evidence: dict[str, object],
    tokenizer_ref: str | None,
) -> DatasetManifest:
    dedup = dict(evidence["deduplication"])
    contamination = dict(evidence["contamination"])
    tokenizer = dict(evidence["tokenizer_compatibility"])
    source_ref = candidate.artifact_ref or str(evidence.get("source_content_hash") or "") or None
    return DatasetManifest.from_dict(
        {
            "manifest_id": f"manifest-{run_id}",
            "run_id": run_id,
            "lineage_id": lineage_id,
            "stage_name": stage_name,
            "artifact_ref": processed_dataset_ref,
            "datasets": [
                {
                    "dataset_id": candidate.dataset_id,
                    "source": source_ref,
                    "version": revision,
                    "split": candidate.splits[0] if candidate.splits else "train",
                    "row_count": output_rows,
                    "byte_size": processed_bytes,
                    "content_hash": processed_content_hash,
                    "hash_type": "sha256",
                    "license": candidate.license,
                    "trust_level": candidate.trust_level,
                    "domain": ",".join(candidate.domains) or None,
                    "notes": f"candidate_id={candidate.candidate_id}",
                }
            ],
            "mixture_weights": {candidate.dataset_id: mixture_weight},
            "sampling_policy": {"kind": "selected_mixture_weight"},
            "filtering_profile": dict(evidence["filtering"]),
            "preprocessing_profile": dict(evidence["preprocessing"]),
            "deduplication_profile": dedup,
            "contamination_checks": contamination,
            "chunking_policy": dict(evidence["chunking"]),
            "wrapper_policy": dict(evidence["wrapper"]),
            "prompt_target_boundary_policy": dict(evidence["prompt_target_boundary"]),
            "tokenizer_ref": tokenizer_ref,
            "tokenizer_compatibility": tokenizer,
            "uses_synthetic_data": bool(candidate.metadata.get("synthetic", False)),
            "synthetic_data_profile": {
                "declared_by_provider": bool(candidate.metadata.get("synthetic", False))
            },
            "uses_hard_negatives": bool(candidate.metadata.get("hard_negative", False)),
            "hard_negative_profile": {
                "declared_by_provider": bool(candidate.metadata.get("hard_negative", False))
            },
            "uses_support_sets": bool(candidate.metadata.get("support_set", False)),
            "support_set_profile": {
                "declared_by_provider": bool(candidate.metadata.get("support_set", False))
            },
            "metadata": {
                "candidate_id": candidate.candidate_id,
                "provider_id": candidate.provider_id,
                "audit_scope": evidence["audit_scope"],
                "processing_evidence_ref": evidence["processing_evidence_ref"],
                "approval_refs": evidence["approval_refs"],
                "source_content_hash": evidence["source_content_hash"],
                "source_acquisition": dict(evidence.get("source_acquisition", {})),
            },
        }
    )
