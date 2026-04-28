# Strict data manifest contract

Every Hephaestus run must persist a data manifest describing what entered the run and how it was prepared.

## Contract

Each persisted manifest uses a strict, stable schema with explicit keys for:

- identity (`manifest_id`, `run_id`, `lineage_id`, `stage_name`, `created_at`, `artifact_ref`)
- dataset entries (`datasets[*].dataset_id`, source/version/split/row_count/hash/license/trust/domain/notes)
- mixture and stage policy (`mixture_weights`, `sampling_policy`, `stage_data_policy_ref`)
- filtering/preprocessing (`filtering_profile`, `preprocessing_profile`, `deduplication_profile`, `contamination_checks`)
- formatting (`chunking_policy`, `wrapper_policy`, `prompt_target_boundary_policy`, `tokenizer_ref`, `tokenizer_compatibility`)
- special flags (`uses_synthetic_data`, `synthetic_data_profile`, `uses_hard_negatives`, `hard_negative_profile`, `uses_support_sets`, `support_set_profile`)
- integrity signals (`manifest_integrity_level`, `completeness_score`, `missing_fields`, `warnings`)
- metadata (`metadata`)

Unknown values are allowed, but the keys must still be present. Unknowns must be represented as `null`, empty objects, or empty lists as appropriate.

## Honesty semantics

- `completeness_score` is an honesty signal for manifest completeness, not a data quality score.
- `missing_fields` and `warnings` make unknowns and weak references explicit.
- `manifest_integrity_level` can be `complete`, `partial`, `reference_only`, or `insufficient`.
- Dry-run paths may produce `reference_only` / `partial` manifests, which are acceptable for simulation but insufficient for strong reproducibility claims.

## Scope boundaries

A strict data manifest records provenance, transformation policy, and reproducibility evidence for a run.
It does **not** itself prove data quality, model quality, or evaluation validity.
