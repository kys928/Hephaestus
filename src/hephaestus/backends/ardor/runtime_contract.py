from __future__ import annotations

from typing import Any

ARDOR_RUNTIME_CONTRACT_VERSION = "ardor_runtime_contract.v1"


def _as_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_ardor_runtime_contract(payload: dict[str, object], contract_ref: str | None = None) -> dict[str, object]:
    raw = dict(payload)
    missing_fields: list[str] = []
    warnings: list[str] = []

    contract_version = _as_str(raw.get("contract_version"))
    run_id = _as_str(raw.get("run_id"))
    status = _as_str(raw.get("status"))

    if not run_id:
        missing_fields.append("run_id")
    if not status:
        missing_fields.append("status")

    artifacts_raw = raw.get("artifacts", {})
    if not isinstance(artifacts_raw, dict):
        artifacts_raw = {}
        missing_fields.append("artifacts")
        warnings.append("malformed_artifacts")

    artifact_keys = (
        "metrics_ref",
        "probe_ref",
        "deterministic_ref",
        "runtime_log_ref",
        "dataset_manifest_ref",
        "training_recipe_ref",
        "tokenizer_ref",
        "architecture_config_ref",
        "eval_report_ref",
        "eval_pack_ref",
    )
    artifacts: dict[str, object] = {key: _as_str(artifacts_raw.get(key)) for key in artifact_keys}

    checkpoint_candidates: list[dict[str, object]] = []
    raw_candidates = raw.get("checkpoint_candidates", [])
    if isinstance(raw_candidates, list):
        for idx, item in enumerate(raw_candidates):
            if not isinstance(item, dict):
                warnings.append(f"malformed_checkpoint_candidate:{idx}")
                continue
            ref = _as_str(item.get("checkpoint_ref"))
            if not ref:
                warnings.append(f"malformed_checkpoint_candidate:{idx}")
                continue
            try:
                step = int(item["step"]) if item.get("step") is not None else None
            except (TypeError, ValueError):
                step = None
                warnings.append(f"malformed_checkpoint_candidate_step:{idx}")
            def _as_float(name: str) -> float | None:
                val = item.get(name)
                if val is None:
                    return None
                try:
                    return float(val)
                except (TypeError, ValueError):
                    warnings.append(f"malformed_checkpoint_candidate_{name}:{idx}")
                    return None
            score = _as_float("score")
            probe_score = _as_float("probe_score")
            content_hash = _as_str(item.get("content_hash"))
            hash_type = _as_str(item.get("hash_type"))
            integrity_level = "content_hash" if content_hash and hash_type else "ref"
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            extra_fields = {k: v for k, v in item.items() if k not in {"checkpoint_ref", "step", "score", "probe_score", "content_hash", "hash_type", "integrity_level", "metadata"}}
            if extra_fields:
                metadata = dict(metadata)
                metadata["extra_fields"] = extra_fields
            checkpoint_candidates.append(
                {
                    "checkpoint_ref": ref,
                    "step": step,
                    "score": score,
                    "probe_score": probe_score,
                    "content_hash": content_hash,
                    "hash_type": hash_type,
                    "integrity_level": integrity_level,
                    "metadata": metadata,
                }
            )
    else:
        warnings.append("malformed_checkpoint_candidates")

    if not checkpoint_candidates:
        legacy_refs = artifacts_raw.get("checkpoint_refs", raw.get("checkpoint_refs", []))
        legacy_scores = raw.get("checkpoint_scores", {})
        if isinstance(legacy_refs, list):
            if not contract_version:
                warnings.append("legacy_contract_shape")
            scores = legacy_scores if isinstance(legacy_scores, dict) else {}
            for ref in legacy_refs:
                ref_str = _as_str(ref)
                if not ref_str:
                    warnings.append("empty_checkpoint_ref")
                    continue
                score = None
                if ref_str in scores:
                    try:
                        score = float(scores[ref_str])
                    except (TypeError, ValueError):
                        warnings.append(f"malformed_checkpoint_score:{ref_str}")
                checkpoint_candidates.append(
                    {
                        "checkpoint_ref": ref_str,
                        "step": None,
                        "score": score,
                        "probe_score": score,
                        "content_hash": None,
                        "hash_type": None,
                        "integrity_level": "ref",
                        "metadata": {},
                    }
                )

    artifacts["checkpoint_refs"] = [str(item["checkpoint_ref"]) for item in checkpoint_candidates]

    unknown = {k: v for k, v in raw.items() if k not in {"contract_version", "run_id", "status", "stage_name", "lineage_id", "created_at", "artifacts", "checkpoint_candidates", "checkpoint_scores", "metrics"}}

    integrity = "partial"
    if not contract_version and run_id and status:
        integrity = "legacy"
    if missing_fields:
        integrity = "insufficient"
    elif run_id and status and artifacts.get("metrics_ref") and artifacts.get("deterministic_ref") and checkpoint_candidates:
        integrity = "complete"

    normalized: dict[str, Any] = {
        "contract_version": contract_version,
        "run_id": run_id,
        "status": status,
        "stage_name": _as_str(raw.get("stage_name")),
        "lineage_id": _as_str(raw.get("lineage_id")),
        "created_at": _as_str(raw.get("created_at")),
        "artifacts": artifacts,
        "metrics": raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {},
        "checkpoint_scores": raw.get("checkpoint_scores") if isinstance(raw.get("checkpoint_scores"), dict) else {},
        "checkpoint_candidates": checkpoint_candidates,
        "contract_integrity_level": integrity,
        "missing_fields": missing_fields,
        "warnings": warnings,
        "metadata": {"contract_ref": contract_ref, "extra_fields": unknown},
    }
    return normalized
