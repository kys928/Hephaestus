from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hephaestus.config_loader import ConfigError, load_named_config
from hephaestus.schemas.eval_pack import EvalPack


_DEFAULT_BUNDLES: dict[str, list[str]] = {
    "promotion": ["probe_score_gate", "toxicity_gate"],
    "certification": ["probe_score_gate", "toxicity_gate"],
}

_ALLOWED_FIELDS = {
    "pack_name",
    "eval_pack_id",
    "version",
    "name",
    "description",
    "stage_name",
    "created_at",
    "frozen",
    "content_hash",
    "hash_type",
    "source_ref",
    "mutation_policy",
    "integrity_level",
    "generation_probes",
    "continuation_prompts",
    "ranking_sets",
    "regression_prompts",
    "structure_tests",
    "repetition_checks",
    "length_termination_checks",
    "human_review_bundle_refs",
    "decoding_config",
    "scoring_config",
    "deterministic_gate_config",
    "required_evidence",
    "stage_thresholds",
    "metadata",
    "warnings",
    "required_metrics",
    "supports_generation_signals",
    "regression_bundles",
    "certification_bundle",
    "minimum_evidence",
    "recheck_requirements",
    "repeatability_requirements",
    "stage_tolerances",
    "task_bundles",
    "success_semantics",
    "failure_semantics",
    "stage_applicability",
    "expected_evidence",
}


def _verified_content_hash(payload: dict[str, object], pack_name: str) -> str | None:
    claimed = str(payload.get("content_hash") or "").strip()
    if not claimed:
        return None
    hash_type = str(payload.get("hash_type") or "").strip()
    if hash_type != "sha256-canonical-json-v1":
        raise ConfigError(
            f"eval pack '{pack_name}' has content_hash but unsupported hash_type "
            f"'{hash_type or 'missing'}'"
        )
    canonical = dict(payload)
    canonical.pop("content_hash", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    observed = hashlib.sha256(encoded).hexdigest()
    if observed != claimed:
        raise ConfigError(
            f"eval pack '{pack_name}' content hash mismatch: expected {claimed}, observed {observed}"
        )
    return observed


def _as_float_map(value: object, field_name: str, pack_name: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ConfigError(f"eval pack '{pack_name}' {field_name} must be an object")
    try:
        return {str(key): float(item) for key, item in value.items()}
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"eval pack '{pack_name}' {field_name} must contain numeric values") from exc


def _as_int(value: object, field_name: str, pack_name: str, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"eval pack '{pack_name}' {field_name} must be an integer") from exc
    if parsed < minimum:
        raise ConfigError(f"eval pack '{pack_name}' {field_name} must be >= {minimum}")
    return parsed


def _as_float(value: object, field_name: str, pack_name: str, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"eval pack '{pack_name}' {field_name} must be numeric") from exc
    if parsed < minimum or parsed > maximum:
        raise ConfigError(f"eval pack '{pack_name}' {field_name} must be between {minimum} and {maximum}")
    return parsed


def load_eval_pack(pack_name: str, config_dir: Path = Path("configs")) -> dict[str, object]:
    payload = load_named_config(config_dir, "eval_packs", pack_name)
    unknown = sorted(set(payload) - _ALLOWED_FIELDS)
    if unknown:
        raise ConfigError(f"eval pack '{pack_name}' has unknown fields: {', '.join(unknown)}")

    required = ("pack_name", "required_metrics", "regression_bundles", "certification_bundle", "minimum_evidence")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ConfigError(f"eval pack '{pack_name}' missing fields: {', '.join(missing)}")

    verified_content_hash = _verified_content_hash(payload, pack_name)

    metrics = payload["required_metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise ConfigError(f"eval pack '{pack_name}' required_metrics must be a non-empty list")

    bundles = payload["regression_bundles"]
    if not isinstance(bundles, dict):
        raise ConfigError(f"eval pack '{pack_name}' regression_bundles must be an object")

    normalized_bundles: dict[str, list[str]] = {}
    for bundle_name, checks in bundles.items():
        if not isinstance(checks, list) or not checks:
            raise ConfigError(f"eval pack '{pack_name}' regression_bundles.{bundle_name} must be a non-empty list")
        normalized_bundles[str(bundle_name)] = [str(check) for check in checks]

    cert_bundle = payload["certification_bundle"]
    if not isinstance(cert_bundle, dict):
        raise ConfigError(f"eval pack '{pack_name}' certification_bundle must be an object")
    cert_required_metrics = cert_bundle.get("required_metrics", metrics)
    if not isinstance(cert_required_metrics, list) or not cert_required_metrics:
        raise ConfigError(f"eval pack '{pack_name}' certification_bundle.required_metrics must be a non-empty list")

    minimum = payload["minimum_evidence"]
    if not isinstance(minimum, dict):
        raise ConfigError(f"eval pack '{pack_name}' minimum_evidence must be an object")

    recheck = payload.get("recheck_requirements", {})
    if not isinstance(recheck, dict):
        raise ConfigError(f"eval pack '{pack_name}' recheck_requirements must be an object")

    repeatability = payload.get("repeatability_requirements", {})
    if not isinstance(repeatability, dict):
        raise ConfigError(f"eval pack '{pack_name}' repeatability_requirements must be an object")

    variance_sensitivity = str(repeatability.get("variance_sensitivity", "medium"))
    if variance_sensitivity not in {"low", "medium", "high"}:
        raise ConfigError(
            f"eval pack '{pack_name}' repeatability_requirements.variance_sensitivity must be low, medium, or high"
        )
    recheck_policy = str(repeatability.get("certification_recheck_policy", "required_if_repeatability_unmet"))
    allowed_policies = {"always", "never", "required_if_repeatability_unmet", "required_if_variance"}
    if recheck_policy not in allowed_policies:
        raise ConfigError(
            f"eval pack '{pack_name}' repeatability_requirements.certification_recheck_policy must be one of: {', '.join(sorted(allowed_policies))}"
        )

    stage_tolerances_raw = payload.get("stage_tolerances", {})
    if not isinstance(stage_tolerances_raw, dict):
        raise ConfigError(f"eval pack '{pack_name}' stage_tolerances must be an object")

    stage_tolerances = {
        str(stage_key): _as_float_map(value, f"stage_tolerances.{stage_key}", pack_name)
        for stage_key, value in stage_tolerances_raw.items()
    }

    metadata = dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {}
    metadata.update(
        {
            "task_bundles": dict(payload.get("task_bundles", {}))
            if isinstance(payload.get("task_bundles", {}), dict)
            else {},
            "success_semantics": dict(payload.get("success_semantics", {}))
            if isinstance(payload.get("success_semantics", {}), dict)
            else {},
            "failure_semantics": dict(payload.get("failure_semantics", {}))
            if isinstance(payload.get("failure_semantics", {}), dict)
            else {},
            "stage_applicability": [str(item) for item in payload.get("stage_applicability", [])],
            "expected_evidence": [str(item) for item in payload.get("expected_evidence", [])],
        }
    )

    normalized = EvalPack.normalize(
        {
            "eval_pack_id": payload.get("eval_pack_id", payload["pack_name"]),
            "version": payload.get("version", "v1"),
            "name": payload.get("name", payload["pack_name"]),
            "description": payload.get("description", ""),
            "stage_name": payload.get("stage_name"),
            "created_at": payload.get("created_at"),
            "frozen": payload.get("frozen", True),
            "content_hash": payload.get("content_hash"),
            "hash_type": payload.get("hash_type"),
            "source_ref": payload.get("source_ref"),
            "mutation_policy": payload.get("mutation_policy", "immutable_without_approval"),
            "generation_probes": payload.get("generation_probes", []),
            "continuation_prompts": payload.get("continuation_prompts", []),
            "ranking_sets": payload.get("ranking_sets", []),
            "regression_prompts": payload.get("regression_prompts", []),
            "structure_tests": payload.get("structure_tests", []),
            "repetition_checks": payload.get("repetition_checks", []),
            "length_termination_checks": payload.get("length_termination_checks", []),
            "human_review_bundle_refs": payload.get("human_review_bundle_refs", []),
            "decoding_config": payload.get("decoding_config", {}),
            "scoring_config": payload.get("scoring_config", {}),
            "deterministic_gate_config": payload.get("deterministic_gate_config", {}),
            "required_evidence": payload.get("required_evidence", minimum),
            "stage_thresholds": payload.get("stage_thresholds", {}),
            "metadata": metadata,
            "warnings": payload.get("warnings", []),
        },
        stage_name=None,
    )

    if normalized.integrity_level == "content_hash_verified" and not normalized.content_hash:
        raise ConfigError(f"eval pack '{pack_name}' cannot claim content hash verification without content_hash")

    return {
        "pack_name": str(payload["pack_name"]),
        "eval_pack_id": normalized.eval_pack_id,
        "eval_pack_version": normalized.version,
        "eval_pack_integrity_level": normalized.integrity_level,
        "frozen": normalized.frozen,
        "mutation_policy": normalized.mutation_policy,
        "content_hash": normalized.content_hash,
        "content_hash_verified": bool(verified_content_hash),
        "hash_type": normalized.hash_type,
        "source_ref": normalized.source_ref,
        "required_metrics": [str(metric) for metric in metrics],
        "description": str(payload.get("description", "")),
        "supports_generation_signals": bool(payload.get("supports_generation_signals", False)),
        "regression_bundles": {
            **_DEFAULT_BUNDLES,
            **normalized_bundles,
        },
        "certification_bundle": {
            "required_metrics": [str(metric) for metric in cert_required_metrics],
            "min_stability_confidence": _as_float(
                cert_bundle.get("min_stability_confidence", 0.9),
                "certification_bundle.min_stability_confidence",
                pack_name,
            ),
            "required_bundle": str(cert_bundle.get("required_bundle", "certification")),
        },
        "minimum_evidence": {
            "promotion_runs": _as_int(minimum.get("promotion_runs", 1), "minimum_evidence.promotion_runs", pack_name),
            "stable_runs": _as_int(minimum.get("stable_runs", 1), "minimum_evidence.stable_runs", pack_name),
            "certification_runs": _as_int(minimum.get("certification_runs", 2), "minimum_evidence.certification_runs", pack_name),
        },
        "recheck_requirements": {
            "required_for_certification": bool(recheck.get("required_for_certification", False)),
            "min_consistent_runs": _as_int(recheck.get("min_consistent_runs", 1), "recheck_requirements.min_consistent_runs", pack_name),
        },
        "repeatability_requirements": {
            "repeatability_required": bool(repeatability.get("repeatability_required", False)),
            "required_rechecks": _as_int(
                repeatability.get("required_rechecks", 0),
                "repeatability_requirements.required_rechecks",
                pack_name,
                minimum=0,
            ),
            "min_repeat_consistency": _as_float(
                repeatability.get("min_repeat_consistency", 0.0),
                "repeatability_requirements.min_repeat_consistency",
                pack_name,
            ),
            "variance_sensitivity": variance_sensitivity,
            "certification_recheck_policy": recheck_policy,
        },
        "stage_tolerances": stage_tolerances,
        "task_bundles": dict(metadata["task_bundles"]),
        "success_semantics": dict(metadata["success_semantics"]),
        "failure_semantics": dict(metadata["failure_semantics"]),
        "stage_applicability": list(metadata["stage_applicability"]),
        "expected_evidence": list(metadata["expected_evidence"]),
        "eval_pack": normalized.to_dict(),
    }
