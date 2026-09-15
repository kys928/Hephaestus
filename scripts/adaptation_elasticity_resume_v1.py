#!/usr/bin/env python3
"""Validate and plan resumable Adaptation Elasticity V1 role evidence.

The module is intentionally independent of torch/transformers so completion
state can be tested on CPU.  The GPU runner supplies the frozen scorer and
summary callback when it validates persisted post-samples.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PRIOR_MANIFEST_VERSION = "adaptation-elasticity-lora.v1"
CURRENT_MANIFEST_VERSION = "adaptation-elasticity-lora.v2"
SUPPORTED_MANIFEST_VERSIONS = {PRIOR_MANIFEST_VERSION, CURRENT_MANIFEST_VERSION}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class RoleEvidenceLocation:
    evidence_id: str
    kind: str
    role_results_dir: Path
    adapters_dir: Path
    post_samples_dir: Path

    def result_path(self, role: str, dose: int) -> Path:
        return self.role_results_dir / f"{role}-dose-{dose}.json"

    def adapter_path(self, role: str, dose: int) -> Path:
        return self.adapters_dir / role / f"dose-epoch-{dose}"

    def sample_path(self, role: str, dose: int) -> Path:
        return self.post_samples_dir / role / f"dose-epoch-{dose}.jsonl"

    def to_dict(self) -> dict[str, str]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "role_results_dir": str(self.role_results_dir),
            "adapters_dir": str(self.adapters_dir),
            "post_samples_dir": str(self.post_samples_dir),
        }


def standard_location(proof_root: Path, model_slug: str) -> RoleEvidenceLocation:
    return RoleEvidenceLocation(
        evidence_id=f"standard:{model_slug}",
        kind="standard",
        role_results_dir=proof_root / "role_results" / model_slug,
        adapters_dir=proof_root / "adapters" / model_slug,
        post_samples_dir=proof_root / "post_samples" / model_slug,
    )


def reconstruction_location(
    proof_root: Path,
    model_slug: str,
    role: str,
    attempt: int,
) -> RoleEvidenceLocation:
    root = proof_root / "reconstructions" / model_slug / role / f"attempt-{attempt}"
    return RoleEvidenceLocation(
        evidence_id=f"reconstruction:{model_slug}:{role}:attempt-{attempt}",
        kind="reconstruction",
        role_results_dir=root / "role_results",
        adapters_dir=root / "adapters",
        post_samples_dir=root / "post_samples",
    )


def discover_reconstructions(
    proof_root: Path,
    model_slug: str,
    role: str,
) -> list[RoleEvidenceLocation]:
    parent = proof_root / "reconstructions" / model_slug / role
    rows: list[tuple[int, RoleEvidenceLocation]] = []
    if not parent.is_dir():
        return []
    for path in parent.glob("attempt-*"):
        try:
            attempt = int(path.name.removeprefix("attempt-"))
        except ValueError:
            continue
        rows.append((attempt, reconstruction_location(proof_root, model_slug, role, attempt)))
    return [location for _, location in sorted(rows, key=lambda item: item[0])]


def _read_json(path: Path, errors: list[str], label: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{label} is not valid JSON: {type(exc).__name__}: {exc}")
        return None
    if not isinstance(payload, dict):
        errors.append(f"{label} must be a JSON object")
        return None
    return payload


def _same(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _require_equal(errors: list[str], label: str, observed: object, expected: object) -> None:
    if not _same(observed, expected):
        errors.append(f"{label} mismatch: observed={observed!r} expected={expected!r}")


def _validate_configured_targets(
    configured_targets: object,
    expected_suffixes: list[str],
    errors: list[str],
) -> None:
    """Validate PEFT's lossless path compression against the suffix allowlist.

    PEFT may serialize resolved targets as architecture-qualified patterns such
    as ``self_attn.q_proj`` even when the governed selector is ``q_proj``.  The
    terminal module name is the frozen scientific variable; the manifest keeps
    the exact resolved-module count and hash.  Visual/multimodal paths remain
    forbidden exactly as they are in the training-time selector.
    """

    if not isinstance(configured_targets, list):
        errors.append("adapter_config target_modules must be a list")
        return
    blocked = ("vision", "visual", "image", "pixel", "projector", "multi_modal", "multimodal")
    paths: list[str] = []
    suffixes: list[str] = []
    for index, target in enumerate(configured_targets):
        if not isinstance(target, str) or not target.strip():
            errors.append(f"adapter_config target_modules[{index}] must be a non-empty string")
            continue
        path = target.strip()
        paths.append(path)
        if any(token in path.casefold() for token in blocked):
            errors.append(f"adapter_config target module enters a forbidden visual path: {path!r}")
        suffixes.append(path.rsplit(".", 1)[-1])
    if len(paths) != len(set(paths)):
        errors.append("adapter_config target_modules contains duplicate paths")
    _require_equal(
        errors,
        "adapter_config target module suffixes",
        sorted(set(suffixes)),
        sorted(set(expected_suffixes)),
    )


def _validate_adapter(
    adapter_dir: Path,
    manifest: dict[str, Any],
    adapter_config: dict[str, Any],
    *,
    model_id: str,
    revision: str,
    role: str,
    dose: int,
    protocol: dict[str, Any],
    identity: dict[str, str],
    errors: list[str],
) -> dict[str, str]:
    version = manifest.get("manifest_version")
    if version not in SUPPORTED_MANIFEST_VERSIONS:
        errors.append(f"unsupported adapter manifest version: {version!r}")
    _require_equal(errors, "adapter model_id", manifest.get("model_id"), model_id)
    _require_equal(errors, "adapter role", manifest.get("role"), role)
    _require_equal(errors, "adapter dose", manifest.get("dose_epoch"), dose)
    if not isinstance(manifest.get("trainable_parameters"), int) or manifest["trainable_parameters"] <= 0:
        errors.append("adapter trainable_parameters must be a positive integer")
    if not isinstance(manifest.get("target_module_count"), int) or manifest["target_module_count"] <= 0:
        errors.append("adapter target_module_count must be a positive integer")
    if not isinstance(manifest.get("target_modules_sha256"), str):
        errors.append("adapter target_modules_sha256 is missing")

    if version == CURRENT_MANIFEST_VERSION:
        for key, expected in identity.items():
            _require_equal(errors, f"adapter {key}", manifest.get(key), expected)
        _require_equal(errors, "adapter revision", manifest.get("revision"), revision)

    components = manifest.get("components")
    component_hashes: dict[str, str] = {}
    byte_size = 0
    if not isinstance(components, dict):
        errors.append("adapter components must be an object")
        components = {}
    required_components = {"adapter_config.json", "adapter_model.safetensors"}
    if not required_components.issubset(components):
        errors.append(f"adapter components lack {sorted(required_components - set(components))}")
    for name, expected_hash in sorted(components.items()):
        component = Path(str(name))
        if component.is_absolute() or ".." in component.parts or component.as_posix() != str(name):
            errors.append(f"unsafe adapter component path: {name!r}")
            continue
        path = adapter_dir / component
        if not path.is_file():
            errors.append(f"adapter component is missing: {name}")
            continue
        observed_hash = "sha256:" + sha256_file(path)
        component_hashes[str(name)] = observed_hash
        byte_size += path.stat().st_size
        _require_equal(errors, f"adapter component hash {name}", observed_hash, expected_hash)
    _require_equal(errors, "adapter byte size", manifest.get("adapter_bytes"), byte_size)
    expected_manifest_hash = hashlib.sha256(
        json.dumps(components, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _require_equal(errors, "adapter component-manifest hash", manifest.get("manifest_sha256"), expected_manifest_hash)

    train = protocol["training"]
    config_expectations = {
        "peft_type": "LORA",
        "peft_version": train["peft_version"],
        "r": train["rank"],
        "lora_alpha": train["alpha"],
        "lora_dropout": train["dropout"],
        "bias": train["bias"],
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    for key, expected in config_expectations.items():
        _require_equal(errors, f"adapter_config {key}", adapter_config.get(key), expected)
    _validate_configured_targets(
        adapter_config.get("target_modules"),
        list(train["target_module_suffixes"]),
        errors,
    )
    base_path = str(adapter_config.get("base_model_name_or_path") or "")
    if revision not in base_path:
        errors.append("adapter_config does not identify the immutable base revision")
    return component_hashes


def inspect_dose(
    location: RoleEvidenceLocation,
    *,
    model_slug: str,
    candidate: dict[str, Any],
    role: str,
    dose: int,
    protocol: dict[str, Any],
    topology: dict[str, Any],
    baseline_summary: dict[str, Any],
    identity: dict[str, str],
    score_response: Callable[[dict[str, Any], dict[str, Any], str], dict[str, Any]],
    summarize_role: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    del model_slug  # included in the call contract to make accidental cross-model use explicit
    result_path = location.result_path(role, dose)
    adapter_dir = location.adapter_path(role, dose)
    manifest_path = adapter_dir / "hephaestus_adapter_manifest.json"
    adapter_config_path = adapter_dir / "adapter_config.json"
    sample_path = location.sample_path(role, dose)
    paths = [result_path, manifest_path, adapter_config_path, sample_path]
    any_existing = any(path.exists() for path in paths) or adapter_dir.exists()
    if not any_existing:
        return {
            "dose_epoch": dose,
            "state": "missing",
            "errors": [],
            "record": None,
            "samples": [],
            "evidence_hashes": {},
            "location": location.to_dict(),
        }

    errors: list[str] = []
    for path, label in (
        (result_path, "role result"),
        (manifest_path, "adapter manifest"),
        (adapter_config_path, "adapter config"),
        (sample_path, "post-sample JSONL"),
    ):
        if not path.is_file():
            errors.append(f"{label} is missing: {path}")

    record = _read_json(result_path, errors, "role result") if result_path.is_file() else None
    manifest = _read_json(manifest_path, errors, "adapter manifest") if manifest_path.is_file() else None
    adapter_config = _read_json(adapter_config_path, errors, "adapter config") if adapter_config_path.is_file() else None

    component_hashes: dict[str, str] = {}
    if manifest is not None and adapter_config is not None:
        component_hashes = _validate_adapter(
            adapter_dir,
            manifest,
            adapter_config,
            model_id=candidate["model_id"],
            revision=candidate["revision"],
            role=role,
            dose=dose,
            protocol=protocol,
            identity=identity,
            errors=errors,
        )

    samples: list[dict[str, Any]] = []
    if sample_path.is_file():
        try:
            for line_number, line in enumerate(sample_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"line {line_number} is not an object")
                samples.append(row)
        except Exception as exc:
            errors.append(f"post-sample JSONL is invalid: {type(exc).__name__}: {exc}")

    role_cases = [case for case in topology["cases"] if case["role"] == role]
    case_map = {case["case_id"]: case for case in role_cases}
    expected_grid = {
        (case["case_id"], int(seed))
        for seed in topology["generation"]["seeds"]
        for case in role_cases
    }
    observed_grid: set[tuple[str, int]] = set()
    for index, row in enumerate(samples):
        _require_equal(errors, f"sample[{index}] model_id", row.get("model_id"), candidate["model_id"])
        _require_equal(errors, f"sample[{index}] revision", row.get("revision"), candidate["revision"])
        _require_equal(errors, f"sample[{index}] role", row.get("role"), role)
        case_id = row.get("case_id")
        seed = row.get("seed")
        if case_id not in case_map or not isinstance(seed, int):
            errors.append(f"sample[{index}] has an unexpected case/seed identity")
            continue
        key = (str(case_id), seed)
        if key in observed_grid:
            errors.append(f"duplicate post-sample identity: {key}")
        observed_grid.add(key)
        case = case_map[str(case_id)]
        _require_equal(errors, f"sample[{index}] condition", row.get("condition"), case["condition"])
        _require_equal(errors, f"sample[{index}] pair_id", row.get("pair_id"), case.get("pair_id"))
        output = row.get("output")
        if not isinstance(output, str):
            errors.append(f"sample[{index}] output must be a string")
            continue
        recomputed_score = score_response(topology, case, output)
        _require_equal(errors, f"sample[{index}] score", row.get("score"), recomputed_score)
    if observed_grid != expected_grid:
        errors.append(
            f"post-sample grid mismatch: observed={len(observed_grid)} expected={len(expected_grid)}"
        )

    if record is not None:
        _require_equal(errors, "role-result dose", record.get("dose_epoch"), dose)
        _require_equal(
            errors,
            "role-result baseline quality",
            record.get("baseline_quality_100"),
            baseline_summary["quality_100"],
        )
        if manifest is not None:
            _require_equal(errors, "embedded adapter manifest", record.get("adapter_manifest"), manifest)
            _require_equal(errors, "role-result adapter bytes", record.get("adapter_bytes"), manifest.get("adapter_bytes"))
            _require_equal(
                errors,
                "role-result trainable parameters",
                record.get("trainable_parameters"),
                manifest.get("trainable_parameters"),
            )
        if samples and observed_grid == expected_grid:
            recomputed_summary = summarize_role(samples)
            _require_equal(errors, "adapted role summary", record.get("adapted_role_summary"), recomputed_summary)
            adapted_quality = recomputed_summary["quality_100"]
            baseline_quality = baseline_summary["quality_100"]
            _require_equal(errors, "adapted quality", record.get("adapted_quality_100"), adapted_quality)
            _require_equal(
                errors,
                "quality delta",
                record.get("delta_quality_points"),
                adapted_quality - baseline_quality,
            )
            for output_key, summary_key in (
                ("schema_compliance_delta", "schema_compliance"),
                ("evidence_grounding_delta", "evidence_grounding"),
                ("confidence_calibration_delta", "confidence_calibration"),
                ("hallucination_rate_delta", "hallucination_rate"),
            ):
                _require_equal(
                    errors,
                    output_key,
                    record.get(output_key),
                    recomputed_summary[summary_key] - baseline_summary[summary_key],
                )
        if record.get("evidence_identity") is not None:
            _require_equal(errors, "role-result evidence identity", record.get("evidence_identity"), identity)
        for positive in (
            "training_seconds_cumulative",
            "training_tokens_cumulative",
            "trainable_parameters",
            "adapter_bytes",
            "training_peak_vram_bytes",
        ):
            value = record.get(positive)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                errors.append(f"role-result {positive} must be positive")

    evidence_hashes: dict[str, Any] = {}
    for key, path in (
        ("role_result_sha256", result_path),
        ("adapter_manifest_sha256", manifest_path),
        ("post_samples_sha256", sample_path),
    ):
        if path.is_file():
            evidence_hashes[key] = "sha256:" + sha256_file(path)
    evidence_hashes["adapter_components"] = component_hashes
    return {
        "dose_epoch": dose,
        "state": "complete" if not errors else "invalid",
        "errors": errors,
        "record": record,
        "samples": samples,
        "evidence_hashes": evidence_hashes,
        "location": location.to_dict(),
    }


def inspect_role(
    location: RoleEvidenceLocation,
    *,
    model_slug: str,
    candidate: dict[str, Any],
    role: str,
    protocol: dict[str, Any],
    topology: dict[str, Any],
    baseline_summary: dict[str, Any],
    identity: dict[str, str],
    score_response: Callable[[dict[str, Any], dict[str, Any], str], dict[str, Any]],
    summarize_role: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    doses = [
        inspect_dose(
            location,
            model_slug=model_slug,
            candidate=candidate,
            role=role,
            dose=dose,
            protocol=protocol,
            topology=topology,
            baseline_summary=baseline_summary,
            identity=identity,
            score_response=score_response,
            summarize_role=summarize_role,
        )
        for dose in protocol["training"]["dose_checkpoints"]
    ]
    states = [dose["state"] for dose in doses]
    errors: list[str] = []
    if all(state == "complete" for state in states):
        first, second = (dose["record"] for dose in doses)
        assert isinstance(first, dict) and isinstance(second, dict)
        _require_equal(errors, "cross-dose trainable parameters", first.get("trainable_parameters"), second.get("trainable_parameters"))
        _require_equal(
            errors,
            "cross-dose target module count",
            first.get("adapter_manifest", {}).get("target_module_count"),
            second.get("adapter_manifest", {}).get("target_module_count"),
        )
        _require_equal(
            errors,
            "cross-dose target module hash",
            first.get("adapter_manifest", {}).get("target_modules_sha256"),
            second.get("adapter_manifest", {}).get("target_modules_sha256"),
        )
        expected_second_tokens = int(first["training_tokens_cumulative"]) * 2
        _require_equal(errors, "dose-2 cumulative training tokens", second.get("training_tokens_cumulative"), expected_second_tokens)
        if float(second["training_seconds_cumulative"]) < float(first["training_seconds_cumulative"]):
            errors.append("dose-2 cumulative training time regressed")
        state = "complete" if not errors else "invalid"
    elif all(item == "missing" for item in states):
        state = "missing"
    elif any(item == "invalid" for item in states):
        state = "invalid"
    else:
        state = "partial"
    return {
        "model_id": candidate["model_id"],
        "revision": candidate["revision"],
        "role": role,
        "state": state,
        "valid_doses": [dose["dose_epoch"] for dose in doses if dose["state"] == "complete"],
        "doses": doses,
        "cross_dose_errors": errors,
        "location": location.to_dict(),
    }


def choose_role_evidence(
    proof_root: Path,
    *,
    model_slug: str,
    candidate: dict[str, Any],
    role: str,
    protocol: dict[str, Any],
    topology: dict[str, Any],
    baseline_summary: dict[str, Any],
    identity: dict[str, str],
    score_response: Callable[[dict[str, Any], dict[str, Any], str], dict[str, Any]],
    summarize_role: Callable[[list[dict[str, Any]]], dict[str, Any]],
) -> dict[str, Any]:
    kwargs = {
        "model_slug": model_slug,
        "candidate": candidate,
        "role": role,
        "protocol": protocol,
        "topology": topology,
        "baseline_summary": baseline_summary,
        "identity": identity,
        "score_response": score_response,
        "summarize_role": summarize_role,
    }
    standard = inspect_role(standard_location(proof_root, model_slug), **kwargs)
    reconstructions = [
        inspect_role(location, **kwargs)
        for location in discover_reconstructions(proof_root, model_slug, role)
    ]
    if standard["state"] == "complete":
        return {
            "state": "complete",
            "selected": standard,
            "standard": standard,
            "reconstructions": reconstructions,
        }
    complete = [row for row in reconstructions if row["state"] == "complete"]
    if complete:
        return {
            "state": "complete",
            "selected": complete[0],
            "standard": standard,
            "reconstructions": reconstructions,
        }
    return {
        "state": standard["state"],
        "selected": None,
        "standard": standard,
        "reconstructions": reconstructions,
    }


def role_action(selection: dict[str, Any]) -> dict[str, str]:
    if selection["state"] == "complete":
        return {"action": "reuse", "reason": "complete_validated_role_evidence"}
    if selection["state"] == "missing":
        return {"action": "run_from_base", "reason": "role_evidence_missing"}
    return {
        "action": "reconstruct_from_base",
        "reason": "partial_or_invalid_role_without_optimizer_scheduler_state",
    }


def output_location_for_action(
    proof_root: Path,
    *,
    model_slug: str,
    role: str,
    attempt: int,
    action: str,
) -> RoleEvidenceLocation:
    if action == "run_from_base":
        location = standard_location(proof_root, model_slug)
        paths = [
            location.result_path(role, dose)
            for dose in (1, 2)
        ] + [
            location.adapter_path(role, dose)
            for dose in (1, 2)
        ] + [
            location.sample_path(role, dose)
            for dose in (1, 2)
        ]
        if any(path.exists() for path in paths):
            raise RuntimeError("refusing to overwrite existing standard role evidence")
        return location
    if action == "reconstruct_from_base":
        location = reconstruction_location(proof_root, model_slug, role, attempt)
        root = location.role_results_dir.parent
        if root.exists():
            raise RuntimeError(f"refusing to overwrite an existing reconstruction: {root}")
        return location
    raise ValueError(f"action does not produce evidence: {action}")
