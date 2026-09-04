#!/usr/bin/env python3
"""Retry the second controlled training after an audited pre-step admission failure.

The previous attempt reached a healthy RTX 4090/CUDA runtime but failed before
optimizer step 1 because the prepared instruction dataset contains about 32.3M
tokens while the generic loader safety ceiling was 20M. This wrapper:

1. archives that zero-step failed run evidence intact before reusing the frozen
   Planner run ID;
2. raises only the non-scientific tokenization admission ceiling to 40M so the
   already-approved immutable dataset can be tokenized;
3. leaves model initialization, tokenizer, row order, shuffle policy, seed,
   batch size, context length, optimizer, scheduler, LR, warmup, precision,
   checkpoint cadence, and 100-step budget unchanged.

The first 100 optimizer steps consume the same ordered records regardless of
this ceiling. The ceiling is therefore an execution guard, not the experiment's
primary variable.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

# Installs the explicit DatasetManifest processed-content-hash projection used by
# the reviewed v2 driver, then reuse that driver rather than duplicating it.
import run_second_controlled_training_v2  # noqa: F401
import run_second_controlled_training as base

OLD_TOKEN_CEILING = 20_000_000
NEW_TOKEN_CEILING = 40_000_000
EXPECTED_ZERO_STEP_ERROR = "ValueError: tokenized dataset exceeds max_total_tokens"


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def archive_previous_zero_step_failure() -> dict[str, object] | None:
    """Atomically preserve the failed attempt and free the canonical run path."""
    run_root = base.RUNS / base.RUN_ID
    if not run_root.exists():
        return None
    if not run_root.is_dir():
        raise RuntimeError("existing controlled run evidence is not a directory")

    runtime_result_ref = run_root / "runtime_result.json"
    scientific_result_ref = run_root / "scientific_run_result.json"
    checkpoint_record_ref = run_root / "checkpoint_record.json"
    finalized_checkpoints = [
        path
        for path in run_root.glob("checkpoint_step_*")
        if path.is_dir() and not path.name.endswith(".partial")
    ]
    partial_checkpoints = [path for path in run_root.glob("checkpoint_step_*.partial") if path.exists()]
    if checkpoint_record_ref.exists() or finalized_checkpoints or partial_checkpoints:
        raise RuntimeError("refusing to archive failed run because checkpoint evidence exists")
    if not runtime_result_ref.is_file() or not scientific_result_ref.is_file():
        raise RuntimeError("existing run lacks complete terminal failure evidence")

    runtime_result = json.loads(runtime_result_ref.read_text(encoding="utf-8"))
    scientific_result = json.loads(scientific_result_ref.read_text(encoding="utf-8"))
    if runtime_result.get("status") != "failed":
        raise RuntimeError("existing run is not a terminal failed attempt")
    if runtime_result.get("error") != EXPECTED_ZERO_STEP_ERROR:
        raise RuntimeError("existing failure is not the reviewed token-ceiling admission failure")
    if scientific_result.get("status") != "failed":
        raise RuntimeError("scientific run result does not confirm terminal failure")
    metrics_ref = run_root / "metrics_summary.json"
    if metrics_ref.exists():
        metrics = json.loads(metrics_ref.read_text(encoding="utf-8"))
        if int(metrics.get("optimizer_steps", metrics.get("steps", 0)) or 0) != 0:
            raise RuntimeError("refusing retry archive because optimizer-step evidence is non-zero")

    identity = _hash_file(runtime_result_ref).removeprefix("sha256:")[:16]
    archive_root = base.ROOT / "failed_attempts" / base.RUN_ID / f"zero-step-token-ceiling-{identity}"
    if archive_root.exists():
        raise RuntimeError(f"failed-attempt archive already exists: {archive_root}")
    archive_root.parent.mkdir(parents=True, exist_ok=True)
    os.replace(run_root, archive_root)

    components = {
        path.relative_to(archive_root).as_posix(): _hash_file(path)
        for path in sorted(archive_root.rglob("*"))
        if path.is_file() and path.name != "archive_manifest.json"
    }
    manifest = {
        "archive_version": "controlled-training-failed-attempt.v1",
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "run_id": base.RUN_ID,
        "experiment_id": base.EXPERIMENT_ID,
        "reason": "zero_step_loader_admission_failure",
        "failure": EXPECTED_ZERO_STEP_ERROR,
        "optimizer_steps": 0,
        "checkpoint_evidence_present": False,
        "source_run_root": str(run_root),
        "archive_root": str(archive_root),
        "components": components,
        "components_canonical_sha256": "sha256:" + hashlib.sha256(
            json.dumps(components, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "retry_policy": {
            "canonical_run_id_reused": True,
            "scientific_primary_variable_changed": False,
            "execution_guard_adjustment": {
                "field": "max_total_tokens",
                "from": OLD_TOKEN_CEILING,
                "to": NEW_TOKEN_CEILING,
                "reason": "admit immutable prepared dataset for tokenization before fixed 100-step training",
            },
        },
    }
    _write_json(archive_root / "archive_manifest.json", manifest)
    return manifest


_original_build_proposal = base.build_proposal
_original_result_for = base.result_for


def _build_proposal_with_admission_ceiling(*args, **kwargs):
    proposal = _original_build_proposal(*args, **kwargs)
    constraints = dict(proposal.training_constraints)
    observed = int(constraints.get("max_total_tokens", 0))
    if observed != OLD_TOKEN_CEILING:
        raise RuntimeError(f"unexpected original max_total_tokens: {observed}")
    constraints["max_total_tokens"] = NEW_TOKEN_CEILING
    metadata = {
        **proposal.metadata,
        "execution_guard_adjustment": {
            "field": "max_total_tokens",
            "from": OLD_TOKEN_CEILING,
            "to": NEW_TOKEN_CEILING,
            "scientific_variable": False,
            "affects_fixed_first_100_step_record_order": False,
            "reason": "prepared immutable dataset has >20M total tokens and loader tokenizes all records before training",
        },
    }
    return replace(proposal, training_constraints=constraints, metadata=metadata)


def _result_with_admission_evidence(proposal, handle, sink):
    result = _original_result_for(proposal, handle, sink)
    recipe = result.get("controlled_recipe")
    if isinstance(recipe, dict):
        recipe["max_total_tokens"] = NEW_TOKEN_CEILING
    result["execution_guard_adjustment"] = {
        "field": "max_total_tokens",
        "from": OLD_TOKEN_CEILING,
        "to": NEW_TOKEN_CEILING,
        "scientific_variable": False,
        "optimizer_step_budget": 100,
        "row_order_changed": False,
    }
    return result


base.build_proposal = _build_proposal_with_admission_ceiling
base.result_for = _result_with_admission_evidence


if __name__ == "__main__":
    archive_previous_zero_step_failure()
    raise SystemExit(base.main())
