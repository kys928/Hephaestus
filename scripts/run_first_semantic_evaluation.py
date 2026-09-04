#!/usr/bin/env python3
"""Run the first real random-init-vs-trained semantic comparison on a mounted volume.

This driver performs generation and evaluation only.  It never trains, mutates a
model, applies a Judge action, or promotes a checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from hephaestus.control.semantic_judge import SemanticComparisonJudgeAdapter
from hephaestus.evaluation.experiment_service import ExperimentEvaluationService
from hephaestus.generation.backends import TransformersCausalLMGenerationBackend
from hephaestus.generation.service import EvaluationGenerationService
from hephaestus.schemas.experiment_contract import ExperimentProposal, TrainingRunHandle
from hephaestus.training.hf_lifecycle import directory_content_identity, validate_checkpoint_manifest

SCIENTIFIC_ROOT = Path("/workspace/hephaestus/scientific/v1")
EXPERIMENT_ID = "experiment-60bff7cb4f478f91"
LINEAGE_ID = "lineage-first-scientific"
STAGE_NAME = "smoke_test"
TRAINED_RUN_ID = "first-bounded-scientific-training-001-33866198758"
TRAINED_CHECKPOINT_HASH = "sha256:7a6be1e0cee47f29d5dd47d41bc01beed066c4de64e24ee18544ff4edcb3f4c3"
MODEL_IDENTITY = "sha256:7dbbc38ae31de5075fbf06f1362f17b6ff3b46bc822e85fc9b5f2ea05c6dad39"
TOKENIZER_IDENTITY = "sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce"
EVAL_PACK_HASH = "ee4acffa6d6ac3dadd1705931d65fc02bc4206f2fbddacf71b25af4d1cb5e3ad"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def canonical_hash(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    os.replace(partial, path)


def component_manifest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checkpoint_manifest.json"
    }


def build_baseline_checkpoint(root: Path, model_source: Path, tokenizer_source: Path) -> tuple[Path, str]:
    final = root / "baseline_random_init_checkpoint"
    if final.exists():
        valid, manifest_hash = validate_checkpoint_manifest(final)
        if not valid:
            raise RuntimeError(f"existing baseline checkpoint wrapper is invalid: {manifest_hash}")
        provenance = json.loads((final / "baseline_provenance.json").read_text(encoding="utf-8"))
        if provenance.get("model_directory_identity") != MODEL_IDENTITY or provenance.get("tokenizer_directory_identity") != TOKENIZER_IDENTITY:
            raise RuntimeError("existing baseline checkpoint wrapper provenance drifted")
        return final, manifest_hash

    partial = root / "baseline_random_init_checkpoint.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True, exist_ok=False)
    shutil.copytree(model_source, partial / "model")
    shutil.copytree(tokenizer_source, partial / "tokenizer")
    atomic_json(
        partial / "baseline_provenance.json",
        {
            "kind": "baseline_evaluation_checkpoint_wrapper",
            "created_at": now(),
            "experiment_id": EXPERIMENT_ID,
            "source_training_run_id": TRAINED_RUN_ID,
            "model_directory_identity": MODEL_IDENTITY,
            "tokenizer_directory_identity": TOKENIZER_IDENTITY,
            "trained": False,
            "purpose": "frozen semantic_behavior_v1 baseline generation",
        },
    )
    atomic_json(
        partial / "loading_instructions.json",
        {
            "backend": "transformers_causal_lm",
            "model_artifact_ref": str(final / "model"),
            "tokenizer_artifact_ref": str(final / "tokenizer"),
            "architecture": "GPT2LMHeadModel",
            "model_revision": MODEL_IDENTITY,
            "tokenizer_revision": TOKENIZER_IDENTITY,
            "trust_remote_code": False,
            "integrity_manifest_ref": str(final / "checkpoint_manifest.json"),
            "evaluation_wrapper": True,
            "trained": False,
        },
    )
    components = component_manifest(partial)
    manifest_hash = canonical_hash(components)
    atomic_json(
        partial / "checkpoint_manifest.json",
        {
            "hash_type": "sha256",
            "components": components,
            "manifest_hash": manifest_hash,
            "partial_write": False,
            "checkpoint_kind": "baseline_evaluation_wrapper",
        },
    )
    os.replace(partial, final)
    valid, observed = validate_checkpoint_manifest(final)
    if not valid or observed != manifest_hash:
        raise RuntimeError(f"baseline checkpoint wrapper verification failed: {observed}")
    return final, manifest_hash


def summary_file(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": sha_file(path), "bytes": path.stat().st_size}


def main() -> int:
    run_id = required("HEPHAESTUS_EVAL_RUN_ID")
    repo_sha = required("HEPHAESTUS_REPO_SHA")
    evaluation_root = SCIENTIFIC_ROOT / "evaluations" / run_id
    execution_root = SCIENTIFIC_ROOT / "executions" / run_id
    evaluation_root.mkdir(parents=True, exist_ok=True)
    execution_root.mkdir(parents=True, exist_ok=True)
    terminal = execution_root / "driver_result.json"

    result: dict[str, object] = {
        "result_version": "first-semantic-evaluation-result.v1",
        "created_at": now(),
        "run_id": run_id,
        "repo_sha": repo_sha,
        "experiment_id": EXPERIMENT_ID,
        "lineage_id": LINEAGE_ID,
        "stage_name": STAGE_NAME,
        "status": "running",
        "training_performed": False,
        "action_applied": False,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for real semantic generation")
        result["runtime_environment"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        }

        model_source = SCIENTIFIC_ROOT / "materialized" / "models" / MODEL_IDENTITY.removeprefix("sha256:")
        tokenizer_source = SCIENTIFIC_ROOT / "materialized" / "tokenizers" / TOKENIZER_IDENTITY.removeprefix("sha256:")
        observed_model = directory_content_identity(model_source)
        observed_tokenizer = directory_content_identity(tokenizer_source)
        if observed_model != MODEL_IDENTITY:
            raise RuntimeError(f"random-init model identity drift: {observed_model}")
        if observed_tokenizer != TOKENIZER_IDENTITY:
            raise RuntimeError(f"tokenizer identity drift: {observed_tokenizer}")

        trained_checkpoint = SCIENTIFIC_ROOT / "runs" / TRAINED_RUN_ID / "checkpoint_step_100"
        trained_valid, trained_manifest = validate_checkpoint_manifest(trained_checkpoint)
        if not trained_valid or trained_manifest != TRAINED_CHECKPOINT_HASH:
            raise RuntimeError(f"trained checkpoint integrity failure: {trained_manifest}")

        baseline_checkpoint, baseline_manifest = build_baseline_checkpoint(
            evaluation_root, model_source, tokenizer_source
        )

        backend = TransformersCausalLMGenerationBackend(device="cuda", dtype="float32", batch_size=6)
        generation = EvaluationGenerationService(
            artifact_root=evaluation_root / "generation",
            backend=backend,
        )
        plan = generation.plan()
        if plan.content_hash != EVAL_PACK_HASH:
            raise RuntimeError(f"frozen semantic pack hash drift: {plan.content_hash}")
        if len(plan.tasks) != 18:
            raise RuntimeError(f"expected 18 frozen task/seed pairs, observed {len(plan.tasks)}")

        baseline_run_id = f"random-init-baseline-{run_id}"
        baseline_run = TrainingRunHandle(
            run_id=baseline_run_id,
            experiment_id=EXPERIMENT_ID,
            backend_id="baseline_evaluation_wrapper",
            status="completed",
            checkpoint_refs=[str(baseline_checkpoint)],
            metadata={
                "generation_handoff_ref": str(baseline_checkpoint / "loading_instructions.json"),
                "model_directory_identity": MODEL_IDENTITY,
                "tokenizer_directory_identity": TOKENIZER_IDENTITY,
                "trained": False,
                "evaluation_only": True,
            },
        )
        candidate_run = TrainingRunHandle(
            run_id=TRAINED_RUN_ID,
            experiment_id=EXPERIMENT_ID,
            backend_id="transformers_causal_lm",
            status="completed",
            checkpoint_refs=[str(trained_checkpoint)],
            metadata={
                "generation_handoff_ref": str(trained_checkpoint / "loading_instructions.json"),
                "checkpoint_manifest_hash": trained_manifest,
                "trained": True,
            },
        )

        baseline_generated = generation.generate(
            baseline_run,
            generation_handoff_ref=str(baseline_checkpoint / "loading_instructions.json"),
        )
        candidate_generated = generation.generate(
            candidate_run,
            generation_handoff_ref=str(trained_checkpoint / "loading_instructions.json"),
        )

        proposal = ExperimentProposal(
            experiment_id=EXPERIMENT_ID,
            run_id=TRAINED_RUN_ID,
            lineage_id=LINEAGE_ID,
            stage_name=STAGE_NAME,
            diagnosis_report_id="source-experiment-diagnosis",
            intervention_id="source-experiment-intervention",
            primary_variable="model_parameters_after_100_optimizer_steps",
            baseline_ref=baseline_run_id,
            controlled_variables={
                "eval_pack": "semantic_behavior_v1@1.0.0",
                "generation_settings_id": plan.generation_settings_id,
                "seed_identity": plan.seed_identity,
                "random_init_model_identity": MODEL_IDENTITY,
                "tokenizer_identity": TOKENIZER_IDENTITY,
            },
            required_evidence=["semantic_behavior_v1_generation", "deterministic_regression_evidence"],
            status="evaluating",
            metadata={
                "evaluation_projection": True,
                "source_experiment_id": EXPERIMENT_ID,
                "source_training_run_id": TRAINED_RUN_ID,
                "does_not_authorize_training": True,
                "does_not_authorize_promotion": True,
            },
        )
        evaluator = ExperimentEvaluationService(pack_name="semantic_behavior_v1")
        comparison = evaluator.compare(
            proposal,
            [baseline_generated.run_handle, candidate_generated.run_handle],
        )
        judge = SemanticComparisonJudgeAdapter().decide(
            comparison,
            run_id=TRAINED_RUN_ID,
            lineage_id=LINEAGE_ID,
            candidate_checkpoint_ref=str(trained_checkpoint),
            monitor_outcome="healthy",
            recent_failure_count=0,
            has_stable_checkpoint=False,
        )

        baseline_report_path = evaluation_root / "baseline_generation_report.json"
        candidate_report_path = evaluation_root / "candidate_generation_report.json"
        comparison_path = evaluation_root / "experiment_comparison.json"
        human_review_path = evaluation_root / "human_review_bundle.json"
        judge_path = evaluation_root / "judge_exit.json"
        atomic_json(baseline_report_path, baseline_generated.report.to_dict())
        atomic_json(candidate_report_path, candidate_generated.report.to_dict())
        atomic_json(comparison_path, comparison.to_dict())
        atomic_json(human_review_path, comparison.metadata.get("human_review_bundle", {}))
        atomic_json(judge_path, judge.to_dict())

        result.update(
            {
                "status": "completed",
                "completed_at": now(),
                "verified_inputs": {
                    "random_init_model_directory_identity": observed_model,
                    "tokenizer_directory_identity": observed_tokenizer,
                    "trained_checkpoint_manifest_hash": trained_manifest,
                    "baseline_checkpoint_manifest_hash": baseline_manifest,
                    "eval_pack_id": plan.eval_pack_id,
                    "eval_pack_version": plan.eval_pack_version,
                    "eval_pack_content_hash": plan.content_hash,
                    "generation_settings_id": plan.generation_settings_id,
                    "seed_identity": plan.seed_identity,
                    "task_seed_count": len(plan.tasks),
                },
                "generation": {
                    "baseline": {
                        "run_id": baseline_run_id,
                        "status": baseline_generated.report.completion_status,
                        "sample_count": len(baseline_generated.report.samples),
                        "report_ref": baseline_generated.report.report_ref,
                    },
                    "candidate": {
                        "run_id": TRAINED_RUN_ID,
                        "status": candidate_generated.report.completion_status,
                        "sample_count": len(candidate_generated.report.samples),
                        "report_ref": candidate_generated.report.report_ref,
                    },
                },
                "comparison": comparison.to_dict(),
                "judge_exit": judge.to_dict(),
                "evidence_files": {
                    "baseline_generation_report": summary_file(baseline_report_path),
                    "candidate_generation_report": summary_file(candidate_report_path),
                    "experiment_comparison": summary_file(comparison_path),
                    "human_review_bundle": summary_file(human_review_path),
                    "judge_exit": summary_file(judge_path),
                },
                "training_performed": False,
                "action_applied": False,
            }
        )
        atomic_json(evaluation_root / "evaluation_result.json", result)
        atomic_json(terminal, result)
        print(json.dumps({
            "status": result["status"],
            "run_id": run_id,
            "outcome": comparison.primary_outcome,
            "deterministic_gate_status": comparison.deterministic_gate_status,
            "confidence": comparison.confidence,
            "recommendation": comparison.recommendation,
            "judge_action": judge.next_action.value,
            "baseline_samples": len(baseline_generated.report.samples),
            "candidate_samples": len(candidate_generated.report.samples),
            "action_applied": False,
        }, sort_keys=True))
        return 0
    except BaseException as exc:
        result.update(
            {
                "status": "failed",
                "completed_at": now(),
                "error": f"{type(exc).__name__}: {exc}",
                "training_performed": False,
                "action_applied": False,
            }
        )
        atomic_json(terminal, result)
        print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
