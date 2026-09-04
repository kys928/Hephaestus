#!/usr/bin/env python3
"""Evaluate the controlled dataset intervention against the previous trained baseline.

This is the frozen one-variable comparison for experiment-d0e911d6bd1fb7ae:
- baseline: first 100-step Wikitext-trained checkpoint;
- candidate: planned-run-b8e558e54effac85, trained from the same random init;
- only primary variable: dataset_mixture;
- frozen semantic_behavior_v1 generation/evaluation settings;
- Judge decision is recorded but never applied by this driver.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from hephaestus.control.semantic_judge import SemanticComparisonJudgeAdapter
from hephaestus.evaluation.experiment_service import ExperimentEvaluationService
from hephaestus.generation.backends import TransformersCausalLMGenerationBackend
from hephaestus.generation.service import EvaluationGenerationService
from hephaestus.schemas.experiment_contract import ExperimentProposal, TrainingRunHandle
from hephaestus.training.hf_lifecycle import directory_content_identity, validate_checkpoint_manifest

SCIENTIFIC_ROOT = Path("/workspace/hephaestus/scientific/v1")
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_FILE = REPO_ROOT / "docs/evidence/first-autonomous-dataset-discovery-001-33876486327/experiment.json"
EXPERIMENT_FILE_HASH = "sha256:530032d5bf2ab1b443592fbd8d3bb61616ca11e293a6ca30d327cc020ff73ffc"
EXPERIMENT_ID = "experiment-d0e911d6bd1fb7ae"
LINEAGE_ID = "lineage-first-scientific"
STAGE_NAME = "smoke_test"
PRIMARY_VARIABLE = "dataset_mixture"
BASELINE_RUN_ID = "first-bounded-scientific-training-001-33866198758"
BASELINE_CHECKPOINT_HASH = "sha256:7a6be1e0cee47f29d5dd47d41bc01beed066c4de64e24ee18544ff4edcb3f4c3"
CANDIDATE_RUN_ID = "planned-run-b8e558e54effac85"
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


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(partial, path)


def summary_file(path: Path) -> dict[str, object]:
    return {"path": str(path), "sha256": sha_file(path), "bytes": path.stat().st_size}


def load_frozen_proposal() -> ExperimentProposal:
    if sha_file(EXPERIMENT_FILE) != EXPERIMENT_FILE_HASH:
        raise RuntimeError("frozen controlled ExperimentProposal evidence drifted")
    payload = json.loads(EXPERIMENT_FILE.read_text(encoding="utf-8"))
    raw = payload.get("experiment_proposal")
    if not isinstance(raw, dict):
        raise RuntimeError("frozen controlled ExperimentProposal is missing")
    proposal = ExperimentProposal.from_dict(raw)
    if proposal.experiment_id != EXPERIMENT_ID or proposal.run_id != CANDIDATE_RUN_ID:
        raise RuntimeError("controlled ExperimentProposal identity drifted")
    if proposal.primary_variable != PRIMARY_VARIABLE:
        raise RuntimeError("controlled ExperimentProposal primary variable drifted")
    if proposal.baseline_ref != f"run://{BASELINE_RUN_ID}":
        raise RuntimeError("controlled ExperimentProposal baseline drifted")
    controlled = dict(proposal.controlled_variables)
    if controlled.get("architecture") != MODEL_IDENTITY or controlled.get("tokenizer") != TOKENIZER_IDENTITY:
        raise RuntimeError("architecture/tokenizer controlled variables drifted")
    if controlled.get("evaluation_reference") != f"sha256:{EVAL_PACK_HASH}":
        raise RuntimeError("frozen evaluation reference drifted")
    return replace(proposal, status="evaluating")


def main() -> int:
    eval_run_id = required("HEPHAESTUS_EVAL_RUN_ID")
    repo_sha = required("HEPHAESTUS_REPO_SHA")
    candidate_expected = required("HEPHAESTUS_CANDIDATE_CHECKPOINT_HASH")
    evaluation_root = SCIENTIFIC_ROOT / "evaluations" / eval_run_id
    execution_root = SCIENTIFIC_ROOT / "executions" / eval_run_id
    evaluation_root.mkdir(parents=True, exist_ok=True)
    execution_root.mkdir(parents=True, exist_ok=True)
    terminal = execution_root / "driver_result.json"

    result: dict[str, object] = {
        "result_version": "second-controlled-semantic-evaluation-result.v1",
        "created_at": now(),
        "run_id": eval_run_id,
        "repo_sha": repo_sha,
        "experiment_id": EXPERIMENT_ID,
        "lineage_id": LINEAGE_ID,
        "stage_name": STAGE_NAME,
        "primary_variable": PRIMARY_VARIABLE,
        "baseline_training_run_id": BASELINE_RUN_ID,
        "candidate_training_run_id": CANDIDATE_RUN_ID,
        "status": "running",
        "training_performed": False,
        "action_applied": False,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for real controlled semantic generation")
        result["runtime_environment"] = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        }

        model_source = SCIENTIFIC_ROOT / "materialized/models" / MODEL_IDENTITY.removeprefix("sha256:")
        tokenizer_source = SCIENTIFIC_ROOT / "materialized/tokenizers" / TOKENIZER_IDENTITY.removeprefix("sha256:")
        observed_model = directory_content_identity(model_source)
        observed_tokenizer = directory_content_identity(tokenizer_source)
        if observed_model != MODEL_IDENTITY:
            raise RuntimeError(f"random-init model identity drift: {observed_model}")
        if observed_tokenizer != TOKENIZER_IDENTITY:
            raise RuntimeError(f"tokenizer identity drift: {observed_tokenizer}")

        baseline_checkpoint = SCIENTIFIC_ROOT / "runs" / BASELINE_RUN_ID / "checkpoint_step_100"
        candidate_checkpoint = SCIENTIFIC_ROOT / "runs" / CANDIDATE_RUN_ID / "checkpoint_step_100"
        baseline_valid, baseline_manifest = validate_checkpoint_manifest(baseline_checkpoint)
        candidate_valid, candidate_manifest = validate_checkpoint_manifest(candidate_checkpoint)
        if not baseline_valid or baseline_manifest != BASELINE_CHECKPOINT_HASH:
            raise RuntimeError(f"previous trained baseline checkpoint integrity failure: {baseline_manifest}")
        if not candidate_valid or candidate_manifest != candidate_expected:
            raise RuntimeError(f"new controlled candidate checkpoint integrity failure: {candidate_manifest}")
        if candidate_manifest == baseline_manifest:
            raise RuntimeError("baseline and candidate checkpoint manifest identities unexpectedly match")

        backend = TransformersCausalLMGenerationBackend(device="cuda", dtype="float32", batch_size=6)
        generation = EvaluationGenerationService(artifact_root=evaluation_root / "generation", backend=backend)
        plan = generation.plan()
        if plan.content_hash != EVAL_PACK_HASH:
            raise RuntimeError(f"frozen semantic pack hash drift: {plan.content_hash}")
        if len(plan.tasks) != 18:
            raise RuntimeError(f"expected 18 frozen task/seed pairs, observed {len(plan.tasks)}")

        baseline_run = TrainingRunHandle(
            run_id=BASELINE_RUN_ID,
            experiment_id=EXPERIMENT_ID,
            backend_id="transformers_causal_lm",
            status="completed",
            checkpoint_refs=[str(baseline_checkpoint)],
            metadata={
                "generation_handoff_ref": str(baseline_checkpoint / "loading_instructions.json"),
                "checkpoint_manifest_hash": baseline_manifest,
                "trained": True,
                "dataset_role": "previous_wikitext_training_baseline",
            },
        )
        candidate_run = TrainingRunHandle(
            run_id=CANDIDATE_RUN_ID,
            experiment_id=EXPERIMENT_ID,
            backend_id="transformers_causal_lm",
            status="completed",
            checkpoint_refs=[str(candidate_checkpoint)],
            metadata={
                "generation_handoff_ref": str(candidate_checkpoint / "loading_instructions.json"),
                "checkpoint_manifest_hash": candidate_manifest,
                "trained": True,
                "dataset_role": "selected_instruction_data_candidate",
            },
        )

        baseline_generated = generation.generate(
            baseline_run,
            generation_handoff_ref=str(baseline_checkpoint / "loading_instructions.json"),
        )
        candidate_generated = generation.generate(
            candidate_run,
            generation_handoff_ref=str(candidate_checkpoint / "loading_instructions.json"),
        )

        proposal = load_frozen_proposal()
        evaluator = ExperimentEvaluationService(pack_name="semantic_behavior_v1")
        comparison = evaluator.compare(
            proposal,
            [baseline_generated.run_handle, candidate_generated.run_handle],
        )
        judge = SemanticComparisonJudgeAdapter().decide(
            comparison,
            run_id=CANDIDATE_RUN_ID,
            lineage_id=LINEAGE_ID,
            candidate_checkpoint_ref=str(candidate_checkpoint),
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
                    "baseline_checkpoint_manifest_hash": baseline_manifest,
                    "candidate_checkpoint_manifest_hash": candidate_manifest,
                    "eval_pack_id": plan.eval_pack_id,
                    "eval_pack_version": plan.eval_pack_version,
                    "eval_pack_content_hash": plan.content_hash,
                    "generation_settings_id": plan.generation_settings_id,
                    "seed_identity": plan.seed_identity,
                    "task_seed_count": len(plan.tasks),
                    "primary_variable": PRIMARY_VARIABLE,
                },
                "generation": {
                    "baseline": {
                        "run_id": BASELINE_RUN_ID,
                        "status": baseline_generated.report.completion_status,
                        "sample_count": len(baseline_generated.report.samples),
                        "report_ref": baseline_generated.report.report_ref,
                    },
                    "candidate": {
                        "run_id": CANDIDATE_RUN_ID,
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
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "run_id": eval_run_id,
                    "baseline_run_id": BASELINE_RUN_ID,
                    "candidate_run_id": CANDIDATE_RUN_ID,
                    "primary_variable": PRIMARY_VARIABLE,
                    "outcome": comparison.primary_outcome,
                    "deterministic_gate_status": comparison.deterministic_gate_status,
                    "confidence": comparison.confidence,
                    "recommendation": comparison.recommendation,
                    "judge_action": judge.next_action.value,
                    "baseline_samples": len(baseline_generated.report.samples),
                    "candidate_samples": len(candidate_generated.report.samples),
                    "action_applied": False,
                },
                sort_keys=True,
            )
        )
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
