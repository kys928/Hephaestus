#!/usr/bin/env python3
"""Guarded RunPod launcher for Evaluator semantic-invariance repair v1."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import launch_interface_repair_v2 as base
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/experiments/hephaestus_evaluator_semantic_invariance_repair_v1.json"
PARENT_PATH = ROOT / "configs/experiments/hephaestus_evaluator_certification_candidate_v1.json"
AUTH_ENV = "HEPHAESTUS_EVALUATOR_REPAIR_AUTHORIZED"
AUTH_PHRASE = "LAUNCH_HEPHAESTUS_EVALUATOR_SEMANTIC_INVARIANCE_REPAIR_V1"
ORIGINAL_POD_SHELL = base.pod_shell


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def authorize(cfg: dict[str, Any], parent: dict[str, Any], mode: str) -> None:
    if base.required(AUTH_ENV) != AUTH_PHRASE:
        raise RuntimeError("semantic repair authorization phrase missing or mismatched")
    gov = cfg["governance"]
    if gov.get("training_approved") is not True:
        raise RuntimeError("semantic repair training is not approved")
    if mode == "train" and gov.get("paid_training_allowed") is not True:
        raise RuntimeError("paid semantic repair training is disabled")
    if mode == "evaluate" and gov.get("paid_checkpoint_evaluation_allowed") is not True:
        raise RuntimeError("paid checkpoint evaluation is disabled")
    if gov.get("parent_adapter_mutation_allowed") is not False:
        raise RuntimeError("parent adapter mutation must remain forbidden")
    if gov.get("burned_sealed_partition_use_allowed") is not False:
        raise RuntimeError("burned sealed partition use must remain forbidden")
    if gov.get("certification_claim_allowed") or gov.get("production_promotion_allowed") or gov.get("automatic_role_dispatch_allowed"):
        raise RuntimeError("repair launcher may not certify, promote, or enable dispatch")
    if parent.get("candidate_id") != cfg["required_parent_candidate_id"]:
        raise RuntimeError("parent candidate identity drift")
    if parent.get("model", {}).get("adapter_sha256") != cfg["required_parent_adapter_sha256"]:
        raise RuntimeError("parent adapter digest drift")
    if parent.get("governance", {}).get("weights_frozen") is not True:
        raise RuntimeError("parent candidate is not frozen")


def pod_shell(_mode: str, venv_cache: str, pip_cache: str, adapter_cache: str, *, launch_mode: str) -> str:
    shell = ORIGINAL_POD_SHELL("preflight", venv_cache, pip_cache, adapter_cache)
    old_body = 'export HEPHAESTUS_REPAIR_ROLE=planner\n"$PY" scripts/run_interface_repair_v2_cached.py --preflight'
    if launch_mode == "train":
        new_body = '"$PY" scripts/run_evaluator_semantic_invariance_repair_v1.py --train'
    elif launch_mode == "evaluate":
        new_body = '"$PY" scripts/run_evaluator_semantic_invariance_repair_v1.py --evaluate-checkpoint "$HEPHAESTUS_EVALUATOR_REPAIR_CHECKPOINT_STEP"'
    else:
        raise ValueError(launch_mode)
    if old_body not in shell:
        raise RuntimeError("V2 bootstrap body changed; refusing unsafe repair adaptation")
    shell = shell.replace(old_body, new_body)

    old_build = '"$PY" scripts/build_interface_repair_v1.py\nwrite_bootstrap runner_started'
    new_build = (
        '"$PY" scripts/build_evaluator_semantic_invariance_repair_v1.py\n'
        '"$PY" -m py_compile scripts/build_evaluator_semantic_invariance_repair_v1.py '
        'scripts/run_evaluator_semantic_invariance_repair_v1.py scripts/evaluator_semantic_repair_common.py '
        'scripts/evaluator_semantic_repair_train.py scripts/evaluator_semantic_repair_eval.py '
        'scripts/run_evaluator_v3_1_development_baseline.py scripts/run_evaluator_sealed_certification_v1.py '
        'scripts/run_interface_repair_v3_evaluator.py scripts/run_interface_repair_v3_1_evaluator.py '
        'scripts/interface_repair_v2_bootstrap.py src/hephaestus/control/evaluator_boundary.py '
        'src/hephaestus/control/evaluator_evidence_refs.py\n'
        'write_bootstrap runner_started'
    )
    if old_build not in shell:
        raise RuntimeError("V2 bootstrap pack-build block changed; refusing unsafe repair adaptation")
    return shell.replace(old_build, new_build)


def request_body(cfg: dict[str, Any], repo_sha: str, run_id: str, mode: str, step: int | None, *, real_env: bool) -> dict[str, object]:
    original = base.pod_shell
    try:
        base.pod_shell = lambda m, v, p, a: pod_shell(m, v, p, a, launch_mode=mode)
        body = base.request_body(cfg, repo_sha, run_id, "preflight", real_env=real_env)
    finally:
        base.pod_shell = original
    prefix = f"{str(cfg['execution']['s3_prefix']).rstrip('/')}/{run_id}"
    if mode == "train":
        suffix = "training"
        name = f"hephaestus-evaluator-semantic-repair-train-{run_id}"
    else:
        if step is None:
            raise RuntimeError("checkpoint evaluation requires --step")
        suffix = f"checkpoints/step-{step:03d}/evaluation"
        name = f"hephaestus-evaluator-semantic-repair-eval-{step:03d}-{run_id}"
    env = dict(body["env"])
    env["HEPHAESTUS_EVALUATOR_REPAIR_RUN_ID"] = run_id
    env["HEPHAESTUS_BOOTSTRAP_PROGRESS_KEY"] = f"{prefix}/{suffix}/bootstrap/progress.json"
    if step is not None:
        env["HEPHAESTUS_EVALUATOR_REPAIR_CHECKPOINT_STEP"] = str(step)
    body["env"] = env
    body["name"] = name[:180]
    return body


def execute(cfg: dict[str, Any], parent: dict[str, Any], mode: str, step: int | None) -> dict[str, Any]:
    authorize(cfg, parent, mode)
    if mode == "evaluate":
        allowed = [int(x) for x in cfg["training"]["checkpoint_steps"]]
        if step not in allowed:
            raise RuntimeError(f"checkpoint step must be one of {allowed}")
    repo_sha = base.required("GITHUB_SHA")
    github_run_id = base.required("GITHUB_RUN_ID")
    run_id = (os.environ.get("HEPHAESTUS_EVALUATOR_REPAIR_RUN_ID") or f"evaluator-semantic-repair-{github_run_id}").strip()
    prefix = f"{str(cfg['execution']['s3_prefix']).rstrip('/')}/{run_id}"
    if mode == "train":
        result_key = f"{prefix}/training/result.json"
        progress_key = f"{prefix}/training/progress.json"
        bootstrap_key = f"{prefix}/training/bootstrap/progress.json"
        expected_status = "training_complete"
        hard_wall = float(cfg["execution"]["hard_wall_seconds_train"])
        cost_ceiling = float(cfg["execution"]["max_estimated_total_usd_train"])
    else:
        assert step is not None
        result_key = f"{prefix}/checkpoints/step-{step:03d}/evaluation/result.json"
        progress_key = f"{prefix}/checkpoints/step-{step:03d}/evaluation/progress.json"
        bootstrap_key = f"{prefix}/checkpoints/step-{step:03d}/evaluation/bootstrap/progress.json"
        expected_status = "evaluation_complete"
        hard_wall = float(cfg["execution"]["hard_wall_seconds_evaluate"])
        cost_ceiling = float(cfg["execution"]["max_estimated_total_usd_evaluate"])

    client = base.s3_client()
    client.head_bucket(Bucket=base.required("RUNPOD_NETWORK_VOLUME_ID"))
    if mode == "evaluate":
        training = base.maybe_json(client, f"{prefix}/training/result.json")
        if training is None or training.get("status") != "training_complete":
            raise RuntimeError("checkpoint evaluation requires completed training evidence")
        if str(training.get("repo_sha")) != repo_sha:
            raise RuntimeError("training/evaluation repository SHA mismatch")

    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
    body = request_body(cfg, repo_sha, run_id, mode, step, real_env=True)
    pod_id: str | None = None
    started: float | None = None
    last_progress_stage: str | None = None
    last_bootstrap_stage: str | None = None
    metadata_printed = False
    record: dict[str, Any] = {
        "run_id": run_id,
        "repo_sha": repo_sha,
        "mode": mode,
        "checkpoint_step": step,
        "status": "starting",
        "result_key": result_key,
    }
    try:
        pod = base.create_with_capacity_retries(
            execution,
            body,
            int(cfg["execution"]["capacity_retry_attempts"]),
            float(cfg["execution"]["capacity_retry_seconds"]),
        )
        pod_id = str(pod["id"])
        record["pod_id"] = pod_id
        started = time.monotonic()
        print("EVALUATOR_SEMANTIC_REPAIR_POD_JSON " + json.dumps({
            "pod_id": pod_id, "run_id": run_id, "mode": mode, "checkpoint_step": step,
            "gpu_allowlist": cfg["execution"]["gpu_type_ids"],
        }, sort_keys=True), flush=True)
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= hard_wall:
                raise TimeoutError(f"semantic repair {mode} hard wall reached")
            snapshot = execution.get_pod(pod_id)
            metadata = base.safe_pod_metadata(snapshot)
            if metadata and not metadata_printed:
                record["pod_metadata"] = metadata
                metadata_printed = True
                print("EVALUATOR_SEMANTIC_REPAIR_POD_METADATA_JSON " + json.dumps(metadata, sort_keys=True), flush=True)
            cost = snapshot.get("adjustedCostPerHr", snapshot.get("costPerHr"))
            if cost is not None:
                hourly = float(cost)
                estimated = hourly * elapsed / 3600.0
                record.update(cost_per_hr=hourly, estimated_cost_usd=estimated)
                if hourly > float(cfg["execution"]["max_hourly_usd_per_pod"]):
                    raise RuntimeError(f"semantic repair hourly cost ceiling exceeded: {hourly:.4f}")
                if estimated > cost_ceiling:
                    raise RuntimeError(f"semantic repair {mode} total cost ceiling exceeded: {estimated:.4f}")
            bootstrap = base.maybe_json(client, bootstrap_key)
            if bootstrap is not None:
                stage = str(bootstrap.get("stage", ""))
                if stage and stage != last_bootstrap_stage:
                    record["last_bootstrap"] = bootstrap
                    last_bootstrap_stage = stage
                    print("EVALUATOR_SEMANTIC_REPAIR_BOOTSTRAP_JSON " + json.dumps(bootstrap, sort_keys=True), flush=True)
            progress = base.maybe_json(client, progress_key)
            if progress is not None:
                stage = str(progress.get("stage", ""))
                if stage and stage != last_progress_stage:
                    record["last_progress"] = progress
                    last_progress_stage = stage
                    print("EVALUATOR_SEMANTIC_REPAIR_PROGRESS_JSON " + json.dumps(progress, sort_keys=True), flush=True)
            terminal = base.maybe_json(client, result_key)
            if terminal is not None:
                record["result"] = terminal
                status = str(terminal.get("status", ""))
                if status == "failed":
                    raise RuntimeError(f"semantic repair runner failed: {terminal.get('error_type')}: {terminal.get('error')}")
                if status == expected_status:
                    record["status"] = expected_status
                    return record
            if str(snapshot.get("desiredStatus", "")).upper() in base.TERMINAL:
                raise RuntimeError("semantic repair pod became terminal before result persistence")
            time.sleep(float(cfg["execution"]["poll_seconds"]))
    finally:
        if pod_id:
            teardown = base.delete_with_retries(execution, pod_id)
            record["teardown"] = teardown
            print("EVALUATOR_SEMANTIC_REPAIR_TEARDOWN_JSON " + json.dumps(teardown, sort_keys=True), flush=True)
            if not teardown.get("verified_absent"):
                raise RuntimeError("semantic repair pod teardown could not be verified")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate"], required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = load_json(CFG_PATH)
    parent = load_json(PARENT_PATH)
    run_id = (os.environ.get("HEPHAESTUS_EVALUATOR_REPAIR_RUN_ID") or f"evaluator-semantic-repair-{os.environ.get('GITHUB_RUN_ID', '<run>')}").strip()
    if args.mode == "evaluate" and args.step is None:
        parser.error("--step is required for --mode evaluate")
    if not args.execute:
        print(json.dumps({
            "run_id": run_id,
            "mode": args.mode,
            "checkpoint_step": args.step,
            "request": request_body(cfg, os.environ.get("GITHUB_SHA", "<sha>"), run_id, args.mode, args.step, real_env=False),
        }, indent=2, sort_keys=True))
        return 0
    result = execute(cfg, parent, args.mode, args.step)
    print("EVALUATOR_SEMANTIC_REPAIR_LAUNCH_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
    expected = "training_complete" if args.mode == "train" else "evaluation_complete"
    return 0 if result.get("status") == expected else 3


if __name__ == "__main__":
    raise SystemExit(main())
