#!/usr/bin/env python3
"""Guarded paid launcher for the frozen Evaluator V3.1 development baseline."""
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
CFG_PATH = ROOT / "configs/experiments/hephaestus_evaluator_sealed_certification_v1.json"
CANDIDATE_PATH = ROOT / "configs/experiments/hephaestus_evaluator_certification_candidate_v1.json"
AUTH_ENV = "HEPHAESTUS_EVALUATOR_DEV_BASELINE_AUTHORIZED"
AUTH_PHRASE = "LAUNCH_HEPHAESTUS_EVALUATOR_V3_1_DEV_BASELINE"
S3_ROOT = "hephaestus/scientific/v4/evaluator_development_baseline"
EXPECTED_CANDIDATE_ID = "evaluator-v3.1-boundary-v1"
EXPECTED_ADAPTER_SHA256 = "913797ddb8d9d95f83d09a244e8efe430d7bfc4383e589499c2fcc2d943487ed"
ORIGINAL_POD_SHELL = base.pod_shell


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain an object")
    return value


def authorize(cfg: dict[str, Any], candidate: dict[str, Any]) -> None:
    if base.required(AUTH_ENV) != AUTH_PHRASE:
        raise RuntimeError("development baseline authorization phrase missing or mismatched")
    governance = cfg["governance"]
    if governance.get("training_allowed") is not False:
        raise RuntimeError("development baseline may not train")
    if governance.get("adapter_mutation_allowed") is not False:
        raise RuntimeError("development baseline may not mutate adapters")
    if governance.get("candidate_weights_mutation_allowed") is not False:
        raise RuntimeError("development baseline may not mutate candidate weights")
    if governance.get("production_promotion_allowed") or governance.get("automatic_role_dispatch_allowed"):
        raise RuntimeError("development baseline may not promote or enable dispatch")
    if candidate.get("candidate_id") != EXPECTED_CANDIDATE_ID:
        raise RuntimeError("candidate identity drift")
    if candidate.get("governance", {}).get("weights_frozen") is not True:
        raise RuntimeError("candidate weights are not frozen")
    if candidate.get("model", {}).get("adapter_sha256") != EXPECTED_ADAPTER_SHA256:
        raise RuntimeError("V3.1 adapter digest drift")


def pod_shell(_mode: str, venv_cache: str, pip_cache: str, adapter_cache: str) -> str:
    shell = ORIGINAL_POD_SHELL("preflight", venv_cache, pip_cache, adapter_cache)
    old_body = 'export HEPHAESTUS_REPAIR_ROLE=planner\n"$PY" scripts/run_interface_repair_v2_cached.py --preflight'
    new_body = '"$PY" scripts/run_evaluator_v3_1_development_baseline.py'
    if old_body not in shell:
        raise RuntimeError("V2 bootstrap body changed; refusing unsafe baseline adaptation")
    shell = shell.replace(old_body, new_body)
    old_build = '"$PY" scripts/build_interface_repair_v1.py\nwrite_bootstrap runner_started'
    new_build = (
        '"$PY" scripts/build_evaluator_semantic_invariance_repair_v1.py\n'
        '"$PY" -m py_compile scripts/build_evaluator_semantic_invariance_repair_v1.py '
        'scripts/run_evaluator_v3_1_development_baseline.py scripts/run_evaluator_sealed_certification_v1.py '
        'scripts/run_interface_repair_v3_evaluator.py scripts/run_interface_repair_v3_1_evaluator.py '
        'src/hephaestus/control/evaluator_boundary.py src/hephaestus/control/evaluator_evidence_refs.py\n'
        'write_bootstrap runner_started'
    )
    if old_build not in shell:
        raise RuntimeError("V2 bootstrap pack-build block changed; refusing unsafe baseline adaptation")
    return shell.replace(old_build, new_build)


def request_body(cfg: dict[str, Any], repo_sha: str, run_id: str, *, real_env: bool) -> dict[str, object]:
    original = base.pod_shell
    try:
        base.pod_shell = pod_shell
        body = base.request_body(cfg, repo_sha, run_id, "preflight", real_env=real_env)
    finally:
        base.pod_shell = original
    body["name"] = f"hephaestus-evaluator-v3-1-dev-baseline-{run_id}"[:180]
    env = dict(body["env"])
    env["HEPHAESTUS_EVALUATOR_DEV_BASELINE_RUN_ID"] = run_id
    body["env"] = env
    return body


def execute(cfg: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    authorize(cfg, candidate)
    repo_sha = base.required("GITHUB_SHA")
    github_run_id = base.required("GITHUB_RUN_ID")
    run_id = os.environ.get("HEPHAESTUS_EVALUATOR_DEV_BASELINE_RUN_ID", f"evaluator-v3-1-dev-baseline-{github_run_id}")
    prefix = f"{S3_ROOT}/{run_id}"
    result_key = f"{prefix}/result.json"
    progress_key = f"{prefix}/progress.json"
    bootstrap_key = f"{prefix}/bootstrap/progress.json"
    client = base.s3_client()
    client.head_bucket(Bucket=base.required("RUNPOD_NETWORK_VOLUME_ID"))
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
    body = request_body(cfg, repo_sha, run_id, real_env=True)
    pod_id: str | None = None
    started: float | None = None
    metadata_printed = False
    last_progress_stage: str | None = None
    last_bootstrap_stage: str | None = None
    record: dict[str, Any] = {
        "run_id": run_id,
        "repo_sha": repo_sha,
        "status": "starting",
        "result_key": result_key,
        "bootstrap_progress_key": bootstrap_key,
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
        print("EVALUATOR_DEV_BASELINE_POD_JSON " + json.dumps({
            "pod_id": pod_id,
            "run_id": run_id,
            "gpu_allowlist": cfg["execution"]["gpu_type_ids"],
        }, sort_keys=True), flush=True)

        while True:
            elapsed = time.monotonic() - started
            if elapsed >= float(cfg["execution"]["hard_wall_seconds"]):
                raise TimeoutError("development baseline total hard wall reached")
            snapshot = execution.get_pod(pod_id)
            metadata = base.safe_pod_metadata(snapshot)
            if metadata and not metadata_printed:
                record["pod_metadata"] = metadata
                metadata_printed = True
                print("EVALUATOR_DEV_BASELINE_POD_METADATA_JSON " + json.dumps(metadata, sort_keys=True), flush=True)

            cost = snapshot.get("adjustedCostPerHr", snapshot.get("costPerHr"))
            if cost is not None:
                hourly = float(cost)
                estimated = hourly * elapsed / 3600.0
                record.update(cost_per_hr=hourly, estimated_cost_usd=estimated)
                if hourly > float(cfg["execution"]["max_hourly_usd_per_pod"]):
                    raise RuntimeError(f"development baseline hourly cost ceiling exceeded: {hourly:.4f}")
                if estimated > float(cfg["execution"]["max_estimated_total_usd"]):
                    raise RuntimeError(f"development baseline total cost ceiling exceeded: {estimated:.4f}")

            bootstrap = base.maybe_json(client, bootstrap_key)
            if bootstrap is not None:
                stage = str(bootstrap.get("stage", ""))
                if stage and stage != last_bootstrap_stage:
                    record["last_bootstrap"] = bootstrap
                    last_bootstrap_stage = stage
                    print("EVALUATOR_DEV_BASELINE_BOOTSTRAP_JSON " + json.dumps(bootstrap, sort_keys=True), flush=True)

            progress = base.maybe_json(client, progress_key)
            if progress is not None:
                stage = str(progress.get("stage", ""))
                if stage and stage != last_progress_stage:
                    record["last_progress"] = progress
                    last_progress_stage = stage
                    print("EVALUATOR_DEV_BASELINE_PROGRESS_JSON " + json.dumps(progress, sort_keys=True), flush=True)

            terminal = base.maybe_json(client, result_key)
            if terminal is not None:
                record["result"] = terminal
                status = str(terminal.get("status", ""))
                if status == "failed":
                    raise RuntimeError(f"development baseline runner failed: {terminal.get('error_type')}: {terminal.get('error')}")
                if status in {"baseline_complete", "preflight_rejected"}:
                    record["status"] = status
                    return record
            if str(snapshot.get("desiredStatus", "")).upper() in base.TERMINAL:
                raise RuntimeError("development baseline pod became terminal before result persistence")
            time.sleep(float(cfg["execution"]["poll_seconds"]))
    finally:
        if pod_id:
            teardown = base.delete_with_retries(execution, pod_id)
            record["teardown"] = teardown
            print("EVALUATOR_DEV_BASELINE_TEARDOWN_JSON " + json.dumps(teardown, sort_keys=True), flush=True)
            if not teardown.get("verified_absent"):
                raise RuntimeError("development baseline pod teardown could not be verified")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = load_json(CFG_PATH)
    candidate = load_json(CANDIDATE_PATH)
    repo_sha = os.environ.get("GITHUB_SHA", "<sha>")
    run_id = os.environ.get("HEPHAESTUS_EVALUATOR_DEV_BASELINE_RUN_ID", f"evaluator-v3-1-dev-baseline-{os.environ.get('GITHUB_RUN_ID', '<run>')}")
    if not args.execute:
        print(json.dumps({
            "run_id": run_id,
            "request": request_body(cfg, repo_sha, run_id, real_env=False),
            "execution": cfg["execution"],
            "candidate_id": candidate.get("candidate_id"),
            "candidate_adapter_sha256": candidate.get("model", {}).get("adapter_sha256"),
            "training_allowed": False,
            "certification_claim_allowed": False,
        }, indent=2, sort_keys=True))
        return 0
    result = execute(cfg, candidate)
    print("EVALUATOR_DEV_BASELINE_LAUNCH_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result.get("status") == "baseline_complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
