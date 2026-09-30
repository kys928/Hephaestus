#!/usr/bin/env python3
"""Guarded RunPod launcher for Evaluator-only Interface Repair V3."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import launch_interface_repair_v2 as base
from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "configs/experiments/hephaestus_interface_repair_v3_evaluator.json"
MARKER_PATH = ROOT / "configs/experiments/interface_repair_v3_evaluator.launch.json"
AUTH_ENV = "HEPHAESTUS_INTERFACE_REPAIR_V3_EVALUATOR_AUTHORIZED"
PREFLIGHT_PHRASE = "LAUNCH_INTERFACE_REPAIR_V3_EVALUATOR_PREFLIGHT"
FULL_PHRASE = "LAUNCH_INTERFACE_REPAIR_V3_EVALUATOR_FULL"
ALLOWED_AUTH_CHANGED_FILES = {
    "configs/experiments/hephaestus_interface_repair_v3_evaluator.json",
    "configs/experiments/interface_repair_v3_evaluator.launch.json",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain an object")
    return value


def authorize(cfg: MappingLike, marker: MappingLike, mode: str) -> None:
    governance = cfg["governance"]
    if governance.get("production_promotion_allowed") or governance.get("automatic_role_dispatch_allowed"):
        raise RuntimeError("V3 cannot enable production promotion or automatic role dispatch")
    if governance.get("preserved_planner_mutation_allowed") is not False:
        raise RuntimeError("V3 must keep the certified Phase-II Planner candidate frozen")
    if list(cfg.get("trainable_roles", [])) != ["evaluator"]:
        raise RuntimeError("V3 paid execution must remain evaluator-only")

    if mode == "preflight":
        allowed = governance.get("paid_preflight_allowed")
        marked = marker.get("preflight_authorized")
        expected_phrase = PREFLIGHT_PHRASE
        phrase = marker.get("authorization_phrase_preflight")
    elif mode == "full":
        allowed = governance.get("paid_full_launch_allowed")
        marked = marker.get("full_launch_authorized")
        expected_phrase = FULL_PHRASE
        phrase = marker.get("authorization_phrase_full")
        if not marker.get("preflight_evidence_key") or not marker.get("preflight_repo_sha"):
            raise RuntimeError("V3 full launch requires exact persisted preflight evidence binding")
    else:
        raise ValueError(mode)

    if allowed is not True or marked is not True:
        raise RuntimeError(f"paid Interface Repair V3 {mode} is not authorized")
    if phrase != expected_phrase or base.required(AUTH_ENV) != expected_phrase:
        raise RuntimeError(f"Interface Repair V3 {mode} authorization phrase missing or mismatched")


def pod_shell(mode: str, venv_cache: str, pip_cache: str, adapter_cache: str) -> str:
    # Reuse the already-debugged V2 CUDA/venv/bootstrap shell, but execute exactly
    # one Evaluator V3 role and build the fresh V3 pack instead of the V1/V2 pack.
    shell = base.pod_shell("preflight", venv_cache, pip_cache, adapter_cache)
    old_body = 'export HEPHAESTUS_REPAIR_ROLE=planner\n"$PY" scripts/run_interface_repair_v2_cached.py --preflight'
    run_flag = " --preflight" if mode == "preflight" else ""
    new_body = f'export HEPHAESTUS_REPAIR_ROLE=evaluator\n"$PY" scripts/run_interface_repair_v3_evaluator.py{run_flag}'
    if old_body not in shell:
        raise RuntimeError("V2 bootstrap shell body changed; refusing unsafe V3 string adaptation")
    shell = shell.replace(old_body, new_body)
    old_build = '"$PY" scripts/build_interface_repair_v1.py\nwrite_bootstrap runner_started'
    new_build = (
        '"$PY" scripts/build_interface_repair_v3_evaluator.py\n'
        '"$PY" -m py_compile scripts/build_interface_repair_v3_evaluator.py scripts/run_interface_repair_v3_evaluator.py\n'
        'write_bootstrap runner_started'
    )
    if old_build not in shell:
        raise RuntimeError("V2 bootstrap pack-build block changed; refusing unsafe V3 string adaptation")
    return shell.replace(old_build, new_build)


def request_body(cfg: dict[str, Any], repo_sha: str, run_id: str, mode: str, *, real_env: bool) -> dict[str, object]:
    original = base.pod_shell
    try:
        base.pod_shell = pod_shell
        body = base.request_body(cfg, repo_sha, run_id, mode, real_env=real_env)
    finally:
        base.pod_shell = original
    body["name"] = f"hephaestus-interface-repair-v3-evaluator-{mode}-{run_id}"[:180]
    return body


def _normalized_science_config(value: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(value))
    governance = clone.get("governance", {})
    governance.pop("paid_preflight_allowed", None)
    governance.pop("paid_full_launch_allowed", None)
    return clone


def verify_preflight_evidence(client: Any, marker: dict[str, Any], current_repo_sha: str) -> dict[str, Any]:
    key = str(marker.get("preflight_evidence_key") or "")
    bound_sha = str(marker.get("preflight_repo_sha") or "")
    if not key or not bound_sha:
        raise RuntimeError("missing V3 preflight evidence binding")
    result = base.maybe_json(client, key)
    if result is None:
        raise RuntimeError("bound V3 preflight evidence does not exist")
    if result.get("status") != "completed" or result.get("approved_for_full_run") is not True:
        raise RuntimeError("bound V3 preflight did not approve a full run")
    if str(result.get("repo_sha", "")) != bound_sha:
        raise RuntimeError("V3 preflight evidence repository SHA does not match marker binding")

    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", bound_sha, current_repo_sha],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("bound V3 preflight SHA is not an ancestor of the full authorization commit")
    diff = subprocess.check_output(
        ["git", "diff", "--name-only", f"{bound_sha}..{current_repo_sha}"],
        cwd=ROOT,
        text=True,
    )
    changed = {line.strip() for line in diff.splitlines() if line.strip()}
    unexpected = sorted(changed - ALLOWED_AUTH_CHANGED_FILES)
    if unexpected:
        raise RuntimeError(f"V3 science/execution files changed after preflight: {unexpected}")

    old_raw = subprocess.check_output(
        ["git", "show", f"{bound_sha}:configs/experiments/hephaestus_interface_repair_v3_evaluator.json"],
        cwd=ROOT,
        text=True,
    )
    old_cfg = json.loads(old_raw)
    current_cfg = load_json(CFG_PATH)
    if _normalized_science_config(old_cfg) != _normalized_science_config(current_cfg):
        raise RuntimeError("V3 scientific or execution configuration changed after preflight")
    return result


# Simple alias used to keep annotations readable without importing typing.Mapping at runtime.
MappingLike = dict[str, Any]


def execute(cfg: dict[str, Any], marker: dict[str, Any], mode: str) -> dict[str, Any]:
    authorize(cfg, marker, mode)
    repo_sha = base.required("GITHUB_SHA")
    github_run_id = base.required("GITHUB_RUN_ID")
    run_id = os.environ.get("HEPHAESTUS_INTERFACE_REPAIR_RUN_ID", f"interface-repair-v3-evaluator-{mode}-{github_run_id}")
    prefix = f"{str(cfg['execution']['s3_prefix']).rstrip('/')}/{run_id}"
    bootstrap_key = f"{prefix}/bootstrap/progress.json"
    role = "evaluator"
    if mode == "preflight":
        result_key = f"{prefix}/preflight/{role}/result.json"
        progress_key = f"{prefix}/preflight/{role}/progress.json"
        wall = int(cfg["execution"]["hard_wall_seconds_preflight_total"])
        bootstrap_wall = int(cfg["execution"]["hard_wall_seconds_preflight_bootstrap"])
        benchmark_wall = int(cfg["execution"]["hard_wall_seconds_preflight_benchmark"])
        max_total = float(cfg["execution"]["max_estimated_total_usd_preflight"])
    else:
        result_key = f"{prefix}/roles/{role}/result.json"
        progress_key = f"{prefix}/roles/{role}/progress.json"
        wall = int(cfg["execution"]["hard_wall_seconds_stack"])
        bootstrap_wall = benchmark_wall = 0
        max_total = float(cfg["execution"]["max_estimated_total_usd_stack"])

    client = base.s3_client()
    client.head_bucket(Bucket=base.required("RUNPOD_NETWORK_VOLUME_ID"))
    preflight = verify_preflight_evidence(client, marker, repo_sha) if mode == "full" else None
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
    body = request_body(cfg, repo_sha, run_id, mode, real_env=True)
    pod_id: str | None = None
    started: float | None = None
    benchmark_started: float | None = None
    last_progress_stage: str | None = None
    last_bootstrap_stage: str | None = None
    metadata_printed = False
    record: dict[str, Any] = {
        "mode": mode,
        "run_id": run_id,
        "repo_sha": repo_sha,
        "status": "starting",
        "result_key": result_key,
        "bootstrap_progress_key": bootstrap_key,
        "preflight_evidence_key": marker.get("preflight_evidence_key") if mode == "full" else None,
        "preflight_projection_seconds": preflight.get("projected_full_role_seconds") if preflight else None,
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
        print("INTERFACE_REPAIR_V3_POD_JSON " + json.dumps({
            "mode": mode,
            "pod_id": pod_id,
            "created_at_unix": time.time(),
            "gpu_allowlist": cfg["execution"]["gpu_type_ids"],
        }, sort_keys=True), flush=True)

        while True:
            elapsed = time.monotonic() - started
            if elapsed >= wall:
                raise TimeoutError(f"V3 {mode} total hard wall reached")
            snapshot = execution.get_pod(pod_id)
            metadata = base.safe_pod_metadata(snapshot)
            if metadata and not metadata_printed:
                print("INTERFACE_REPAIR_V3_POD_METADATA_JSON " + json.dumps(metadata, sort_keys=True), flush=True)
                record["pod_metadata"] = metadata
                metadata_printed = True

            cost = snapshot.get("adjustedCostPerHr", snapshot.get("costPerHr"))
            if cost is not None:
                hourly = float(cost)
                estimated = hourly * elapsed / 3600.0
                record.update(cost_per_hr=hourly, estimated_cost_usd=estimated)
                if hourly > float(cfg["execution"]["max_hourly_usd_per_pod"]):
                    raise RuntimeError(f"hourly cost ceiling exceeded: {hourly:.4f}")
                if estimated > max_total:
                    raise RuntimeError(f"estimated {mode} cost ceiling exceeded: {estimated:.4f}")

            bootstrap = base.maybe_json(client, bootstrap_key)
            if bootstrap is not None:
                stage = str(bootstrap.get("stage", ""))
                if stage and stage != last_bootstrap_stage:
                    print("INTERFACE_REPAIR_V3_BOOTSTRAP_PROGRESS_JSON " + json.dumps(bootstrap, sort_keys=True), flush=True)
                    record["last_bootstrap"] = bootstrap
                    last_bootstrap_stage = stage

            progress = base.maybe_json(client, progress_key)
            if progress is not None:
                stage = str(progress.get("stage", ""))
                if stage and stage != last_progress_stage:
                    print("INTERFACE_REPAIR_V3_PROGRESS_JSON " + json.dumps(progress, sort_keys=True), flush=True)
                    record["last_progress"] = progress
                    last_progress_stage = stage
                if mode == "preflight" and benchmark_started is None and stage in base.PRELIGHT_BENCHMARK_STAGES:
                    benchmark_started = time.monotonic()
                    record["benchmark_started_at_unix"] = time.time()

            if mode == "preflight":
                if benchmark_started is None and elapsed >= bootstrap_wall:
                    raise TimeoutError(
                        "V3 preflight bootstrap/model-init hard wall reached before benchmark start; "
                        f"last_bootstrap_stage={last_bootstrap_stage!r} last_runner_stage={last_progress_stage!r}"
                    )
                if benchmark_started is not None and (time.monotonic() - benchmark_started) >= benchmark_wall:
                    raise TimeoutError("V3 preflight benchmark hard wall reached")

            terminal = base.maybe_json(client, result_key)
            if terminal is not None:
                record["result"] = terminal
                if terminal.get("status") == "completed":
                    if mode == "preflight" and terminal.get("approved_for_full_run") is not True:
                        record["status"] = "preflight_rejected"
                    elif mode == "full" and terminal.get("certification", {}).get("certified") is not True:
                        record["status"] = "scientific_rejected"
                    else:
                        record["status"] = "completed"
                    return record
                if terminal.get("status") == "failed":
                    raise RuntimeError(f"V3 runner failed: {terminal.get('error_type')}: {terminal.get('error')}")
            if str(snapshot.get("desiredStatus", "")).upper() in base.TERMINAL:
                raise RuntimeError("V3 pod became terminal before terminal S3 result appeared")
            time.sleep(float(cfg["execution"]["poll_seconds"]))
    finally:
        if pod_id:
            teardown = base.delete_with_retries(execution, pod_id)
            record["teardown"] = teardown
            print("INTERFACE_REPAIR_V3_TEARDOWN_JSON " + json.dumps(teardown, sort_keys=True), flush=True)
            if not teardown.get("verified_absent"):
                raise RuntimeError("V3 pod teardown could not be verified")


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--preflight", action="store_true")
    group.add_argument("--full", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    mode = "preflight" if args.preflight else "full"
    cfg = load_json(CFG_PATH)
    marker = load_json(MARKER_PATH)
    repo_sha = os.environ.get("GITHUB_SHA", "<sha>")
    run_id = os.environ.get("HEPHAESTUS_INTERFACE_REPAIR_RUN_ID", f"interface-repair-v3-evaluator-{mode}-{os.environ.get('GITHUB_RUN_ID', '<run>')}")
    if not args.execute:
        print(json.dumps({
            "mode": mode,
            "run_id": run_id,
            "request": request_body(cfg, repo_sha, run_id, mode, real_env=False),
            "execution": cfg["execution"],
            "governance": cfg["governance"],
            "preflight_evidence_key": marker.get("preflight_evidence_key"),
            "preflight_repo_sha": marker.get("preflight_repo_sha"),
        }, indent=2, sort_keys=True))
        return 0
    result = execute(cfg, marker, mode)
    print("INTERFACE_REPAIR_V3_LAUNCH_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
    if result.get("status") in {"preflight_rejected", "scientific_rejected"}:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
