#!/usr/bin/env python3
"""Guarded paid launcher for frozen Evaluator sealed certification v1."""
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
CFG_PATH = ROOT / "configs/experiments/hephaestus_evaluator_sealed_certification_v1.json"
MARKER_PATH = ROOT / "configs/experiments/hephaestus_evaluator_sealed_certification_v1.launch.json"
CANDIDATE_PATH = ROOT / "configs/experiments/hephaestus_evaluator_certification_candidate_v1.json"
AUTH_ENV = "HEPHAESTUS_EVALUATOR_SEALED_CERTIFICATION_AUTHORIZED"
AUTH_PHRASE = "LAUNCH_HEPHAESTUS_EVALUATOR_SEALED_CERTIFICATION_V1"
ORIGINAL_POD_SHELL = base.pod_shell
ALLOWED_POST_PREP_CHANGED_FILES = {
    "configs/experiments/hephaestus_evaluator_sealed_certification_v1.json",
    "configs/experiments/hephaestus_evaluator_sealed_certification_v1.launch.json",
}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain an object")
    return value


def _normalized_cfg(value: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(value))
    clone["governance"].pop("paid_launch_allowed", None)
    return clone


def verify_prepared_binding(cfg: dict[str, Any], marker: dict[str, Any], current_repo_sha: str) -> None:
    prepared_sha = str(marker.get("authorized_repo_sha") or "")
    if not prepared_sha:
        raise RuntimeError("sealed certification launch is not bound to a validated prepared repository SHA")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", prepared_sha, current_repo_sha],
        cwd=ROOT, check=False, capture_output=True, text=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("prepared sealed-certification SHA is not an ancestor of the launch commit")
    changed = {
        line.strip()
        for line in subprocess.check_output(
            ["git", "diff", "--name-only", f"{prepared_sha}..{current_repo_sha}"], cwd=ROOT, text=True
        ).splitlines()
        if line.strip()
    }
    unexpected = sorted(changed - ALLOWED_POST_PREP_CHANGED_FILES)
    if unexpected:
        raise RuntimeError(f"sealed certification code/science changed after preparation: {unexpected}")
    old_cfg = json.loads(subprocess.check_output(
        ["git", "show", f"{prepared_sha}:{CFG_PATH.relative_to(ROOT)}"], cwd=ROOT, text=True
    ))
    if _normalized_cfg(old_cfg) != _normalized_cfg(cfg):
        raise RuntimeError("sealed certification protocol changed after validated preparation")


def authorize(cfg: dict[str, Any], marker: dict[str, Any], current_repo_sha: str) -> None:
    governance = cfg["governance"]
    candidate = load_json(CANDIDATE_PATH)
    if governance.get("paid_launch_allowed") is not True or marker.get("launch_authorized") is not True:
        raise RuntimeError("paid sealed certification launch is locked")
    if marker.get("authorization_phrase") != AUTH_PHRASE or base.required(AUTH_ENV) != AUTH_PHRASE:
        raise RuntimeError("sealed certification authorization phrase missing or mismatched")
    if governance.get("training_allowed") is not False or governance.get("adapter_mutation_allowed") is not False:
        raise RuntimeError("sealed certification may not train or mutate adapters")
    if governance.get("candidate_weights_mutation_allowed") is not False:
        raise RuntimeError("sealed certification may not mutate candidate weights")
    if governance.get("production_promotion_allowed") or governance.get("automatic_role_dispatch_allowed"):
        raise RuntimeError("sealed certification may not promote or enable automatic dispatch")
    if candidate["governance"].get("weights_frozen") is not True:
        raise RuntimeError("candidate weights are not frozen")
    if candidate["sealed_evidence"].get("canonical_sha256") != cfg["pack"].get("canonical_sha256"):
        raise RuntimeError("candidate/config sealed pack binding mismatch")
    verify_prepared_binding(cfg, marker, current_repo_sha)


def pod_shell(_mode: str, venv_cache: str, pip_cache: str, adapter_cache: str) -> str:
    shell = ORIGINAL_POD_SHELL("preflight", venv_cache, pip_cache, adapter_cache)
    old_body = 'export HEPHAESTUS_REPAIR_ROLE=planner\n"$PY" scripts/run_interface_repair_v2_cached.py --preflight'
    new_body = '"$PY" scripts/run_evaluator_sealed_certification_v1.py'
    if old_body not in shell:
        raise RuntimeError("V2 bootstrap body changed; refusing unsafe certification adaptation")
    shell = shell.replace(old_body, new_body)
    old_build = '"$PY" scripts/build_interface_repair_v1.py\nwrite_bootstrap runner_started'
    new_build = (
        '"$PY" scripts/build_evaluator_certification_sealed_v1.py\n'
        '"$PY" -m py_compile scripts/build_evaluator_certification_sealed_v1.py '
        'scripts/run_evaluator_sealed_certification_v1.py scripts/run_interface_repair_v3_evaluator.py '
        'src/hephaestus/control/evaluator_boundary.py src/hephaestus/control/evaluator_evidence_refs.py\n'
        'write_bootstrap runner_started'
    )
    if old_build not in shell:
        raise RuntimeError("V2 bootstrap pack-build block changed; refusing unsafe certification adaptation")
    return shell.replace(old_build, new_build)


def request_body(cfg: dict[str, Any], repo_sha: str, run_id: str, *, real_env: bool) -> dict[str, object]:
    original = base.pod_shell
    try:
        base.pod_shell = pod_shell
        body = base.request_body(cfg, repo_sha, run_id, "preflight", real_env=real_env)
    finally:
        base.pod_shell = original
    body["name"] = f"hephaestus-evaluator-sealed-certification-{run_id}"[:180]
    env = dict(body["env"])
    env["HEPHAESTUS_EVALUATOR_CERTIFICATION_RUN_ID"] = run_id
    body["env"] = env
    return body


def execute(cfg: dict[str, Any], marker: dict[str, Any]) -> dict[str, Any]:
    repo_sha = base.required("GITHUB_SHA")
    authorize(cfg, marker, repo_sha)
    github_run_id = base.required("GITHUB_RUN_ID")
    run_id = os.environ.get("HEPHAESTUS_EVALUATOR_CERTIFICATION_RUN_ID", f"evaluator-sealed-cert-{github_run_id}")
    prefix = f"{str(cfg['execution']['s3_prefix']).rstrip('/')}/{run_id}"
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
        print("EVALUATOR_SEALED_CERT_POD_JSON " + json.dumps({
            "pod_id": pod_id,
            "run_id": run_id,
            "gpu_allowlist": cfg["execution"]["gpu_type_ids"],
        }, sort_keys=True), flush=True)

        while True:
            elapsed = time.monotonic() - started
            if elapsed >= float(cfg["execution"]["hard_wall_seconds"]):
                raise TimeoutError("sealed certification total hard wall reached")
            snapshot = execution.get_pod(pod_id)
            metadata = base.safe_pod_metadata(snapshot)
            if metadata and not metadata_printed:
                record["pod_metadata"] = metadata
                metadata_printed = True
                print("EVALUATOR_SEALED_CERT_POD_METADATA_JSON " + json.dumps(metadata, sort_keys=True), flush=True)

            cost = snapshot.get("adjustedCostPerHr", snapshot.get("costPerHr"))
            if cost is not None:
                hourly = float(cost)
                estimated = hourly * elapsed / 3600.0
                record.update(cost_per_hr=hourly, estimated_cost_usd=estimated)
                if hourly > float(cfg["execution"]["max_hourly_usd_per_pod"]):
                    raise RuntimeError(f"sealed certification hourly cost ceiling exceeded: {hourly:.4f}")
                if estimated > float(cfg["execution"]["max_estimated_total_usd"]):
                    raise RuntimeError(f"sealed certification total cost ceiling exceeded: {estimated:.4f}")

            bootstrap = base.maybe_json(client, bootstrap_key)
            if bootstrap is not None:
                stage = str(bootstrap.get("stage", ""))
                if stage and stage != last_bootstrap_stage:
                    record["last_bootstrap"] = bootstrap
                    last_bootstrap_stage = stage
                    print("EVALUATOR_SEALED_CERT_BOOTSTRAP_JSON " + json.dumps(bootstrap, sort_keys=True), flush=True)

            progress = base.maybe_json(client, progress_key)
            if progress is not None:
                stage = str(progress.get("stage", ""))
                if stage and stage != last_progress_stage:
                    record["last_progress"] = progress
                    last_progress_stage = stage
                    print("EVALUATOR_SEALED_CERT_PROGRESS_JSON " + json.dumps(progress, sort_keys=True), flush=True)

            terminal = base.maybe_json(client, result_key)
            if terminal is not None:
                record["result"] = terminal
                status = str(terminal.get("status", ""))
                if status == "failed":
                    raise RuntimeError(f"sealed certification runner failed: {terminal.get('error_type')}: {terminal.get('error')}")
                if status in {"certified", "scientifically_rejected", "preflight_rejected"}:
                    record["status"] = status
                    return record
            if str(snapshot.get("desiredStatus", "")).upper() in base.TERMINAL:
                raise RuntimeError("sealed certification pod became terminal before result persistence")
            time.sleep(float(cfg["execution"]["poll_seconds"]))
    finally:
        if pod_id:
            teardown = base.delete_with_retries(execution, pod_id)
            record["teardown"] = teardown
            print("EVALUATOR_SEALED_CERT_TEARDOWN_JSON " + json.dumps(teardown, sort_keys=True), flush=True)
            if not teardown.get("verified_absent"):
                raise RuntimeError("sealed certification pod teardown could not be verified")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = load_json(CFG_PATH)
    marker = load_json(MARKER_PATH)
    repo_sha = os.environ.get("GITHUB_SHA", "<sha>")
    run_id = os.environ.get("HEPHAESTUS_EVALUATOR_CERTIFICATION_RUN_ID", f"evaluator-sealed-cert-{os.environ.get('GITHUB_RUN_ID', '<run>')}")
    if not args.execute:
        print(json.dumps({
            "run_id": run_id,
            "request": request_body(cfg, repo_sha, run_id, real_env=False),
            "execution": cfg["execution"],
            "governance": cfg["governance"],
            "launch_marker": marker,
        }, indent=2, sort_keys=True))
        return 0
    result = execute(cfg, marker)
    print("EVALUATOR_SEALED_CERT_LAUNCH_RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result.get("status") == "certified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
