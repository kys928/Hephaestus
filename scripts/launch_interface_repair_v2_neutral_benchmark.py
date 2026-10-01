#!/usr/bin/env python3
"""Guarded launcher for the frozen V2 neutral parent benchmark."""
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
CFG_PATH = ROOT / "configs/experiments/hephaestus_interface_repair_v2_neutral_benchmark.json"
MARKER_PATH = ROOT / "configs/experiments/interface_repair_v2_neutral_benchmark.launch.json"
AUTH_ENV = "HEPHAESTUS_INTERFACE_REPAIR_V2_NEUTRAL_AUTHORIZED"
AUTH_PHRASE = "LAUNCH_INTERFACE_REPAIR_V2_NEUTRAL_BENCHMARK"
ORIGINAL_POD_SHELL = base.pod_shell
ALLOWED_POST_PREP_CHANGED_FILES = {str(CFG_PATH.relative_to(ROOT)), str(MARKER_PATH.relative_to(ROOT))}


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
        raise RuntimeError("V2 neutral benchmark is not bound to a validated SHA")
    if subprocess.run(["git","merge-base","--is-ancestor",prepared_sha,current_repo_sha], cwd=ROOT, check=False).returncode != 0:
        raise RuntimeError("validated SHA is not an ancestor of launch commit")
    changed = {x.strip() for x in subprocess.check_output(["git","diff","--name-only",f"{prepared_sha}..{current_repo_sha}"], cwd=ROOT, text=True).splitlines() if x.strip()}
    unexpected = sorted(changed - ALLOWED_POST_PREP_CHANGED_FILES)
    if unexpected:
        raise RuntimeError(f"benchmark code/science changed after validation: {unexpected}")
    old_cfg = json.loads(subprocess.check_output(["git","show",f"{prepared_sha}:{CFG_PATH.relative_to(ROOT)}"], cwd=ROOT, text=True))
    if _normalized_cfg(old_cfg) != _normalized_cfg(cfg):
        raise RuntimeError("benchmark protocol changed after validation")


def authorize(cfg: dict[str, Any], marker: dict[str, Any], repo_sha: str) -> None:
    g = cfg["governance"]
    if g.get("paid_launch_allowed") is not True or marker.get("launch_authorized") is not True:
        raise RuntimeError("paid V2 neutral benchmark is locked")
    if marker.get("authorization_phrase") != AUTH_PHRASE or base.required(AUTH_ENV) != AUTH_PHRASE:
        raise RuntimeError("benchmark authorization phrase mismatch")
    if g.get("training_allowed") is not False or g.get("adapter_mutation_allowed") is not False:
        raise RuntimeError("benchmark may not train or mutate")
    if g.get("production_promotion_allowed") or g.get("automatic_role_dispatch_allowed") or g.get("certification_claim_allowed"):
        raise RuntimeError("benchmark may not promote, dispatch, or certify")
    verify_prepared_binding(cfg, marker, repo_sha)


def pod_shell(_mode: str, venv_cache: str, pip_cache: str, adapter_cache: str) -> str:
    shell = ORIGINAL_POD_SHELL("preflight", venv_cache, pip_cache, adapter_cache)
    old_body = 'export HEPHAESTUS_REPAIR_ROLE=planner\n"$PY" scripts/run_interface_repair_v2_cached.py --preflight'
    new_body = '"$PY" scripts/run_interface_repair_v2_neutral_benchmark.py'
    if old_body not in shell:
        raise RuntimeError("V2 bootstrap body changed")
    shell = shell.replace(old_body, new_body)
    old_build = '"$PY" scripts/build_interface_repair_v1.py\nwrite_bootstrap runner_started'
    new_build = '"$PY" scripts/build_interface_repair_v3_1_neutral_holdout.py\n"$PY" -m py_compile scripts/run_interface_repair_v2_neutral_benchmark.py\nwrite_bootstrap runner_started'
    if old_build not in shell:
        raise RuntimeError("V2 bootstrap pack-build block changed")
    return shell.replace(old_build, new_build)


def request_body(cfg: dict[str, Any], repo_sha: str, run_id: str, *, real_env: bool) -> dict[str, object]:
    original = base.pod_shell
    try:
        base.pod_shell = pod_shell
        body = base.request_body(cfg, repo_sha, run_id, "preflight", real_env=real_env)
    finally:
        base.pod_shell = original
    body["name"] = f"hephaestus-v2-neutral-benchmark-{run_id}"[:180]
    return body


def execute(cfg: dict[str, Any], marker: dict[str, Any]) -> dict[str, Any]:
    repo_sha = base.required("GITHUB_SHA")
    authorize(cfg, marker, repo_sha)
    run_id = os.environ.get("HEPHAESTUS_INTERFACE_REPAIR_RUN_ID", f"v2-neutral-benchmark-{base.required('GITHUB_RUN_ID')}")
    prefix = f"{str(cfg['execution']['s3_prefix']).rstrip('/')}/{run_id}"
    result_key = f"{prefix}/result.json"
    progress_key = f"{prefix}/progress.json"
    bootstrap_key = f"{prefix}/bootstrap/progress.json"
    client = base.s3_client()
    client.head_bucket(Bucket=base.required("RUNPOD_NETWORK_VOLUME_ID"))
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
    body = request_body(cfg, repo_sha, run_id, real_env=True)
    pod_id = None
    started = None
    record: dict[str, Any] = {"run_id": run_id, "repo_sha": repo_sha, "status": "starting", "result_key": result_key}
    try:
        pod = base.create_with_capacity_retries(execution, body, int(cfg["execution"]["capacity_retry_attempts"]), float(cfg["execution"]["capacity_retry_seconds"]))
        pod_id = str(pod["id"])
        started = time.monotonic()
        record["pod_id"] = pod_id
        print("INTERFACE_REPAIR_V2_NEUTRAL_POD_JSON " + json.dumps({"pod_id":pod_id,"run_id":run_id},sort_keys=True), flush=True)
        last_stage = None
        last_boot = None
        meta_printed = False
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= float(cfg["execution"]["hard_wall_seconds"]):
                raise TimeoutError("V2 neutral benchmark hard wall reached")
            snap = execution.get_pod(pod_id)
            if not meta_printed:
                meta = base.safe_pod_metadata(snap)
                if meta:
                    record["pod_metadata"] = meta
                    print("INTERFACE_REPAIR_V2_NEUTRAL_POD_METADATA_JSON " + json.dumps(meta,sort_keys=True), flush=True)
                    meta_printed = True
            cost = snap.get("adjustedCostPerHr", snap.get("costPerHr"))
            if cost is not None:
                hourly = float(cost)
                estimated = hourly * elapsed / 3600.0
                record.update(cost_per_hr=hourly, estimated_cost_usd=estimated)
                if hourly > float(cfg["execution"]["max_hourly_usd_per_pod"]):
                    raise RuntimeError("hourly cost ceiling exceeded")
                if estimated > float(cfg["execution"]["max_estimated_total_usd"]):
                    raise RuntimeError("total cost ceiling exceeded")
            boot = base.maybe_json(client, bootstrap_key)
            if boot is not None and str(boot.get("stage")) != last_boot:
                last_boot = str(boot.get("stage"))
                print("INTERFACE_REPAIR_V2_NEUTRAL_BOOTSTRAP_JSON " + json.dumps(boot,sort_keys=True), flush=True)
            prog = base.maybe_json(client, progress_key)
            if prog is not None and str(prog.get("stage")) != last_stage:
                last_stage = str(prog.get("stage"))
                record["last_progress"] = prog
                print("INTERFACE_REPAIR_V2_NEUTRAL_PROGRESS_JSON " + json.dumps(prog,sort_keys=True), flush=True)
            terminal = base.maybe_json(client, result_key)
            if terminal is not None:
                record["result"] = terminal
                if terminal.get("status") == "failed":
                    raise RuntimeError(f"runner failed: {terminal.get('error_type')}: {terminal.get('error')}")
                if terminal.get("status") == "completed":
                    record["status"] = "completed"
                    return record
            if str(snap.get("desiredStatus","")).upper() in base.TERMINAL:
                raise RuntimeError("pod became terminal before result persistence")
            time.sleep(float(cfg["execution"]["poll_seconds"]))
    finally:
        if pod_id:
            teardown = base.delete_with_retries(execution, pod_id)
            record["teardown"] = teardown
            print("INTERFACE_REPAIR_V2_NEUTRAL_TEARDOWN_JSON " + json.dumps(teardown,sort_keys=True), flush=True)
            if not teardown.get("verified_absent"):
                raise RuntimeError("pod teardown could not be verified")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg, marker = load_json(CFG_PATH), load_json(MARKER_PATH)
    repo_sha = os.environ.get("GITHUB_SHA", "<sha>")
    run_id = os.environ.get("HEPHAESTUS_INTERFACE_REPAIR_RUN_ID", "v2-neutral-benchmark-<run>")
    if not args.execute:
        print(json.dumps({"run_id":run_id,"request":request_body(cfg,repo_sha,run_id,real_env=False),"execution":cfg["execution"],"governance":cfg["governance"],"launch_marker":marker}, indent=2, sort_keys=True))
        return 0
    result = execute(cfg, marker)
    print("INTERFACE_REPAIR_V2_NEUTRAL_LAUNCH_RESULT_JSON " + json.dumps(result,sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
