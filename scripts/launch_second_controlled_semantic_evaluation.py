#!/usr/bin/env python3
"""Launch and independently verify the frozen semantic comparison for the second controlled experiment."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter

import launch_first_bounded_scientific_training as base

VOLUME_ID = base.VOLUME_ID
DATACENTER_ID = base.DATACENTER_ID
SCIENTIFIC_PREFIX = base.SCIENTIFIC_PREFIX
IMAGE = base.IMAGE
EXPERIMENT_ID = "experiment-d0e911d6bd1fb7ae"
PRIMARY_VARIABLE = "dataset_mixture"
BASELINE_RUN_ID = "first-bounded-scientific-training-001-33866198758"
BASELINE_CHECKPOINT_HASH = "sha256:7a6be1e0cee47f29d5dd47d41bc01beed066c4de64e24ee18544ff4edcb3f4c3"
CANDIDATE_RUN_ID = "planned-run-b8e558e54effac85"
EVAL_PACK_HASH = "ee4acffa6d6ac3dadd1705931d65fc02bc4206f2fbddacf71b25af4d1cb5e3ad"
APPROVAL_REF = "approval://operator/chat-2026-09-04-second-controlled-experiment"
MAX_SECONDS = 1200
POLL_SECONDS = 5


def pod_shell() -> str:
    return r'''set -Eeuo pipefail
RUN_DIR="/workspace/hephaestus/scientific/v1/executions/${HEPHAESTUS_EVAL_RUN_ID}"
mkdir -p "$RUN_DIR"
exec >"$RUN_DIR/pod_runtime.log" 2>&1
write_bootstrap_failure() {
  code=$?
  if [ "$code" -ne 0 ] && [ ! -f "$RUN_DIR/driver_result.json" ]; then
    python - "$RUN_DIR/driver_result.json" "$code" <<'PYFAIL'
import json, os, sys
from datetime import datetime, timezone
path, code = sys.argv[1], int(sys.argv[2])
payload = {
    "result_version": "second-controlled-semantic-evaluation-result.v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_id": os.environ.get("HEPHAESTUS_EVAL_RUN_ID", "unknown"),
    "status": "pod_bootstrap_failed",
    "exit_code": code,
    "repo_sha": os.environ.get("HEPHAESTUS_REPO_SHA", "unknown"),
    "training_performed": False,
    "action_applied": False,
}
with open(path + ".partial", "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
os.replace(path + ".partial", path)
PYFAIL
  fi
}
trap write_bootstrap_failure EXIT
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates python3-venv
rm -rf /opt/hephaestus-src /opt/hephaestus-venv
git clone --filter=blob:none https://github.com/kys928/Hephaestus.git /opt/hephaestus-src
cd /opt/hephaestus-src
git checkout "$HEPHAESTUS_REPO_SHA"
python -m venv --system-site-packages /opt/hephaestus-venv
PY=/opt/hephaestus-venv/bin/python
"$PY" -m pip install --disable-pip-version-check -e . 'transformers>=4.46,<6' 'tokenizers>=0.20,<1' 'safetensors>=0.4,<1'
"$PY" - <<'PYCHECK'
import torch
assert torch.cuda.is_available(), "CUDA unavailable after controlled-evaluation venv bootstrap"
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
PYCHECK
"$PY" -m py_compile src/hephaestus/control/semantic_judge.py scripts/run_second_controlled_semantic_evaluation.py
"$PY" scripts/run_second_controlled_semantic_evaluation.py
'''


def create_pod(
    execution: RunPodExecutionAdapter,
    eval_run_id: str,
    repo_sha: str,
    candidate_checkpoint_hash: str,
) -> dict[str, Any]:
    return execution.create_bounded_gpu_pod(
        name=f"hephaestus-{eval_run_id}"[:180],
        image_name=IMAGE,
        gpu_type_ids=[
            "NVIDIA GeForce RTX 3070",
            "NVIDIA GeForce RTX 3080",
            "NVIDIA GeForce RTX 3090",
            "NVIDIA L4",
            "NVIDIA GeForce RTX 4090",
        ],
        docker_start_cmd=["bash", "-lc", pod_shell()],
        env={
            "HEPHAESTUS_EVAL_RUN_ID": eval_run_id,
            "HEPHAESTUS_REPO_SHA": repo_sha,
            "HEPHAESTUS_CANDIDATE_CHECKPOINT_HASH": candidate_checkpoint_hash,
            "HEPHAESTUS_OPERATOR_APPROVAL_REF": APPROVAL_REF,
        },
        container_disk_in_gb=20,
        interruptible=False,
    )


def wait_for_result(
    client: Any,
    execution: RunPodExecutionAdapter,
    eval_run_id: str,
    pod_id: str,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    key = f"{SCIENTIFIC_PREFIX}/executions/{eval_run_id}/driver_result.json"
    deadline = time.monotonic() + MAX_SECONDS
    observations: list[dict[str, object]] = []
    last_status: str | None = None
    while time.monotonic() < deadline:
        raw = base.maybe_read_key(client, key)
        if raw is not None:
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("controlled evaluation terminal record is not a JSON object")
            return payload, observations
        pod = base.pod_snapshot(execution, pod_id)
        status = str(pod.get("desiredStatus", "unknown")) if pod else "unknown"
        if status != last_status:
            observations.append({"at": datetime.now(timezone.utc).isoformat(), "desired_status": status})
            last_status = status
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"controlled semantic evaluation result did not appear within {MAX_SECONDS} seconds")


def list_prefix(client: Any, prefix: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    token: str | None = None
    while True:
        kwargs: dict[str, object] = {"Bucket": VOLUME_ID, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        for item in page.get("Contents", []) or []:
            rows.append({"key": str(item["Key"]), "size": int(item.get("Size", 0))})
        if not page.get("IsTruncated"):
            break
        token = str(page.get("NextContinuationToken") or "")
    rows.sort(key=lambda item: str(item["key"]))
    return rows


def volume_key(path: str) -> str:
    if not path.startswith("/workspace/"):
        raise RuntimeError(f"evidence path is not volume-local: {path}")
    return path[len("/workspace/"):]


def verify_evaluation(
    client: Any,
    eval_run_id: str,
    candidate_checkpoint_hash: str,
    result: dict[str, Any],
) -> dict[str, object]:
    if result.get("status") != "completed":
        raise RuntimeError(f"controlled evaluation did not complete: {result.get('status')}: {result.get('error', '')}")
    if result.get("training_performed") is not False or result.get("action_applied") is not False:
        raise RuntimeError("controlled evaluation crossed a forbidden training/action boundary")
    if result.get("experiment_id") != EXPERIMENT_ID or result.get("primary_variable") != PRIMARY_VARIABLE:
        raise RuntimeError("controlled evaluation experiment identity drifted")
    if result.get("baseline_training_run_id") != BASELINE_RUN_ID or result.get("candidate_training_run_id") != CANDIDATE_RUN_ID:
        raise RuntimeError("controlled evaluation baseline/candidate identity drifted")

    inputs = result.get("verified_inputs")
    if not isinstance(inputs, dict):
        raise RuntimeError("controlled evaluation terminal record lacks verified inputs")
    if inputs.get("eval_pack_content_hash") != EVAL_PACK_HASH:
        raise RuntimeError("frozen evaluation pack identity drifted")
    if inputs.get("baseline_checkpoint_manifest_hash") != BASELINE_CHECKPOINT_HASH:
        raise RuntimeError("previous trained baseline checkpoint identity drifted")
    if inputs.get("candidate_checkpoint_manifest_hash") != candidate_checkpoint_hash:
        raise RuntimeError("controlled candidate checkpoint identity drifted")
    if int(inputs.get("task_seed_count", 0)) != 18:
        raise RuntimeError("frozen task/seed plan is incomplete")

    baseline_path = f"/workspace/{SCIENTIFIC_PREFIX}/runs/{BASELINE_RUN_ID}/checkpoint_step_100"
    candidate_path = f"/workspace/{SCIENTIFIC_PREFIX}/runs/{CANDIDATE_RUN_ID}/checkpoint_step_100"
    baseline = base.verify_checkpoint(client, baseline_path)
    candidate = base.verify_checkpoint(client, candidate_path)
    if baseline["checkpoint_manifest_hash"] != BASELINE_CHECKPOINT_HASH:
        raise RuntimeError("independent baseline checkpoint verification disagrees")
    if candidate["checkpoint_manifest_hash"] != candidate_checkpoint_hash:
        raise RuntimeError("independent candidate checkpoint verification disagrees")

    evidence_files = result.get("evidence_files")
    if not isinstance(evidence_files, dict):
        raise RuntimeError("controlled evaluation lacks evidence-file identities")
    verified_files: dict[str, object] = {}
    for name, item in sorted(evidence_files.items()):
        if not isinstance(item, dict):
            raise RuntimeError(f"evidence file metadata is invalid: {name}")
        verified_files[name] = base.verify_key(
            client,
            volume_key(str(item.get("path") or "")),
            str(item.get("sha256") or ""),
            int(item.get("bytes", 0)),
        )

    eval_prefix = f"{SCIENTIFIC_PREFIX}/evaluations/{eval_run_id}/"
    objects = list_prefix(client, eval_prefix)
    samples: list[dict[str, object]] = []
    for item in objects:
        key = str(item["key"])
        if not key.endswith(".json") or "/semantic_generation/" not in key:
            continue
        raw = base.read_key(client, key)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or not payload.get("sample_id"):
            continue
        output = str(payload.get("output") or "")
        output_hash = f"sha256:{hashlib.sha256(output.encode('utf-8')).hexdigest()}"
        if payload.get("output_hash") != output_hash:
            raise RuntimeError(f"semantic sample output hash mismatch: {key}")
        samples.append(
            {
                "key": key,
                "sample_id": payload.get("sample_id"),
                "run_id": payload.get("run_id"),
                "task_id": payload.get("task_id"),
                "seed": payload.get("seed"),
                "output_hash": output_hash,
                "bytes": len(raw),
            }
        )
    if len(samples) != 36:
        raise RuntimeError(f"expected 36 independently readable semantic samples, observed {len(samples)}")
    by_run: dict[str, int] = {}
    for sample in samples:
        name = str(sample.get("run_id") or "")
        by_run[name] = by_run.get(name, 0) + 1
    if by_run != {BASELINE_RUN_ID: 18, CANDIDATE_RUN_ID: 18}:
        raise RuntimeError(f"semantic sample baseline/candidate split is invalid: {by_run}")

    comparison = json.loads(base.read_key(client, f"{eval_prefix}experiment_comparison.json").decode("utf-8"))
    judge = json.loads(base.read_key(client, f"{eval_prefix}judge_exit.json").decode("utf-8"))
    terminal_comparison = result.get("comparison")
    terminal_judge = result.get("judge_exit")
    if not isinstance(terminal_comparison, dict) or comparison.get("primary_outcome") != terminal_comparison.get("primary_outcome"):
        raise RuntimeError("comparison readback disagrees with terminal result")
    if not isinstance(terminal_judge, dict) or judge.get("next_action") != terminal_judge.get("next_action"):
        raise RuntimeError("Judge readback disagrees with terminal result")

    return {
        "verification_version": "second-controlled-semantic-evaluation-verification.v1",
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "status": "verified",
        "run_id": eval_run_id,
        "experiment_id": EXPERIMENT_ID,
        "primary_variable": PRIMARY_VARIABLE,
        "volume_id": VOLUME_ID,
        "datacenter_id": DATACENTER_ID,
        "baseline_checkpoint": baseline,
        "candidate_checkpoint": candidate,
        "evidence_files": verified_files,
        "sample_count": len(samples),
        "sample_counts_by_run": by_run,
        "sample_inventory_hash": base.canonical_hash(samples),
        "evaluation_object_count": len(objects),
        "evaluation_total_bytes": sum(int(item["size"]) for item in objects),
        "evaluation_inventory_hash": base.canonical_hash(objects),
        "evaluation_objects": objects,
        "comparison": comparison,
        "judge_exit": judge,
        "terminal_result": result,
    }


def atomic_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    base.required("RUNPOD_API_KEY")
    repo_sha = base.required("GITHUB_SHA")
    workflow_run = os.environ.get("GITHUB_RUN_ID", str(int(time.time())))
    eval_run_id = f"second-controlled-semantic-evaluation-001-{workflow_run}"
    client = base.s3_client()
    client.head_bucket(Bucket=VOLUME_ID)

    baseline_preflight = base.verify_checkpoint(
        client,
        f"/workspace/{SCIENTIFIC_PREFIX}/runs/{BASELINE_RUN_ID}/checkpoint_step_100",
    )
    if baseline_preflight["checkpoint_manifest_hash"] != BASELINE_CHECKPOINT_HASH:
        raise RuntimeError("preflight previous trained baseline checkpoint hash mismatch")
    candidate_preflight = base.verify_checkpoint(
        client,
        f"/workspace/{SCIENTIFIC_PREFIX}/runs/{CANDIDATE_RUN_ID}/checkpoint_step_100",
    )
    candidate_hash = str(candidate_preflight["checkpoint_manifest_hash"])
    if candidate_hash == BASELINE_CHECKPOINT_HASH:
        raise RuntimeError("controlled candidate checkpoint unexpectedly equals previous baseline")

    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
    started = time.monotonic()
    created_at = datetime.now(timezone.utc).isoformat()
    pod: dict[str, Any] | None = None
    pod_id: str | None = None
    final_pod: dict[str, Any] | None = None
    teardown: dict[str, object] = {"deleted": False, "not_created": True}
    result: dict[str, Any] | None = None
    verification: dict[str, object] | None = None
    observations: list[dict[str, object]] = []
    launcher_error: str | None = None
    funds_unavailable = False

    try:
        pod = create_pod(execution, eval_run_id, repo_sha, candidate_hash)
        pod_id = str(pod["id"])
        teardown = {"deleted": False, "not_created": False}
        result, observations = wait_for_result(client, execution, eval_run_id, pod_id)
        final_pod = base.pod_snapshot(execution, pod_id)
        verification = verify_evaluation(client, eval_run_id, candidate_hash, result)
    except BaseException as exc:
        launcher_error = f"{type(exc).__name__}: {exc}"
        lowered = launcher_error.lower()
        funds_unavailable = "http 402" in lowered or any(
            phrase in lowered
            for phrase in ("insufficient funds", "insufficient credit", "insufficient balance", "not enough credit")
        )
        if pod_id:
            final_pod = base.pod_snapshot(execution, pod_id)
    finally:
        if pod_id:
            teardown = base.delete_pod(execution, pod_id)

    elapsed = time.monotonic() - started
    pod_view = final_pod or pod or {}
    cost_per_hr = pod_view.get("adjustedCostPerHr", pod_view.get("costPerHr")) if isinstance(pod_view, dict) else None
    try:
        estimated_cost = float(cost_per_hr) * elapsed / 3600.0 if cost_per_hr is not None else None
    except (TypeError, ValueError):
        estimated_cost = None

    launcher = {
        "launcher_version": "second-controlled-semantic-evaluation-launcher.v1",
        "created_at": created_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "run_id": eval_run_id,
        "experiment_id": EXPERIMENT_ID,
        "repo_sha": repo_sha,
        "volume_id": VOLUME_ID,
        "datacenter_id": DATACENTER_ID,
        "image": IMAGE,
        "operator_approval_ref": APPROVAL_REF,
        "primary_variable": PRIMARY_VARIABLE,
        "baseline_training_run_id": BASELINE_RUN_ID,
        "candidate_training_run_id": CANDIDATE_RUN_ID,
        "baseline_checkpoint_verified_before_launch": baseline_preflight,
        "candidate_checkpoint_verified_before_launch": candidate_preflight,
        "candidate_checkpoint_manifest_hash": candidate_hash,
        "pod_id": pod_id,
        "pod": pod_view,
        "pod_status_observations": observations,
        "elapsed_seconds": elapsed,
        "cost_per_hour": cost_per_hr,
        "estimated_cost_for_observed_elapsed_time": estimated_cost,
        "estimated_cost_is_not_billing_record": True,
        "funds_unavailable": funds_unavailable,
        "teardown": teardown,
        "terminal_result": result,
        "verification_status": verification.get("status") if verification else None,
        "error": launcher_error,
    }
    atomic_json(Path("second_controlled_semantic_evaluation_launcher.json"), launcher)
    if verification is not None:
        verification["launcher"] = {
            "pod_id": pod_id,
            "repo_sha": repo_sha,
            "elapsed_seconds": elapsed,
            "cost_per_hour": cost_per_hr,
            "estimated_cost_for_observed_elapsed_time": estimated_cost,
            "teardown": teardown,
        }
        atomic_json(Path("second_controlled_semantic_evaluation_verification.json"), verification)

    if launcher_error:
        print(json.dumps(launcher, indent=2, sort_keys=True))
        return 1
    if not verification or verification.get("status") != "verified":
        print(json.dumps(launcher, indent=2, sort_keys=True))
        return 1
    comparison = verification["comparison"]
    judge = verification["judge_exit"]
    print(
        json.dumps(
            {
                "status": "verified",
                "run_id": eval_run_id,
                "candidate_checkpoint_manifest_hash": candidate_hash,
                "outcome": comparison.get("primary_outcome") if isinstance(comparison, dict) else None,
                "deterministic_gate_status": comparison.get("deterministic_gate_status") if isinstance(comparison, dict) else None,
                "judge_action": judge.get("next_action") if isinstance(judge, dict) else None,
                "pod_deleted": teardown.get("deleted"),
                "funds_unavailable": funds_unavailable,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
