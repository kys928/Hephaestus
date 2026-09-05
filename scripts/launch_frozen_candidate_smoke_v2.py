#!/usr/bin/env python3
"""Bounded RunPod smoke launcher with resilient immutable-model prefetch."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
from datetime import datetime, timezone
from pathlib import Path

from hephaestus.infrastructure.secrets import EnvironmentSecretsProvider
from hephaestus.providers.runpod import RunPodConfig, RunPodExecutionAdapter

import launch_positive_promotion_proof as launcher


def required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def main() -> int:
    model_id = required("HEPHAESTUS_CANDIDATE_MODEL_ID")
    revision = required("HEPHAESTUS_CANDIDATE_REVISION")
    license_name = required("HEPHAESTUS_CANDIDATE_LICENSE").lower()
    repo_sha = required("GITHUB_SHA")
    github_run_id = required("GITHUB_RUN_ID")
    proof_run_id = f"candidate-frozen-continuation-smoke-{github_run_id}"
    attempt = 1

    q_model = shlex.quote(model_id)
    q_revision = shlex.quote(revision)
    q_license = shlex.quote(license_name)
    shell = rf'''set -Eeuo pipefail
ATTEMPT_DIR="/workspace/hephaestus/scientific/v1/executions/${{HEPHAESTUS_PROOF_RUN_ID}}/attempt-${{HEPHAESTUS_ATTEMPT}}"
mkdir -p "$ATTEMPT_DIR"
exec >"$ATTEMPT_DIR/pod_runtime.log" 2>&1
write_failure() {{
  code=$?
  if [ "$code" -ne 0 ] && [ ! -f "$ATTEMPT_DIR/driver_result.json" ]; then
    python - "$ATTEMPT_DIR/driver_result.json" "$code" <<'PYFAIL'
import json, os, sys
from datetime import datetime, timezone
path, code = sys.argv[1], int(sys.argv[2])
payload = {{
    "result_version": "frozen-candidate-continuation-smoke.v2",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "proof_run_id": os.environ.get("HEPHAESTUS_PROOF_RUN_ID", "unknown"),
    "attempt": os.environ.get("HEPHAESTUS_ATTEMPT", "unknown"),
    "status": "pod_failed",
    "stage": "pod_shell",
    "exit_code": code,
    "repo_sha": os.environ.get("HEPHAESTUS_REPO_SHA", "unknown"),
}}
with open(path + ".partial", "w", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2, sort_keys=True)
os.replace(path + ".partial", path)
PYFAIL
  fi
}}
trap write_failure EXIT
apt-get update >/dev/null
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git ca-certificates python3-venv >/dev/null
rm -rf /opt/hephaestus-src /opt/hephaestus-venv
git clone --filter=blob:none https://github.com/kys928/Hephaestus.git /opt/hephaestus-src >/dev/null
cd /opt/hephaestus-src
git checkout "$HEPHAESTUS_REPO_SHA" >/dev/null
python -m venv --system-site-packages /opt/hephaestus-venv
PY=/opt/hephaestus-venv/bin/python
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=30
export HEPHAESTUS_CANDIDATE_MODEL_ID={q_model}
export HEPHAESTUS_CANDIDATE_REVISION={q_revision}
export HEPHAESTUS_CANDIDATE_LICENSE={q_license}
"$PY" -m pip install --disable-pip-version-check -e . 'transformers>=4.57,<6' 'tokenizers>=0.20,<1' 'safetensors>=0.4,<1' 'huggingface_hub>=0.26,<2' 'jinja2>=3.1,<4' >/dev/null

# Transport-only prefetch: resume the shared immutable cache with lower
# concurrency and a longer read timeout. No model/eval semantics are changed.
"$PY" - <<'PYPREFETCH'
import json, os, traceback
from datetime import datetime, timezone
from pathlib import Path
from huggingface_hub import snapshot_download

model_id = os.environ['HEPHAESTUS_CANDIDATE_MODEL_ID']
revision = os.environ['HEPHAESTUS_CANDIDATE_REVISION']
cache_dir = Path('/workspace/hephaestus/scientific/v1/model_cache/huggingface')
out = Path('/workspace/hephaestus/scientific/v1/executions') / os.environ['HEPHAESTUS_PROOF_RUN_ID'] / f"attempt-{{os.environ['HEPHAESTUS_ATTEMPT']}}" / 'driver_result.json'
try:
    snapshot = snapshot_download(
        repo_id=model_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=False,
        max_workers=2,
        allow_patterns=[
            'config.json', 'generation_config.json',
            'model.safetensors', 'model-*.safetensors', 'model.safetensors.index.json',
            'tokenizer.json', 'tokenizer_config.json', 'tokenizer.model',
            'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt',
            'chat_template.jinja', '*.jinja',
        ],
    )
    print(json.dumps({{'prefetch_status': 'completed', 'snapshot': snapshot, 'max_workers': 2, 'download_timeout': 300}}, sort_keys=True))
except Exception as exc:
    payload = {{
        'result_version': 'frozen-candidate-continuation-smoke.v2',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'proof_run_id': os.environ['HEPHAESTUS_PROOF_RUN_ID'],
        'attempt': os.environ['HEPHAESTUS_ATTEMPT'],
        'repo_sha': os.environ['HEPHAESTUS_REPO_SHA'],
        'status': 'pod_failed',
        'stage': 'model_prefetch',
        'model_id': model_id,
        'revision': revision,
        'error_type': type(exc).__name__,
        'error': str(exc),
        'traceback': traceback.format_exc(),
        'transport': {{'hf_hub_disable_xet': True, 'download_timeout': 300, 'etag_timeout': 30, 'max_workers': 2}},
    }}
    tmp = out.with_suffix('.json.partial')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(tmp, out)
    raise
PYPREFETCH

"$PY" -m py_compile scripts/run_frozen_candidate_smoke.py
"$PY" scripts/run_frozen_candidate_smoke.py
'''

    launcher.pod_shell = lambda: shell
    execution = RunPodExecutionAdapter(RunPodConfig.from_env(), EnvironmentSecretsProvider())
    client = launcher.base.s3_client()
    client.head_bucket(Bucket=launcher.VOLUME_ID)
    row: dict[str, object] = {
        "proof_run_id": proof_run_id,
        "repo_sha": repo_sha,
        "attempt": attempt,
        "model_id": model_id,
        "revision": revision,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "transport_hardening": {"hf_hub_disable_xet": True, "download_timeout": 300, "etag_timeout": 30, "max_workers": 2},
    }
    pod_id: str | None = None
    try:
        pod, capacity = launcher.create_pod(execution, proof_run_id=proof_run_id, repo_sha=repo_sha, attempt=attempt)
        pod_id = str(pod["id"])
        row["pod_id"] = pod_id
        row["capacity_selection"] = capacity
        result, observations = launcher.wait_for_result(
            client, execution, proof_run_id=proof_run_id, attempt=attempt, pod_id=pod_id
        )
        row["observations"] = observations
        key = f"{launcher.SCIENTIFIC_PREFIX}/executions/{proof_run_id}/attempt-{attempt}/driver_result.json"
        raw = launcher.base.read_key(client, key)
        observed = json.loads(raw.decode("utf-8"))
        if observed != result:
            raise RuntimeError("independent S3 readback differs from waiter result")
        row["s3_key"] = key
        row["s3_sha256"] = "sha256:" + hashlib.sha256(raw).hexdigest()
        row["verified_s3_readback"] = True
        row["result"] = result
        if result.get("status") != "completed":
            raise RuntimeError(f"candidate smoke did not complete: {result}")
        if result.get("model_id") != model_id or result.get("revision") != revision:
            raise RuntimeError("candidate identity drifted")
        if str(result.get("license") or "").lower() != license_name:
            raise RuntimeError("candidate license drifted")
        if result.get("frozen_eval_pack_hash") != launcher.EVAL_PACK_HASH:
            raise RuntimeError("frozen pack identity drifted")
        if result.get("sample_count") != 9:
            raise RuntimeError("candidate smoke did not produce nine samples")
        row["all_hard_gates_passed"] = bool(result.get("all_hard_gates_passed"))
    finally:
        if pod_id:
            row["teardown"] = launcher.base.delete_pod(execution, pod_id)
        row["completed_at"] = datetime.now(timezone.utc).isoformat()
        Path("frozen_candidate_smoke.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "proof_run_id": proof_run_id,
        "model_id": model_id,
        "revision": revision,
        "verified_s3_readback": row.get("verified_s3_readback", False),
        "all_hard_gates_passed": row.get("all_hard_gates_passed", False),
        "hard_gate_summary": (row.get("result") or {}).get("hard_gate_summary") if isinstance(row.get("result"), dict) else None,
    }, sort_keys=True))
    return 0 if row.get("all_hard_gates_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
