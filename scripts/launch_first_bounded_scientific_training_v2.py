#!/usr/bin/env python3
"""Run the existing bounded RunPod launcher with a PEP-668-safe Pod bootstrap."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SOURCE = Path(__file__).with_name("launch_first_bounded_scientific_training.py")
spec = importlib.util.spec_from_file_location("hephaestus_first_training_launcher_v1", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load first scientific training launcher")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def _pod_shell() -> str:
    return r'''set -Eeuo pipefail
RUN_DIR="/workspace/hephaestus/scientific/v1/executions/${HEPHAESTUS_RUN_ID}"
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
    "result_version": "first-bounded-scientific-training-result.v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "run_id": os.environ.get("HEPHAESTUS_RUN_ID", "unknown"),
    "status": "pod_bootstrap_failed",
    "exit_code": code,
    "repo_sha": os.environ.get("HEPHAESTUS_REPO_SHA", "unknown"),
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
assert torch.cuda.is_available(), "CUDA unavailable after venv bootstrap"
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
PYCHECK
"$PY" -m py_compile scripts/run_first_bounded_scientific_training.py scripts/run_first_bounded_scientific_training_v2.py
"$PY" scripts/run_first_bounded_scientific_training_v2.py
'''


module.pod_shell = _pod_shell

if __name__ == "__main__":
    raise SystemExit(module.main())
