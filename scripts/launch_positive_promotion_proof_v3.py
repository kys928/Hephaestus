#!/usr/bin/env python3
"""RunPod launcher/verifier for the evidence-selected V3 promotion wave."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import launch_positive_promotion_proof as launcher

ALLOWED_REVISION_LICENSES = {
    "2db69c1c3e91a05d2c64a3185acfbaf36f744e25": "mit",
    "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8": "apache-2.0",
}
ALLOWED_REVISIONS = set(ALLOWED_REVISION_LICENSES)
V3_MAX_SECONDS = 6000

# Dual-3090 inference can be slower than the original single high-memory GPU path.
# This extends only the infrastructure wait budget; scientific evaluation stays fixed.
launcher.MAX_SECONDS = V3_MAX_SECONDS


def pod_shell_v3() -> str:
    return r'''set -Eeuo pipefail
ATTEMPT_DIR="/workspace/hephaestus/scientific/v1/executions/${HEPHAESTUS_PROOF_RUN_ID}/attempt-${HEPHAESTUS_ATTEMPT}"
mkdir -p "$ATTEMPT_DIR"
# Keep immutable model downloads and Xet reconstruction scratch on the Pod's
# enlarged ephemeral container disk. The mounted network volume is reserved for
# proof evidence/state, avoiding its persistent storage quota. This remains cache
# storage only: the FP16 model itself is still sharded exclusively across 2 GPUs.
export HF_HOME="/opt/hephaestus-cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_XET_CACHE="$HF_HOME/xet"
export XDG_CACHE_HOME="/opt/hephaestus-cache/xdg"
export TMPDIR="/opt/hephaestus-tmp/${HEPHAESTUS_PROOF_RUN_ID}/attempt-${HEPHAESTUS_ATTEMPT}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_XET_CACHE" "$XDG_CACHE_HOME" "$TMPDIR"
exec >"$ATTEMPT_DIR/pod_runtime.log" 2>&1
write_bootstrap_failure() {
  code=$?
  if [ "$code" -ne 0 ] && [ ! -f "$ATTEMPT_DIR/driver_result.json" ]; then
    python - "$ATTEMPT_DIR/driver_result.json" "$code" <<'PYFAIL'
import json, os, sys
from datetime import datetime, timezone
path, code = sys.argv[1], int(sys.argv[2])
payload = {
    "result_version": "positive-real-model-promotion-proof.v3",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "proof_run_id": os.environ.get("HEPHAESTUS_PROOF_RUN_ID", "unknown"),
    "attempt": os.environ.get("HEPHAESTUS_ATTEMPT", "unknown"),
    "status": "pod_bootstrap_failed",
    "exit_code": code,
    "repo_sha": os.environ.get("HEPHAESTUS_REPO_SHA", "unknown"),
    "training_performed": False,
    "original_research_lineage_mutated": False,
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
rm -rf /var/lib/apt/lists/*
rm -rf /opt/hephaestus-src /opt/hephaestus-venv
git clone --filter=blob:none https://github.com/kys928/Hephaestus.git /opt/hephaestus-src
cd /opt/hephaestus-src
git checkout "$HEPHAESTUS_REPO_SHA"
python -m venv --system-site-packages /opt/hephaestus-venv
PY=/opt/hephaestus-venv/bin/python
"$PY" -m pip install --no-cache-dir --disable-pip-version-check -e . 'transformers>=4.47,<6' 'accelerate>=1,<2' 'tokenizers>=0.20,<1' 'safetensors>=0.4,<1' 'huggingface_hub>=0.26,<2' 'hf_xet>=1,<2'
"$PY" - <<'PYCHECK'
import json
import torch

assert torch.cuda.is_available(), "CUDA unavailable after positive-proof bootstrap"
count = torch.cuda.device_count()
assert count >= 2, f"V3 dual-GPU proof requires at least 2 CUDA devices; found {count}"
gpus = []
for index in range(2):
    name = torch.cuda.get_device_name(index)
    total_gib = torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
    assert "3090" in name, f"V3 expected RTX 3090 at cuda:{index}; found {name}"
    assert total_gib >= 23.0, f"V3 expected ~24GB VRAM at cuda:{index}; found {total_gib:.2f} GiB"
    gpus.append({"index": index, "name": name, "total_memory_gib": round(total_gib, 2)})
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu_count": count, "gpus": gpus}))
PYCHECK
"$PY" -m py_compile scripts/run_positive_promotion_proof.py scripts/run_positive_promotion_proof_v2.py scripts/run_positive_promotion_proof_v3.py
"$PY" scripts/run_positive_promotion_proof_v3.py
'''


def verify_proof_v3(client: Any, proof_run_id: str, result: dict[str, Any]) -> dict[str, object]:
    if result.get("status") != "completed":
        raise RuntimeError(f"positive proof did not complete: {result.get('status')}: {result.get('error', '')}")
    if result.get("training_performed") is not False:
        raise RuntimeError("positive promotion proof unexpectedly performed training")
    if result.get("original_research_lineage_mutated") is not False:
        raise RuntimeError("positive promotion proof reports original research lineage mutation")

    verification_path = str(result.get("independent_verification_ref") or "")
    expected_verification_hash = str(result.get("independent_verification_sha256") or "")
    raw_verification = launcher.base.read_key(client, launcher.volume_key(verification_path))
    observed_hash = f"sha256:{hashlib.sha256(raw_verification).hexdigest()}"
    if observed_hash != expected_verification_hash:
        raise RuntimeError("independent verification S3 hash disagrees with terminal result")
    verification = json.loads(raw_verification.decode("utf-8"))
    if not isinstance(verification, dict) or verification.get("status") != "verified":
        raise RuntimeError("independent verification record is not verified")
    if verification.get("frozen_eval_pack_hash") != launcher.EVAL_PACK_HASH:
        raise RuntimeError("positive proof frozen eval-pack identity drifted")
    if verification.get("operator_approval_ref") != launcher.APPROVAL_REF:
        raise RuntimeError("positive proof operator approval reference drifted")

    lineage = verification.get("lineage")
    manifest = verification.get("certified_model_manifest")
    if not isinstance(lineage, dict) or not isinstance(manifest, dict):
        raise RuntimeError("positive proof verification lacks lineage/model manifest")
    certified = str(verification.get("certified_checkpoint_ref") or "")
    if not certified:
        raise RuntimeError("positive proof has no certified checkpoint")
    if lineage.get("certified_stable_checkpoint_ref") != certified:
        raise RuntimeError("lineage certified checkpoint differs from verification")
    if lineage.get("best_checkpoint_ref") != certified or lineage.get("last_stable_checkpoint_ref") != certified:
        raise RuntimeError("lineage best/stable/certified refs disagree")
    if lineage.get("last_certification_result") != "certification_passed":
        raise RuntimeError("lineage certification state is not passed")

    revision = str(manifest.get("revision") or "")
    expected_license = ALLOWED_REVISION_LICENSES.get(revision)
    if manifest.get("status") != "verified" or expected_license is None:
        raise RuntimeError("certified model immutable revision is unexpected")
    if str(manifest.get("license") or "").lower() != expected_license:
        raise RuntimeError("certified model license disagrees with the immutable V3 allowlist")

    proof_result_key = f"{launcher.SCIENTIFIC_PREFIX}/positive_promotion/{proof_run_id}/proof_result.json"
    proof_raw = launcher.base.read_key(client, proof_result_key)
    proof_payload = json.loads(proof_raw.decode("utf-8"))
    if not isinstance(proof_payload, dict) or proof_payload.get("certified_checkpoint_ref") != certified:
        raise RuntimeError("proof-result readback differs from independent verification")

    return {
        "verification_version": "positive-real-model-promotion-launcher-verification.v3",
        "status": "verified",
        "proof_run_id": proof_run_id,
        "volume_id": launcher.VOLUME_ID,
        "datacenter_id": launcher.DATACENTER_ID,
        "proof_result": {
            "key": proof_result_key,
            "sha256": f"sha256:{hashlib.sha256(proof_raw).hexdigest()}",
            "bytes": len(proof_raw),
        },
        "independent_verification": {
            "key": launcher.volume_key(verification_path),
            "sha256": observed_hash,
            "bytes": len(raw_verification),
        },
        "certified_checkpoint_ref": certified,
        "certified_model_manifest": manifest,
        "lineage": lineage,
        "program_state": verification.get("program_state"),
        "terminal_result": result,
    }


# Compatibility aliases allow the generic V2 driver implementation to be
# reused without changing its recovery semantics when the V3 wrapper swaps the
# module-level launcher dependency.
pod_shell_v2 = pod_shell_v3
verify_proof_v2 = verify_proof_v3

launcher.pod_shell = pod_shell_v3
launcher.verify_proof = verify_proof_v3

if __name__ == "__main__":
    raise SystemExit(launcher.main())
