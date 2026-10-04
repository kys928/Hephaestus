#!/usr/bin/env python3
"""RunPod launcher adapter for the stronger 7B positive-promotion wave."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import launch_positive_promotion_proof as launcher

ALLOWED_REVISIONS = {
    "ddb6b63a3f61ac6c557eb55619b0a5e125129302",
    "c170c708c41dac9275d15a8fff4eca08d52bab71",
}


def pod_shell_v2() -> str:
    return r'''set -Eeuo pipefail
ATTEMPT_DIR="/workspace/hephaestus/scientific/v1/executions/${HEPHAESTUS_PROOF_RUN_ID}/attempt-${HEPHAESTUS_ATTEMPT}"
mkdir -p "$ATTEMPT_DIR"
exec >"$ATTEMPT_DIR/pod_runtime.log" 2>&1
write_bootstrap_failure() {
  code=$?
  if [ "$code" -ne 0 ] && [ ! -f "$ATTEMPT_DIR/driver_result.json" ]; then
    python - "$ATTEMPT_DIR/driver_result.json" "$code" <<'PYFAIL'
import json, os, sys
from datetime import datetime, timezone
path, code = sys.argv[1], int(sys.argv[2])
payload = {
    "result_version": "positive-real-model-promotion-proof.v2",
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
rm -rf /opt/hephaestus-src /opt/hephaestus-venv
git clone --filter=blob:none https://github.com/kys928/Hephaestus.git /opt/hephaestus-src
cd /opt/hephaestus-src
git checkout "$HEPHAESTUS_REPO_SHA"
python -m venv --system-site-packages /opt/hephaestus-venv
PY=/opt/hephaestus-venv/bin/python
"$PY" -m pip install --disable-pip-version-check -e . 'transformers>=4.46,<6' 'tokenizers>=0.20,<1' 'safetensors>=0.4,<1' 'huggingface_hub>=0.26,<2' 'hf_xet>=1,<2'
"$PY" - <<'PYCHECK'
import torch
assert torch.cuda.is_available(), "CUDA unavailable after positive-proof bootstrap"
print({"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)})
PYCHECK
"$PY" -m py_compile scripts/run_positive_promotion_proof.py scripts/run_positive_promotion_proof_v2.py
"$PY" scripts/run_positive_promotion_proof_v2.py
'''


def verify_proof_v2(client: Any, proof_run_id: str, result: dict[str, Any]) -> dict[str, object]:
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
    if manifest.get("status") != "verified" or manifest.get("revision") not in ALLOWED_REVISIONS:
        raise RuntimeError("certified model immutable revision is unexpected")
    if str(manifest.get("license") or "").lower() != "apache-2.0":
        raise RuntimeError("certified model is not Apache-2.0")

    proof_result_key = f"{launcher.SCIENTIFIC_PREFIX}/positive_promotion/{proof_run_id}/proof_result.json"
    proof_raw = launcher.base.read_key(client, proof_result_key)
    proof_payload = json.loads(proof_raw.decode("utf-8"))
    if not isinstance(proof_payload, dict) or proof_payload.get("certified_checkpoint_ref") != certified:
        raise RuntimeError("proof-result readback differs from independent verification")

    return {
        "verification_version": "positive-real-model-promotion-launcher-verification.v2",
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


launcher.pod_shell = pod_shell_v2
launcher.verify_proof = verify_proof_v2

if __name__ == "__main__":
    raise SystemExit(launcher.main())
