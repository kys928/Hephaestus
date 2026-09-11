#!/usr/bin/env python3
"""RunPod launcher/verifier for the evidence-selected V4 promotion wave."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import launch_positive_promotion_proof_v3 as wave3

launcher = wave3.launcher

ALLOWED_REVISION_LICENSES = {
    "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b": "apache-2.0",
    "3a5c85baefbb1896a54d56fe2e76c0395627ddf4": "apache-2.0",
}
ALLOWED_REVISIONS = set(ALLOWED_REVISION_LICENSES)
V4_MAX_SECONDS = wave3.V3_MAX_SECONDS
launcher.MAX_SECONDS = V4_MAX_SECONDS


def pod_shell_v4() -> str:
    """Reuse the proven dual-3090 bootstrap while executing the V4 proof."""
    shell = wave3.pod_shell_v3()
    shell = shell.replace(
        '"$PY" -m py_compile scripts/run_positive_promotion_proof.py scripts/run_positive_promotion_proof_v2.py scripts/run_positive_promotion_proof_v3.py\n"$PY" scripts/run_positive_promotion_proof_v3.py',
        '"$PY" -m py_compile scripts/run_positive_promotion_proof.py scripts/run_positive_promotion_proof_v2.py scripts/run_positive_promotion_proof_v3.py scripts/run_positive_promotion_proof_v4.py\n"$PY" scripts/run_positive_promotion_proof_v4.py',
    )
    shell = shell.replace("positive-real-model-promotion-proof.v3", "positive-real-model-promotion-proof.v4")
    shell = shell.replace("V3 dual-GPU proof", "V4 dual-GPU proof")
    shell = shell.replace("V3 expected RTX 3090", "V4 expected RTX 3090")
    shell = shell.replace("V3 expected ~24GB VRAM", "V4 expected ~24GB VRAM")
    return shell


def verify_proof_v4(client: Any, proof_run_id: str, result: dict[str, Any]) -> dict[str, object]:
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
        raise RuntimeError("certified model license disagrees with the immutable V4 allowlist")

    proof_result_key = f"{launcher.SCIENTIFIC_PREFIX}/positive_promotion/{proof_run_id}/proof_result.json"
    proof_raw = launcher.base.read_key(client, proof_result_key)
    proof_payload = json.loads(proof_raw.decode("utf-8"))
    if not isinstance(proof_payload, dict) or proof_payload.get("certified_checkpoint_ref") != certified:
        raise RuntimeError("proof-result readback differs from independent verification")

    return {
        "verification_version": "positive-real-model-promotion-launcher-verification.v4",
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


# Compatibility aliases let the generic production-loop driver use this module
# without changing infrastructure-recovery semantics.
pod_shell_v2 = pod_shell_v4
verify_proof_v2 = verify_proof_v4

launcher.pod_shell = pod_shell_v4
launcher.verify_proof = verify_proof_v4

if __name__ == "__main__":
    raise SystemExit(launcher.main())
