#!/usr/bin/env python3
"""V5 bootstrap using the dependency lock that passed immutable CPU admission."""
from __future__ import annotations

import json
from pathlib import Path
import launch_positive_promotion_proof_v4 as previous

launcher = previous.launcher
SPEC = json.loads((Path(__file__).resolve().parents[1] / "configs/models/promotion_wave_v5.json").read_text())
ALLOWED_REVISION_LICENSES = {c["revision"]: c["license"] for c in SPEC["candidates"]}
ALLOWED_REVISIONS = set(ALLOWED_REVISION_LICENSES)


def pod_shell_v5():
    shell = previous.pod_shell_v4()
    old_install = next(line for line in shell.splitlines()
                       if '"$PY" -m pip install ' in line and "'transformers>=" in line)
    locked_install = r'''ADMISSION_DIR="/workspace/hephaestus/scientific/v1/model_admission/$HEPHAESTUS_REPO_SHA"
"$PY" - "$ADMISSION_DIR" <<'PYLOCK'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
admission = json.loads((root / "admission.json").read_text())
assert hashlib.sha256((root / "requirements.txt").read_bytes()).hexdigest() == admission["requirements_sha256"]
PYLOCK
"$PY" -m pip install --no-cache-dir --disable-pip-version-check -e . -r "$ADMISSION_DIR/requirements.txt"'''
    shell = shell.replace(old_install, locked_install)
    shell = shell.replace('"$PY" scripts/run_positive_promotion_proof_v4.py',
                          '"$PY" scripts/run_positive_promotion_proof_v5.py')
    shell = shell.replace("positive-real-model-promotion-proof.v4", "positive-real-model-promotion-proof.v5")
    return shell


def verify_proof_v5(client, proof_run_id, result):
    saved = previous.ALLOWED_REVISION_LICENSES
    try:
        previous.ALLOWED_REVISION_LICENSES = ALLOWED_REVISION_LICENSES
        verified = previous.verify_proof_v4(client, proof_run_id, result)
    finally:
        previous.ALLOWED_REVISION_LICENSES = saved
    verified["verification_version"] = "positive-real-model-promotion-launcher-verification.v5"
    verified["wave_disposition"] = result.get("disposition")
    return verified


pod_shell_v2 = pod_shell_v5
verify_proof_v2 = verify_proof_v5
