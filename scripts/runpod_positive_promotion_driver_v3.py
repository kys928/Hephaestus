#!/usr/bin/env python3
"""Generic RunPod production-loop driver wrapper for promotion wave V3."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import launch_positive_promotion_proof_v3 as launcher_v3  # noqa: E402
import runpod_positive_promotion_driver as base  # noqa: E402

# Reuse the already live-proven generic recovery implementation, but swap only
# the scientific Pod shell and independent verifier to V3.
# Launch marker: V3 preflight passed before this audited real-compute commit.
base.launcher_v2 = launcher_v3


class RunPodPositivePromotionDriverV3(base.RunPodPositivePromotionDriver):
    def _write_launcher_record(self) -> None:
        base.launcher.atomic_json(
            Path("positive_promotion_launcher.json"),
            {
                "launcher_version": "positive-real-model-promotion-generic-loop.v3",
                "created_at": base._now(),
                "repo_sha": self.repo_sha,
                "proof_run_id": self.proof_run_id,
                "volume_id": base.launcher.VOLUME_ID,
                "datacenter_id": base.launcher.DATACENTER_ID,
                "generic_cli": "hephaestus run",
                "generic_recovery_owned": True,
                "scientific_variables_changed_on_retry": False,
                "proof_driver": "scripts/run_positive_promotion_proof_v3.py",
                "allowed_candidate_revisions": sorted(launcher_v3.ALLOWED_REVISIONS),
                "allowed_revision_licenses": dict(sorted(launcher_v3.ALLOWED_REVISION_LICENSES.items())),
                "attempts": self.attempt_rows,
                "error": self.last_error,
                "status": "verified" if self.verification is not None else "running",
            },
        )

    def execute_cycle(self, *, runtime: Any, state: Any, cycle_index: int):
        result = super().execute_cycle(runtime=runtime, state=state, cycle_index=cycle_index)
        result.evidence["proof_driver"] = "scripts/run_positive_promotion_proof_v3.py"
        result.evidence["allowed_candidate_revisions"] = sorted(launcher_v3.ALLOWED_REVISIONS)
        result.evidence["allowed_revision_licenses"] = dict(sorted(launcher_v3.ALLOWED_REVISION_LICENSES.items()))
        return result


def build_driver(config: dict[str, object]) -> RunPodPositivePromotionDriverV3:
    return RunPodPositivePromotionDriverV3(
        prove_bootstrap_recovery_once=bool(config.get("prove_bootstrap_recovery_once", False)),
    )
