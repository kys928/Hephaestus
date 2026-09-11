#!/usr/bin/env python3
"""Fourth real-model promotion wave selected from exact V3 gate evidence.

V3 showed both 14B candidates were materially better than the random-init
baseline, perfectly repeatable, and low-variance, but were rejected by strict
hard deterministic formatting/termination gates. Phi-4 failed continuation
termination and strict JSON; Qwen2.5-14B failed exact punctuation and strict
JSON typing/wrapping. V4 therefore changes only candidate-model identity to
models selected specifically for instruction-following and structured-output
behavior while retaining the frozen eval pack, decoding, baseline, fixed
independent reviewer, Judge, certification, promotion policy, and the proven
2x RTX 3090 balanced-FP16 execution topology.
"""
from __future__ import annotations

import run_positive_promotion_proof_v3 as wave3

proof = wave3.proof

PRIOR_V3_REJECTION_EVIDENCE = (
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34561911749/cycles/cycle-01/cycle_summary.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34561911749/cycles/cycle-01/experiment_comparison.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34561911749/cycles/cycle-01/promotion_gate_report.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34561911749/cycles/cycle-01/certification_decision.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34561911749/cycles/cycle-02/cycle_summary.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34561911749/cycles/cycle-02/experiment_comparison.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34561911749/cycles/cycle-02/promotion_gate_report.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34561911749/cycles/cycle-02/certification_decision.json",
)

# Keep the independent reviewer fixed so candidate-model identity remains the
# only scientific variable changed by V4.
JUDGE_MODEL_ID = wave3.JUDGE_MODEL_ID
JUDGE_REVISION = wave3.JUDGE_REVISION

GRANITE_REVISION = "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b"
OLMO_REVISION = "3a5c85baefbb1896a54d56fe2e76c0395627ddf4"

proof.CANDIDATES = (
    {
        "model_id": "ibm-granite/granite-3.3-8b-instruct",
        "revision": GRANITE_REVISION,
        "license": "apache-2.0",
        "judge_model_id": JUDGE_MODEL_ID,
        "judge_revision": JUDGE_REVISION,
        "judge_license": "apache-2.0",
        "prior_rejection_evidence": list(PRIOR_V3_REJECTION_EVIDENCE),
        "selection_reason": (
            "V3 failures were strict exact-format/termination failures rather than broad capability failures; "
            "Granite 3.3 is selected for instruction-following and structured-output behavior while fitting "
            "comfortably inside the unchanged dual-3090 FP16 topology"
        ),
    },
    {
        "model_id": "allenai/OLMo-2-1124-13B-Instruct",
        "revision": OLMO_REVISION,
        "license": "apache-2.0",
        "judge_model_id": JUDGE_MODEL_ID,
        "judge_revision": JUDGE_REVISION,
        "judge_license": "apache-2.0",
        "prior_rejection_evidence": list(PRIOR_V3_REJECTION_EVIDENCE),
        "selection_reason": (
            "OLMo 2 13B Instruct is post-trained with Tulu-3/DPO/RLVR and selected as an independent model "
            "family aimed at strong instruction following, including IFEval-like behavior, while remaining "
            "within the unchanged dual-3090 FP16 execution envelope"
        ),
    },
)

if __name__ == "__main__":
    raise SystemExit(proof.main())
