#!/usr/bin/env python3
"""Third real-model promotion wave selected from V2 rejection evidence.

V2 showed deterministic, perfectly repeatable rejection rather than noisy
semantic uncertainty. Qwen2.5-7B failed instruction_exact and continuation
termination; Mistral-7B failed instruction_exact, brief length, and strict JSON.
This wave therefore changes only model identity to stronger permissively
licensed instruction-following candidates while retaining the frozen eval pack,
decoding settings, baseline, Judge, certification, and promotion policy.
"""
from __future__ import annotations

import run_positive_promotion_proof_v2 as wave2

proof = wave2.proof

PRIOR_V2_REJECTION_EVIDENCE = (
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34104954995/cycles/cycle-01/cycle_summary.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34104954995/cycles/cycle-01/experiment_comparison.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34104954995/cycles/cycle-01/certification_decision.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34104954995/cycles/cycle-02/cycle_summary.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34104954995/cycles/cycle-02/experiment_comparison.json",
    "/workspace/hephaestus/scientific/v1/positive_promotion/positive-real-model-promotion-001-34104954995/cycles/cycle-02/certification_decision.json",
)

# Keep the existing independent reviewer fixed so candidate-model identity is
# the single scientific variable changed by this wave.
JUDGE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
JUDGE_REVISION = "582efe62d7cfafd242bffca71ecbde1bcecc1bcc"

proof.CANDIDATES = (
    {
        "model_id": "microsoft/phi-4",
        "revision": "2db69c1c3e91a05d2c64a3185acfbaf36f744e25",
        "license": "mit",
        "judge_model_id": JUDGE_MODEL_ID,
        "judge_revision": JUDGE_REVISION,
        "judge_license": "apache-2.0",
        "prior_rejection_evidence": list(PRIOR_V2_REJECTION_EVIDENCE),
        "selection_reason": "14B permissive model explicitly aligned for precise instruction adherence; targets V2 exact-instruction/format failures",
    },
    {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "license": "apache-2.0",
        "judge_model_id": JUDGE_MODEL_ID,
        "judge_revision": JUDGE_REVISION,
        "judge_license": "apache-2.0",
        "prior_rejection_evidence": list(PRIOR_V2_REJECTION_EVIDENCE),
        "selection_reason": "same Qwen2.5 instruction family as V2 but doubled scale, preserving architecture while testing whether strict adherence failures are capability-limited",
    },
)

if __name__ == "__main__":
    raise SystemExit(proof.main())
