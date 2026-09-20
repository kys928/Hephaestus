# Planner/Judge Bakeoff V1 — PREPARED

Status: **prepared, not launched**.

The bakeoff is frozen and ready for an explicit launch command. No paid/GPU bakeoff has been initiated.

## Frozen curriculum

Canonical pack SHA-256:

`1723aa1da329e0440d27073a62f74c1e0666a7649c55a00f9f9f55d6400ed749`

Each role contains 32 zero-shot cases, 80 micro-LoRA training cases, and 48 untouched held-out cases across eight independent capability families. Total deterministic cases: 320.

## Planner roster

- NVIDIA Nemotron-Cascade-8B — `276a779fe0c1fe2d63a68fd573fec7b4b0a1907e`
- AllenAI OLMo-3-7B-Instruct — `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`
- HuggingFaceTB SmolLM3-3B — `a07cc9a04f16550a088caea529712d1d335b0ac1`
- Mistral Ministral-3-8B-Reasoning-2512 — `81eaece1948f3875421d9a45bc55487d10e2d894`

Leading hypothesis only: Nemotron-Cascade-8B. This is not a selection result.

## Judge roster

- AllenAI OLMo-3-7B-Instruct — `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`
- Microsoft Phi-4-mini-instruct — `cfbefacb99257ffa30c83adab238a50856ac3083`
- HuggingFaceTB SmolLM3-3B — `a07cc9a04f16550a088caea529712d1d335b0ac1`
- NVIDIA Nemotron-Cascade-8B — `276a779fe0c1fe2d63a68fd573fec7b4b0a1907e`

Leading hypothesis only: OLMo-3-7B-Instruct. This is not a selection result.

## Matched intervention

Every candidate-role pair receives:
- zero-shot evaluation on 32 role cases;
- held-out pre-LoRA evaluation on 48 untouched cases;
- 80-case role-specific micro-LoRA curriculum;
- rank 8, alpha 16, dropout 0;
- 32 optimizer steps, AdamW, learning rate 5e-5, seed 11;
- gradient accumulation 4, microbatch 1;
- uniform max sequence length 1024;
- exactly 131,072 padded token slots per candidate-role pair;
- post-LoRA evaluation on the same untouched 48 held-out cases.

The experiment reports zero-shot fit, post-LoRA quality, adaptation gain, gain per supervised token, schema compliance, evidence grounding, hallucination/reference error, calibration, latency, per-skill deltas, and mastered-capability regressions.

## Preflight evidence

- Dataset/static validation passed.
- Candidate immutable revisions and parameter counts resolved.
- All candidates are below 10B parameters.
- LoRA target surfaces were verified on meta-device models.
- Ministral LoRA is restricted to its language model; vision tower excluded.
- Chat-template generation/training interfaces validated.
- All 640 candidate-role training transcripts were tokenized successfully.
- Maximum observed training transcript: 863 tokens, below the frozen 1024-token budget.
- Nemotron is marked license-review-required before any later production promotion because Hugging Face metadata reports its license as `other`.

## Launch boundary

Current launch marker: `authorized=false`.

Current protocol: `paid_launch_allowed=false`.

The launcher additionally requires the explicit runtime authorization environment variable. All three gates must be enabled before a RunPod pod can be created.

No automatic model-selection commit, production promotion, certification claim, or lineage mutation is permitted by this bakeoff.
