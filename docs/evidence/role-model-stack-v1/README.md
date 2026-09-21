# Hephaestus Role Model Stack V1

Frozen on 2026-09-21.

This record fixes the foundation-model selection for the five cognitive roles that will enter role-specific fine-tuning:

| Role | Selected foundation |
| --- | --- |
| Controller | `Qwen/Qwen3-4B-Instruct-2507` |
| Diagnosis | `mistralai/Ministral-3-3B-Reasoning-2512` |
| Planner | `mistralai/Ministral-3-8B-Reasoning-2512` |
| Evaluator | `ibm-granite/granite-3.3-8b-instruct` |
| Judge | `microsoft/Phi-4-mini-instruct` |

All model revisions are pinned in `selection.json` and `configs/models/hephaestus_role_model_stack_v1.json`.

## Evidence posture

Diagnosis was selected through the dedicated diagnosis foundation/adaptability experiments.

Planner and Judge were selected through `planner-judge-bakeoff-v1-35538704330`. Planner Ministral reached 100.0 post-LoRA held-out quality with perfect schema, grounding, calibration and exact-contract pass across all eight Planner skill families. Judge Phi-4-mini-instruct ranked first at 93.0208 post-LoRA held-out quality.

Controller Qwen3-4B and Evaluator Granite 3.3 8B preserve the earlier role-specialization selections from the cognitive-topology, adaptation-elasticity and distance-to-role work.

## Boundary

This is a **model-selection freeze for role-specific fine-tuning**, not production certification.

The experimental LoRA adapters referenced in the evidence are retained as scientific evidence and initialization/reference artifacts only. They are not declared final production adapters.

The next phase is to construct governed, role-specific training corpora, train each frozen foundation model for its selected role, and certify the resulting role adapters/checkpoints on independent held-out evidence.
