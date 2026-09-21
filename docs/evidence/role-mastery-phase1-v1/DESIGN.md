# Phase I — Role Mastery V1

This program trains the five frozen Hephaestus foundation models independently before any pipeline/co-adaptation stage.

Each role receives:
- 600 role-specific SFT cases across ten capability families;
- 100 separate development cases used only to select epoch 1 vs epoch 2;
- 150 untouched certification cases;
- certification evaluated at seeds 11, 29 and 47.

The split surfaces, case IDs, and evidence references are disjoint. Certification cases use independent production-like wording and are never used for adapter selection.

Training is LoRA rank 16 / alpha 32 / dropout 0.05, BF16, AdamW 2e-5, gradient accumulation 4, max sequence length 1024. Epoch checkpoints are selected on development evidence only.

Certification gates:
- quality >= 92/100;
- schema compliance = 100%;
- evidence grounding >= 98%;
- hallucination/inadmissible-reference rate <= 2%;
- exact-contract pass >= 70%;
- every skill >= 80/100.

No Phase-I result promotes a production model or mutates lineage truth. Passing means the role adapter is eligible to enter Phase II interface/pipeline training.
