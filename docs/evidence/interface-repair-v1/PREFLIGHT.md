# Interface Repair V1 preflight

Status: prepared for targeted paid training after Phase-II V1 failure analysis.

Scientific hypothesis: the Phase-II V1 score of 56.25/100 was primarily caused by missing interface curriculum rather than insufficient base-model capability. Evidence grounding was 100% and hallucinated evidence references were 0%; dominant failures were role-vocabulary translation, semantic escalation, and action-authority transfer.

Frozen intervention:
- Diagnosis stays byte-identical to its certified Phase-I adapter and receives no gradient updates.
- Planner, Evaluator, Judge, and Controller continue training from their certified Phase-I LoRA adapters.
- The parent adapters remain immutable S3 artifacts; repaired adapters are new descendants.
- Training data is 75% interface repair and 25% role rehearsal.
- Old Phase-II V1 cases are not used for new certification.
- Judge/Controller curriculum makes machine state authoritative and upstream model text explicitly advisory.

Training geometry:
- Planner: 1,280 cases / 160 optimizer steps.
- Evaluator: 1,280 / 160.
- Judge: 2,048 / 256.
- Controller: 1,280 / 160.
- micro-batch 1, gradient accumulation 8, 2,048-token context, BF16, AdamW, LR 1e-5.

Certification sequence:
1. fresh per-role interface certification (128 cases/role),
2. fresh role-regression certification (128 cases/role),
3. only if all four pass, a 128-case live model-to-model rollout (32/interface),
4. deterministic Controller boundary must pass 100%, semantic escalation and upstream-copy violation rates must both be zero.

Production promotion and automatic role dispatch remain disabled regardless of experimental outcome.
