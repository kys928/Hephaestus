# Prometheus — proposed public six-state Granite counterfactual diagnostic

**Status: proposed, not executed, not independently accepted.** This file defines an *available measurement option*, not an intervention that Sol selected for Hephaestus. The research Planner must be able to request further evidence and may choose this diagnostic only after policy admission and independent protocol review. No automatic dispatch is enabled.

## Why propose it

The **original certified Granite Evaluator** returned **200/200 exact six-state decisions** on its original selected step-250 development set in read-only forensic workflow [41645367124](https://github.com/kys928/Hephaestus/actions/runs/41645367124). That is useful evidence of narrow original-task competence, but not of novel semantic generalization. The later eight-state D variant's failures do not establish a weakness in that frozen production parent.

We therefore need a predeclared, independently interpretable six-state probe before claiming that the actual certified parent needs any fine-tuning.

## Draft prospective public development protocol

Pure builder: `src/hephaestus/development/granite_sixstate_public_probe.py`.

- Target the exact frozen `ibm-granite/granite-3.3-8b-instruct`, revision `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`, adapter SHA-256 `e60497854ea64655fc18be7962dec29f13091c75ea91e576726b0fa5c6eb2325`.
- Use the **existing production six-state decision ontology**; do not introduce the old research states `equivalent`, `recheck_required` or `certification_ready`.
- Generate **120 original publicly authored development cases**, exactly **20 per state**: 48 two-case controlled causal contrasts and 12 two-case irrelevant-nuisance contrasts.
- Each causal pair changes precisely **one** scientific fact, keeping the situation and other measured factors constant, and expects the finite oracle's label to change.
- Each nuisance pair changes an explicitly irrelevant vanity metric while preserving the deterministic scientific label.
- Cover evaluation protocol mismatch, checkpoint provenance, hard gate status, sample completeness, measurement repeatability, and quality delta.
- Use four presentation forms to examine lexical robustness and avoid hardcoding one natural-language pattern.
- Never include gold labels, hidden state, adapter references or oracle variables in the material presented to the model. Cases have separate `situation`, `evidence`, `oracle_factors` and `reference` structures; only the first two are model-visible.
- The draft oracle declares precedence among invalid evaluation provenance, non-waivable gate failure, incomplete/unstable evidence, regression, practical equivalence and improvement. **This precedence is a proposed research assumption for scientific review, not a newly certified rule of the production Evaluator.**

The probe is intentionally **public development / feasibility**, not a final holdout. Even excellent scores would not grant certification. Source is versioned and deterministic; the full pack and hash must be committed before model outputs exist.

## Prospective measurement, not post-hoc repair

If independently accepted and genuinely selected by Hephaestus as an admissible information-gathering action, execute against the **verified, unchanged original parent**, under the real certified-role runtime with pinned tokenizer/configuration. Use the frozen production decoding profile (deterministic, `do_sample=false`) and predeclare a sufficient generation budget, EOS accounting, and time limits before execution.

Predeclare outputs:

- Six-state exact decision accuracy, including all per-state confusion matrices.
- Pairwise **causal sensitivity**: both opposite labels must be right for a pair.
- Pairwise **nuisance invariance**: the label must remain correct despite irrelevant changes.
- Strict seven-key JSON/schema and permitted-vocabulary compliance.
- Evidence reference grounding, missing answers, truncation/EOS behavior.
- Full original-model and raw generation receipts, with no substitution of another Granite or a synthetic stand-in.

Use the finite oracle only as a proposed reference to be reviewed; the model does not grade itself. Do not treat aggregate improvement over the **different old eight-state D candidate** as a matched comparison. Any subsequent repair must be measured against the original parent on identical six-state cases under identical decoding.

## No-credit, no-spend posture

**No** role model is loaded or called by the builder; no model is trained, no Hugging Face download or S3 operations are needed, no GPU/cloud authorization is implied, and no protected certification material is exposed. Tests establish only correctness, balance, pairing, determinism and scope of the proposed public probe.

A passing CPU test suite is **not** a scientific result. Until model-backed scoring and independent design review are available, remain in `collect_more_evidence` or an appropriate blocked state.

Example generator:

```bash
python -m hephaestus.development.granite_sixstate_public_probe \
  --output /tmp/proposed-public-sixstate-granite-probe.json
```

Record the canonical pack SHA-256 displayed by the generator *before* any real model is evaluated.
