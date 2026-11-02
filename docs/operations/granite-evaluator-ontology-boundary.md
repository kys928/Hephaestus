# Granite Evaluator: two scientific ontologies, no implicit mapping

## Contract identity matters

Hephaestus's *current frozen production* role-mastery Evaluator and the
historical Evaluator semantic-repair experiments do not use the same decision
vocabulary.

- **Frozen role mastery v1 (six decisions)**:
  improved, regressed, no_material_change, scientific_rejection,
  incomplete_evidence, evaluation_invalid.
  Normative source: `configs/models/hephaestus_role_mastery_runtime_v1.json`,
  pinned by `src/hephaestus/llm/role_contract.py`.
- **Historical semantic-invariance research (eight states)**:
  scientific_rejection, incomplete_evidence, inconclusive,
  recheck_required, regressed, equivalent, certification_ready, improved.
  Historical source: the draft Evaluator research branches and their source
  revision-specific scientific-state oracle, not the frozen production registry.

The historical eight-state results are valuable for diagnosis, but they must
**not** be treated as the production Evaluator's direct performance measurements.
In particular, there is no documented validated one-to-one translation from
the research states `equivalent`, `recheck_required` and
`certification_ready` to the frozen production six-state labels.

## What was implemented

`evaluation/granite_dev_scoring.py` provides pure strict raw-output scoring
with an explicit ontology declaration, strict response envelope and matched
case-set/partition requirements. It separately records invalid JSON, invalid
envelopes and decisions outside the declared ontology. The original wrong
labels are never "fixed" by coercing them into the other vocabulary.

Reference cases and model generations require separately verified source
identities and authentic inference provenance. The scorer by itself **does
not** verify source files, model weights, correctness of supplied labels,
scorer provenance, or contamination. Its return value explicitly states this.

All input partitions must be previously exposed development or burned
diagnostic material. Protected holdout material is refused. No source here
authorizes training, promotion, independent certification, or changing the
original Evaluator.

## Next work

The next integration must combine this scorer with *independently pinned
raw per-case outputs and reference labels* and the actual original-model
inference receipts. Only then may it create read-only, typed performance
observations for Hephaestus's Diagnosis and Planner. The downstream research
system must distinguish the actual six-state production objective from
exploratory eight-state repair studies before selecting experiments.

The previously sealed/consumed V3.1 research data cannot be re-sealed and
reclaimed as fresh certification just because their raw predictions are
successfully rescored.
