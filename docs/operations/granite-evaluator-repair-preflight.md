# Granite Evaluator self-repair: read-only evidence preflight

**Status:** engineering first step only. No Granite weights were loaded, no model
inference/training occurred, no Planner-selected training action was executed,
and no scientific repair is demonstrated.

This route deliberately targets the frozen, certified Granite Evaluator
(`ibm-granite/granite-3.3-8b-instruct`) rather than SmolLM2.
The frozen model and adapter identity are read from
`configs/models/hephaestus_role_model_stack_v1.json`.

## Why the boundary exists

Historical Evaluator experiments in draft PRs #70–#74 contain valuable **exposed**
observations and rejected candidate results. Their reported data-shift,
effect-polarity and decision-precedence explanations are hypotheses rather than
verified causal conclusions. A hash of an archived report proves which bytes
were imported; it does **not** prove a report's scoring, interpretation or
claimed `eval_integrity_verified` flag.

The existing generic JSON evidence loader can read files by path without an
independently retained byte identity. The new
`PinnedHistoricalEvidenceAdapter` accepts a narrowly scoped, read-only
flat directory of individually SHA-256-pinned JSON/JSONL files. It refuses
unapproved names, symlinks, oversized files, duplicate JSON keys, nonfinite
numbers, tampered bytes and missing records. It exposes only inert summaries
and source identities to diagnosis; historical assertions cannot become
authoritative diagnostic signals simply by appearing inside a JSON document.

## Preflight inputs

The operator first obtains **authorized historical exposed source files**
in a local directory, without copying protected final material. File names must
be flat (one directory, not arbitrary recursive paths). Independently record the
SHA-256 of every file.

Create an index outside the repository containing:

- `schema`: `granite-repair-historical-index.v1`
- `historical_exposed`: `true`
- `model_id`: exact frozen Evaluator model ID from registry
- `model_revision`: exact frozen Evaluator revision
- `adapter_sha256`: exact frozen Evaluator adapter hash
- `files`: mapping of filename to independently retained `sha256:<64 hex>` digest

Hash the final index bytes and retain that digest separately. Neither self-reported
source metadata nor a digest saved **inside** a file authorizes its contents.

Example invocation (replace placeholders with actual identities):

`python -m hephaestus.development.granite_repair_preflight \
  --registry configs/models/hephaestus_role_model_stack_v1.json \
  --index /admitted/private/historical-index.json \
  --index-sha256 sha256:REPLACE_WITH_REAL_INDEX_SHA256 \
  --evidence-root /admitted/private/historical-source`

The preflight uses the existing deterministic
`EvidenceBasedDiagnosisService` and `ClosedLoopExperimentPlanner`,
and prints a JSON diagnosis/proposal record to stdout. It performs no writes,
CUDA calls, model inference or training. It explicitly records zero updates and
no scientific claim.

With only historical summaries available, the correct admissible outcome is
**collect more evidence**, not automatically select a data-balancing recipe,
longer training or another favored intervention. A real scientific decision
requires a later **independently verified raw-output scoring binding** and
policy-admissible research services.

## Next engineering acceptance

1. Recover and verify the original Granite model/adapter and authorized raw
   historical prediction and scorer artifacts.
2. Implement independent raw-generation/label/scorer admission without taking
   pre-existing failure explanations as authoritative.
3. Bind those authenticated observations to the actual diagnosis and planning
   path and record the owner of each scientific decision.
4. Test materially different evidence states to show a later Hephaestus
   intervention is evidence-dependent, not a hardcoded script.
5. Implement a safely bounded real next-experiment handoff, using the current
   eight-phase production spine, before claiming autonomous repair.

Protect the historical frozen Evaluator and its certification record. No source
in this preflight authorizes protected assessment access, paid RunPod spending,
checkpoint promotion or active Evaluator replacement.
