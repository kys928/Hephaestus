# Evaluator Semantic Repair V2 — implementation and scientific boundary

## Verdict

**No-go for scientific improvement or certification.** V2 curriculum generation,
explicit supervision, offline oracle, validation and reporting are implemented.
No V2 8B training or inference has been purchased. The checked-in protocol refuses
paid training/evaluation, certification, promotion and authoritative dispatch.
The dry-run-first V2 launcher and remote training/evaluation entrypoints are
implemented and disabled by the checked-in approval flags. An explicit-marker-only workflow reuses existing GitHub secrets; it cannot buy
compute under the checked-in flags and null budget approval ID. The launcher enforces
price/time limits, reserves six minutes for bounded polling/teardown, and requires actual HTTP 404 evidence
of deletion. A transport failure does not establish absence. Conditional S3 launch
claims, scoped to a unique budget approval ID rather than a run ID, reject
duplicate/concurrent paid attempts and reuse of a spent approval; if the storage endpoint rejects
conditional writes, allocation fails closed.

## Verified evidence and corrected diagnosis

Inspected main and the diagnostic branch. This PR is intentionally based on
`diagnostic/evaluator-sealed-v1-autopsy` at
`e141a9a30cb40a750445444bcb6a6517bc15cf5a`, rather than importing its 113-file
diagnostic stack into main as part of this repair.

Reviewed original run 37005231044 (workflow failed), recovered checkpoint run
38910181535 (completed), and decision-error audit 38925834575. The new CPU-only
GET/HEAD audit 38929135305 independently retrieved 56 persisted S3 objects,
checked all 1,536 checkpoint sample identities, and checked archive sizes and
checkpoint lineage against the completed training result. It records byte hashes
for read JSON/sample objects; HEAD size checks do **not** claim a fresh full
archive-content SHA verification. The existing materializer verifies archive
bytes and SHA before any later model loading.

Frozen baseline: 39.0625% decision accuracy. Checkpoints 16/32/48/64/80/96:
40.234375%, 43.359375%, 44.921875%, 44.921875%, 45.3125%, 45.703125%.
All failed the unchanged scientific acceptance gates. At step 96, 23/32 equivalent
and 16/32 recheck-required cases became inconclusive; recheck accuracy declined
from 15/32 to 3/32. This is scientific classification failure, not JSON success.

An important earlier diagnosis was disproved. Although `v2.training_example`
calls `v1.target_answer`, importing `evaluator_semantic_repair_train` imports
`common -> sealed -> v3`, and V3 replaces that function with `target_answer_v3`.
The actual training path **already uses state-specific rationale**. On the frozen
Granite tokenizer, all 768 B inputs and labels are exactly equal to V1. B is
therefore an import-dependency repair/equivalence check, not a new learning
intervention. A paid B rerun would be scientifically redundant. V2 now serializes
its own targets explicitly without depending on mutation of a shared module.

The other two findings are verified: factorial_generalization calls a canonical
latent template per state, and 176/256 old contrast pairs differ in **two**
scientific variables (80 differ in one). Existing validators did not test this.
These defects motivate an experiment; they do not establish its outcome.

## Implemented controls

- Typed six-factor domain with strict booleans/effect vocabulary. Exhaustive 96-way
  oracle preserves V1 precedence: gate, completeness, provenance, variance, effect,
  then confirmation for positive effects only. It is offline supervision/test
  logic and does not replace model scientific-state authority.
- 256 corrected minimal pairs, one declared factor and one observation changed,
  with identical context, refs and advisory content within each pair.
- All 96 independent combinations in factorial D/E corpora; simultaneous defects,
  negation, changing quantitative thresholds, contradictory non-authoritative
  secondary metrics, and prose/JSON/record evidence presentations.
- 1,024 advisory quartets in E: absent, correct, incorrect and confidently wrong,
  with identical evidence, machine state and target in each quartet.
- Every selected arm remains 768 examples, 96 per state, with frozen optimizer,
  seed, LR, base revision, parent SHA and decoding. E keeps all 512 contrast rows
  and uses the remaining 256 slots for matched advisory quartets.
- Unchanged original development partition and canonical hash
  `ae8b2bd702d9b0751f9b1b04958d5a4559188971fcaf1dd818be4da233cd6912`.
  No burned sealed cases enter a training schedule. Future certification is out of
  scope and must be generated independently after checkpoint selection.
- Data checks cover refs, targets, machine/numeric consistency, identifiers,
  group membership, exact duplicates and cross-split trigram near duplicates.
  D has no repeated complete evidence bundles; E repetitions are explicitly
  declared same-evidence advisory controls. This is a lexical contamination check,
  not a claim that train and diagnostic occupy different scientific ontologies.
- Fail-closed full-prefix token masking (no permissive short-LCP fallback), exact
  assistant serialization, no silent truncation and assistant-only labels.
- Real pinned Granite tokenizer verification across all 3,072 selected examples:
  maximum lengths B/C/D/E = 1252/1260/1318/1299, below the frozen 2048-token limit.
  Tiny Granite CPU forward, shifted loss and backward work. Full 8B/PEFT loading
  remains a GPU preflight requirement; the CPU test does not validate 8B runtime.
- Enhanced reporting independently scores persisted raw generations: full decision
  confusion matrix, state accuracy, pair accuracy, state-change consistency,
  advisory invariance and invariant correctness, misleading-advice strata,
  erroneous copying versus benign agreement, model/system action mappings,
  schema/citation grounding and regressions against the frozen reference.
  Invariance can be 100% while every answer is wrong; tests demonstrate this.
  Citation metrics do not establish free-text factual truth.

## Interpretable experiment sequence

A is the frozen baseline plus the persisted V1 checkpoint evidence. B has no
changed targets/tokens and needs no paid experiment. C replaces contrast data;
its first comparison is against frozen A under identical development settings.
D adds factorial/precedence data and controlled difficulty ordering: this is a
combined curriculum intervention, not an isolated estimate of either component.

E changes the distribution of repeated scientific examples to support exact
advisory matching. Comparing ordinary D directly to E would confound this change.
Use the separately prepared **D_matched** control with precisely E's evidence,
order and multiplicity, removing only advisory records. Compare E to that control.
Each experiment starts from the same frozen parent, not from the preceding arm.
Training loss and deterministic boundary corrections are never improvement gates.

## Reproduction

Set `PYTHONPATH=scripts:src:.` and install core/test dependencies plus
`transformers==5.17.0` and CPU torch. Run:

```bash
pytest -q
python scripts/validate_evaluator_semantic_repair_v2.py
python scripts/check_evaluator_semantic_v2_tokenizer.py --output tokenizer-proof.json
python scripts/prepare_evaluator_semantic_repair_v2.py --arm C --output manifest-C.json
python scripts/launch_evaluator_semantic_repair_v2.py --mode train # dry run only
python scripts/prepare_evaluator_semantic_repair_v2.py --arm E --matched-handoff-control --output manifest-D-matched.json
python scripts/evaluate_evaluator_semantic_v2_outputs.py --inputs shard-0001.jsonl shard-0002.jsonl --output rescored.json
```

The scoring CLI requires the complete declared diagnostic, not the two example
shards alone. It rejects missing, duplicate and undeclared outputs, re-scores raw
outputs rather than trusting supplied scores, and refuses training partitions as
development evidence. Pack, schedule, target serialization and scientific config
hashes appear in immutable prepared manifests. Generated evidence is retained as
workflow artifacts, not committed as datasets/state in this PR.

## Spending boundary and next justified action

The earlier $2.49/hour and $0.90/checkpoint recovery approval does not authorize
new training. The proposal file requests at most **$2.50 total** for one C training
run (cap $1.60) and one predeclared step-96 development evaluation (cap $0.90),
retaining the stricter configured $2.10/hour ceiling. These are spending caps,
not a guarantee the job will finish within them. Startup/download/teardown time
counts; current capacity/price must be checked before allocation. The implemented launcher is covered by mocked
lifecycle/failure tests; its full 8B execution remains pending this approval. No additional arms launch automatically. Stop and diagnose a negative
result; do not silently extend the budget, change model families, or relax gates.

CPU validation run 38930553151 passed on GitHub, including the S3 rescoring, full
repository suite, real tokenizer proof and all immutable manifest preparations.
The subsequent launcher hardening is verified by the final PR-head CPU run.

Next action: approve or decline this bounded C experiment after reviewing the PR.
Until then the scientific verdict remains unproven and production remains blocked.
