# Approved C proof: rejected checkpoint, incomplete diagnostic

**C step 96 is a scientific no-go.** Training completed, but only 128/256 frozen
diagnostic cases were saved before the evaluation cost guard stopped the pod.
The checkpoint answered 61/128 correctly. Even 128 further correct answers would
yield only **189/256 = 73.828125%**, below the unchanged 95% acceptance gate.
The remaining cases are unnecessary to establish rejection of this checkpoint;
their metrics remain unknown. No checkpoint is selected or certified.

## Identities and evidence

- Paid workflow: [38935682810](https://github.com/kys928/Hephaestus/actions/runs/38935682810).
- Immutable experiment source: `80366f51dac74141134ffdeb3f19c4f8a942c943`.
- Run: `evaluator-semantic-v2-C-38935682810`.
- S3 root: `hephaestus/scientific/v5/evaluator_semantic_repair_v2/evaluator-semantic-v2-C-38935682810`.
- Step-96 adapter: 61,534,393 bytes;
  SHA256 `fef6b5622c4e33a490c00d8c989d153a6d7f4b37feba682b91255b3d013a6d6b`.
- Base revision: `51dd4bc2ade4059a6bd87649d68aa11e4fb2529b`;
  frozen parent adapter SHA256:
  `913797ddb8d9d95f83d09a244e8efe430d7bfc4383e589499c2fcc2d943487ed`.
- Frozen diagnostic SHA256:
  `ae8b2bd702d9b0751f9b1b04958d5a4559188971fcaf1dd818be4da233cd6912`.
- Training manifest SHA256:
  `a071abe3217fca2185715086e88284a3dea01962a247043928de533c388771e3`.
- Independent read-only assessment and resource verification:
  [38938765419](https://github.com/kys928/Hephaestus/actions/runs/38938765419),
  artifact `semantic-v2-independent-assessment`.

The training result, manifest, all six checkpoint archives and four 32-case
evaluation shards remain in S3. The full assessment artifact contains object
byte hashes, training runtime, full confusion/action tables and paired counts.
The repository's `scripts/assess_evaluator_semantic_v2_run.py` reproduces the
assessment without allocating a GPU or writing S3. It verifies case identities,
order, checkpoint lineage, archive size, and every saved C score by rescoring its
persisted generation. A final CPU workflow also publishes this assessment.

## Comparable results

The saved prefix contains all 32 cases for each of four states. It contains no
`equivalent`, `recheck_required`, `inconclusive` or `certification_ready` cases.
It is not a balanced sample of the eight-state diagnostic. Full-diagnostic
baseline/V1 accuracies must not be compared directly to this prefix accuracy.

| Model | Correct on identical saved 128 cases | Accuracy |
|---|---:|---:|
| Frozen parent | 39/128 | 30.46875% |
| V1 step 96 | 63/128 | 49.21875% |
| V2 C step 96 | 61/128 | 47.65625% |

Paired C versus V1: four reference-only correct, two C-only correct,
59 both correct, 63 both wrong; exploratory exact paired p = 0.6875.
There is no demonstrated C advantage over V1 on these cases. Against the frozen
parent, the prefix gains 17.1875 percentage points (24 C-only, two parent-only;
exploratory p = 0.0000104904). These unadjusted diagnostic comparisons cannot
replace the fixed acceptance gates or independent certification.

| Observed state | C correct | C accuracy |
|---|---:|---:|
| scientific_rejection | 18/32 | 56.25% |
| incomplete_evidence | 20/32 | 62.5% |
| improved | 14/32 | 43.75% |
| regressed | 9/32 | 28.125% |

Each fully observed state already fails its fixed per-state gate. Other prefix
metrics: schema compliance 100%; evidence grounding 98.4375%; citation
hallucination 2.34375%; upstream-copy violations 24.21875%; system-action accuracy
69.53125%. Direct decision accuracy is 40/64 (62.5%); advisory accuracy is 21/64
(32.8125%). `inconclusive` is predicted on 41/128 cases despite none of these
cases having that target. The full hallucination gate is also unreachable:
three observed violations already exceed 1% of all 256 cases.

Boundary and primary-variable projection are 100%, with zero system semantic
escalation. Those deterministic protections do not credit wrong scientific
decisions. The diagnostic has no declared V2 single-variable pairs or advisory
quartets, so pair accuracy and matched invariance are unavailable here. The
new harness supports those measurements on declared independent packs; they
must not be inferred from direct/advisory aggregate differences. Citation
integrity does not establish free-text rationale truth.

## Execution and cost

Training consumed exactly 768 selected cases and 96 optimizer steps, preserving
the frozen dose, parent, seed, hyperparameters and base revision. All six scheduled
checkpoints were saved. No additional checkpoint was evaluated or selected.
The shared frozen loader and the checkpoint's adapter materializer executed on
the required 96 GB Blackwell GPU. Training recorded BF16 base parameters and the
pinned PEFT loader's adapter precision; weights were not silently recast.

| Allocation | Duration including teardown | Estimated GPU cost | Outcome |
|---|---:|---:|---|
| Training `xsj2mcd8uw29rl` | 448.999 s | $0.310557 | Complete |
| Evaluation `i8m100fixm1gsy` | 943.769 s | $0.652774 | Cost guard stopped |

Replacement estimate: **$0.963331** at $2.49/hour. These are elapsed-time estimates,
not billing-ledger charges. The earlier refused allocation `lji9euultcursa`
lasted 3.0468 seconds; its rate was not logged, so its exact charge remains unknown.
The replacement approval retained a $0.10 reserve for that startup.

The evaluation guard reserves six minutes of its $0.90 ceiling for teardown;
it therefore stopped below $0.90 rather than spending the reserve. The 128 saved
generations consumed 420.915 seconds and 23,907 tokens. The remaining elapsed time
includes startup/materialization/loading and any unsaved final batch; existing
telemetry cannot attribute that overhead precisely. No limit or decoding setting
was relaxed. Evaluation ended with a host cost timeout, not a successful full
evaluation result. The missing terminal evaluation JSON is not treated as success.

Deletion was verified by the launcher. A separate read-only run confirmed actual
HTTP 404 for the refused, training and evaluation pod IDs and no matching live pod.
Both immutable approval records and conditional S3 claims are retained. Paid
flags are now disabled and the active approval ID cleared; the completed source
commit and evidence preserve the exact authorized configuration.

## Diagnosis and next justified action

Correcting false contrasts alone did not repair the scientific classifier at the
frozen training dose. It does not disprove the value of genuine counterfactuals,
nor establish which other curriculum intervention will work. The original
supervision-mismatch hypothesis remains disproved: B is token/mask equivalent to
effective V1 supervision and merits no paid rerun.

The implemented D factorial/precedence curriculum is the next defensible small
ablation to consider after review, starting again from the frozen parent. Its
budget needs separate authorization; no D/E run or blind C continuation is allowed.
Improved startup phase timing and cost feasibility should precede another paid
evaluation. Finishing the rejected C diagnostic may inform error analysis, but
cannot rescue its acceptance outcome and is not the immediate spending priority.
E still requires its prepared D_matched control to isolate advisory supervision.

Engineering implementation and CPU validation are complete. Scientific repair
is not demonstrated; C step 96 is rejected. The PR remains a diagnostic draft,
with no production merge, parent mutation, certification or authoritative dispatch.

Final local repository suite: **576 passed, two skipped**. The targeted V2 suite
passed all 61 tests. CPU workflow validation also covers real frozen-tokenizer
masking, tiny Granite loss/backward, all curriculum hashes and S3 rescoring; the
PR records the final remote run and its artifacts.
