# Prometheus: independent raw-source contrast and actual coverage-probe discrepancy — 2026-11-14

## Result: real evidence acquired; original Granite failure gate still protected

We did **not** stop at the missing-source inventory. Two independently coded, read-only audit jobs re-downloaded and SHA-256-verified both original dataset byte streams from the **original Czech RunPod S3 storage**, parsed all JSONL records, cross-verified archival SHA256SUMS, and independently confirmed the source-linked contrast.

[Read-only audit #43637529964](https://github.com/kys928/Hephaestus/actions/runs/43637529964) — job A `132577519475`, job B `132577519642`, **both success**. Jobs check out the immutable source at `f27882fc08aa6d7c162a4b12c5dc522e9cdb69e2`, use two separate full-archive readers, and access the existing regional bucket with read-only API operations only. No GPU, model change, dataset mutation, final holdout, or paid Pod was involved.

### Independently verified source facts

| Property | Original bounded training | Later controlled dataset-mix |
| --- | --- | --- |
| Original run | `first-bounded-scientific-training-001-33866198758` | `planned-run-b8e558e54effac85` |
| SHA-256 | `f7c512199b6a34ce07fabcd4bdbd45a613aad650190c11bd32c0bbb979910b5c` | `bac39c4c25394e32e86d0e73fe410123e38fcd0d67064e2e1b59a1e31e822fac` |
| Bytes read and rehashed | 15,920,505 | 157,151,627 |
| JSONL rows parsed | 54,878 | 171,295 |
| Separate `prompt`/`target` fields | 0 | 0 |
| Embedded `<\|prompt\|>` and `<\|target\|>` tags in `text` | **0** | **35,774** |
| Independent readback outcomes | PASS in A and B | PASS in A and B |

The original diagnosed `data_coverage` gap is genuinely source supported: the prior probe found 54,878 records without prompt/target supervision and the present independent byte-level rescore agrees. The later dataset is a different content-addressed source under the **same project training lineage**, deliberately changed by the earlier `dataset_mixture` intervention. This is authentic source change, not a hand-authored synthetic case.

### Executed the actual original diagnostic probe (not a hypothetical simulation)

On the later rehashed S3 dataset, the unmodified `analyze_dataset_task_coverage` implementation was invoked against the original pinned `semantic_behavior_v1` pack. It returned:

- `prompt_target_rows = 0`;
- `structured_target_rows = 0`;
- `data_coverage_gap = true`.

Meanwhile, both separately coded raw-file readers measured **35,774 genuine embedded prompt/target wrappers** on exactly the same verified source SHA-256. The probe's source-level predicate checks only `row["prompt"]` and `row["target"]`; it does not recognize properly embedded text wrappers. This is a **verified instruction-coverage measurement blind spot**.

**Important qualification:** the overall `data_coverage_gap` can still be justified by other requirements, particularly the frozen structured-target tasks. Thus the verified defect is the probe's false **zero count of embedded instruction wrappers**, not a demonstrated false **overall data-coverage conclusion**. Neither measurement proves the second candidate solved its generalization problem. The later evaluation still independently recorded `regressed` / hard gate `failed`, despite improved soft aggregate score.

### Honest classification and research use

1. **Authentic positive data signal:** source-reverified baseline missing instructions -> `data_coverage` support on the original training lineage.
2. **Authentic changed-source counterfactual:** candidate text wrappers increase from zero to 35,774; independent SHA identities differ and correspond to the controlled dataset-mix intervention. Only the subclaim of embedded instruction supervision has changed.
3. **Authentic instrument discrepancy:** on candidate source, the unchanged diagnostic probe reports zero separate prompt/target fields, while source-aware readers count 35,774 embedded pairs. This can challenge the Diagnostician on **measurement semantics**, provided evaluation labels carefully delimit the question and do not imply Granite's evaluator failed.
4. **Authentic mixed metrics:** the second controlled model has better soft aggregate but failed hard checks; these are different criteria and cannot honestly be labeled a contradiction about an identical claim.
5. **Only controlled, not authentic failure:** an in-memory hash-mismatch detection check is a fault-injection contract test. It is **not** evidence that any stored original dataset or Granite evaluation was corrupted.

These rehashes and measurements are **fresh independent computations over historical development data**. They are not new held-out production-parent outcomes, not independent original Granite positive eval-integrity failures, and do not complete the eight-slot original-parent comparison preregistered in PR #208. No numeric confidence in any observed report is a calibrated probability. No model outputs were rewritten, downgraded or auto-admitted.

### Research decision

Admit this as a **retrospective source-linked development contrast** and a concrete diagnostic-probe quality issue. Build a separate prospective, clearly scoped measurement-semantics pilot if the frozen role is to diagnose this case. Preserve original-parent PR #208 source requirements for any **certified six-state Granite autonomy** claim; do not contaminate its positive eval-failure slots with evidence from a different model lineage or training instrument.

The exact non-secret source inventory and branch decisions are preserved in [`prometheus-authenticated-contrast-20261114.json`](../evidence/prometheus-authenticated-contrast-20261114.json). It records two separate job IDs, source SHA/size/row identities, the legacy probe discrepancy, scientific limitations and all forbidden actions.

**No Planner, Granite repair, training, final holdout or promotion is authorized by this evidence report.**
