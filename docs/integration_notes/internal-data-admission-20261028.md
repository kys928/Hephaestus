# Internal prospective admission and external certification

This policy redesign is explicitly authorized by the attached implementation mandate. It is additive to PR #116; no historical approvals, manifests or signatures are recreated.

| Concern | Contract | Authority |
| --- | --- | --- |
| Source eligibility | Exact source hashes, rights, processing/tokenizer commitments and complete screening | Auditable evidence, rechecked at consumption |
| Tier A internal authorization | `InternalResearchPolicy` and `InternalDatasetAdmissionPolicy` | Exact policy digest injected by composition owner; bounded experiment/workflow/run, CPU time and optimizer limits |
| Tier B review | Failed or unresolved source review | Remains blocked; discovery continues |
| Tier C historical/external certification | Existing `IndependentDatasetReviewPolicy`, custody and signature verifiers | Existing external trust roots, unchanged |

`IndependentDatasetReviewPolicy.for_claim` selects an explicit scope. Unknown scopes fail. Existing callers retain the historical external behavior and four-manifest requirements. Tier A cannot mint an external review or promotion grant. Automatic decisions identify their machine-policy authority, rather than masquerading as an independent person's signature.

Prospective protocols freeze newly created evaluation and development artifacts in the existing SQLite repository before training exposure. They identify original example bytes, methods and limitations. They are not replacements for missing historical references. Re-freezing an experiment after exposure is forbidden. This provides internal reproducibility, not independent custody or proof that a pretrained model never saw related examples.

Migration is opt-in on a new experiment with a new protocol. Historical campaigns cannot be relabeled; their claim scope and original evidence stay intact. WikiText's existing HOLD and exact-subset guard remain intact. CPU research requires a separate runtime profile; none of these records authorize CUDA replay, promotion or external certification.
