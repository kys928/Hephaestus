# Governed Evaluator role activation and rollback — CPU integration

This branch follows draft PR #89, which itself follows #88. It does not
turn on production automatic role dispatch.

## Concrete behavior

- Registers `activate_role_adapter` and `rollback_active_role` as
  **distinct high-risk approval-required actions**; an approved
  `promote_checkpoint` grant cannot authorize a live routing transition.
- `ActionAuthorizer` pins activation approval to the exact finalized
  checkpoint manifest hash and the exact current lineage digest.
- `EvaluatorRoleActivation` requires an already **applied**,
  independently certified `promote_checkpoint` action in the trusted action
  journal, plus a separate valid activation approval.
- Validates the checkpoint's full component manifest, PEFT adapter,
  certified Granite base identity/revision and adapter tree identity.
- Performs a real `RoleGenerator.generate` health check before publishing
  active routing. Production mode requires `PinnedTransformersRoleGenerator`
  and measured GPU provenance. Rejected/malformed probes never replace the
  active route.
- Activation is append-only in the existing state repository with a durable
  prepared event; an interrupted probe leaves the previous routing effective.
  Retrying an applied event is idempotent while a rolled-back event cannot be
  replayed to re-activate it.
- `CertifiedRoleReasoner` optionally resolves the active local Evaluator
  adapter at inference time, verifies the content-addressed checkpoint manifest
  plus the exact executable adapter files (without rehashing multi-GB optimizer
  state per request), and loads it
  through the existing Transformers/PEFT runtime. Other roles keep the frozen
  registry. With no activation manager provided, old behavior is unchanged.
- A separately authorized rollback restores the previous active Evaluator
  or the frozen certified role. Rollback is idempotent and does not mutate
  the immutable five-role registry.

## Scientific and deployment honesty

- All positive activation tests use **synthetic finalized checkpoint bytes**
  and explicitly injected CPU generator test doubles. These are marked
  `active_simulated`; the default production resolver **refuses** them.
- The production setting `automatic_role_dispatch_enabled` remains false.
- No actual Granite GPU role probe has run. A valid code path is not proof
  that the model or PEFT loader works on the chosen physical GPU.
- A promotion recorded in the action journal is assumed to be produced by the
  trusted `GovernedActionExecutor` under its existing scientific gates.
  A host operator with arbitrary writes to the authoritative state repository
  remains inside the trust boundary; there is no distributed attestation.
- The new scientific-state scorer in #89 is *advisory eligibility* and does
  not itself grant a certification or a promotion.
- Multi-controller/remote routing consistency, GPU health checks, and the
  full autonomous campaign are not demonstrated.
- A production deployment must configure one fenced controller, durable
  shared state, pinned base model snapshots, and a real operator approval
  process for each protected role transition.

## Verified tests

`tests/test_governed_evaluator_activation.py` checks:
scoped approval acceptance/refusal, activation into the actual
`CertifiedRoleReasoner` execution path, no S3 fallback for locally
activated adapters, corrupted adapter refusal, unsuccessful health probe
preserving old routing, explicit simulated/prod separation, separate rollback
approval, retry idempotency, and repeated activation/rollback behavior.

The full CPU suite runs unchanged in the dedicated workflow.

No GPU or cloud resources were created and no production model was activated.

- The existing frozen registry certification metrics remain metadata about the
  original certified role adapter. They must not be interpreted as measured
  quality metrics for a newly activated Evaluator. Scientific promotion evidence
  is recorded separately in the governed action journal.
