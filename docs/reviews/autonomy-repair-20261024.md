# Autonomous recovery repairs — evidence ledger

Baseline inspected: PR #81 `796785b87797c6019499a03d1ec80fd2bc31f87e`,
PR #82 `f1a8470e5b0f6e5b296eca1c2b7d705878d98bfd`, main
`304875f0c7d65650553aafcce608b5d345d93d9d`. Both PRs were draft/unmerged.
PR #82 CI runs 40082119343/40082119348 passed. Independent local baseline:
485 passed, 6 skipped. No weights, cloud jobs, training or production activation.

## Governance repair

- P0-02 **confirmed**: nonempty caller approval strings and a promotion boolean
  authorized sensitive actions. Seven initial adversarial cases reproduced this
  and contract/vocabulary defects before changes.
- Executor now resolves an actual `DecisionStore` approval event, requires its
  latest decision/request, operator identity, correct outcome/type, expiry,
  run/lineage/action/checkpoint/child scope and current lineage digest. Approval
  requests can be prepared using `build_action_approval_request`; this never
  grants approval. Legacy unscoped approvals intentionally fail closed.
- Promotion re-reads hash-addressed evaluation/manifest bytes, verifies the
  candidate's persisted run/lineage and recalculates promotion and certification
  policy. Caller booleans, status, confidence and certification claims cannot
  grant promotion. This is lineage metadata mutation, never active-role routing.
- P1-07 **confirmed**: Controller `noop` was rejected as an unknown action. It is
  now explicitly non-executable, with no expansion of the action registry.
- P1-11 **confirmed, single-host scope**: all lineage writes and action admission,
  preparation, recovery and application share a kernel-owned process lock.
  No lease expiry can permit a paused old writer to continue past takeover.
  Startup reconciles prepared intents. Before/after divergence blocks recovery.
  SQLite and JSON remain separate stores; no distributed atomicity is claimed.
- P1-12 **confirmed**: entire runtime contract is independently hash-pinned;
  returned profiles cannot mutate live rules. Base identities/revisions are
  pinned to original training config. `scripts/verify_frozen_role_contract.py`
  independently compares five profiles and byte-for-byte prompt rendering with
  immutable training commit c831bbfc0e2054d54fccd1ef374fd84722612c79.
  Adapter configuration compatibility remains in the subsequent cache repair.

Validation: full CPU regression **503 passed, 6 skipped**. Dedicated tests cover
forged, stale, expired, superseded and mismatched grants, unknown operators,
changed targets, valid retries, cross-process writer exclusion, startup healing,
existing mid-branch and post-write crashes, mutable profiles and contract edits.

Deployment limits: local-filesystem flock semantics are required. Multi-host
controllers remain unsupported. The approval store is a trusted operator control
plane; access to append decisions must be restricted by deployment. No adversary
with arbitrary filesystem/source write access is modeled as an untrusted caller.
Real inference, PEFT continuation and independent scientific certification remain
unverified. The remaining findings are tracked in subsequent repair commits.
