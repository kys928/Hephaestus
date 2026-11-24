# Project Anvil: blackboard semantics and persistence contract

**Status:** CPU contract rehearsal and review. Production activation and certification remain gated. Related: #217, #213, #219. Baseline source: `f5201350ddf05c4023ff3ca55b1a72f8714ffc8c`. Phases 0 and 1 remain open.

## 1. Architectural invariant

Anvil is the shared, evolving scientific blackboard. Blacksmiths independently read, question, propose, criticize, revise and test knowledge against it. The semantic protocol and knowledge state do not depend on PostgreSQL, SQLite, LanceDB, S3 or a blockchain. The Forge performs admitted actions; deterministic governance decides authorization; independently verified evidence decides scientific acceptance. No model gains authority by submitting a record. Parallel inference/submission is distinct from physical commit coordination. A writer is infrastructure, not a privileged thinker.

The existing eight-phase, 45-operation staged workflow remains an admitted capability. Do not add a second scheduler, production root, role registry or independent memory service. The existing five certified identities and frozen v1 response contract remain valid; the separate provisional Coder is not certified by this design.

## 2. Knowledge representation

Use an append-only logical contribution history and deterministic, versioned projections of each investigation. These form a typed, traversable knowledge graph; no graph database is required. Each node has an immutable ID, schema version, investigation identity and scope. Nodes: objective with frozen acceptance criteria; observation; verified evidence binding; hypothesis; question; challenge; intervention proposal; prediction; task request; experiment result; revision; uncertainty/abstention; conclusion/verdict. Edges: supports, contradicts, challenges, tests, answers, depends_on, derived_from, revises, supersedes. Validate endpoints, scope and relation types. Allow explicit reviewed cross-investigation citations without silently merging lineage. Preserve disagreement and prior records.

Distinguish epistemic status (proposed, observed, independently verified, contradicted, superseded, inconclusive) from action status (proposed, admitted, executed, independently verified, rejected). A model assertion never becomes a machine-verified observation merely by being stored. A challenge is a first-class contribution, not an overwrite. Evidence resolution uses registered, hash-verified artifacts and lineage bindings. Preserve exact generation receipts; mark claims and hypotheses as such.

Suggested logical collections: investigations; contributions; relations; evidence_bindings (reuse scientific evidence store); tasks/results (references to existing job and action ledgers); projections; embedding_index (derived and rebuildable). Protected material and raw receipts stay under existing custody rules. Do not index protected material into an unrestricted retrieval corpus. Large artifacts stay in ArtifactStore, with content-addressed references, not embedded in hot knowledge records.

## 3. Shared message protocol (proposed v2)

Universal envelope, role-specific typed payloads. Model-authored fields: message_kind, claim/question/proposal payload, referenced contribution/evidence IDs, rationale summary, uncertainty, requested capability and anticipated test (as appropriate). Runtime-assigned fields: schema version, contribution ID, idempotency key, investigation/run/lineage, exact model and adapter identity, inference receipt reference, context snapshot version and hash, validated parent IDs, canonical payload hash, provenance and commit sequence/time. A model must not forge these fields. Persist raw accepted/rejected inference receipts under current policy. Canonicalize using a specified stable encoding before hashing; version it.

Supported intents: observation claim, hypothesis, evidence request, capability request, intervention proposal, critique, revision, verification result reference, abstention, verdict proposal. Validate schema, scope, authorization, evidence identity and references separately. The runtime may reject malformed or unsupported messages, never silently reinterpret a wrong scientific choice as the desired one. Frozen v1 outputs may be faithfully projected with their original bytes; missing semantic information stays unknown. Future v2 fine-tuning requires separate scientific acceptance.

A Blacksmith receives the objective, exact committed snapshot identity, relevant evidence and hypotheses, unresolved questions, prior challenges, outstanding tasks and available capability descriptors. It may independently submit multiple distinct contributions within its role/scope. Context assembly is deterministic for replay, bounded and provenance-linked. Exact filters/graph traversal first; vector retrieval later when useful, with embedding model/revision and index version recorded. Embeddings are derived, never a source of truth. Stale context is permissible for advisory claims if explicitly bound; stale decisions or actions require revalidation against current state and evidence.

## 4. Investigation coordination and convergence

Existing control plane selects eligible specialist/capability invocations from unresolved questions, contradictions, evidence deficits and progress. Model requests capabilities, not arbitrary shell access or hardcoded next-agent routing. Independent scientific and governance gates admit actual operations. Scheduler may invoke multiple specialists for the same snapshot. Submission acknowledgements only follow durable committed acceptance, not inference completion or enqueue. Once new knowledge commits, eligible readers can retrieve it; no fixed dialogue order.

For convergence, use preregistered objective criteria, discriminating predictions, evidence quality, coverage of material unresolved challenges and independently verified results. Stop with distinct terminal states: resolved, inconclusive (no defensible next experiment / exhausted evidence), blocked (external dependency or approval), budget/deadline exhausted, or cancelled. Record reason, evidence and outstanding uncertainty. A unanimous vote, model confidence or repeated hypothesis is not sufficient. Detect repeated proposal/context signatures, duplicate requests, lack of new evidence, stale action inputs, dependency cycles and diminishing value. Bound per-investigation attempts, inference tokens, elapsed time and authorized compute; never spend on an unadmitted proposal. No autonomous promotion from an Anvil conclusion.

## 5. Storage-independent contract (proposed extensions, not a second backend)

Reuse `storage/base.py::StateRepository`, `ArtifactStore`, `DistributedLockProvider` and existing transaction/idempotence ownership. Introduce only the narrow blackboard semantics missing from the current interface, exposed through the existing `ProductionCompositionRoot`:

- append_contribution(validated envelope, idempotency key, expected dependencies): durable receipt with ID, canonical hash, scope and monotonic committed sequence; duplicate same identity and payload returns original receipt, conflicting payload rejected.
- read_snapshot(investigation, as_of_version): consistent, attributable knowledge projection and exact committed frontier.
- query(snapshot, typed filter, relation traversal, optional embedding query): scope-correct results with provenance and index freshness. Retrieval may be eventually indexed; it cannot invent an accepted record.
- replay(investigation, committed range): reconstruct accepted logical history and projection; verify integrity and sequence/gaps.
- checkpoint/restore: independently verified sealed artifacts plus an independently retained accepted frontier; never acknowledge rollback of accepted records.
- reconcile(interrupted/unknown submission identity): recover committed or uncommitted outcome without duplicating accepted records; unknown remains unknown if no safe proof.

Invariants: no acknowledged data loss, unique logical identities, atomic contribution plus relationship admission (or explicit pending state hidden until commit), durable acceptance before acknowledgement, deterministic replay, explicit snapshot/freshness, conflicting retry rejection, isolation across investigations, authorization at operational boundaries. A stale reader must not be able to commit a decision under a changed prerequisite. A replacement writer may not overlap an old writer; qualify fencing/ownership independently of object-storage conditional writes. A read replica/index may be rebuilt from verified committed events. Actual job, grant, lease, approval and execution transactions remain in the existing qualified governance path until separately replaced and proven. Do not present LanceDB retrieval capability as a transactional replacement for these operations.

## 6. Integration map

- `llm/role_contract.py`, certified runtime and inference receipts: retain v1; add versioned semantic admission without breaking certified models.
- `state/evidence_store.py` and `state/experiment_memory.py`: extend existing evidence bindings, supersession and historical retrieval into the knowledge projection; no separate evidence authority.
- `storage/base.py`: extend the narrow protocols where missing semantics require it; keep existing adapters.
- `production/composition.py`: compose Anvil within the existing control plane.
- `jobs/durable_worker.py` and durable queue: reuse eligibility, job claims, recovery and leases; do not create a competing scheduler.
- Existing action registry/authorization, staged control, and independent verifier: retain actual execution/approval boundaries. Issue #219 is a deferred optional provenance experiment, not a persistence dependency.

## 7. LanceDB qualification against this contract

Actual evidence (2026-11-24): [basic run](https://github.com/kys928/Hephaestus/actions/runs/45438393585), [fault controls](https://github.com/kys928/Hephaestus/actions/runs/45438759956), source `a485cebf85f3686a58286776a97666956a9d6dcc` on `ops/anvil-lancedb-s3-qualification`. LanceDB 0.25.3, synthetic 8D vectors, GitHub CPU runners, Czech S3, no Pods. Direct S3 create/read/range/list, structured inserts, metadata filtering and vector nearest-neighbor passed. Stale reader saw six records until explicit refresh, then seven. Separate runner recovered seven. S3 If-None-Match conditional creation was ignored; second object bytes visible. Four direct concurrent writers acknowledged four appends, three of those appends were missing on independent readback (75% loss in this trial, not an estimated general loss rate). In isolated single-writer kill control, seven previously committed rows survived; pending 256-row batch was unacknowledged and invisible; independent runner confirmed. Serialized control retained all ten rows. This proves bounded feasibility, not production safety.

| Contract requirement | Status |
| --- | --- |
| Structured and vector retrieval | Demonstrated on synthetic rows |
| Explicit reader refresh | Demonstrated |
| Fresh-process and independent-runner reconstruction | Demonstrated for observed rows |
| Serialized appends | Demonstrated in small control |
| Single process interruption | Demonstrated in one bounded fault trial |
| Direct concurrent durable acceptance | Failed; denied |
| Provider conditional create | Failed; denied |
| Idempotent retry and conflicting identity rejection | Not qualified |
| Atomic typed graph/event acceptance, scope and stale dependency handling | Not qualified |
| Crash-safe multi-producer intake, acknowledgements, unknown outcomes | Not qualified |
| Writer failover with independent fencing | Not qualified |
| Full history replay, snapshot/restore, independent frontier/rollback protection | Not qualified |
| Job/grant/approval/lease transaction equivalence | Not qualified |

**Disposition:** candidate structured/vector persistence or rebuildable retrieval projection only. Not approved as the authoritative operational database. No direct concurrent writes. Conditional-create dependent S3 artifact publication must also not be assumed safe on this endpoint; assess the existing S3 adapter's assumptions separately. For future qualification, evaluate the minimum existing durable admission/serialization mechanism with a clearly scoped independent owner, then inject duplicate and conflicting submissions, simultaneous producers, delayed old writers, process kills before/during/after commits, acknowledgement loss, stale context, corrupt/missing records, interrupted projection, restart on a new runner and independently verified snapshot restoration. Multiple trials with preregistered outcomes and immutable receipts; zero acknowledged loss/duplicate/conflicting accepted IDs, no stale or unauthorized action, exact reconstructed accepted frontier. Do not weaken existing SQLite guards or treat a CI concurrency group alone as production fencing. If durable writer authority and replay cannot be established without another bespoke database, reject LanceDB as authoritative rather than change blackboard semantics.

## 8. Review and next gates

Design is storage-independent and may be reviewed while Phase 0/1 gates remain open. No deployment, physical migration, training, paid allocation, blockchain anchoring, or production activation authorized by this document. Do not implement Phase 2 against unqualified production foundations. Preserve Czech/Global volumes and current archives. Review semantic schema and persistence contract first, implement CPU-only contract tests against existing supported storage, then qualify candidate adapters in isolation. Compare against existing deterministic state/evidence paths; report results separately from actual specialist collaboration and scientific certification.


## 9. Implemented rehearsal boundary

`schemas/anvil.py` defines advisory contribution kinds, typed links, runtime provenance and immutable acceptance receipts. The model parser rejects runtime fields, objective creation, machine evidence, results and supersession. The trusted caller supplies verifiers for inference/context provenance and scoped evidence. The canonical production runtime exposes `anvil_rehearsal()` only under an explicit bounded-rehearsal storage profile.

`state/anvil.py` reuses the admitted host-local SQLite repository and writer fence. An append logically commits the contribution and its edges together. For identical investigation/idempotency identities, the original sequence and hash survive retries. Replay validates the scoped history hash chain, known frontiers, sequence and references. Link targets must have existed in the submitted context and belong to the same lineage. Advisory contributions can cite older snapshots. Proposals, capability requests and verdicts require the current frontier and still carry no execution authority. Query operates on a typed immutable snapshot. Checkpoint restore requires a separately retained frontier and appends only a verified suffix in one transaction. Missing acceptance remains unknown on reconciliation.

The rehearsal currently verifies restart, conflicting retry, stale proposals, scoped links, lost acknowledgement and stale restore. Trusted machine-event admission, real model receipt binding, independent frontier retention, cross-runner recovery, replacement writer fencing and index reconstruction remain qualification targets. LanceDB may be tested as a rebuildable projection; the original direct-writer configuration remains denied.

The V1 compatibility view preserves the frozen decision, action, variable, confidence and evidence fields. The test replays the complete JSON excerpts recovered in [the independent triad observation](https://github.com/kys928/Hephaestus/actions/runs/45376969755). The Diagnostician still recommends a preprocessing change and the Planner still recommends an evaluation recheck; the historical Coder admission remains unsatisfied.
