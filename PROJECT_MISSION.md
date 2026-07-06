# PROJECT_MISSION.md — Hephaestus Project Mission

## Purpose

Hephaestus is an autonomous research-engineering system for training, repairing, evaluating, and stabilizing language models from scratch.

Its long-term goal is to help run the full model-development lifecycle with minimal human intervention while preserving scientific discipline, reproducibility, rollback safety, and honest uncertainty.

Hephaestus is allowed to aim big. It is not only an experiment tracker, checkpoint comparator, or narrow repair loop. The intended direction is a system that can eventually help manage:

- data sourcing and validation
- tokenization and formatting
- architecture and configuration control
- training launch and monitoring
- checkpoint validation
- evaluation and ranking
- regression detection
- recovery from instability
- next-experiment decision-making
- promotion, rejection, rollback, branching, or restart of model lineages

However, Hephaestus must never fake certainty. Any autonomy must be earned through evidence.

## Current reality

The current implementation should be treated as an audit-first control and governance shell, not a complete autonomous training factory.

What is currently real:

- a mandatory staged control spine
- role-separated workflow execution
- dry-run training lifecycle simulation
- dataset profile and manifest records
- preprocessing reports and trainable data contracts
- training plans and launch configs
- runtime monitoring records
- evaluation, repeatability, promotion, and certification logic
- approval and governance metadata
- replay verification
- persistent JSON-line state stores
- an early governed operator console
- bounded code-edit proposal governance and dry-run execution records

What is not yet complete:

- general real model training
- production-grade dataset acquisition
- full preprocessing implementation
- rich evaluator metric coverage
- polished operator UX
- production packaging and deployment
- fully autonomous recovery/self-healing behavior

When in doubt, treat the system as a control tower whose factory machinery is partly simulated and partly scaffolded.

## Core mission

Build a disciplined multi-agent research-engineering system that can eventually:

- pretrain models from random initialization
- recover from training failures and regressions
- compare candidate model families against baselines
- diagnose whether problems come from data, tokenizer, architecture, optimizer, runtime, checkpoint lineage, or evaluation
- decide when to continue, pause, roll back, branch, quarantine, restart, or stop
- drive a model family toward a stable, general-purpose baseline with minimal human input

Hephaestus should help answer:

- What is broken?
- How confident are we?
- Which layer is most likely responsible?
- What action has the highest expected value?
- Did the model truly improve, or did it only get smoother?
- Should this run continue or be killed?
- Should this family be repaired, branched, quarantined, or restarted?
- Are we actually moving toward a stable base model?

If a proposed feature does not improve these decisions, it is probably not core.

## Operating posture

Hephaestus must behave like a disciplined research lab, not a freeform agent swarm.

The system must be:

- schema-driven
- audit-friendly
- rollback-capable
- strongly versioned
- backend-aware
- evidence-first
- able to say “inconclusive” when signal is weak
- conservative in claims even when ambitious in goals

A recommendation is not truth. Agent agreement is not evidence. Confidence should increase only when supported by repeated outcomes, controlled comparisons, stable loading, reproducibility, deterministic regressions not firing, consistent qualitative improvement, and alignment between reward-model and deterministic signals.

## Mandatory control spine

All workflow execution must preserve the explicit staged control spine:

1. Judge entry
2. Planner
3. Data acquisition and audit
4. Data preprocessor
5. Training engineer
6. Runtime monitor
7. Evaluator
8. Judge exit

Do not collapse these roles into a monolithic “smart orchestrator.” Coordination belongs in `src/hephaestus/control/`. Role behavior belongs in `src/hephaestus/roles/`. Decision-critical records must persist through `src/hephaestus/state/`. Cross-component payloads should use explicit schemas in `src/hephaestus/schemas/`. Policy decisions belong in `src/hephaestus/policy/`.

## Multi-agent design principles

Hephaestus should use specialists only when they map to real workflow needs.

Useful responsibilities include:

- Planner: chooses the next experiment or intervention
- Data curator: controls dataset choice, mixture, filtering, weighting, preprocessing, and stage-specific data policy
- Training engineer: owns training recipes, launch configs, optimizer settings, masking rules, scheduler changes, and backend-compatible execution plans
- Runtime monitor: watches launches, events, incidents, metrics, and failure conditions
- Evaluator: reads metrics, logs, probes, deterministic tests, and outputs to summarize actual behavioral change
- Judge: combines deterministic checks, policy, ranking, and evidence into a verdict
- Lineage manager: tracks checkpoint families and decides whether to continue, branch, quarantine, archive, or kill a lineage
- Incident responder: handles crashes, corrupt artifacts, missing metrics, invalid comparisons, and broken resumes
- Memory/retrieval agent: finds related prior runs, failures, interventions, dead ends, and promotion blocks
- Coder agent: optional and constrained; only edits code when necessary, scoped, approved, and tested

Do not add agents because they sound impressive. Add them only when there is a concrete workflow boundary.

## Full training lifecycle target

Hephaestus should eventually support staged model development from scratch, including:

- tokenizer validation
- smoke-test model training
- early-base pretraining
- scale-up continuation
- stabilization and variance testing
- targeted repair
- ranking or preference repair
- alignment or wrapper specialization
- final baseline selection

Each stage should have:

- entry conditions
- exit conditions
- data policy
- evaluation pack
- failure triggers
- rollback, restart, branch, or quarantine rules

## Failure domains Hephaestus must reason about

Hephaestus must not immediately blame “the model” when a run fails. It should help isolate the likely failure source.

It must reason about:

- data quality, duplication, contamination, formatting, prompt-target boundaries, wrappers, diversity, and curriculum
- tokenizer mismatch, vocabulary drift, special-token misuse, segmentation errors, EOS/EOT bias, and representation bottlenecks
- architecture and config issues such as scaling choices, head/dimension mismatch, context length, initialization, dropout, normalization, tying assumptions, loader contracts, and schema mismatch
- training dynamics such as exploding/vanishing gradients, optimizer pathologies, unstable warmup/decay, objective mismatch, repetition loops, undertraining, collapse, batch construction issues, and checkpoint transition instability
- systems/runtime issues such as failed runs, corrupt checkpoints, bad resumes, missing artifacts, inconsistent seeds, hardware interruptions, throughput bottlenecks, data-loader faults, I/O corruption, wrong environment variables, and backend-specific mismatches
- evaluation issues such as weak eval packs, reward-model overtrust, scoring the wrong thing, eval overfitting, misleading variance, decoding artifacts, false regressions, and benchmark vanity
- decision-making issues such as repeating dead ends, changing too many variables at once, continuing poisoned lineages, restarting without diagnosis, and making confident but poorly grounded recommendations

Preferred debugging order:

1. Eval integrity
2. Reproducibility and launch contract
3. Data integrity
4. Tokenizer and wrapper consistency
5. Checkpoint loading and architecture contract
6. Training dynamics
7. Model-family limitations

Do not jump to the hardest explanation first.

## Hard safety rules

Hephaestus must not:

- mutate frozen eval packs without explicit approval
- silently rewrite critical loader logic
- use non-strict checkpoint loading as the default in critical paths
- edit forbidden files
- promote a checkpoint that fails hard regression gates
- allow unapproved datasets into the training mixture
- continue a clearly poisoned lineage just because compute has already been spent
- launch real training from UI mutation routes
- apply patches directly from the operator console
- bypass approval governance
- commit generated state, artifacts, caches, checkpoints, secrets, datasets, or model weights

If a change weakens replay verification, approval policy, eval-pack safety, lineage tracking, or role boundaries, it is probably wrong.

## Lineage policy

Checkpoint families are first-class objects.

For every lineage, Hephaestus should track:

- origin recipe
- architecture contract
- tokenizer contract
- data policy
- major interventions
- known pathologies
- best candidate
- current trust level
- rollback candidates
- certification state

A lineage can be exploratory, promising, stable, suspect, poisoned, archived, or deprecated.

The system must be able to quarantine bad families instead of endlessly repairing them.

## Data policy

Hephaestus must know exactly what data enters every run.

Every experiment should have a data manifest that records:

- dataset IDs
- versions or hashes
- row counts
- mixture weights
- filtering profile
- preprocessing profile
- chunking/windowing policy
- wrapper policy
- hard-negative/support/synthetic-data usage
- license and provenance information
- contamination and integrity risk where known

Vague labels are unacceptable.

## Evaluation policy

Every target should define frozen evaluation packs where appropriate.

Evaluation may include:

- generation probes
- continuation prompts
- ranking sets
- regression prompts
- length and termination checks
- repetition checks
- structure-sensitive tests
- stage-specific validation tasks
- sample-based human review bundles
- reward-model or judge scoring

Reward-model scoring is allowed, but it must sit beside deterministic scoring. If deterministic regressions fire, the candidate cannot be blindly promoted even if reward is high.

## Current Codex/agent guidance

When working in this repository, agents should prefer small, reviewable, PR-sized changes.

Before making architectural changes, read:

- `PROJECT_MISSION.md`
- `AGENTS.md`
- `docs/architecture.md`
- `docs/control_spine.md`
- `docs/stage_policy.md`
- `docs/evaluation_policy.md`

Agents must not pretend incomplete systems are finished. Prefer typed stubs and TODO markers over fake-complete implementations.

Good agent tasks:

- add tests for an existing governance boundary
- document what is implemented versus audit-only
- improve replay verification evidence
- strengthen schema validation
- improve CLI smoke checks
- add a narrow backend contract
- add a deterministic utility
- expand evaluation policy carefully
- make safety checks more enforceable only where tests prove the behavior

Bad agent tasks:

- “make Hephaestus autonomous”
- “implement full training” in one pass
- broad rewrites without tests
- new agents without workflow need
- changing eval packs casually
- weakening approval gates to make tests pass
- hiding missing functionality behind optimistic docs

## Development priorities

Build in this order:

1. Schemas, registries, and lineage tracking
2. Backend contracts
3. Data manifests and stage definitions
4. Frozen eval packs and deterministic scorecards
5. Reward-model/judge scoring
6. Promotion, rollback, branch, quarantine, and restart logic
7. Retrieval and experiment memory
8. Constrained code-editing and broader autonomy
9. Advanced recovery and self-healing behavior
10. Production packaging and operator UX

## Final standard

Hephaestus should become a system that can eventually run a full training program with minimal user intervention.

But it must get there through disciplined capability, not fantasy.

Always prefer:

- trustworthy over flashy
- recoverable over clever
- diagnosable over automatic
- controlled autonomy over uncontrolled ambition
- evidence over agent confidence
- rollback safety over speed

If forced to choose between bigger scope and better judgment, choose better judgment.
