# Integration note: execution infrastructure

## Branch and scope

Branch: `feature/execution-infrastructure`.

This branch adds adapter boundaries and safe local implementations. It does not wire the orchestrator, change domain policy, launch training, migrate existing state stores, or claim production authentication/distributed guarantees.

PR #35 is merged in the branch base. No diagnosis, data, evaluation, training, or planner integration notes were present on `main` when this work began. The stable `autonomous-experiment.v1` service protocols are therefore consumed as read-only boundaries; final feature-service wiring remains deferred.

## Queue implementation and adapter

`hephaestus.jobs.JobQueue` defines submission, lookup, leasing, heartbeat, and cancellation signaling. `InMemoryJobQueue` provides deterministic FIFO selection by creation time and job ID inside one process.

Job records include:

- stable job, owner, run, and experiment identity;
- payload reference rather than inline heavy payloads;
- queued, leased, running, succeeded, failed, cancelled, and expired states;
- attempt count and idempotency key;
- lease owner/expiration;
- created, updated, started, and finished timestamps;
- result/error references and cancellation-request status.

Repeated submission with the same idempotency key and identity returns the original job. Reusing a key for a different identity is rejected. Lease-owner checks prevent another local worker from starting, heartbeating, completing, failing, or acknowledging cancellation for the job. Expiration is explicit; retry requeues the same stable job ID and the next lease increments its attempt count.

This adapter has thread safety only within one Python process. It is not durable and makes no cross-process or cross-host ordering, delivery, or exactly-once claim. A future external queue must implement `JobQueue` and preserve these visible state/ownership semantics.

## Worker implementation

`LocalWorker.run_once()` polls one job, leases it, transitions it to running, executes a supplied handler, and records success, failure, or acknowledged cancellation. Handler exceptions are reduced to an exception-type reference so arbitrary exception messages are not persisted as infrastructure errors.

Long-running handlers must call the queue heartbeat method from their execution integration and must cooperate with `cancellation_requested`; the local one-shot worker cannot preempt Python code safely. The final training integration should use the training lifecycle service for semantic control and use the job queue only as execution transport.

## Storage implementation

`ArtifactStore` and `FileSystemArtifactStore` provide:

- immutable `sha256:<digest>` artifact identity;
- expected-hash validation;
- content-addressed paths;
- atomic same-filesystem replacement from a temporary file;
- byte/file put, read, and verification operations;
- structured storage-failure and verification events.

`StateRepository` and `JsonLineStateRepository` preserve JSON/JSONL compatibility. The local repository provides locked, flushed, fsynced single-record appends. POSIX uses `flock` in addition to per-process thread locks. It does not provide multi-record transactions, snapshot isolation, schema migration, or database durability. Existing `src/hephaestus/state/` stores remain unchanged and usable.

`DistributedLockProvider` is the adapter boundary for a future database/coordination-backed lock. `InMemoryLockProvider` is deliberately named and documented as process-local; it supports acquisition, owner-safe heartbeat, expiration, and release without claiming distributed safety.

## Secret provider

`SecretReference` persists only a provider ID and key. `SecretsProvider` resolves a reference at runtime. `EnvironmentSecretsProvider` is the development implementation and reads from the process environment only when `resolve()` is called.

Secret values are not fields of configuration, job, artifact, event, or reference records. This branch does not load `.env` files and contains no credentials.

## Observability entry points

`StructuredEvent` and `EventSink` are the common structured-event boundary. Local sinks include:

- `InMemoryEventSink` for tests;
- `JsonLineEventSink` for local structured logs;
- `MetricsCollector` for in-process lifecycle counters and queue-delay/execution-duration observations;
- `CompositeEventSink` to fan events out to multiple adapters.

Events cover job queued/leased/running/succeeded/failed/cancelled/expired/retry states, cancellation requests, worker polling/idleness/heartbeats, artifact put/verification/storage failures, and health readiness. Events report execution facts only and do not make promotion, approval, lineage, or experiment decisions.

## Configuration

`InfrastructureConfig.from_env()` supports:

- `HEPHAESTUS_STATE_ROOT`;
- `HEPHAESTUS_ARTIFACT_ROOT`;
- `HEPHAESTUS_EVENT_LOG`;
- `HEPHAESTUS_WORKER_ID`;
- `HEPHAESTUS_JOB_LEASE_SECONDS`.

Lease duration is validated as a positive integer. Secret values are intentionally absent from this configuration schema; callers provide `SecretReference` records separately.

## Local development commands

```bash
PYTHONPATH=src python -m hephaestus.infrastructure.cli health --json
pytest -q tests/test_execution_jobs.py tests/test_execution_storage.py tests/test_execution_infrastructure.py
python -m compileall -q src/hephaestus
```

The unit tests require no network service.

## Container behavior

Build from the repository root with `docker build -f docker/Dockerfile -t hephaestus-infrastructure .`. The image:

- runs as non-root UID `10001`;
- copies only application source;
- exposes state and artifact mount points;
- uses environment variables for paths;
- embeds no secrets;
- exposes no unrestricted mutation or training-launch endpoint;
- runs the health CLI as its entry point and health check.

## Contracts consumed and produced

Consumed without modification:

- `autonomous-experiment.v1` identity vocabulary by reference;
- existing backend evidence rule that heavy artifacts are path/hash references;
- existing JSON/JSONL state behavior;
- existing approval, action-boundary, replay, lineage, promotion, evaluation, and control-spine policy.

Produced:

- process-neutral protocols for queue, artifacts, state repository, locks, secrets, events, and telemetry;
- typed local job/artifact/lock/health records;
- local reference adapters and container/health entry points.

No new domain decision contract is produced.

## Known non-production guarantees

- In-memory jobs and locks disappear on process exit and are not safe across processes/hosts.
- JSONL provides single-record append safety only; there is no multi-record transaction or distributed consensus.
- Filesystem atomic replacement assumes temporary and final paths share a filesystem.
- The one-shot local worker is cooperative, not preemptive.
- JSONL event logs and in-process metrics are not centralized observability systems.
- Health checks prove local path readiness, not external dependency readiness.
- Authentication and authorization are not implemented by this branch.
- No cloud queue, object store, database, scheduler, secret manager, or telemetry service is assumed to exist.

## Exact final-integration wiring

The `integration/real-autonomous-loop` branch should:

1. Instantiate concrete `JobQueue`, `ArtifactStore`, `StateRepository`, `DistributedLockProvider`, `SecretsProvider`, and `EventSink` adapters in a composition root outside subsystem/domain logic.
2. Serialize a validated `ExperimentProposal` to an immutable artifact and submit only its artifact reference plus owner/run/experiment identities.
3. Have the execution worker resolve that artifact, then call the stable `TrainingLifecycleService.launch()` entry point; do not call backend or orchestrator internals from the queue adapter.
4. Persist returned `TrainingRunHandle` and execution evidence through the existing governed state paths, using artifact references for heavy evidence.
5. Route cancel intent first through existing approval/action-boundary governance, then call `TrainingLifecycleService.control()` and signal the infrastructure job cancellation cooperatively.
6. Feed infrastructure events to operator inspection/telemetry only; never translate them directly into promotion, rollback, restart, or planner decisions.
7. Replace local queue/lock implementations before multi-process deployment, while retaining local adapters for unit tests.
8. Add end-to-end tests proving proposal-reference submission, governed launch/control, result-reference persistence, replay evidence, cancellation, and duplicate submission without changing feature-owned semantics.
