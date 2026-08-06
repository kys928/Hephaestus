# Integration note: production execution infrastructure

## Branch, base, and scope

- Branch: `feature/production-execution-infrastructure`
- Immutable base: `552200b95b027ed91ac76167619ac9478100e811`
- Base meaning: merged PR #42 (`integration/real-autonomous-loop`)

This branch adds durable execution adapters beside the existing local adapters. It
does not modify the orchestrator, control spine, shared schemas/interfaces, training
lifecycle semantics, diagnosis, planning, evaluation, approval, promotion, lineage,
replay, or action-boundary policy.

The following remain available and behavior-compatible:

- `InMemoryJobQueue`
- `LocalWorker`
- `FileSystemArtifactStore`
- `JsonLineStateRepository`
- `InMemoryLockProvider`
- `EnvironmentSecretsProvider`

## Queue backends

### Process-local reference

`InMemoryJobQueue` remains the deterministic development/test adapter. Its lock and
state disappear with the process. It makes no process, host, durability, delivery, or
distributed-ordering guarantee.

### Durable SQLite baseline

`SQLiteJobQueue` is the fully implemented and locally exercised durable backend. It
uses a file database, WAL mode, `synchronous=FULL`, busy timeouts, foreign keys,
schema version 1, and `BEGIN IMMEDIATE` write transactions.

Persisted fields include payload/owner/run/experiment identities, idempotency key,
status, attempts, lease owner/token/expiry, start/finish/update timestamps,
cancellation request/acknowledgement, result/error references, lease-expiration
count, maximum attempts, dead-letter reason/time, and replay count. An append-only
`job_audit` table preserves transition, actor, reason, and bounded evidence.

Transactions are explicit for:

- idempotent submission: database uniqueness plus identity comparison;
- lease acquisition: expired-lease recovery, FIFO selection, attempt increment, and
  lease-token creation in one writer transaction;
- heartbeat: exact owner/token and unexpired-lease predicate;
- start/complete/fail/cancellation acknowledgement: exact fenced ownership;
- retry and dead letter: atomic state plus audit row;
- dead-letter replay: explicit actor/reason plus preserved prior evidence;
- cancellation request: immediate queued cancellation or persisted active signal.

SQLite writer serialization prevents double leasing between processes sharing the
same correctly locked database file. Every lease has a random fencing token. A late
worker cannot start, heartbeat, complete, fail, or acknowledge after ownership is
lost, even if it still holds an old in-memory `JobRecord`.

This is safe for multiple processes on one host. It is not a cross-host consensus
queue and must not be placed on a network filesystem with unknown lock semantics.
The existing `JobQueue` protocol remains the external queue/PostgreSQL adapter
boundary. This branch does not pretend that a PostgreSQL queue was exercised.

## Worker, dead letter, and restart recovery

`DurableWorker` uses a stable configured worker ID, bounded polling, per-job lease
tokens, a heartbeat thread, cooperative cancellation, exception-type-only error
references, and graceful shutdown signaling.

- `RetryableJobError` is the explicit transport/execution retry signal.
- `NonRetryableJobError` and unclassified handler exceptions dead-letter instead of
  creating an infrastructure retry policy from domain behavior.
- A persisted cancellation request may be acknowledged by the owner worker.
- A cancellation exception without persisted intent cannot manufacture authority.
- Heartbeat failure fences terminal persistence.
- `WorkerShutdown` leaves unfinished work leased; lease expiry performs a clean,
  auditable handoff instead of marking success.
- Maximum attempts, non-retryable failures, malformed payload references, and
  repeated lease expiration move jobs to dead letter.
- Replay requires an actor and reason. Prior dead-letter evidence remains in audit.

After restart, queued/terminal/dead-letter rows and idempotency keys remain. An
unexpired lease cannot be stolen. Expired active leases are transactionally requeued
or dead-lettered according to the persisted attempt/expiration ceilings.

The queue cannot safely detect a database commit failure that prevents the database
from recording the failure itself. It therefore makes no claim to dead-letter
"terminal persistence repeatedly fails" when no durable observation can exist.

## State repository

`SQLiteStateRepository` persists finite JSON-object records with:

- schema version 1;
- globally unique operation IDs with idempotent same-record replay and conflict
  rejection;
- stable timezone-aware timestamps;
- database sequence ordering;
- validated collection, operation, and optional record-key identifiers;
- ordered payload retrieval and envelope retrieval;
- indexed `latest_by_key`;
- compatibility `all` and `get_latest` methods;
- explicit multi-record transaction context with rollback.

It is an adapter and migration target only. No existing `src/hephaestus/state/` store
is replaced in this branch. Migration must be selected and verified collection by
collection in later integration work.

## Locking

`SQLiteLockProvider` is a non-process-local lock for processes sharing the same
SQLite database. Acquisition/replacement is serialized by database transactions;
heartbeat/release require exact owner and lease token. It supports expiry, timeout,
contention events, owner assertions, context-manager cleanup, and lost-lock errors.

`PostgresLeaseLockProvider` is the optional cross-host lock. It uses PostgreSQL lease
rows with `INSERT ... ON CONFLICT ... WHERE expires_at <= ... RETURNING` and exact
owner/token predicates for renewal/release. `psycopg` is loaded only through the
optional adapter constructor. No real PostgreSQL service was available or exercised
for this branch.

## Artifact storage

`FileSystemArtifactStore` remains the local immutable SHA-256 adapter.

`S3ArtifactStore` is an optional injected-client adapter with:

- `sha256:<digest>` identity and content-addressed keys;
- expected-hash checking before upload;
- existing-object metadata integrity checks;
- streamed file upload;
- multipart upload with atomic final visibility at completion;
- cancellation and abort on every incomplete multipart failure;
- cleanup discovery for dangling uploads;
- existence, read, full-content verification, and bounded telemetry;
- configurable server-side encryption;
- presigned GET only through an explicit caller method.

Artifact metadata contains hashes and media type only; it contains no secret values.
Unit tests use an injected fake S3 client. No real object store was exercised.

## Secrets

- `EnvironmentSecretsProvider` remains the development adapter.
- `FileMountedSecretsProvider` rejects traversal, symlinks, non-regular files,
  group/other permissions, and (where available) wrong ownership.
- `InjectedSecretsProvider` is the cloud-neutral client boundary.
- `AwsSecretsManagerProvider` accepts an injected client or optional boto3 client.

Persisted configuration uses `SecretReference(provider, key)` only. Raw values are
not job/config/artifact/event fields. Event construction redacts secret-like attribute
names and bounds unstructured strings. No real secret manager was exercised.

## Telemetry

Existing JSONL, in-memory, metrics, composite, and null sinks remain. New adapters and
events cover submission, queue delay, lease/recovery, heartbeat failure, execution,
retry, dead letter/replay, upload/download/verification, state failure, lock
contention, worker start/stop/handoff, and component readiness.

`emit_safely` prevents a telemetry exporter failure from changing an already durable
infrastructure result. `OpenTelemetryEventSink` imports OpenTelemetry only through
`from_installed`; the core package has no mandatory telemetry dependency. No real
collector/exporter was exercised.

## Health, configuration, and migrations

`DependencyHealthService` reports database, queue, artifact store, lock service,
secret provider, worker ability, and migration state separately, plus configured
root checks. Required checks alone determine readiness; an unavailable required
dependency cannot report ready.

`InfrastructureConfig` validates backend choices, SQLite path, DSN reference,
poll/lease/attempt/expiry settings, artifact bucket/prefix/encryption, secret backend,
telemetry backend, required health components, and shutdown timeout. It accepts a
database DSN reference but no raw DSN field.

`hephaestus-infrastructure migrate --json` initializes the three SQLite schemas.
`health --json` does not migrate. `service` emits readiness without mutation APIs.
`worker --handler module:function` starts the durable worker with an installed,
deployment-controlled handler; executable callables are never serialized into jobs.

## Packaging and container

`pyproject.toml` adds a lightweight console entry point and optional extras for
PostgreSQL, S3/AWS secrets, telemetry, and tests. Core dependencies remain empty.

The Dockerfile:

- runs as non-root UID `10001`;
- installs the core package without optional SDKs;
- declares mounted state/artifact paths;
- defaults to SQLite and explicit full readiness;
- uses a migration command separate from service/worker commands;
- handles SIGTERM directly;
- embeds no credentials and exposes no unrestricted mutation API.

The image is a deployment boundary, not hardened orchestration. A derived image must
install the selected optional extras and composition-root handler.

## Consumed and produced contracts

Consumed without modification:

- shared autonomous-experiment payloads only through immutable payload references;
- `TrainingLifecycleService.launch/status/control` as the eventual governed handler
  boundary;
- backend heavy-evidence reference/hash rules;
- approval, action, stage, promotion, lineage, replay, and control-spine policy.

Produced:

- durable infrastructure `JobRecord` state and transition audit;
- immutable artifact records;
- JSON-safe state envelopes;
- fenced lock leases;
- secret references and runtime-only resolutions;
- non-authoritative operational events/readiness evidence.

No infrastructure status promotes, retries a domain intervention, approves a launch,
changes a lineage, or selects an experiment.

## Exact integration wiring

1. In a composition root outside domain/orchestrator code, select queue, state,
   artifact, lock, secret, event, and health adapters from validated configuration.
2. Resolve database/object-store credentials from `SecretReference` at runtime only.
3. Serialize an already validated and approved `ExperimentProposal` to an immutable
   artifact. Submit only its artifact reference plus owner/run/experiment identity.
4. Configure a worker handler that verifies the artifact hash, deserializes the known
   schema, rechecks current approval/action-boundary evidence, and calls the stable
   `TrainingLifecycleService.launch()` entry point.
5. Persist the returned `TrainingRunHandle` and heavy evidence references through the
   existing governed state paths. Do not infer promotion from job success.
6. Route cancel intent through existing governance first. Only then persist queue
   cancellation and call `TrainingLifecycleService.control()` as applicable.
7. Feed operational events to inspection/telemetry. Diagnosis/planning may consume
   persisted evidence through their own contracts; observability makes no decision.
8. Migrate existing JSONL collections one at a time with operation IDs, count/hash
   verification, rollback, and replay tests. Do not perform a blanket store swap.
9. Use SQLite only for one-host multi-process deployments. Select and conformance-test
   a PostgreSQL queue before cross-host workers; the PostgreSQL lock alone does not
   make the SQLite queue distributed.
10. Add opt-in live-service tests in the deployment environment before claiming
    PostgreSQL, S3, secret-manager, or OpenTelemetry production readiness.

## Tested guarantees and non-guarantees

Fully exercised locally without network services:

- SQLite durable/idempotent submission and restart;
- competing-process atomic lease;
- heartbeat, fencing, expiry recovery, cancellation, retry, dead letter, and replay;
- durable worker heartbeat, normalized failures, cancellation, and shutdown;
- SQLite state ordering, idempotency, concurrency, transaction rollback;
- SQLite lock contention, renewal, expiry, timeout, lost-lock, exception cleanup;
- filesystem and fake-S3 hash/immutability/multipart/abort behavior;
- mounted/injected secret behavior and telemetry redaction;
- migration and component readiness;
- missing-optional-dependency core imports.

Not exercised against real external services:

- PostgreSQL;
- S3-compatible object storage;
- AWS Secrets Manager or Vault;
- OpenTelemetry collector/exporter;
- Docker daemon/image runtime.

No exactly-once execution is claimed. The durable queue provides at-least-once
delivery with idempotent submission and fenced terminal persistence. A handler may
perform an external side effect before crashing; that side effect must have its own
idempotency contract. SQLite is not cross-host distributed. S3 behavior ultimately
depends on the selected compatible service. Authentication/authorization and cloud
service hardening remain deployment responsibilities.
