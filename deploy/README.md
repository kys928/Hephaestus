# Production execution adapter deployment boundary

This directory describes reviewable local and container operation. It is not a
hardened cloud deployment, an authentication system, or authority to launch an
experiment. Infrastructure transports already-governed payload references only.

## Durable local baseline

The supported dependency-free durable baseline is one SQLite database on a local
filesystem with trustworthy POSIX locking. It supports multiple worker processes on
one host. Do not place the database on NFS or claim cross-host consensus.

```bash
export HEPHAESTUS_STATE_ROOT=state
export HEPHAESTUS_ARTIFACT_ROOT=artifacts
export HEPHAESTUS_QUEUE_BACKEND=sqlite
export HEPHAESTUS_DATABASE_BACKEND=sqlite
export HEPHAESTUS_SQLITE_PATH=state/infrastructure.sqlite3
export HEPHAESTUS_HEALTH_REQUIRED=state_root,artifact_root,database,queue,artifact_store,lock_service,worker,migrations

hephaestus-infrastructure migrate --json
hephaestus-infrastructure health --json
```

Migration is explicit. Health does not call a migration and reports readiness false
when required schemas or adapters are unavailable.

## Worker and service commands

The service command emits recurring dependency readiness and exposes no mutation or
training-launch API:

```bash
hephaestus-infrastructure service --interval-seconds 30
```

The worker requires a deployment-controlled `module:function` handler. The handler
is installed code, not a serialized job callable. It receives a `JobRecord` and
`JobExecutionContext`; persisted jobs contain only payload references.

```bash
hephaestus-infrastructure worker \
  --handler your_composition_root:handle_job
```

SIGTERM/SIGINT stops new leasing. A running handler may finish, acknowledge an
already persisted cancellation request, or cooperatively hand off by raising
`WorkerShutdown`. Unfinished work is never marked successful and becomes recoverable
after lease expiry.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `HEPHAESTUS_STATE_ROOT` | `state` | Local state/event root |
| `HEPHAESTUS_ARTIFACT_ROOT` | `artifacts` | Filesystem artifact root |
| `HEPHAESTUS_EVENT_LOG` | `<state>/infrastructure_events.jsonl` | Local structured events |
| `HEPHAESTUS_WORKER_ID` | hostname | Stable worker-process identity |
| `HEPHAESTUS_QUEUE_BACKEND` | `memory` | `memory`, `sqlite`, or configured `postgres` boundary |
| `HEPHAESTUS_DATABASE_BACKEND` | `sqlite` | `sqlite` or `postgres` |
| `HEPHAESTUS_SQLITE_PATH` | `<state>/infrastructure.sqlite3` | Durable queue/state/lock database |
| `HEPHAESTUS_DATABASE_DSN_REF` | unset | `provider:key` reference; never a raw DSN |
| `HEPHAESTUS_JOB_LEASE_SECONDS` | `60` | Lease duration |
| `HEPHAESTUS_QUEUE_POLL_SECONDS` | `1` | Bounded idle poll interval |
| `HEPHAESTUS_JOB_MAXIMUM_ATTEMPTS` | `3` | Transport/execution attempt ceiling |
| `HEPHAESTUS_MAXIMUM_LEASE_EXPIRATIONS` | `3` | Crash/lease-loss ceiling before dead letter |
| `HEPHAESTUS_ARTIFACT_BACKEND` | `filesystem` | `filesystem` or `s3` |
| `HEPHAESTUS_OBJECT_STORE_BUCKET` | unset | Required for S3 |
| `HEPHAESTUS_OBJECT_STORE_PREFIX` | `hephaestus` | Content-addressed object prefix |
| `HEPHAESTUS_OBJECT_STORE_SSE` | `AES256` | Server-side encryption selection |
| `HEPHAESTUS_SECRETS_BACKEND` | `environment` | `environment`, `file`, AWS, or injected provider |
| `HEPHAESTUS_SECRETS_ROOT` | `/run/secrets` | Strict-permission mounted secret root |
| `HEPHAESTUS_TELEMETRY_BACKEND` | `jsonl` | `jsonl`, `none`, or `opentelemetry` |
| `HEPHAESTUS_HEALTH_REQUIRED` | roots only | Comma-separated readiness requirements |
| `HEPHAESTUS_SHUTDOWN_TIMEOUT_SECONDS` | `30` | Deployment shutdown budget |

Persist references such as `file:postgres_dsn`; resolve them only in the composition
root. Never put passwords, tokens, private keys, or raw database URLs in examples,
job records, artifact metadata, event attributes, or committed configuration.

## Container operation

```bash
docker build -f docker/Dockerfile -t hephaestus-infrastructure .

docker run --rm \
  -v "$(pwd)/state:/var/lib/hephaestus/state" \
  -v "$(pwd)/artifacts:/var/lib/hephaestus/artifacts" \
  hephaestus-infrastructure migrate --json

docker run --rm \
  -v "$(pwd)/state:/var/lib/hephaestus/state" \
  -v "$(pwd)/artifacts:/var/lib/hephaestus/artifacts" \
  hephaestus-infrastructure service
```

The image runs as UID `10001`, uses SIGTERM directly, declares state/artifact
volumes, installs no optional cloud SDK by default, and reports unhealthy until the
configured migrations and required dependencies are ready. Build a derived image
with the required optional extra and composition-root handler for PostgreSQL, S3, or
OpenTelemetry use.

## Optional adapters

```bash
python -m pip install '.[postgres]'
python -m pip install '.[s3,aws-secrets]'
python -m pip install '.[telemetry]'
```

- `PostgresLeaseLockProvider` supplies a real cross-host database lease lock using
  fenced owner/token rows. The queue remains on the fully exercised SQLite backend
  until a PostgreSQL queue adapter is selected and tested by the integration layer.
- `S3ArtifactStore` supports content-addressed final keys, expected hashes,
  streaming/multipart upload, abort/cleanup, verification, configurable server-side
  encryption, and explicit presigning.
- File-mounted, injected-client, and AWS Secrets Manager providers resolve values at
  runtime. Only `SecretReference` records are persistent.
- OpenTelemetry is an optional injected/exporter bridge; JSONL and in-process metrics
  remain available without it.

External PostgreSQL, S3, secret-manager, and telemetry services are not required for
unit tests. Their credentials and lifecycle belong to the deployment environment.
