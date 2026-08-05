# Execution infrastructure deployment boundary

This directory documents the minimal local/container boundary. It does not claim that a production queue, database, distributed lock, authentication service, or telemetry backend exists.

## Local health check

From the repository root:

```bash
PYTHONPATH=src \
HEPHAESTUS_STATE_ROOT=state \
HEPHAESTUS_ARTIFACT_ROOT=artifacts \
python -m hephaestus.infrastructure.cli health --json
```

Configuration is loaded only from explicit environment variables. This adapter does not read `.env` files or accept secret values as configuration records.

| Variable | Default | Purpose |
| --- | --- | --- |
| `HEPHAESTUS_STATE_ROOT` | `state` | Mounted JSON/JSONL and event state root |
| `HEPHAESTUS_ARTIFACT_ROOT` | `artifacts` | Mounted content-addressed artifact root |
| `HEPHAESTUS_EVENT_LOG` | `<state-root>/infrastructure_events.jsonl` | Structured local event log |
| `HEPHAESTUS_WORKER_ID` | hostname | Local worker identity |
| `HEPHAESTUS_JOB_LEASE_SECONDS` | `60` | Positive lease duration for local workers |

## Container smoke check

```bash
docker build -f docker/Dockerfile -t hephaestus-infrastructure .
docker run --rm \
  -v "$(pwd)/state:/var/lib/hephaestus/state" \
  -v "$(pwd)/artifacts:/var/lib/hephaestus/artifacts" \
  hephaestus-infrastructure
```

The image runs as UID `10001`, exposes no network port, embeds no credentials, and requires explicit state/artifact mounts for durable data. Its entry point currently supports the health command only. A production deployment must replace the process-local queue and lock adapters and provide authentication, authorization, durable scheduling, and external telemetry through the documented protocols.
