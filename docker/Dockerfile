FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HEPHAESTUS_STATE_ROOT=/var/lib/hephaestus/state \
    HEPHAESTUS_ARTIFACT_ROOT=/var/lib/hephaestus/artifacts \
    HEPHAESTUS_EVENT_LOG=/var/lib/hephaestus/state/infrastructure_events.jsonl \
    HEPHAESTUS_QUEUE_BACKEND=sqlite \
    HEPHAESTUS_DATABASE_BACKEND=sqlite \
    HEPHAESTUS_SQLITE_PATH=/var/lib/hephaestus/state/infrastructure.sqlite3 \
    HEPHAESTUS_HEALTH_REQUIRED=state_root,artifact_root,database,queue,artifact_store,lock_service,worker,migrations

RUN useradd --create-home --uid 10001 hephaestus \
    && mkdir -p /app /var/lib/hephaestus/state /var/lib/hephaestus/artifacts \
    && chown -R hephaestus:hephaestus /app /var/lib/hephaestus

WORKDIR /app
COPY --chown=hephaestus:hephaestus pyproject.toml ./
COPY --chown=hephaestus:hephaestus src/ ./src/
RUN python -m pip install --no-cache-dir --no-deps .

USER hephaestus

VOLUME ["/var/lib/hephaestus/state", "/var/lib/hephaestus/artifacts"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["python", "-m", "hephaestus.infrastructure.cli", "health", "--json"]

STOPSIGNAL SIGTERM

ENTRYPOINT ["hephaestus-infrastructure"]
CMD ["service"]
