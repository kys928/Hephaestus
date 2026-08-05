FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HEPHAESTUS_STATE_ROOT=/var/lib/hephaestus/state \
    HEPHAESTUS_ARTIFACT_ROOT=/var/lib/hephaestus/artifacts \
    HEPHAESTUS_EVENT_LOG=/var/lib/hephaestus/state/infrastructure_events.jsonl

RUN useradd --create-home --uid 10001 hephaestus \
    && mkdir -p /app /var/lib/hephaestus/state /var/lib/hephaestus/artifacts \
    && chown -R hephaestus:hephaestus /app /var/lib/hephaestus

WORKDIR /app
COPY --chown=hephaestus:hephaestus src/ ./src/

USER hephaestus

VOLUME ["/var/lib/hephaestus/state", "/var/lib/hephaestus/artifacts"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD ["python", "-m", "hephaestus.infrastructure.cli", "health", "--json"]

ENTRYPOINT ["python", "-m", "hephaestus.infrastructure.cli"]
CMD ["health", "--json"]
