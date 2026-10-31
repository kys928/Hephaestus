"""Explicit test-only opt-in; the actual fixed fixture subprocess still runs."""
from hephaestus.control.orchestrator import build_orchestrator as _build


def build_orchestrator(*args, **kwargs):
    return _build(*args, **{**kwargs, "legacy_fixture_mode": True})
