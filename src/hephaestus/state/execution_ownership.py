"""Kernel-fenced ownership for supported host-local state repositories.

No timeout takeover: a paused controller retains its fence; process death
releases it. Shared-network-filesystem and cross-host execution are unsupported.
"""
from contextlib import contextmanager
import hashlib
import threading

from hephaestus.storage.filesystem import JsonLineStateRepository
from hephaestus.storage.sqlite import SQLiteStateRepository
from .single_writer import single_writer, WriterBusyError

_held = threading.local()


@contextmanager
def execution_owner(repository, scope: str, identity: str):
    if isinstance(repository, SQLiteStateRepository):
        location = repository.path.resolve()
        root = location.parent / ".execution-ownership"
    elif isinstance(repository, JsonLineStateRepository):
        location = repository.root.resolve()
        root = location / ".execution-ownership"
    else:
        raise NotImplementedError("execution ownership requires a supported host-local state repository")
    name = hashlib.sha256(f"{location}:{scope}:{identity}".encode()).hexdigest()
    active = getattr(_held, "active", set())
    _held.active = active
    if name in active:
        raise WriterBusyError("recursive execution of an already owned operation")
    with single_writer(root, name=name):
        active.add(name)
        try:
            yield
        finally:
            active.remove(name)
