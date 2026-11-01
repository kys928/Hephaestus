"""Kernel-fenced ownership for supported host-local state repositories.

No timeout takeover: a paused controller retains its fence; process death
releases it. Shared-network-filesystem and cross-host execution are unsupported.
"""
from contextlib import contextmanager, nullcontext
import hashlib
import threading

from hephaestus.storage.filesystem import JsonLineStateRepository
from hephaestus.storage.sqlite import SQLiteStateRepository
from .single_writer import single_writer, WriterBusyError

_held = threading.local()


@contextmanager
def execution_owner(repository, scope: str, identity: str):
    if isinstance(repository, SQLiteStateRepository):
        if repository._db.read_only:
            raise PermissionError('read-only state inspectors cannot own research execution')
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
    witness = repository._db.witness() if isinstance(repository, SQLiteStateRepository) else None
    with witness.owner(blocking=False) if witness else nullcontext():
        with single_writer(root, name=name):
            if not active and isinstance(repository,SQLiteStateRepository):
                repository.assert_integrity()
                if witness:
                    witness.verify_chain()
                    with repository._db.connection() as connection:
                        repository._db.assert_witness(connection, witness)
            active.add(name)
            try:
                yield
            finally:
                active.remove(name)
