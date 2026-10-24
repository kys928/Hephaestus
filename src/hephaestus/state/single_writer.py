"""Host-local, process-safe ownership of a state directory.

The kernel lock has no time-based takeover: a paused writer cannot outlive its
fence. Process death releases ownership. This requires reliable local flock
semantics; it is deliberately not a multi-host lease.
"""
from contextlib import contextmanager
from pathlib import Path
import threading
from uuid import uuid4

_guard = threading.Lock()
_locks: dict[str, threading.RLock] = {}
_held = threading.local()


class WriterBusyError(RuntimeError):
    pass


@contextmanager
def single_writer(root: Path, name: str = "lineage"):
    import fcntl  # Fail closed on platforms without process-safe flock.

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = str(root / f".{name}-writer.lock")
    with _guard:
        lock = _locks.setdefault(path, threading.RLock())
    if not lock.acquire(blocking=False):
        raise WriterBusyError("state directory already has an active writer")
    held = getattr(_held, "locks", {})
    _held.locks = held
    try:
        if path in held:
            yield held[path]
            return
        with open(path, "a+b") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise WriterBusyError("state directory already has an active writer") from exc
            token = uuid4().hex
            held[path] = token
            try:
                yield token
            finally:
                held.pop(path)
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        lock.release()
