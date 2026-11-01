"""Kernel-owned producer fence; a PID is not a portable worker identity."""
from contextlib import contextmanager
from pathlib import Path
from hephaestus.policy.storage_admission import reject_symlinks
from .single_writer import single_writer

WORKER_LOCK_NAME='hf-worker'
WORKER_LOCK_FILE='.hf-worker-writer.lock'


@contextmanager
def worker_owner(root, *, prove_quiescent=False):
    with single_writer(reject_symlinks(Path(root)),name=WORKER_LOCK_NAME,blocking=False,
                       reentrant=not prove_quiescent):
        yield
