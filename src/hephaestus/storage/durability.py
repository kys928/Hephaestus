"""POSIX file publication barriers; success means bytes and names were flushed.

These operations require a filesystem that honors fsync and atomic local links/
renames. They cannot prevent external snapshots, storage rollback or bad hardware.
Interrupted publications may leave explicitly named partial files for inspection.
"""
from pathlib import Path
import os
import stat
import tempfile


def sync_directory_chain(directory: Path) -> None:
    """Persist directory entries, including every newly created ancestor name."""
    current = Path(directory).absolute()
    while True:
        descriptor = os.open(current, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if current.parent == current:
            break
        current = current.parent


def sync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("durable evidence must be a regular file")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes, *, overwrite: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if overwrite:
        try:
            os.replace(temporary, path)
        finally:
            # Preserve the established mutable-document failure contract. This
            # removes only our uncommitted sibling, never the old destination.
            if os.path.exists(temporary):
                os.unlink(temporary)
    else:
        # A concurrent publisher may win; immutable destinations are never replaced.
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            os.unlink(temporary)
            sync_directory_chain(path.parent)
            raise
        os.unlink(temporary)
    sync_directory_chain(path.parent)


def sync_tree(root: Path) -> None:
    """Flush a completed checkpoint before its directory is published."""
    root = Path(root)
    directories = [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("checkpoint durability cannot depend on symlinked components")
        if path.is_file():
            sync_file(path)
        elif path.is_dir():
            directories.append(path)
        else:
            raise ValueError("checkpoint contains a nonregular component")
    for path in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    sync_directory_chain(root.parent)


def append_bytes(path: Path, data: bytes) -> None:
    """One serialized durable append; a torn trailing record remains an error."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    sync_directory_chain(path.parent)
