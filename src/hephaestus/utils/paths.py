"""Path normalization helpers for artifact and state references."""

from __future__ import annotations

from pathlib import Path


def normalize_ref(path: str | Path) -> str:
    """Return a stable POSIX-style path reference without resolving symlinks."""

    return Path(path).as_posix()


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for a file path and return the path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def artifact_path(artifact_root: str | Path, *parts: str | Path) -> Path:
    """Build a normalized artifact path under an artifact root."""

    return Path(artifact_root).joinpath(*(Path(part) for part in parts))


def is_within(path: str | Path, root: str | Path) -> bool:
    """Return whether ``path`` is contained by ``root`` after resolution."""

    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True
