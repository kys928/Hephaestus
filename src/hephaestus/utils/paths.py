from __future__ import annotations

from pathlib import PurePosixPath


def normalize_repo_path(path: str) -> str:
    return PurePosixPath(str(path).strip()).as_posix().lstrip("./")


def is_relative_safe_path(path: str) -> bool:
    normalized = normalize_repo_path(path)
    return bool(normalized) and not normalized.startswith("../") and "//" not in normalized
