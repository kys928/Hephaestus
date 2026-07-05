from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def hash_text(text: str, *, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def hash_json(payload: Any, *, algorithm: str = "sha256") -> str:
    return hash_text(stable_json_dumps(payload), algorithm=algorithm)


def hash_file(path: str | Path, *, algorithm: str = "sha256", chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
