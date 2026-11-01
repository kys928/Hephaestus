"""Standalone text-template utility with no active production call sites.

Frozen specialist inference uses RoleMasteryContract.prompt/parse instead.
The top-level prompts/ TODO files are inactive scaffolds, not those contracts.
"""
from __future__ import annotations

from pathlib import Path


def load_prompt(path: str | Path, *, variables: dict[str, object] | None = None) -> str:
    text = Path(path).read_text(encoding="utf-8")
    for key, value in dict(variables or {}).items():
        text = text.replace("{{" + key + "}}", str(value))
    return text
