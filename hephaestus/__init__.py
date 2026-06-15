"""Checkout import shim for the src-layout Hephaestus package."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "hephaestus"
if _SRC_PACKAGE.is_dir():
    _SRC_TEXT = str(_SRC_PACKAGE)
    if _SRC_TEXT not in __path__:
        __path__.append(_SRC_TEXT)
