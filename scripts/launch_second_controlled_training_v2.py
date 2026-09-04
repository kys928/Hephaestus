#!/usr/bin/env python3
"""Launch the second controlled training using the reviewed v2 runtime projection."""
from __future__ import annotations

import launch_second_controlled_training as base

_original_shell = base.pod_shell


def _pod_shell() -> str:
    shell = _original_shell()
    shell = shell.replace(
        '"$PY" -m py_compile scripts/run_second_controlled_training.py',
        '"$PY" -m py_compile scripts/run_second_controlled_training.py scripts/run_second_controlled_training_v2.py',
    )
    shell = shell.replace(
        '"$PY" scripts/run_second_controlled_training.py',
        '"$PY" scripts/run_second_controlled_training_v2.py',
    )
    return shell


base.pod_shell = _pod_shell


if __name__ == "__main__":
    raise SystemExit(base.main())
