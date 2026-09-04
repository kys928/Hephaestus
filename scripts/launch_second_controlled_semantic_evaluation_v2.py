#!/usr/bin/env python3
"""Launch the controlled semantic evaluation using the reviewed v2 baseline projection."""
from __future__ import annotations

import launch_second_controlled_semantic_evaluation as base

_original_shell = base.pod_shell


def _pod_shell() -> str:
    shell = _original_shell()
    shell = shell.replace(
        '"$PY" -m py_compile src/hephaestus/control/semantic_judge.py scripts/run_second_controlled_semantic_evaluation.py',
        '"$PY" -m py_compile src/hephaestus/control/semantic_judge.py scripts/run_second_controlled_semantic_evaluation.py scripts/run_second_controlled_semantic_evaluation_v2.py',
    )
    shell = shell.replace(
        '"$PY" scripts/run_second_controlled_semantic_evaluation.py',
        '"$PY" scripts/run_second_controlled_semantic_evaluation_v2.py',
    )
    return shell


base.pod_shell = _pod_shell


if __name__ == "__main__":
    raise SystemExit(base.main())
