#!/usr/bin/env python3
"""Guarded launcher wrapper for Evaluator Interface Repair V3.1."""
from __future__ import annotations

import launch_interface_repair_v3_evaluator as base

_ORIGINAL_POD_SHELL = base.pod_shell


def pod_shell(mode: str, venv_cache: str, pip_cache: str, adapter_cache: str) -> str:
    shell = _ORIGINAL_POD_SHELL(mode, venv_cache, pip_cache, adapter_cache)
    shell = shell.replace(
        'scripts/build_interface_repair_v3_evaluator.py',
        'scripts/build_interface_repair_v3_1_evaluator.py',
    )
    shell = shell.replace(
        'scripts/run_interface_repair_v3_evaluator.py',
        'scripts/run_interface_repair_v3_1_evaluator.py',
    )
    return shell


base.pod_shell = pod_shell


if __name__ == "__main__":
    raise SystemExit(base.main())
