#!/usr/bin/env python3
"""Launch distance-to-role V1 with the explicit FP16->BF16 dose-zero bridge driver."""
from __future__ import annotations

import launch_distance_to_role_v1 as base

_ORIGINAL_POD_SHELL = base.pod_shell


def bridged_pod_shell() -> str:
    shell = _ORIGINAL_POD_SHELL()
    needle = '"$PY" scripts/run_distance_to_role_v1.py'
    replacement = '"$PY" scripts/run_distance_to_role_v1_bridge.py'
    if needle not in shell:
        raise RuntimeError("distance launcher shell no longer contains the expected driver invocation")
    return shell.replace(needle, replacement, 1)


def main() -> int:
    base.pod_shell = bridged_pod_shell
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
