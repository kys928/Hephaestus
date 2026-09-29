#!/usr/bin/env python3
"""Interface Repair V2 entrypoint with persistent immutable parent caches."""
from __future__ import annotations

import run_interface_repair_v2 as v2
from interface_repair_v2_bootstrap import load_stack_compatible, materialize_parent_cached


# Execution-only substitutions. The frozen model revision, certified Phase-I
# source run, and adapter digest are unchanged. These adapters only normalize the
# registry schema and reuse persistent immutable caches.
v2.v1.load_stack = load_stack_compatible
v2.v1.materialize_parent = materialize_parent_cached


if __name__ == "__main__":
    raise SystemExit(v2.main())
