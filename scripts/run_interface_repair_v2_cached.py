#!/usr/bin/env python3
"""Interface Repair V2 entrypoint with persistent immutable parent caches."""
from __future__ import annotations

import run_interface_repair_v2 as v2
from interface_repair_v2_bootstrap import materialize_parent_cached


# Execution-only substitution. The frozen model revision and adapter digest are
# still supplied by the same certified Phase-I registry and verified before use.
v2.v1.materialize_parent = materialize_parent_cached


if __name__ == "__main__":
    raise SystemExit(v2.main())
