#!/usr/bin/env python3
"""V2 execution wrapper for the unchanged live Interface Repair certification."""
from __future__ import annotations

from pathlib import Path

import run_interface_repair_live_v1 as live_v1
import run_interface_repair_v2 as exec_v2

ROOT = Path(__file__).resolve().parents[1]
live_v1.CFG_PATH = ROOT / "configs/experiments/hephaestus_interface_repair_v2.json"
# Keep the live scientific logic unchanged while using V2's cheaper persistence
# and contract-aware generation stopping mechanics.
live_v1.put_json = exec_v2.put_json
live_v1.generate = exec_v2.generate


if __name__ == "__main__":
    raise SystemExit(live_v1.main())
