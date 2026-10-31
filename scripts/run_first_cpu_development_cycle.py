#!/usr/bin/env python3
"""One actual CPU development experiment. Does NOT resume the 45-slot campaign."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from hephaestus.development.cpu_cycle import run_cycle, verify_cycle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("cpu-development-cycle"))
    parser.add_argument("--steps", type=int, default=85)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_cycle(args.output), sort_keys=True))
    else:
        print(json.dumps(run_cycle(args.output, steps=args.steps), sort_keys=True))


if __name__ == "__main__":
    main()
