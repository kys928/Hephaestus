#!/usr/bin/env python3
"""Run or verify real PEFT CPU *development* loop; never the 45-slot campaign."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from hephaestus.development.existing_peft_bridge import (
    run_existing_peft_cycle, verify_peft_cycle,
)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=Path("cpu-peft-development-cycle"))
    parser.add_argument("--steps",type=int,default=8)
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    print(json.dumps(
        verify_peft_cycle(args.output) if args.verify_only else
        run_existing_peft_cycle(args.output,steps=args.steps),
        sort_keys=True
    ))


if __name__=="__main__":
    main()
