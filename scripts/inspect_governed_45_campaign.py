#!/usr/bin/env python3
"""Read-only inspection of an ACTUAL persisted governed 45-stage campaign.

Never substitutes copied Markdown stage numbers, issue comments or synthetic
test fixtures for the real SQLite and content-addressed artifact stores.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from hephaestus.production.governed_campaign_admission import inspect_governed_campaign
from hephaestus.storage.filesystem import FileSystemArtifactStore
from hephaestus.storage.sqlite import SQLiteStateRepository


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--lineage-id", required=True)
    parser.add_argument("--stage-name", required=True)
    args = parser.parse_args()
    if not args.sqlite.is_file():
        raise FileNotFoundError("actual campaign state sqlite file is missing")
    if not args.artifacts.is_dir():
        raise FileNotFoundError("actual campaign content-addressed artifact directory is missing")
    result=inspect_governed_campaign(
        SQLiteStateRepository(args.sqlite),
        FileSystemArtifactStore(args.artifacts),
        workflow_id=args.workflow_id,run_id=args.run_id,
        lineage_id=args.lineage_id,stage_name=args.stage_name,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
