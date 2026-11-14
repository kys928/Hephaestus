"""Submit bounded natural-language research objectives to the existing queue.

Submission itself never runs a model, opens remote S3, edits code, launches
training or grants authority. A trusted, separately configured durable worker
owns execution and evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hephaestus.jobs.sqlite import SQLiteJobQueue
from hephaestus.storage.filesystem import FileSystemArtifactStore

from .durable_jobs import submit_research_objective


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(prog="hephaestus-research")
    parser.add_argument("--state-root",required=True,type=Path,
                        help="absolute durable local SQLite/artifact directory")
    child=parser.add_subparsers(dest="command",required=True)
    submit=child.add_parser("submit",help="submit a bounded scientific objective")
    submit.add_argument("--objective",required=True)
    submit.add_argument("--run-id",required=True)
    submit.add_argument("--lineage-id",required=True)
    args=parser.parse_args(argv)
    if not args.state_root.is_absolute() or args.state_root.is_symlink():
        parser.error("--state-root must be a non-symlink absolute durable path")
    root=args.state_root
    artifacts=FileSystemArtifactStore(root/"artifacts")
    queue=SQLiteJobQueue(root/"state.db")
    if args.command=="submit":
        job=submit_research_objective(
            question=args.objective,run_id=args.run_id,
            lineage_id=args.lineage_id,queue=queue,artifacts=artifacts)
        print(json.dumps({
            "job_id":job.job_id,"payload_ref":job.payload_ref,
            "status":job.status.value,
            "research_mode":"registered_capabilities_only",
            "execution_started":False,
            "training_authorized":False,
            "automatic_merge_authorized":False,
        },sort_keys=True))
        return 0
    return 2


if __name__=="__main__":
    raise SystemExit(main())
