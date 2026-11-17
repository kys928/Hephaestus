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
    serve=child.add_parser("serve-signed-cuda",
        help="opt-in, actual frozen-role CUDA worker; requires external signed grant")
    serve.add_argument("--run-id",required=True)
    serve.add_argument("--lineage-id",required=True)
    serve.add_argument("--source-catalog",required=True,type=Path)
    serve.add_argument("--catalog-sha256",required=True)
    serve.add_argument("--signed-grant",required=True,type=Path)
    serve.add_argument("--trusted-public-key",required=True,type=Path)
    serve.add_argument("--public-key-sha256",required=True)
    serve.add_argument("--role-registry-sha256",required=True)
    serve.add_argument("--eval-pack",required=True,type=Path)
    serve.add_argument("--model-cache",required=True,type=Path)
    serve.add_argument("--adapter-cache",required=True,type=Path)
    serve.add_argument("--max-idle-polls",type=int,default=None)
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
    if args.command=="serve-signed-cuda":
        from .deploy_worker import DeployedResearchWorkerScope, compose_live_queue_worker
        if args.max_idle_polls is not None and args.max_idle_polls<1:
            parser.error("--max-idle-polls must be positive")
        scope=DeployedResearchWorkerScope(
            state_root=root,run_id=args.run_id,lineage_id=args.lineage_id,
            source_catalog_path=args.source_catalog,
            pinned_source_catalog_sha256=args.catalog_sha256,
            signed_grant_path=args.signed_grant,
            trusted_public_key_path=args.trusted_public_key,
            pinned_public_key_sha256=args.public_key_sha256,
            trusted_role_registry_sha256=args.role_registry_sha256,
            eval_pack_path=args.eval_pack,
            original_model_cache=args.model_cache,
            original_adapter_cache=args.adapter_cache,
            s3_endpoint="https://s3api-eu-cz-1.runpod.io/",
            s3_region="EU-CZ-1")
        worker=compose_live_queue_worker(scope)
        worker.run_forever(maximum_idle_polls=args.max_idle_polls)
        return 0
    return 2


if __name__=="__main__":
    raise SystemExit(main())
