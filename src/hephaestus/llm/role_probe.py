"""CLI for a bounded, read-only *actual weights* smoke of certified roles.

Run inside a separately approved one-GPU worker with all five pinned Hugging
Face revisions pre-cached. No pod allocation, training, registry mutation,
actions, network HF downloads or S3 writes are performed here.

Example:
  python -m hephaestus.llm.role_probe \\
    --case-file /workspace/verified-role-probes.json \\
    --s3-endpoint https://s3api-eu-cz-1.runpod.io/ \\
    --s3-bucket <network-volume-id> \\
    --adapter-cache /workspace/role-adapters \\
    --hf-cache /workspace/hf-cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from hephaestus.control.deterministic_role_boundary import (
    DeterministicRoleContext, EvidenceRecord, EvidenceRegistry,
)
from hephaestus.providers.models.role_stack import (
    REQUIRED_ROLES, load_certified_role_model_stack,
)
from .certified_runtime import (
    CertifiedRoleReasoner, PinnedTransformersRoleGenerator, VerifiedRoleAdapters,
)
from .role_contract import RoleEvidence, RoleMasteryContract, RoleTask
from hephaestus.production.evidence import evidence_excerpt
from hephaestus.storage.filesystem import FileSystemArtifactStore


DEFAULT_REGISTRY = Path("configs/models/hephaestus_role_model_stack_v1.json")
DEFAULT_CONTRACT = Path("configs/models/hephaestus_role_mastery_runtime_v1.json")


def checked_cases(case_file: Path):
    """Validate local evidence bytes before connecting to any remote provider."""
    payload = json.loads(case_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("case document must be an object")
    run_id = str(payload.get("run_id") or "")
    lineage_id = str(payload.get("lineage_id") or "")
    stage_name = str(payload.get("stage_name") or "")
    if not all((run_id, lineage_id, stage_name)):
        raise ValueError("missing case workflow identity")
    situations = payload.get("situations")
    if not isinstance(situations, dict) or set(situations) != set(REQUIRED_ROLES):
        raise ValueError("case document must contain situations for all five roles")
    raws = payload.get("evidence")
    if not isinstance(raws, list) or not raws:
        raise ValueError("case document is missing independently hashed evidence files")
    records = {}
    task_evidence = []
    for item in raws:
        if not isinstance(item, dict):
            raise ValueError("evidence entry must be an object")
        ref, source, expected = (
            str(item.get("ref") or ""), str(item.get("source_file") or ""),
            str(item.get("sha256") or ""),
        )
        if not ref or not source or ref in records:
            raise ValueError("evidence has missing or duplicate fields")
        content = Path(source).read_bytes()
        observed = hashlib.sha256(content).hexdigest()
        if len(expected) != 64 or observed != expected:
            raise ValueError(f"local source evidence SHA-256 mismatch: {ref}")
        source_doc = json.loads(content)
        if not isinstance(source_doc, dict):
            raise ValueError("probe scientific evidence must be a JSON object")
        # Human/model-authored fact strings are not substituted for source bytes.
        task_evidence.append(RoleEvidence(ref, evidence_excerpt(source_doc)))
        records[ref] = EvidenceRecord(
            ref=ref, kind="verified_local_source",
            run_id=run_id, lineage_id=lineage_id,
            sha256=expected, observed_sha256=observed,
        )
    return run_id, lineage_id, stage_name, situations, tuple(task_evidence), EvidenceRegistry(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--s3-endpoint", required=True)
    parser.add_argument("--s3-bucket", required=True)
    parser.add_argument("--adapter-cache", type=Path, required=True)
    parser.add_argument("--hf-cache", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args(argv)
    run_id, lineage, stage, situations, facts, registry = checked_cases(args.case_file)
    stack = load_certified_role_model_stack(args.registry)
    contract = RoleMasteryContract.load(args.contract)

    # Credentials are provided by boto3's normal environment/instance chain;
    # never print them or place them in a subprocess command. GET only.
    import boto3
    s3 = boto3.client("s3", endpoint_url=args.s3_endpoint)
    generator = PinnedTransformersRoleGenerator(hf_cache_dir=args.hf_cache)
    reasoner = CertifiedRoleReasoner(
        stack=stack, contract=contract,
        adapters=VerifiedRoleAdapters(
            client=s3, bucket=args.s3_bucket, cache_root=args.adapter_cache),
        generator=generator,
        artifact_store=FileSystemArtifactStore(args.adapter_cache / "inference-receipts"),
    )
    try:
        for role in REQUIRED_ROLES:
            task = RoleTask(
                role=role, situation=str(situations[role]),
                evidence=facts, run_id=run_id, lineage_id=lineage,
                stage_name=stage,
            )
            context = DeterministicRoleContext(
                run_id=run_id, lineage_id=lineage, stage_name=stage,
                approval_status="none",
                # This smoke never authorizes actions, even when a model
                # produces a schema-valid action proposal.
                stage_allowed_actions=("observe_state", "read_memory", "request_recheck"),
            )
            result = reasoner.propose(
                task, registry, context=context, seed=args.seed,
                max_new_tokens=args.max_new_tokens,
            )
            print(json.dumps({
                "role": role, "run_id": run_id, "lineage_id": lineage,
                "model_id": result.model_id, "revision": result.revision,
                "adapter_sha256": result.adapter_sha256,
                "prompt_sha256": result.prompt_sha256,
                "output_sha256": result.output_sha256,
                "output": result.answer.to_dict(), "advisory_only": True,
                "control_gate": ({
                    "allowed": result.action_gate.allowed,
                    "failures": list(result.action_gate.failures),
                } if result.action_gate is not None else None),
            }, sort_keys=True), flush=True)
    finally:
        generator.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
