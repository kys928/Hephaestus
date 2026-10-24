"""Immutable role inference inputs, raw outputs and execution provenance."""
from dataclasses import dataclass, field
import hashlib
from typing import Mapping
from hephaestus.storage.base import ArtifactStore
from hephaestus.storage.verified import read_verified_json


@dataclass(frozen=True, slots=True)
class RoleGeneration:
    text: str
    finish_reason: str = "completed"
    metadata: Mapping[str, object] = field(default_factory=dict)


def verify_inference_receipt(store: ArtifactStore, ref: str, contract) -> dict:
    doc = read_verified_json(store, ref)
    if (doc.get("schema") != "role-inference.v1" or
            hashlib.sha256(doc["prompt"].encode()).hexdigest() != doc["prompt_sha256"] or
            hashlib.sha256(doc["raw_output"].encode()).hexdigest() != doc["output_sha256"]):
        raise ValueError("role inference receipt input/output mismatch")
    if doc["status"] == "accepted":
        parsed = contract.parse(doc["role"], doc["raw_output"], set(doc["evidence_refs"]))
        if parsed.to_dict() != doc["answer"]:
            raise ValueError("role inference receipt decision differs from raw output")
    return doc
