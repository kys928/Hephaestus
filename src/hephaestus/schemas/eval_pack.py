from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from ._base import JsonSchema


@dataclass(slots=True)
class EvalPack(JsonSchema):
    eval_pack_id: str
    version: str
    name: str
    description: str | None = None
    stage_name: str | None = None
    created_at: str | None = None
    frozen: bool = True
    content_hash: str | None = None
    hash_type: str | None = None
    source_ref: str | None = None
    mutation_policy: str = "immutable_without_approval"
    integrity_level: str = "insufficient"
    generation_probes: list[dict[str, object]] = field(default_factory=list)
    continuation_prompts: list[dict[str, object]] = field(default_factory=list)
    ranking_sets: list[dict[str, object]] = field(default_factory=list)
    regression_prompts: list[dict[str, object]] = field(default_factory=list)
    structure_tests: list[dict[str, object]] = field(default_factory=list)
    repetition_checks: list[dict[str, object]] = field(default_factory=list)
    length_termination_checks: list[dict[str, object]] = field(default_factory=list)
    human_review_bundle_refs: list[str] = field(default_factory=list)
    decoding_config: dict[str, object] = field(default_factory=dict)
    scoring_config: dict[str, object] = field(default_factory=dict)
    deterministic_gate_config: dict[str, object] = field(default_factory=dict)
    required_evidence: dict[str, int] = field(default_factory=dict)
    stage_thresholds: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvalPack":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"eval pack payload has unknown fields: {', '.join(unknown)}")
        return cls(**payload)

    @classmethod
    def normalize(cls, payload: dict[str, Any], stage_name: str | None = None) -> "EvalPack":
        eval_pack_id = str(payload.get("eval_pack_id") or payload.get("pack_name") or payload.get("name") or "")
        version = str(payload.get("version") or "v1")
        name = str(payload.get("name") or payload.get("pack_name") or eval_pack_id or "unnamed_eval_pack")
        description = payload.get("description")
        if description is not None:
            description = str(description)

        content_hash = payload.get("content_hash")
        source_ref = payload.get("source_ref")
        if content_hash is not None:
            content_hash = str(content_hash)
        if source_ref is not None:
            source_ref = str(source_ref)

        integrity_level = str(payload.get("integrity_level") or "")
        warnings: list[str] = [str(item) for item in payload.get("warnings", []) if item is not None]
        if content_hash:
            integrity_level = "content_hash_verified"
        elif source_ref:
            integrity_level = "reference_only"
        elif eval_pack_id:
            integrity_level = "inline_unhashed"
        else:
            integrity_level = "insufficient"
            warnings.append("eval_pack_identity_missing")

        if integrity_level == "content_hash_verified" and not content_hash:
            integrity_level = "inline_unhashed"
            warnings.append("content_hash_missing_for_claimed_hash_integrity")

        mutation_policy = str(payload.get("mutation_policy") or "immutable_without_approval")

        return cls(
            eval_pack_id=eval_pack_id,
            version=version,
            name=name,
            description=description,
            stage_name=str(payload.get("stage_name") or stage_name) if (payload.get("stage_name") or stage_name) else None,
            created_at=str(payload.get("created_at")) if payload.get("created_at") is not None else None,
            frozen=bool(payload.get("frozen", True)),
            content_hash=content_hash,
            hash_type=str(payload.get("hash_type")) if payload.get("hash_type") is not None else None,
            source_ref=source_ref,
            mutation_policy=mutation_policy,
            integrity_level=integrity_level,
            generation_probes=[dict(item) for item in payload.get("generation_probes", []) if isinstance(item, dict)],
            continuation_prompts=[dict(item) for item in payload.get("continuation_prompts", []) if isinstance(item, dict)],
            ranking_sets=[dict(item) for item in payload.get("ranking_sets", []) if isinstance(item, dict)],
            regression_prompts=[dict(item) for item in payload.get("regression_prompts", []) if isinstance(item, dict)],
            structure_tests=[dict(item) for item in payload.get("structure_tests", []) if isinstance(item, dict)],
            repetition_checks=[dict(item) for item in payload.get("repetition_checks", []) if isinstance(item, dict)],
            length_termination_checks=[dict(item) for item in payload.get("length_termination_checks", []) if isinstance(item, dict)],
            human_review_bundle_refs=[str(item) for item in payload.get("human_review_bundle_refs", [])],
            decoding_config=dict(payload.get("decoding_config", {})) if isinstance(payload.get("decoding_config", {}), dict) else {},
            scoring_config=dict(payload.get("scoring_config", {})) if isinstance(payload.get("scoring_config", {}), dict) else {},
            deterministic_gate_config=dict(payload.get("deterministic_gate_config", {}))
            if isinstance(payload.get("deterministic_gate_config", {}), dict)
            else {},
            required_evidence={str(k): int(v) for k, v in dict(payload.get("required_evidence", {})).items()},
            stage_thresholds={str(k): float(v) for k, v in dict(payload.get("stage_thresholds", {})).items()},
            metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
            warnings=warnings,
        )
