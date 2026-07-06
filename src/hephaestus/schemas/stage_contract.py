from __future__ import annotations

from dataclasses import dataclass, field

from ._base import JsonSchema


@dataclass(slots=True)
class StageContract(JsonSchema):
    contract_id: str
    stage_name: str
    eval_pack_ref: str
    allowed_backends: list[str] = field(default_factory=list)
    required_manifest_fields: list[str] = field(default_factory=list)
    required_data_contract_fields: list[str] = field(default_factory=list)
    required_contract_refs: list[str] = field(default_factory=list)
    accepted_eval_pack_integrity_levels: list[str] = field(default_factory=list)
    min_manifest_completeness: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_stage_profile(cls, stage_profile: dict[str, object]) -> "StageContract":
        stage_name = str(stage_profile.get("name") or "")
        eval_pack = str(stage_profile.get("eval_pack") or stage_profile.get("eval_pack_ref") or "")
        return cls(
            contract_id=f"stage-contract-{stage_name or 'unknown'}",
            stage_name=stage_name,
            eval_pack_ref=str(stage_profile.get("eval_pack_ref") or eval_pack),
            allowed_backends=[str(item) for item in stage_profile.get("allowed_backends", [])]
            if isinstance(stage_profile.get("allowed_backends", []), list)
            else [],
            required_manifest_fields=[
                "manifest_id",
                "run_id",
                "lineage_id",
                "datasets",
                "mixture_weights",
            ],
            required_data_contract_fields=[
                "contract_id",
                "run_id",
                "manifest_id",
                "processed_dataset_ref",
                "schema_version",
                "min_tokens",
            ],
            required_contract_refs=[
                "stage.eval_pack_ref",
                "manifest.stage_data_policy_ref",
                "manifest.tokenizer_ref",
            ],
            accepted_eval_pack_integrity_levels=[
                "content_hash_verified",
                "reference_only",
                "inline_unhashed",
            ],
            min_manifest_completeness=float(stage_profile.get("min_manifest_completeness", 0.45) or 0.45),
            metadata={
                "strictness": str(stage_profile.get("strictness", "")),
                "eval_pack": eval_pack,
            },
        )
