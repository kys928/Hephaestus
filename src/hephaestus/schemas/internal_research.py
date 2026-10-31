"""Internal prospective research is a distinct claim from external certification."""
from dataclasses import dataclass
from ._base import JsonSchema


@dataclass(slots=True)
class InternalResearchPolicy(JsonSchema):
    policy_id: str
    authorized_by: str
    experiment_id: str
    workflow_id: str
    run_id: str
    protocol_ref: str
    processing_profile_sha256: str
    tokenizer_sha256: str
    maximum_bytes: int
    maximum_rows: int
    maximum_training_steps: int
    maximum_cpu_seconds: int
    allowed_licenses: tuple[str, ...]
    trusted_rights_refs: tuple[str, ...] = ()
    allowed_generator_code_sha256: tuple[str, ...] = ()
    allowed_capabilities: tuple[str, ...] = ()
    schema: str = "internal-research-policy.v1"

    def validate(self):
        import re
        if self.schema != "internal-research-policy.v1" or not all(
            isinstance(v, str) and v.strip() for v in (
                self.policy_id, self.authorized_by, self.experiment_id,
                self.workflow_id, self.run_id)):
            raise ValueError("explicit internal research authority and identities required")
        for digest in (self.protocol_ref, self.processing_profile_sha256, self.tokenizer_sha256):
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
                raise ValueError("immutable internal policy commitments required")
        for value in (self.maximum_bytes, self.maximum_rows, self.maximum_training_steps,
                      self.maximum_cpu_seconds):
            if type(value) is not int or value <= 0:
                raise ValueError("positive bounded internal research limits required")
        if not self.allowed_licenses or not self.allowed_capabilities:
            raise ValueError("explicit licenses and capability scope required")


@dataclass(slots=True)
class InternalDatasetReview(JsonSchema):
    candidate_sha256: str
    source_file_hashes: dict[str, str]
    processing_profile_sha256: str
    tokenizer_sha256: str
    protocol_ref: str
    capability: str
    rights_ref: str
    scan_ref: str
    generator_ref: str | None = None
    schema: str = "internal-dataset-review.v1"
