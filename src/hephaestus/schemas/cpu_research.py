from dataclasses import dataclass
from ._base import JsonSchema


@dataclass(slots=True)
class CpuResearchExecutionProfile(JsonSchema):
    profile_id: str
    policy_ref: str
    model_id: str
    model_revision: str
    snapshot_hashes: dict[str, str]
    tokenizer_sha256: str
    maximum_training_steps: int
    maximum_cpu_seconds: int
    seed: int = 20261028
    cpu_threads: int = 2
    role_adapter: str = "none"
    device: str = "cpu"
    external_certification: bool = False
    frozen_cuda_identity: bool = False
    schema: str = "cpu-research-execution-profile.v1"

    def validate(self):
        from hephaestus.llm.cpu_causal_research import MODEL_ID, MODEL_REVISION
        if (self.schema != "cpu-research-execution-profile.v1" or self.device != "cpu" or
                self.model_id != MODEL_ID or self.model_revision != MODEL_REVISION or
                self.role_adapter != "none" or self.external_certification is not False or
                self.frozen_cuda_identity is not False or self.cpu_threads != 2 or
                not {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}.issubset(self.snapshot_hashes)):
            raise PermissionError("CPU research profile cannot impersonate certified CUDA roles")
        if (type(self.maximum_training_steps) is not int or not 1 <= self.maximum_training_steps <= 8 or
                type(self.maximum_cpu_seconds) is not int or not 0 < self.maximum_cpu_seconds <= 3600):
            raise PermissionError("bounded CPU profile required")
