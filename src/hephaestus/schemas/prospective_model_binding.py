"""Physical locations are separate from the signed logical model identities."""
from dataclasses import dataclass
from ._base import JsonSchema


@dataclass(slots=True)
class ProspectiveModelBinding(JsonSchema):
    label: str
    model_path: str
    model_identity: str
    tokenizer_path: str
    tokenizer_identity: str
    seed: int | None = None
    adapter_path: str | None = None
    adapter_identity: str | None = None
    checkpoint_path: str | None = None
    checkpoint_manifest_hash: str | None = None
    job_path: str | None = None
    job_hash: str | None = None

    def identity(self):
        return {k: v for k, v in self.to_dict().items() if not k.endswith("_path")}
