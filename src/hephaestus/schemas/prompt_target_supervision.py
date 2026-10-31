"""Structured supervision retained through preprocessing into the HF worker."""
from dataclasses import dataclass
from ._base import JsonSchema


@dataclass(slots=True)
class PromptTargetSupervision(JsonSchema):
    prompt: str
    target: str
    template: str
    schema_version: str = "prompt-target-structured.v1"

    def __post_init__(self):
        if (self.schema_version != "prompt-target-structured.v1"
                or any(type(v) is not str for v in (self.prompt, self.target, self.template))
                or not self.prompt.strip() or not self.target.strip()):
            raise ValueError("invalid structured prompt-target supervision")
