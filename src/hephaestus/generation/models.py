"""JSON-safe records for frozen-pack model generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from hephaestus.schemas.contract_common import ContractIssue
from hephaestus.schemas.experiment_contract import TrainingRunHandle


@dataclass(frozen=True, slots=True)
class GenerationTask:
    task_id: str
    task_kind: str
    prompt: str
    seed: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GeneratedText:
    output: str
    finish_reason: str = "completed"
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.output.strip():
            raise ValueError("generated output must not be empty")
        if self.finish_reason not in {"completed", "eos", "length"}:
            raise ValueError(f"unsupported generation finish reason: {self.finish_reason}")
        if self.prompt_tokens is not None and self.prompt_tokens < 0:
            raise ValueError("prompt_tokens must be non-negative")
        if self.generated_tokens is not None and self.generated_tokens < 0:
            raise ValueError("generated_tokens must be non-negative")


@dataclass(frozen=True, slots=True)
class GenerationSample:
    sample_id: str
    run_id: str
    task_id: str
    task_kind: str
    seed: int
    prompt_hash: str
    output: str
    output_hash: str
    evidence_ref: str
    checkpoint_ref: str
    finish_reason: str
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GenerationPlan:
    eval_pack_id: str
    eval_pack_version: str
    content_hash: str
    decoding_config: dict[str, object]
    generation_settings_id: str
    seed_identity: str
    tasks: tuple[GenerationTask, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["tasks"] = [task.to_dict() for task in self.tasks]
        return payload


@dataclass(slots=True)
class GenerationReport:
    report_id: str
    run_id: str
    checkpoint_ref: str
    backend_id: str
    eval_pack_id: str
    eval_pack_version: str
    content_hash: str
    decoding_config: dict[str, object]
    generation_settings_id: str
    seed_identity: str
    completion_status: str
    report_ref: str | None = None
    samples: list[GenerationSample] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    issues: list[ContractIssue] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return self.completion_status == "completed" and not any(
            issue.blocking for issue in self.issues
        )

    def semantic_evaluation_metadata(self) -> dict[str, object]:
        return {
            "eval_pack_id": self.eval_pack_id,
            "eval_pack_version": self.eval_pack_version,
            "integrity_level": "content_hash_verified",
            "content_hash": self.content_hash,
            "decoding_config": dict(self.decoding_config),
            "report_ref": self.report_ref,
            "evidence_refs": list(self.evidence_refs),
            "samples": [sample.to_dict() for sample in self.samples],
            "generation_settings_id": self.generation_settings_id,
            "seed_identity": self.seed_identity,
            "completion_status": self.completion_status,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["samples"] = [sample.to_dict() for sample in self.samples]
        payload["issues"] = [issue.to_dict() for issue in self.issues]
        payload["completed"] = self.completed
        return payload


@dataclass(frozen=True, slots=True)
class GenerationResult:
    report: GenerationReport
    run_handle: TrainingRunHandle
