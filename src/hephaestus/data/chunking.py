from __future__ import annotations

from dataclasses import dataclass
from string import Formatter
from hephaestus.schemas.prompt_target_supervision import PromptTargetSupervision


def supervision_segments(supervision: PromptTargetSupervision) -> tuple[str, str]:
    """Render a declared prefix and target without inspecting marker content."""
    parts = list(Formatter().parse(supervision.template))
    if ([field for _, field, _, _ in parts if field is not None] != ["prompt", "target"]
            or any(spec or conversion for _, field, spec, conversion in parts if field is not None)):
        raise ValueError("prompt-target template requires one prompt followed by one target")
    prefix, target, in_target = "", "", False
    for literal, field, _, _ in parts:
        if in_target:
            target += literal
        else:
            prefix += literal
        if field == "prompt":
            prefix += supervision.prompt
        elif field == "target":
            target += supervision.target
            in_target = True
    return prefix, target


@dataclass(frozen=True, slots=True)
class ChunkingResult:
    records: tuple[dict[str, object], ...]
    dropped_below_min_tokens: int
    source_records: int


def render_training_text(record: dict[str, object], *, prompt_target_template: str) -> tuple[str, str]:
    if "text" in record:
        return str(record["text"]), "text"
    supervision = PromptTargetSupervision(record.get("prompt"), record.get("target"), prompt_target_template)
    return "".join(supervision_segments(supervision)), "prompt_target"


def chunk_records(
    records: tuple[dict[str, object], ...],
    *,
    chunk_size_tokens: int,
    min_tokens: int,
    prompt_target_template: str,
) -> ChunkingResult:
    if chunk_size_tokens <= 0 or min_tokens < 0:
        raise ValueError("chunk_size_tokens must be positive and min_tokens must be non-negative")
    chunks: list[dict[str, object]] = []
    dropped = 0
    for source_index, record in enumerate(records):
        text, record_kind = render_training_text(record, prompt_target_template=prompt_target_template)
        tokens = text.split()
        if len(tokens) < min_tokens:
            dropped += 1
            continue
        if record_kind == "prompt_target":
            if len(tokens) > chunk_size_tokens:
                raise ValueError("prompt-target example exceeds chunk limit; structured pairs cannot be split")
            supervision = PromptTargetSupervision(record["prompt"], record["target"], prompt_target_template)
            chunks.append({"text": text, "record_kind": record_kind,
                "source_record_index": source_index, "chunk_index": 0, "token_count": len(tokens),
                "supervision": supervision.to_dict()})
            continue
        for chunk_index, offset in enumerate(range(0, len(tokens), chunk_size_tokens)):
            chunk_tokens = tokens[offset:offset + chunk_size_tokens]
            if len(chunk_tokens) < min_tokens:
                dropped += 1
                continue
            chunks.append(
                {
                    "text": " ".join(chunk_tokens),
                    "source_record_index": source_index,
                    "chunk_index": chunk_index,
                    "token_count": len(chunk_tokens),
                    "record_kind": record_kind,
                }
            )
    return ChunkingResult(records=tuple(chunks), dropped_below_min_tokens=dropped, source_records=len(records))
