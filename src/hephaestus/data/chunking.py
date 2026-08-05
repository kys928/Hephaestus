from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChunkingResult:
    records: tuple[dict[str, object], ...]
    dropped_below_min_tokens: int
    source_records: int


def render_training_text(record: dict[str, object], *, prompt_target_template: str) -> tuple[str, str]:
    if "text" in record:
        return str(record["text"]), "text"
    try:
        rendered = prompt_target_template.format(
            prompt=str(record.get("prompt", "")),
            target=str(record.get("target", "")),
        )
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid prompt_target_template: {exc}") from exc
    return rendered, "prompt_target"


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
