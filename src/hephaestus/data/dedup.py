from __future__ import annotations

from dataclasses import dataclass

from hephaestus.utils.hashing import hash_json


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    records: tuple[dict[str, object], ...]
    exact_duplicates_removed: int
    approximate_duplicates_removed: int
    exact_status: str
    approximate_status: str


def _record_text(record: dict[str, object]) -> str:
    if "text" in record:
        return str(record["text"])
    return f"{record.get('prompt', '')}\n{record.get('target', '')}"


def _shingles(text: str, width: int = 3) -> set[tuple[str, ...]]:
    tokens = text.casefold().split()
    if len(tokens) < width:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index:index + width]) for index in range(len(tokens) - width + 1)}


def deduplicate_records(
    records: list[dict[str, object]],
    *,
    near_duplicate_threshold: float | None = None,
) -> DeduplicationResult:
    seen_hashes: set[str] = set()
    exact_kept: list[dict[str, object]] = []
    exact_removed = 0
    for record in records:
        digest = hash_json(record)
        if digest in seen_hashes:
            exact_removed += 1
            continue
        seen_hashes.add(digest)
        exact_kept.append(record)

    if near_duplicate_threshold is None:
        return DeduplicationResult(
            records=tuple(exact_kept),
            exact_duplicates_removed=exact_removed,
            approximate_duplicates_removed=0,
            exact_status="exact_deduplication_complete",
            approximate_status="approximate_deduplication_not_run",
        )
    threshold = max(0.0, min(1.0, float(near_duplicate_threshold)))
    kept: list[dict[str, object]] = []
    kept_shingles: list[set[tuple[str, ...]]] = []
    approximate_removed = 0
    for record in exact_kept:
        shingles = _shingles(_record_text(record))
        duplicate = False
        for prior in kept_shingles:
            union = shingles | prior
            similarity = len(shingles & prior) / len(union) if union else 1.0
            if similarity >= threshold:
                duplicate = True
                break
        if duplicate:
            approximate_removed += 1
        else:
            kept.append(record)
            kept_shingles.append(shingles)
    return DeduplicationResult(
        records=tuple(kept),
        exact_duplicates_removed=exact_removed,
        approximate_duplicates_removed=approximate_removed,
        exact_status="exact_deduplication_complete",
        approximate_status=f"approximate_jaccard_complete:threshold={threshold:.4f}",
    )
