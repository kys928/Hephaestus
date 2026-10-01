"""Strict fail-closed canonicalization for Evaluator evidence references.

Only one repair is permitted: an exact identifier may differ from an allowed
reference solely by the optional leading ``E-`` marker. No case folding, substring
matching, edit distance, token similarity, prefix guessing, or other fuzzy repair is
performed. If the canonical body maps to zero or multiple allowed references, the
reference remains unresolved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class EvidenceReferenceCanonicalization:
    raw_refs: tuple[str, ...]
    canonical_refs: tuple[str, ...]
    normalized_pairs: tuple[tuple[str, str], ...]
    unresolved_refs: tuple[str, ...]
    ambiguous_refs: tuple[str, ...]

    @property
    def grounded(self) -> bool:
        return not self.unresolved_refs and not self.ambiguous_refs


def _body(ref: str) -> str:
    return ref[2:] if ref.startswith("E-") else ref


def canonicalize_evaluator_evidence_refs(
    cited_refs: Iterable[object],
    allowed_refs: Iterable[object],
) -> EvidenceReferenceCanonicalization:
    """Canonicalize model evidence refs using only exact optional ``E-`` repair.

    Exact allowed references pass through unchanged. Otherwise a citation may be
    repaired only when removing an optional leading ``E-`` from both sides yields an
    exact, case-sensitive body with exactly one allowed match. Ambiguous and unknown
    identifiers fail closed and remain unresolved.
    """

    raw = tuple(str(value) for value in cited_refs)
    allowed = tuple(str(value) for value in allowed_refs)
    allowed_set = set(allowed)

    by_body: dict[str, list[str]] = {}
    for ref in allowed:
        by_body.setdefault(_body(ref), []).append(ref)

    canonical: list[str] = []
    normalized: list[tuple[str, str]] = []
    unresolved: list[str] = []
    ambiguous: list[str] = []

    for ref in raw:
        if ref in allowed_set:
            canonical.append(ref)
            continue

        matches = by_body.get(_body(ref), [])
        if len(matches) == 1:
            target = matches[0]
            # A repair is valid only when the strings differ solely by the literal
            # optional ``E-`` marker. _body is deliberately case-sensitive.
            if {ref, target} == {_body(ref), f"E-{_body(ref)}"}:
                canonical.append(target)
                normalized.append((ref, target))
                continue
        elif len(matches) > 1:
            ambiguous.append(ref)
            continue

        unresolved.append(ref)

    return EvidenceReferenceCanonicalization(
        raw_refs=raw,
        canonical_refs=tuple(canonical),
        normalized_pairs=tuple(normalized),
        unresolved_refs=tuple(unresolved),
        ambiguous_refs=tuple(ambiguous),
    )
