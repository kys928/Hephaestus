"""Source-only measurements. Never emits scientific clearance or source text.

Regex PII and lexical candidates are triage, not privacy/semantic certification.
Protected comparisons require separate custodian-side execution.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import re
import unicodedata

PII = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "ipv4_candidate": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "phone_candidate": re.compile(r"(?<!\w)\+?\d[\d ()-]{7,}\d(?!\w)"),
    "iban_candidate": re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b"),
}


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def pii_counts(rows: list[dict]) -> dict:
    """Inspect every string cell, including fields excluded from supervision."""
    counts = Counter()
    affected = 0
    cells = 0
    for row in rows:
        found = False
        for value in row.values():
            if not isinstance(value, str):
                continue
            cells += 1
            for name, pattern in PII.items():
                count = sum(1 for _ in pattern.finditer(value))
                counts[name] += count
                found |= count > 0
        affected += found
    return {"method": "regex-candidate-triage.v1", "rows_checked": len(rows),
            "string_cells_checked": cells, "rows_flagged": affected,
            "candidate_counts": dict(counts), "status": "human_review_required",
            "limitations": ["false positives and negatives", "names/addresses not comprehensively detected",
                            "no privacy clearance", "no raw matches persisted"]}


def duplicate_counts(texts: list[str]) -> dict:
    """Exact normalized row repeats plus bounded lexical candidate retrieval.

    Near-duplicate retrieval uses four bands of a 64-bit SimHash, followed by
    five-word shingle Jaccard >= .9. This is not exhaustive semantic detection.
    Large candidate buckets are counted and skipped, never reported as clean.
    """
    counts = Counter(normalized(t).casefold() for t in texts if t.strip())
    buckets = defaultdict(list)
    shingles_by_id = []
    matched = comparisons = skipped = 0
    for text in counts:
        words = re.findall(r"\w+", text)
        shingles = {" ".join(words[i:i + 5]) for i in range(len(words) - 4)}
        current = len(shingles_by_id)
        shingles_by_id.append(shingles)
        if not shingles:
            continue
        bits = [0] * 64
        for shingle in shingles:
            h = int.from_bytes(hashlib.blake2b(shingle.encode(), digest_size=8).digest(), "big")
            for bit in range(64):
                bits[bit] += 1 if h & (1 << bit) else -1
        signature = sum(1 << i for i, v in enumerate(bits) if v >= 0)
        keys = [(band, (signature >> (16 * band)) & 65535) for band in range(4)]
        candidates = set()
        for key in keys:
            bucket = buckets[key]
            if len(bucket) > 256:
                skipped += 1
            else:
                candidates.update(bucket)
            bucket.append(current)
        for other_id in candidates:
            other = shingles_by_id[other_id]
            comparisons += 1
            if len(shingles & other) / len(shingles | other) >= .9:
                matched += 1
    return {"method": "NFKC-casefold-exact+simhash64-four-bands+jaccard-five-word.v1",
            "nonempty_rows": sum(counts.values()), "unique_normalized_rows": len(counts),
            "exact_repeated_rows": sum(n - 1 for n in counts.values()),
            "near_duplicate_candidate_pairs": matched, "candidate_pairs_compared": comparisons,
            "overfull_buckets_skipped": skipped, "status": "triage_only",
            "semantic_contamination_checked": False, "exhaustive_near_duplicate_search": False}


def public_probe_overlap(texts: list[str], prompts: list[str]) -> dict:
    """Complete lexical comparison to supplied public prompts, not a holdout."""
    refs = [normalized(p).casefold() for p in prompts]
    shingle_refs = [{" ".join(p.split()[i:i + 5]) for i in range(len(p.split()) - 4)} for p in refs]
    exact = shared = 0
    for text in texts:
        text = normalized(text).casefold()
        words = text.split()
        grams = {" ".join(words[i:i + 5]) for i in range(len(words) - 4)}
        exact += any(p in text for p in refs if p)
        shared += any(grams & reference for reference in shingle_refs)
    return {"method": "normalized-prompt-substring+shared-five-word-shingle.v1",
            "source_rows_checked": len(texts), "public_prompts_checked": len(prompts),
            "rows_with_prompt_substrings": exact, "rows_with_shared_five_word_shingle": shared,
            "coverage": "public_repository_behavior_pack_only",
            "status": "partial_reference_coverage", "sealed_holdout_checked": False,
            "semantic_contamination_checked": False}
