"""Descriptive public-pilot sensitivity; never a confirmatory test or selector."""
from collections import Counter


def paired_template_diagnostics(samples):
    """Compare wrappers with exactly their matching canonical tasks, not all 64.

    Wrapped surfaces are repeated measurements of the same tasks. They never
    increase the number of independent canonical units or select a candidate.
    """
    templates = {name: {} for name in ("original-canonical.v1", "task-wrapper-sensitivity.v1")}
    flags = ("terminated_exact", "typed_exact", "valid_json", "eos", "truncated")
    for sample in samples:
        template = sample["template_id"]
        if template not in templates or sample["sample_id"] in templates[template]:
            raise ValueError("unknown or duplicate public surface")
        if any(type(sample[name]) is not bool for name in flags):
            raise ValueError("public diagnostic requires actual Boolean scores")
        templates[template][sample["sample_id"]] = sample
    canonical, wrapped = templates.values()
    if not wrapped or not set(wrapped) <= set(canonical):
        raise ValueError("every wrapper needs its original canonical measurement")
    ids = sorted(wrapped)
    if any(canonical[sid]["family"] != wrapped[sid]["family"] for sid in ids):
        raise ValueError("paired template family differs")
    counts = lambda rows: {name: sum(row[name] for row in rows) for name in flags}
    return {
        "paired_canonical_units": len(ids),
        "all_canonical_units": len(canonical),
        "family_counts": dict(Counter(wrapped[sid]["family"] for sid in ids)),
        "original_on_same_cases": counts([canonical[sid] for sid in ids]),
        "wrapper_on_same_cases": counts([wrapped[sid] for sid in ids]),
        "terminated_exact_wrapper_wins": sum(
            wrapped[sid]["terminated_exact"] and not canonical[sid]["terminated_exact"] for sid in ids),
        "terminated_exact_wrapper_losses": sum(
            canonical[sid]["terminated_exact"] and not wrapped[sid]["terminated_exact"] for sid in ids),
        "additional_independent_units": 0,
        "scope": "paired predeclared public wrapper sensitivity; no unseen-template inference",
    }
