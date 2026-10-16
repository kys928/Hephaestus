"""Curriculum oracle only. Production scientific-state authority stays with the model.

Precedence preserves the frozen V1 ontology, including variance before effect and
confirmation applying only to a stable positive effect.
"""
from hephaestus.schemas.evaluator_factors import ScientificFactors


def scientific_state(f: ScientificFactors) -> str:
    if not f.gate_pass:
        return 'scientific_rejection'
    if not f.complete:
        return 'incomplete_evidence'
    if not f.provenance_valid:
        return 'inconclusive'
    if not f.variance_stable:
        return 'recheck_required'
    if f.effect == 'negative':
        return 'regressed'
    if f.effect == 'neutral':
        return 'equivalent'
    return 'certification_ready' if f.recheck_satisfied else 'improved'
