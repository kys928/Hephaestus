"""Offline scientific factors; never a replacement for model inference."""
from dataclasses import asdict, dataclass
from itertools import product
from typing import Literal


@dataclass(frozen=True)
class ScientificFactors:
    gate_pass: bool
    complete: bool
    provenance_valid: bool
    effect: Literal['positive', 'neutral', 'negative']
    variance_stable: bool
    recheck_satisfied: bool

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key != 'effect' and type(value) is not bool:
                raise ValueError(f'{key} must be boolean')
        if self.effect not in ('positive', 'neutral', 'negative'):
            raise ValueError('invalid effect')

    def to_dict(self):
        return asdict(self)


def combination_space():
    """All 96 combinations, including simultaneous defects."""
    return tuple(ScientificFactors(*values) for values in product(
        (False, True), (False, True), (False, True),
        ('positive', 'neutral', 'negative'), (False, True), (False, True)))
