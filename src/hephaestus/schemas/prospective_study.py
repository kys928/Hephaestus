"""Prospective design identity; no custody keys, approval or protected samples."""
from dataclasses import dataclass
from hashlib import sha256

from ._base import JsonSchema
from hephaestus.storage.verified import canonical_bytes


@dataclass(slots=True)
class ProspectiveDesign(JsonSchema):
    design_id: str = "original-four-family-range100.v2"
    purpose: str = "confirmatory"
    cases_per_family: int = 512
    difficulty: int = 100
    families: tuple[str, ...] = ("arithmetic", "json", "symbolic", "reasoning")
    training_seeds: tuple[int, ...] = (1117, 2227, 3337)
    minimum_gain: float = 0.05
    maximum_loss_fraction: float = 0.01
    alpha: float = 0.05
    bootstrap_draws: int = 2000
    bootstrap_seed: int = 152
    max_new_tokens: int = 64
    generator_contract: str = "unique-eval-bucket9-range100.v2"
    inference_scope: str = "fixed-four-families-and-templates-only"

    def __post_init__(self):
        self.families = tuple(self.families)
        self.training_seeds = tuple(self.training_seeds)
        if (type(self.difficulty) is not int or type(self.bootstrap_draws) is not int
                or type(self.bootstrap_seed) is not int or type(self.max_new_tokens) is not int
                or any(type(seed) is not int for seed in self.training_seeds)):
            raise ValueError("design counts, seeds and bounds require exact integer types")
        if self.families != ("arithmetic", "json", "symbolic", "reasoning"):
            raise ValueError("prospective design must retain the four declared task strata")
        if self.purpose not in {"confirmatory", "rehearsal"}:
            raise ValueError("unknown prospective study purpose")
        if (self.design_id != "original-four-family-range100.v2" or self.difficulty != 100
                or self.generator_contract != "unique-eval-bucket9-range100.v2"
                or self.inference_scope != "fixed-four-families-and-templates-only"):
            raise ValueError("design distribution/scope requires a new protocol version")
        if type(self.cases_per_family) is not int or (self.purpose == "confirmatory" and self.cases_per_family != 512):
            raise ValueError("confirmatory design requires 512 cases per family")
        if self.purpose == "rehearsal" and not 1 <= self.cases_per_family <= 16:
            raise ValueError("CPU rehearsal is bounded to 1..16 cases per family")
        if (self.training_seeds != (1117, 2227, 3337) or self.minimum_gain != .05
                or self.maximum_loss_fraction != .01 or self.alpha != .05
                or self.bootstrap_draws != 2000 or self.bootstrap_seed != 152):
            raise ValueError("changed analysis/seed plan requires a new protocol version")
        if self.max_new_tokens != (64 if self.purpose == "confirmatory" else 16):
            raise ValueError("decoding limit differs from the declared study purpose")

    def commitment(self) -> str:
        self.__post_init__()
        return "sha256:" + sha256(canonical_bytes(self.to_dict())).hexdigest()
