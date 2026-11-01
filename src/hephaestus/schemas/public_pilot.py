"""Public development-only pilot identities; never final assessment authority."""
from dataclasses import dataclass
from ._base import JsonSchema


@dataclass(slots=True)
class PublicGenerationContract(JsonSchema):
    contract_id: str = 'smol-public-json-cap128.v1'
    max_new_tokens: int = 128
    purpose: str = 'nonconfirmatory-public-development'
    prompt_recipe: str = 'original-instruction-chat-json-system.v1'
    decoding: str = 'greedy-batch1-float32-cpu-no-forced-termination.v1'
    eos_token_id: int = 2
    pad_token_id: int = 2
    malformed_output: str = 'incorrect-no-repair'
    nonterminated_output: str = 'incorrect-even-if-json-parses'

    def __post_init__(self):
        if (type(self.max_new_tokens) is not int or self.max_new_tokens not in (16,64,128)
                or self.contract_id != f'smol-public-json-cap{self.max_new_tokens}.v1'
                or self.purpose != 'nonconfirmatory-public-development'
                or self.prompt_recipe != 'original-instruction-chat-json-system.v1'
                or self.decoding != 'greedy-batch1-float32-cpu-no-forced-termination.v1'
                or any(type(v) is not int for v in (self.eos_token_id,self.pad_token_id))
                or (self.eos_token_id,self.pad_token_id) != (2,2)
                or self.malformed_output != 'incorrect-no-repair'
                or self.nonterminated_output != 'incorrect-even-if-json-parses'):
            raise ValueError('public pilot generation contract requires a new version')


@dataclass(slots=True)
class PublicPilotPlan(JsonSchema):
    pilot_id: str = 'smol-public-development-pilot-20261101.v1'
    purpose: str = 'nonconfirmatory-feasibility-and-power-only'
    sampling_seed: int = 202611010152
    canonical_bucket: int = 8
    cases_per_family: int = 16
    template_pairs_per_family: int = 8
    difficulty: int = 100
    training_seeds: tuple = (1117,2227,3337)
    updates_per_seed: int = 96
    same_seed_replay: int = 1117
    protected_final_material_authorized: bool = False

    def __post_init__(self):
        self.training_seeds=tuple(self.training_seeds)
        if (any(type(v) is not int for v in (self.sampling_seed,self.canonical_bucket,self.cases_per_family,
                    self.template_pairs_per_family,self.difficulty,self.updates_per_seed,self.same_seed_replay,*self.training_seeds))
                or self.pilot_id != 'smol-public-development-pilot-20261101.v1'
                or self.purpose != 'nonconfirmatory-feasibility-and-power-only'
                or (self.sampling_seed,self.canonical_bucket,self.cases_per_family,
                    self.template_pairs_per_family,self.difficulty) != (202611010152,8,16,8,100)
                or self.training_seeds != (1117,2227,3337) or self.updates_per_seed != 96
                or self.same_seed_replay != 1117 or self.protected_final_material_authorized is not False):
            raise ValueError('public pilot sampling, dose and scope are frozen')
