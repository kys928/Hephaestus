from dataclasses import dataclass
from ._base import JsonSchema


@dataclass(slots=True)
class GeneratedDataSpec(JsonSchema):
    seed: int
    train_per_family: int = 24
    development_per_family: int = 6
    evaluation_per_family: int = 6
    difficulty: int = 20
    families: tuple[str, ...] = ("arithmetic", "json", "symbolic", "reasoning")
    generator_version: str = "original-verifiable-tasks.v1"

    def validate(self):
        if type(self.seed) is not int or self.generator_version != "original-verifiable-tasks.v1":
            raise ValueError("versioned original generator and integer seed required")
        if (not self.families or len(set(self.families)) != len(self.families) or
                not set(self.families) <= {"arithmetic", "json", "symbolic", "reasoning"}):
            raise ValueError("unsupported generated task family")
        if any(type(v) is not int or not 1 <= v <= 1000 for v in (
                self.train_per_family, self.development_per_family, self.evaluation_per_family)):
            raise ValueError("bounded nonempty train/development/evaluation samples required")
        if type(self.difficulty) is not int or not 10 <= self.difficulty <= 1000:
            raise ValueError("difficulty must be 10..1000")
