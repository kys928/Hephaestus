"""Immutable, bounded CPU campaign identity and research recipe specification.

A specification is not an operator approval, rights clearance, or certification.
The native admission and policy checks must still succeed before a source is used.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\Z")


@dataclass(frozen=True, slots=True)
class CpuCampaignConfig:
    workflow_id: str
    run_id: str
    lineage_id: str
    stage_name: str
    experiment_id: str
    cpu_seed: int
    dataset_seed: int
    maximum_training_steps: int
    maximum_cpu_seconds: int
    include_public: bool
    schema: str = "cpu-campaign-config.v1"

    def validate(self):
        if self.schema != "cpu-campaign-config.v1":
            raise ValueError("unsupported CPU campaign specification")
        for name in ("workflow_id", "run_id", "lineage_id", "stage_name", "experiment_id"):
            if not isinstance(getattr(self, name), str) or not _ID.fullmatch(getattr(self, name)):
                raise ValueError("invalid CPU campaign " + name)
        if any(type(getattr(self, k)) is not int or not 0 <= getattr(self, k) < 2**32
               for k in ("cpu_seed", "dataset_seed")):
            raise ValueError("CPU/dataset seeds must be explicit unsigned 32-bit integers")
        if (type(self.maximum_training_steps) is not int or
                not 1 <= self.maximum_training_steps <= 8 or
                type(self.maximum_cpu_seconds) is not int or
                not 1 <= self.maximum_cpu_seconds <= 3600 or
                type(self.include_public) is not bool):
            raise ValueError("bounded CPU research resources and source mode required")
        return self

    @property
    def identity(self):
        return {k: getattr(self, k) for k in
                ("workflow_id", "run_id", "lineage_id", "stage_name")}

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("CPU campaign specification must contain exactly the declared keys")
        return cls(**payload).validate()

    @classmethod
    def from_file(cls, path):
        raw = Path(path).read_bytes()
        if len(raw) > 16384:
            raise ValueError("CPU campaign specification too large")
        return cls.from_dict(json.loads(raw))
