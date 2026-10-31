"""Exact reproduction contract. Identity/count evidence is not training approval."""
from dataclasses import dataclass, asdict
import re



@dataclass(frozen=True, slots=True)
class ContinuationSubsetContract:
    dataset_id: str
    revision: str
    source_file: str
    source_file_sha256: str
    source_bytes: int
    source_rows: int
    tokenizer_ref: str
    tokenizer_file_sha256: str
    public_eval_pack_sha256: str
    selection_commitment_sha256: str
    accepted_rows: int
    eligible_tokens: int
    unknown_tokens: int
    windows: int
    prefix_tokens: int = 128
    target_tokens: int = 32
    stride_tokens: int = 64
    schema: str = "continuation-subset.v1"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict):
        return cls(**payload)

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.schema != "continuation-subset.v1" or not self.dataset_id:
            raise ValueError("unsupported exact continuation contract")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("immutable source revision required")
        from pathlib import PurePosixPath
        p = PurePosixPath(self.source_file)
        if p.is_absolute() or ".." in p.parts or str(p) != self.source_file:
            raise ValueError("exact source file must be a safe relative path")
        for value in (self.source_file_sha256, self.tokenizer_ref,
                      self.public_eval_pack_sha256, self.selection_commitment_sha256):
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
                raise ValueError("exact continuation contract requires SHA-256 identities")
        if not re.fullmatch(r"[0-9a-f]{64}", self.tokenizer_file_sha256):
            raise ValueError("exact tokenizer file SHA required")
        for value in (self.source_bytes, self.source_rows, self.accepted_rows,
                      self.eligible_tokens, self.windows, self.prefix_tokens,
                      self.target_tokens, self.stride_tokens):
            if type(value) is not int or value <= 0:
                raise ValueError("positive integer continuation bounds required")
        if (type(self.unknown_tokens) is not int or self.unknown_tokens < 0
                or self.unknown_tokens * 100 > self.eligible_tokens
                or self.accepted_rows > self.source_rows
                or self.windows < self.accepted_rows):
            raise ValueError("inconsistent continuation inventory or unknown-token ceiling")


PINNED_WIKITEXT_SUBSET = ContinuationSubsetContract(
    dataset_id="Salesforce/wikitext",
    revision="b08601e04326c79dfdd32d625aee71d232d685c3",
    source_file="wikitext-2-raw-v1/train-00000-of-00001.parquet",
    source_file_sha256="sha256:e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7",
    source_bytes=6357543, source_rows=36718,
    tokenizer_ref="sha256:123745ffe03aadf5d275c90bceb4e3bfb71678548a5ed936410ebe1e8c85e4ce",
    tokenizer_file_sha256="3ebcc9816398d7a2afa341a9db07de5f0ac30d2625ffe63e4752c2eddce40f25",
    public_eval_pack_sha256="sha256:cbbc110822877a07b5e768d049064dbc9eadd5aaf0d004444ff072c5f0af5601",
    selection_commitment_sha256="sha256:87ae1f4ab0f71693f0f20e733ebd7aa8ff25edc252f51f9ea306ef74ac7fe69e",
    accepted_rows=7232, eligible_tokens=1755120, unknown_tokens=0, windows=13380,
)
