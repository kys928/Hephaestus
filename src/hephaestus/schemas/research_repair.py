"""Review contracts for a single diagnosis-module repair proposal."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Protocol
import difflib
import re

_SOURCE_PREFIX = "src/hephaestus/diagnosis/"
_TESTS = frozenset({
    "tests/test_targeted_post_failure_probes.py",
    "tests/test_evidence_based_diagnosis.py",
})
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
class RepairBoundaryError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RepairScope:
    """Operator-controlled scope; never sourced from the model response."""

    base_commit: str
    allowed_file: str
    fixed_test: str
    maximum_changed_lines: int = 100

    def validate(self) -> None:
        if not _COMMIT.fullmatch(self.base_commit):
            raise RepairBoundaryError("exact approved Git commit is required")
        path = PurePosixPath(self.allowed_file)
        if (self.allowed_file != path.as_posix() or not self.allowed_file.startswith(_SOURCE_PREFIX)
                or len(path.parts) != 4 or path.suffix != ".py" or ".." in path.parts
                or path.name.startswith("_") or not path.name):
            raise RepairBoundaryError("only one approved diagnosis Python module is repairable")
        if self.fixed_test not in _TESTS:
            raise RepairBoundaryError("test command must use a preserved review-approved suite")
        if not 1 <= self.maximum_changed_lines <= 150:
            raise RepairBoundaryError("code change exceeds the bounded repair ceiling")


@dataclass(frozen=True, slots=True)
class PatchCandidate:
    path: str
    expected_source_sha256: str
    replacement: str
    evidence_ref: str
    hypothesis: str

    def validate(self, scope: RepairScope, original: bytes) -> tuple[str, int]:
        scope.validate()
        if self.path != scope.allowed_file or not _HASH.fullmatch(self.evidence_ref):
            raise RepairBoundaryError("patch outside operator-controlled scope or without evidence")
        if (not _HASH.fullmatch(self.expected_source_sha256)
                or "sha256:" + sha256(original).hexdigest() != self.expected_source_sha256):
            raise RepairBoundaryError("repair proposal uses stale or fabricated source bytes")
        if (not isinstance(self.replacement, str) or not 10 <= len(self.replacement.encode()) <= 64_000
                or not 10 <= len(self.hypothesis) <= 500 or "\x00" in self.replacement):
            raise RepairBoundaryError("patch size or hypothesis invalid")
        try:
            compile(self.replacement, self.path, "exec", dont_inherit=True)
        except (SyntaxError, ValueError) as exc:
            raise RepairBoundaryError("replacement does not parse as Python") from exc
        if self.replacement.encode() == original:
            raise RepairBoundaryError("candidate patch changes no bytes")
        old_lines = original.decode("utf-8").splitlines()
        new_lines = self.replacement.splitlines()
        diff = list(difflib.ndiff(old_lines, new_lines))
        touched = sum(line.startswith(("+ ", "- ")) for line in diff)
        if touched > scope.maximum_changed_lines:
            raise RepairBoundaryError("patch changes too many lines")
        return "sha256:" + sha256(self.replacement.encode()).hexdigest(), touched


class PatchModel(Protocol):
    def generate_patch_json(self, prompt: str) -> str: ...
