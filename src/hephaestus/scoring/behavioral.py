"""Deterministic behavioral checks for recorded generation samples."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
_TERMINAL_PUNCTUATION = (".", "!", "?", "}", "]", '"', "'")
_ABRUPT_ENDINGS = {
    "a",
    "an",
    "and",
    "because",
    "but",
    "for",
    "if",
    "of",
    "or",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True, slots=True)
class BehavioralCheck:
    name: str
    dimension: str
    passed: bool
    hard: bool
    score: float
    details: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "passed": self.passed,
            "hard": self.hard,
            "score": self.score,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class BehavioralSampleScore:
    task_id: str
    seed: int | str
    checks: tuple[BehavioralCheck, ...]
    dimension_scores: dict[str, float] = field(default_factory=dict)

    @property
    def deterministic_passed(self) -> bool:
        return not any(check.hard and not check.passed for check in self.checks)

    @property
    def overall_score(self) -> float:
        if not self.dimension_scores:
            return 0.0
        return sum(self.dimension_scores.values()) / len(self.dimension_scores)

    @property
    def failed_hard_checks(self) -> list[str]:
        return [check.name for check in self.checks if check.hard and not check.passed]

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "checks": [check.to_dict() for check in self.checks],
            "dimension_scores": dict(self.dimension_scores),
            "overall_score": self.overall_score,
            "deterministic_passed": self.deterministic_passed,
            "failed_hard_checks": self.failed_hard_checks,
        }


def _normalized_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def _words(text: str) -> list[str]:
    return re.findall(r"[\w'-]+", text.casefold(), flags=re.UNICODE)


def repeated_ngram_fraction(text: str, n: int = 3) -> float:
    words = _words(text)
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    repeated = len(ngrams) - len(set(ngrams))
    return repeated / len(ngrams) if ngrams else 0.0


def _balanced_delimiters(text: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for char in text:
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
    return not stack


def _result(
    *,
    name: str,
    dimension: str,
    passed: bool,
    hard: bool,
    details: str,
) -> BehavioralCheck:
    return BehavioralCheck(
        name=name,
        dimension=dimension,
        passed=passed,
        hard=hard,
        score=1.0 if passed else 0.0,
        details=details,
    )


def _json_check(response: str, spec: dict[str, object]) -> tuple[bool, str]:
    try:
        payload = json.loads(response)
    except (TypeError, json.JSONDecodeError) as exc:
        return False, f"malformed_json:{exc.__class__.__name__}"
    if not isinstance(payload, dict):
        return False, "json_root_not_object"

    required = [str(item) for item in spec.get("required_keys", [])]
    missing = [key for key in required if key not in payload]
    if missing:
        return False, f"missing_json_keys={','.join(sorted(missing))}"

    expected_values = spec.get("expected_values", {})
    if isinstance(expected_values, dict):
        mismatched = [
            str(key)
            for key, expected in expected_values.items()
            if payload.get(str(key)) != expected
        ]
        if mismatched:
            return False, f"json_value_mismatch={','.join(sorted(mismatched))}"

    expected_types = spec.get("required_types", {})
    type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    if isinstance(expected_types, dict):
        bad_types: list[str] = []
        for key, type_name in expected_types.items():
            expected_type = type_map.get(str(type_name))
            value = payload.get(str(key))
            if expected_type is None or not isinstance(value, expected_type):
                bad_types.append(str(key))
        if bad_types:
            return False, f"json_type_mismatch={','.join(sorted(bad_types))}"
    return True, "valid_json_object"


def _evaluate_check(
    task: dict[str, object],
    response: str,
    spec: dict[str, object],
    index: int,
) -> BehavioralCheck:
    check_type = str(spec.get("type", "unsupported"))
    name = str(spec.get("name") or f"{task.get('task_id', 'task')}:{index}:{check_type}")
    dimension = str(spec.get("dimension") or task.get("dimension") or "behavior")
    hard = bool(spec.get("hard", False))
    normalized = _normalized_text(response)
    folded = normalized.casefold()

    if check_type == "exact_match":
        expected = _normalized_text(spec.get("expected", ""))
        case_sensitive = bool(spec.get("case_sensitive", False))
        observed = normalized if case_sensitive else folded
        target = expected if case_sensitive else expected.casefold()
        return _result(name=name, dimension=dimension, passed=observed == target, hard=hard, details=f"expected={expected!r}")

    if check_type in {"contains_all", "excludes_all"}:
        terms = [str(item).casefold() for item in spec.get("terms", [])]
        present = [term for term in terms if term in folded]
        if check_type == "contains_all":
            passed = len(present) == len(terms)
            details = f"missing_terms={','.join(term for term in terms if term not in present)}"
        else:
            passed = not present
            details = f"forbidden_terms_present={','.join(present)}"
        return _result(name=name, dimension=dimension, passed=passed, hard=hard, details=details)

    if check_type in {"max_words", "min_words"}:
        count = len(_words(response))
        limit = int(spec.get("value", 0))
        passed = count <= limit if check_type == "max_words" else count >= limit
        return _result(name=name, dimension=dimension, passed=passed, hard=hard, details=f"word_count={count};limit={limit}")

    if check_type == "repetition":
        n = max(1, int(spec.get("ngram", 3)))
        threshold = float(spec.get("max_fraction", 0.2))
        fraction = repeated_ngram_fraction(response, n=n)
        return _result(
            name=name,
            dimension=dimension,
            passed=fraction <= threshold,
            hard=hard,
            details=f"repeated_{n}gram_fraction={fraction:.4f};max={threshold:.4f}",
        )

    if check_type == "termination":
        require_punctuation = bool(spec.get("require_terminal_punctuation", True))
        forbidden = tuple(str(item).casefold() for item in spec.get("forbidden_endings", []))
        last_word = _words(response)[-1] if _words(response) else ""
        punctuation_ok = bool(normalized) and (not require_punctuation or normalized.endswith(_TERMINAL_PUNCTUATION))
        ending_ok = not last_word or last_word not in forbidden
        passed = punctuation_ok and ending_ok
        return _result(
            name=name,
            dimension=dimension,
            passed=passed,
            hard=hard,
            details=f"terminal_punctuation={punctuation_ok};last_word={last_word!r}",
        )

    if check_type == "continuation_not_echo_prompt":
        prompt = _normalized_text(task.get("prompt", "")).casefold()
        minimum_prompt_chars = int(spec.get("minimum_prompt_chars", 12))
        echo = bool(prompt) and len(prompt) >= minimum_prompt_chars and folded.startswith(prompt)
        return _result(name=name, dimension=dimension, passed=not echo, hard=hard, details=f"prompt_echo={echo}")

    if check_type == "json_object":
        passed, details = _json_check(response, spec)
        return _result(name=name, dimension=dimension, passed=passed, hard=hard, details=details)

    if check_type == "surface_coherence":
        words = _words(response)
        last_word = words[-1] if words else ""
        passed = bool(words) and _balanced_delimiters(response) and last_word not in _ABRUPT_ENDINGS
        return _result(
            name=name,
            dimension=dimension,
            passed=passed,
            hard=hard,
            details=f"nonempty={bool(words)};balanced={_balanced_delimiters(response)};last_word={last_word!r}",
        )

    return _result(
        name=name,
        dimension=dimension,
        passed=False,
        hard=True,
        details=f"unsupported_check_type={check_type}",
    )


def evaluate_behavioral_sample(
    task: dict[str, object],
    response: object,
    seed: int | str,
) -> BehavioralSampleScore:
    text = str(response)
    raw_checks = task.get("checks", [])
    checks = tuple(
        _evaluate_check(task, text, dict(spec), index)
        for index, spec in enumerate(raw_checks)
        if isinstance(spec, dict)
    )
    dimensions: dict[str, list[float]] = {}
    for check in checks:
        dimensions.setdefault(check.dimension, []).append(check.score)
    dimension_scores = {
        dimension: sum(values) / len(values)
        for dimension, values in sorted(dimensions.items())
        if values
    }
    return BehavioralSampleScore(
        task_id=str(task.get("task_id", "")),
        seed=seed,
        checks=checks,
        dimension_scores=dimension_scores,
    )
