"""Versioned finite JSON-object/recursive typed answer scoring for future runs."""
import json
import math

SCORER_VERSION = "strict-json-object-finite-typed.v1"


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("non-finite JSON is not valid")


def _finite_tree(value):
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite_tree(v) for v in value)
    return True


def strict_json_object(raw):
    if not isinstance(raw, str):
        return None
    try:
        value = json.loads(raw, object_pairs_hook=_unique_pairs,
                           parse_constant=_reject_constant)
    except (ValueError, TypeError, json.JSONDecodeError):
        return None
    return value if type(value) is dict and _finite_tree(value) else None


def exact_typed_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            exact_typed_equal(actual[k], v) for k, v in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            exact_typed_equal(a, b) for a, b in zip(actual, expected, strict=True)
        )
    return actual == expected


def score_prediction(raw, target):
    expected = strict_json_object(target)
    if expected is None:
        raise ValueError("reference target is not an exact JSON object")
    parsed = strict_json_object(raw)
    return {
        "valid_json": parsed is not None,
        "exact_correct": parsed is not None and exact_typed_equal(parsed, expected),
    }


