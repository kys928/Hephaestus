"""Independent parser and deterministic validators; never trust supplied answers/specs.

No generator imports. Canonical task equivalence, including commutative
arithmetic and graph edge order, detects semantic duplicates of these grammars.
"""
from fractions import Fraction
import json
import re


def parse_and_solve(prompt):
    match = re.fullmatch(r"Compute (\d+) (plus|minus|times) (\d+)\. Return JSON with key result\.", prompt)
    if match:
        a, op, b = match.groups()
        left, right = Fraction(a), Fraction(b)
        answer = left + right if op == "plus" else left - right if op == "minus" else left * right
        operands = sorted([int(a), int(b)]) if op in {"plus", "times"} else [int(a), int(b)]
        return ("arithmetic", op, tuple(operands)), {"result": int(answer)}
    match = re.fullmatch(r"For integers (\[.*\]), return JSON with keys count and sum\.", prompt)
    if match:
        values = json.loads(match[1])
        if not values or any(type(v) is not int or abs(v) > 10000 for v in values):
            raise ValueError("invalid integer list")
        result = Fraction(0)
        for value in values:
            result += Fraction(value)
        return ("json", tuple(sorted(values))), {"count": len(values), "sum": int(result)}
    match = re.fullmatch(r"Reverse the lowercase string ([a-z]{3,12})\. Return JSON with key text\.", prompt)
    if match:
        text = match[1]
        output = ""
        for character in text:
            output = character + output
        return ("symbolic", text), {"text": output}
    match = re.fullmatch(r"Find shortest directed path length from 0 to 4 in edges (\[.*\])\. Return JSON with key distance\.", prompt)
    if match:
        edges = json.loads(match[1])
        if (not edges or any(not isinstance(e, list) or len(e) != 2 or
                any(type(v) is not int or not 0 <= v <= 4 for v in e) for e in edges)):
            raise ValueError("invalid graph")
        # Floyd-Warshall is independent of the generator's BFS implementation.
        dist = [[0 if i == j else 99 for j in range(5)] for i in range(5)]
        for a, b in edges:
            dist[a][b] = min(1, dist[a][b])
        for k in range(5):
            for i in range(5):
                for j in range(5):
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        if dist[0][4] == 99:
            raise ValueError("unreachable graph target")
        return ("reasoning", tuple(sorted(set(tuple(e) for e in edges)))), {"distance": dist[0][4]}
    raise ValueError("unsupported original task grammar")


def validate_original_example(row):
    if not isinstance(row, dict) or not isinstance(row.get("target"), str):
        raise ValueError("invalid generated sample")
    key, expected = parse_and_solve(row.get("prompt", ""))
    def unique_pairs(pairs):
        if len(dict(pairs)) != len(pairs):
            raise ValueError("duplicate JSON keys")
        return dict(pairs)
    actual = json.loads(row["target"], object_pairs_hook=unique_pairs)
    if (actual != expected or set(actual) != set(expected) or
            any(type(actual[k]) is not type(v) for k, v in expected.items()) or row.get("family") != key[0]):
        raise ValueError("independent deterministic answer validation failed")
    return key


class OriginalTaskSemanticChecker:
    def distinct_task(self, row, reference):
        return validate_original_example(row) != validate_original_example(reference)

    def check(self, rows, references):
        train = [validate_original_example(row) for row in rows]
        heldout = {validate_original_example(row) for row in references}
        return {"overlap": sum(key in heldout for key in train),
            "within_source_semantic_duplicates": len(train) - len(set(train)),
            "method": "independent-parser-canonical-task-equivalence.v1",
            "validator": "fractions-reversal-floyd-warshall.v1",
            "limitations": ["Semantic equivalence is complete only for these supported task grammars.",
                            "Task-family transfer and pretrained exposure are not independent benchmark claims."]}
