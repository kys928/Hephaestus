"""F04/F05 exercise each evaluator's actual measurement block without a GPU.

Only the parser assignments from generate() are executed. No model outcome is
simulated or counted as scientific evidence; model execution has separate tests.
"""
import ast
import inspect
import json

import pytest

from hephaestus.evaluation.cpu_prospective import CpuProspectiveEvaluator
from hephaestus.evaluation.tier_a_cuda_target import CudaProspectiveEvaluator
from hephaestus.evaluation.paired_holdout import strict_json_object


def measurement(cls, raw_name, raw, target):
    module = inspect.getmodule(cls)
    tree = ast.parse(inspect.getsource(module))
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "generate")
    loop = next(n for n in ast.walk(function) if isinstance(n, ast.For) and ast.unparse(n.target) == "row")
    start = next(i for i, n in enumerate(loop.body) if isinstance(n, ast.Assign)
                 and {x.id for t in n.targets for x in ast.walk(t) if isinstance(x, ast.Name)} == {"valid", "correct"})
    end = next(i for i in range(start, len(loop.body))
               if isinstance(loop.body[i], ast.Expr) and ast.unparse(loop.body[i]).startswith("outputs.append"))
    namespace = {**vars(module), raw_name: raw, "row": {"target": target}}
    exec(compile(ast.Module(body=loop.body[start:end], type_ignores=[]), "<actual-evaluator-measurement>", "exec"), namespace)
    return namespace["valid"], namespace["correct"]


@pytest.mark.parametrize("cls,raw_name", [(CpuProspectiveEvaluator, "text"), (CudaProspectiveEvaluator, "answer_text")])
@pytest.mark.parametrize("raw,target,valid,correct", [
    ('{"result":999,"result":2}', '{"result":2}', False, False),
    ('{"x":{"a":1,"a":2}}', '{"x":{"a":2}}', False, False),
    ('{"x":[true]}', '{"x":[1]}', True, False),
    ('{"result":2,"extra":0}', '{"result":2}', True, False),
    ('{"result":1e400}', '{"result":2}', False, False),
    ('{"result":NaN}', '{"result":2}', False, False),
    ('```json\n{"result":2}\n```', '{"result":2}', False, False),
    ('{"result":2}', '{"result":2}', True, True),
])
def test_future_cpu_cuda_measurements_obey_same_strict_contract(cls, raw_name, raw, target, valid, correct):
    assert measurement(cls, raw_name, raw, target) == (valid, correct)


@pytest.mark.parametrize("raw", ['{"x":1e400}', '{"x":-1e400}', '{"x":[{"y":1e400}]}'])
def test_finite_contract_rejects_decoded_overflow(raw):
    assert strict_json_object(raw) is None


def test_finite_large_values_are_distinct_from_overflow():
    assert strict_json_object('{"x":1e300}') == {"x": 1e300}
    assert strict_json_object('{"x":' + '9' * 400 + '}')["x"] == int('9' * 400)
