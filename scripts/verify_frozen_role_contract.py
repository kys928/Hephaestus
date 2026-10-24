"""Independently compare the runtime contract with immutable training source.

Run from a repository checkout containing the frozen commit, without ML packages.
Never imports or executes the training program.
"""
from pathlib import Path
import ast
import hashlib
import json
import subprocess

FROZEN = "c831bbfc0e2054d54fccd1ef374fd84722612c79"
ROOT = Path(__file__).resolve().parents[1]


def frozen(path):
    return subprocess.check_output(["git", "show", f"{FROZEN}:{path}"], cwd=ROOT, text=True)


def main():
    runtime = json.loads((ROOT / "configs/models/hephaestus_role_mastery_runtime_v1.json").read_text())
    builder = ast.parse(frozen("scripts/build_role_mastery_v1.py"))
    definitions = {n.targets[0].id: ast.literal_eval(n.value) for n in builder.body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in {"VOCAB", "ROLE_RULES"}}
    config = json.loads(frozen("configs/experiments/hephaestus_role_mastery_v1.json"))
    trainer = ast.parse(frozen("scripts/run_role_mastery_v1.py"))
    prompt_fn = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "role_prompt")
    namespace = {"Any": object}
    exec(compile(ast.Module(body=[prompt_fn], type_ignores=[]), "frozen-role-prompt", "exec"), namespace)
    from hephaestus.llm.role_contract import RoleMasteryContract, RoleTask, RoleEvidence, BASE_IDENTITIES, KEYS
    contract = RoleMasteryContract(runtime)
    pack = {"contract_vocabulary": definitions["VOCAB"], "role_rules": definitions["ROLE_RULES"],
            "response_schema": {"required_exact_keys": list(KEYS)}}
    for role, profile in runtime["roles"].items():
        assert profile["vocabulary"] == definitions["VOCAB"][role], role
        assert profile["rule"] == definitions["ROLE_RULES"][role], role
        candidate = config["candidates"][role]
        for key in ("load_kind", "decoding", "reasoning_projection", "chat_template_kwargs"):
            assert profile[key] == candidate[key], (role, key)
        assert BASE_IDENTITIES[role] == (candidate["model_id"], candidate["revision"]), role
        task = RoleTask(role, "Independent controlled case.", (RoleEvidence("ref-1", "Measured fact."),),
                        "R", "L", "smoke_test")
        original = namespace["role_prompt"](pack, {"role":role, "situation":task.situation,
            "evidence":[{"ref":"ref-1", "fact":"Measured fact."}]})
        assert contract.prompt(task) == original, role
    print("Five role profiles, base identities and rendered prompts match frozen training source.")


if __name__ == "__main__":
    main()
