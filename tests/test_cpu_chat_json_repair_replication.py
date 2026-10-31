from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("repair",ROOT/"scripts/run_cpu_chat_json_repair.py")
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_precommitted_original_and_replica_seeds_have_exact_same_science():
    orig=mod.validate_protocol(ROOT/"configs/research/cpu_chat_json_repair_v1.json")
    assert orig["train_seed"]==1117
    for seed in (2227,3337):
        rep=mod.validate_protocol(ROOT/f"configs/research/cpu_chat_json_repair_seed{seed}.json")
        expected=dict(orig)
        expected["train_seed"]=seed
        expected["experiment_id"]=f"json-format-repair-development-20261028-seed-{seed}"
        assert rep==expected
