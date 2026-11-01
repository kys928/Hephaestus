"""Inventory known public fixtures, never a protected sample or final seed.

The deliberately large mock scorer fixture is public too. Repository coverage is
bounded; an independent custodian must reconcile external and dynamic exposures.
"""
import ast
from hashlib import sha256
from pathlib import Path

from hephaestus.data.generated_factory import generate_original_splits, generator_code_sha256, validator_code_sha256
from hephaestus.data.generated_validation import parse_and_solve, validate_original_example
from hephaestus.schemas.generated_data import GeneratedDataSpec
from hephaestus.schemas.prospective_study import ProspectiveDesign
from hephaestus.storage.verified import canonical_bytes
from .prospective_design import arithmetic_capacity, build_custodian_holdout


def public_exposure_catalog(repository_root, pilot_rows):
    root = Path(repository_root)
    keys, sources = {}, {}

    def add(key):
        keys["sha256:" + sha256(canonical_bytes(key)).hexdigest()] = key

    def add_rows(rows):
        for row in rows:
            add(validate_original_example(row))

    # These are published, explicitly synthetic unit/development specifications.
    # The 812351 fixture's source explicitly says it is not the real evaluation.
    specs = [GeneratedDataSpec(20261028), GeneratedDataSpec(31, 12, 3, 3),
             GeneratedDataSpec(93, 12, 3, 3), GeneratedDataSpec(413, 12, 3, 3),
             GeneratedDataSpec(812351, 24, 6, 512, 1000)]
    for spec in specs:
        splits, _ = generate_original_splits(spec)
        for rows in splits.values():
            add_rows(rows)
    original, _ = generate_original_splits(GeneratedDataSpec(20261028))
    original_ids = {row["sample_id"] for rows in original.values() for row in rows}
    design = ProspectiveDesign(purpose="rehearsal", cases_per_family=2, max_new_tokens=16)
    add_rows(build_custodian_holdout(design, custodian_seed=152001,
                                    exposure_sample_ids={"sha256:" + "a" * 64}))
    first = build_custodian_holdout(design, custodian_seed=152002,
                                   exposure_sample_ids={"sha256:" + "b" * 64})
    add_rows(first)
    add_rows(build_custodian_holdout(design, custodian_seed=152002,
        exposure_sample_ids={"sha256:" + "b" * 64, first[0]["sample_id"]}))
    add_rows(build_custodian_holdout(
        ProspectiveDesign(purpose="rehearsal", cases_per_family=1, max_new_tokens=16),
        custodian_seed=152200, exposure_sample_ids={row["sample_id"] for row in original["development"][:4]}))
    add_rows(build_custodian_holdout(design, custodian_seed=152100, exposure_sample_ids=original_ids))
    for row in pilot_rows:
        key = validate_original_example(row)
        if int(sha256(canonical_bytes(key)).hexdigest(), 16) % 10 != 8:
            raise ValueError("public pilot catalog accepts only development bucket 8")
        add(key)

    # Literal valid grammar prompts are exposures even in a negative parser test.
    for directory in (root / "tests", root / "scripts"):
        for path in sorted(directory.rglob("*.py")):
            raw = path.read_bytes()
            sources[str(path.relative_to(root))] = "sha256:" + sha256(raw).hexdigest()
            for node in ast.walk(ast.parse(raw)):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    try:
                        key, _ = parse_and_solve(node.value)
                    except ValueError:
                        continue
                    add(key)
    exposed_arithmetic = sum(key[0] == "arithmetic" and all(0 <= n < 100 for n in key[2])
        and int(sid[7:], 16) % 10 == 9 for sid, key in keys.items())
    ids = sorted(keys)
    capacity = arithmetic_capacity(100)["evaluation"]
    return {"schema": "known-public-exposure-catalog.v2", "sample_ids": ids,
            "catalog_commitment": "sha256:" + sha256(canonical_bytes(ids)).hexdigest(),
            "generator_code_sha256": generator_code_sha256(),
            "validator_code_sha256": validator_code_sha256(),
            "pilot_canonical_cases_included": len(pilot_rows),
            "known_unique_cases": len(ids), "public_generator_specs": [spec.to_dict() for spec in specs],
            "repository_source_files": sources, "arithmetic_bucket9_capacity": capacity,
            "known_arithmetic_bucket9_exclusions_in_range100": exposed_arithmetic,
            "remaining_arithmetic_bucket9_capacity": capacity - exposed_arithmetic,
            "protected_final_material_accessed": False,
            "limitations": ["Known repository fixtures and supplied public pilot only; external/dynamic exposures need independent reconciliation.",
                "Canonical aliases are excluded; all semantic near-duplicates and pretrained exposure are not ruled out."]}
