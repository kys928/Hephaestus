"""Read a separately committed public pilot; never generate tasks or select seeds."""
import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from hephaestus.evaluation.public_pilot_analysis import paired_template_diagnostics
from hephaestus.storage.verified import canonical_bytes


def analyse(root, expected_analysis_ref):
    publication = json.loads((root / "analysis.json").read_text())
    analysis = publication["analysis"]
    observed = "sha256:" + sha256(canonical_bytes(analysis)).hexdigest()
    if publication["analysis_ref"] != expected_analysis_ref or observed != expected_analysis_ref:
        raise ValueError("public analysis differs from separately retained commitment")
    if (analysis["schema"] != "public-smol-pilot-analysis.v2"
            or analysis["protected_final_holdout_exposed"] is not False
            or analysis["confirmatory_improvement_established"] is not False):
        raise ValueError("this diagnostic accepts only the declared public pilot")
    outputs = {}
    for label, ref in analysis["generation_refs"].items():
        hexdigest = ref.removeprefix("sha256:")
        raw = (root / "artifacts/objects/sha256" / hexdigest[:2] / hexdigest).read_bytes()
        if "sha256:" + sha256(raw).hexdigest() != ref:
            raise ValueError("public raw generation commitment differs")
        outputs[label] = json.loads(raw)
    pairs = {label: paired_template_diagnostics(output["samples"]) for label, output in outputs.items()}
    if any(row["paired_canonical_units"] != 32 or row["all_canonical_units"] != 64 for row in pairs.values()):
        raise ValueError("public sensitivity roster differs from the predeclared pilot")
    train = json.loads((root / "original-public-train.json").read_text())["rows"]
    pilot = json.loads((root / "public-pilot.json").read_text())["rows"]
    for name, rows in (("train", train), ("pilot", pilot)):
        if "sha256:" + sha256(canonical_bytes(rows)).hexdigest() != analysis["contract"]["source_hashes"][name]:
            raise ValueError("public source differs from the frozen input commitment")
    prior = {}
    for family in ("arithmetic", "json", "symbolic", "reasoning"):
        counts = Counter(row["target"] for row in train if row["family"] == family)
        mode = min(counts, key=lambda target: (-counts[target], target))
        items = [row for row in pilot if row["family"] == family]
        prior[family] = {"training_modal_target": mode, "training_count": counts[mode],
                         "pilot_hits": sum(row["target"] == mode for row in items), "pilot_n": len(items)}
    return {"schema": "public-smol-pilot-supplement.v1", "analysis_ref": expected_analysis_ref,
            "paired_template_sensitivity": pairs, "training_modal_label_diagnostic": prior,
            "constant_diagnostic_scope": "theoretical complete JSON; not a model inference or a selector",
            "protected_final_material_accessed": False, "confirmatory_effect_claimed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-analysis-ref", required=True)
    args = parser.parse_args()
    print(json.dumps(analyse(args.root, args.expected_analysis_ref), indent=2, allow_nan=False))
