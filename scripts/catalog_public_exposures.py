"""Export known public fixture identities only; no final seed or final sample."""
import argparse
from collections import Counter
import json
from pathlib import Path

from hephaestus.evaluation.public_exposures import public_exposure_catalog

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("public_pilot_file", type=Path)
    args = parser.parse_args()
    rows = json.loads(args.public_pilot_file.read_text())["rows"]
    if (Counter(row["family"] for row in rows) != Counter({name: 16 for name in
            ("arithmetic", "json", "symbolic", "reasoning")})
            or any(row.get("purpose") != "nonconfirmatory-feasibility-and-power-only" for row in rows)):
        raise ValueError("catalog CLI requires the complete explicitly public pilot")
    result = public_exposure_catalog(Path(__file__).resolve().parents[1], rows)
    print(json.dumps(result, indent=2, allow_nan=False))
