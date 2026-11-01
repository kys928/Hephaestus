"""Scientific variables cannot drift during an operational bootstrap retry."""
from hashlib import sha256
from hephaestus.storage.verified import canonical_bytes


def scientific_fingerprint(proposal):
    # These fields name run-scoped wrappers or an explicitly declared CPU fault.
    # Actual data bytes, recipes, parent adapters, seeds and all hyperparameters
    # remain included. Unknown constraint keys remain included by default.
    operational = {"trainable_data_contract_ref", "trainable_data_contract_hash", "data_contract_ref",
                   "processing_evidence_ref", "processing_evidence_hash", "fixture_bootstrap_failure"}
    constraints = {k: v for k, v in proposal.training_constraints.items() if k not in operational}
    return "sha256:" + sha256(canonical_bytes({"lineage": proposal.lineage_id, "stage": proposal.stage_name,
        "baseline": proposal.baseline_ref, "primary_variable": proposal.primary_variable,
        "controlled_variables": proposal.controlled_variables, "constraints": constraints})).hexdigest()
