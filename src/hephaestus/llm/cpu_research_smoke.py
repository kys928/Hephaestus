"""Separate CPU-only research role smoke contract: real tiny trained weights.

This contract is NOT the frozen five-role mastery contract and is not admitted
into CertifiedRoleReasoner, ModelBackedStage, or production promotion. It trains
a small Torch model on explicitly synthetic policy-like binary features to
exercise five actual role-conditioned forward passes. Training labels are NOT
scientific judgments, and this is NOT a valid production role certificate.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROLES = ("controller", "diagnosis", "planner", "evaluator", "judge")
VERDICTS = ("proceed_with_review", "needs_evidence", "reject_observed")
SCHEMA = "cpu-research-role-inference-smoke.v1"


def _torch():
    import torch
    return torch


def _features(role: str, *, evidence_present: bool, provenance_verified: bool,
              contradiction: bool, approval_present: bool) -> list[float]:
    if role not in ROLES:
        raise ValueError("unknown CPU research role")
    return [float(role == key) for key in ROLES] + [
        float(evidence_present), float(provenance_verified),
        float(contradiction), float(approval_present),
    ]


def _target(role: str, evidence: bool, provenance: bool, contradiction: bool,
            approval: bool) -> int:
    """Synthetic supervised label generator, NEVER a decision policy or agent."""
    if contradiction:
        return 2
    if not evidence or not provenance:
        return 1
    if role in {"controller", "planner", "judge"} and not approval:
        return 1
    return 0


class TinyResearchRoleNet:
    """Only instantiated by the explicitly scoped CPU research runner."""

    def __init__(self):
        torch = _torch()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(9, 32), torch.nn.Tanh(),
            torch.nn.Linear(32, 24), torch.nn.Tanh(),
            torch.nn.Linear(24, len(VERDICTS)),
        )

    def __call__(self, batch):
        return self.model(batch)


@dataclass(frozen=True, slots=True)
class CpuResearchRoleCheckpoint:
    path: Path
    sha256: str
    optimizer_steps: int
    initial_loss: float
    final_loss: float
    training_accuracy: float
    training_examples: int
    scope: str = "nonproduction_synthetic_binary_feature_smoke"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1_048_576), b""):
            digest.update(data)
    return "sha256:" + digest.hexdigest()


def train_cpu_research_roles(destination: Path, *, seed: int = 97,
                             maximum_epochs: int = 280) -> CpuResearchRoleCheckpoint:
    """Perform genuine autograd/Adam updates and persist actual model weights."""
    torch = _torch()
    if maximum_epochs < 2 or maximum_epochs > 500:
        raise ValueError("CPU research training epochs outside bounded test range")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    net = TinyResearchRoleNet()
    xs, ys = [], []
    for role in ROLES:
        for evidence in (False, True):
            for provenance in (False, True):
                for contradiction in (False, True):
                    for approval in (False, True):
                        xs.append(_features(role, evidence_present=evidence,
                                            provenance_verified=provenance,
                                            contradiction=contradiction,
                                            approval_present=approval))
                        ys.append(_target(role, evidence, provenance,
                                          contradiction, approval))
    inputs = torch.tensor(xs, dtype=torch.float32)
    labels = torch.tensor(ys, dtype=torch.long)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.model.parameters(), lr=0.045, weight_decay=0.001)
    initial_loss = float(criterion(net(inputs), labels).detach().item())
    final_loss = initial_loss
    steps = 0
    for _ in range(maximum_epochs):
        optimizer.zero_grad(set_to_none=True)
        prediction = net(inputs)
        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
        steps += 1
        if final_loss < 0.007:
            break
    with torch.no_grad():
        score = float((net(inputs).argmax(dim=-1) == labels).float().mean().item())
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": SCHEMA, "seed": seed,
        "feature_order": [*ROLES, "evidence", "provenance",
                          "contradiction", "approval"],
        "model": net.model.state_dict(),
        "optimizer_steps": steps, "train_loss_before": initial_loss,
        "train_loss_after": final_loss,
    }, destination)
    # Always reopen the actual persisted weight bytes; no in-memory-only proof.
    checkpoint = torch.load(destination, map_location="cpu", weights_only=True)
    if checkpoint["schema"] != SCHEMA or not checkpoint["model"]:
        raise ValueError("CPU research model checkpoint persistence failed")
    if (steps <= 0 or final_loss >= initial_loss or score < 0.975):
        raise ValueError("real research role optimization insufficient for smoke gate")
    return CpuResearchRoleCheckpoint(
        destination, _sha_file(destination), steps, initial_loss,
        final_loss, score, len(ys),
    )


def infer_cpu_research_role(checkpoint: CpuResearchRoleCheckpoint, *, role: str,
                            evidence_present: bool, provenance_verified: bool,
                            contradiction: bool, approval_present: bool) -> dict[str, Any]:
    """Inference only from reloaded trained tensor weights, no role-label fallback."""
    torch = _torch()
    if checkpoint.scope != "nonproduction_synthetic_binary_feature_smoke":
        raise PermissionError("CPU research checkpoint cannot enter production inference")
    if _sha_file(checkpoint.path) != checkpoint.sha256:
        raise ValueError("trained role checkpoint bytes changed")
    packed = torch.load(checkpoint.path, map_location="cpu", weights_only=True)
    if packed.get("schema") != SCHEMA:
        raise ValueError("CPU smoke model checkpoint identity mismatch")
    net = TinyResearchRoleNet()
    net.model.load_state_dict(packed["model"], strict=True)
    net.model.eval()
    feat = _features(role, evidence_present=evidence_present,
                     provenance_verified=provenance_verified,
                     contradiction=contradiction,
                     approval_present=approval_present)
    with torch.inference_mode():
        logits = net(torch.tensor([feat], dtype=torch.float32))[0]
        probabilities = torch.softmax(logits, dim=-1)
        selected = int(torch.argmax(logits).item())
    return {
        "schema": SCHEMA, "role": role, "checkpoint_sha256": checkpoint.sha256,
        "input_features": feat,
        "logits": [float(x) for x in logits.tolist()],
        "probabilities": [float(x) for x in probabilities.tolist()],
        "selected_index": selected, "selection": VERDICTS[selected],
        "is_certified_model": False, "action_authorized": False,
        "scientific_claim": "no",
    }
