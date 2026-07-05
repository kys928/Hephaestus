from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hephaestus.policy.operator_console_policy import evaluate_operator_action
from hephaestus.schemas.operator_action import OperatorAction
from hephaestus.state._json_store import JsonStore


@dataclass(slots=True)
class OperatorActionStore:
    root: Path

    def append(self, action: OperatorAction | dict[str, object]) -> OperatorAction:
        evaluated = evaluate_operator_action(action)
        JsonStore(self.root, "operator_actions.jsonl").append(evaluated.to_dict())
        return evaluated

    def list_all(self) -> list[dict[str, object]]:
        return JsonStore(self.root, "operator_actions.jsonl").all()
