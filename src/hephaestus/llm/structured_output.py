from __future__ import annotations

import json
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("structured output must be a JSON object")
    return payload


def validate_required_keys(payload: dict[str, Any], required_keys: list[str]) -> list[str]:
    return [key for key in required_keys if key not in payload]
