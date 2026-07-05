"""Structured-output parsing and schema validation for LLM boundaries."""

from __future__ import annotations

import json
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from types import NoneType, UnionType
from typing import Any, get_args, get_origin, get_type_hints

from hephaestus.schemas._base import JsonSchema


class StructuredOutputError(ValueError):
    """Raised when an LLM response is not valid structured output."""


def parse_json_object(output: str | dict[str, Any]) -> dict[str, Any]:
    """Parse a JSON object, accepting fenced JSON while rejecting non-objects."""

    if isinstance(output, dict):
        return dict(output)
    text = output.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(f"response is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise StructuredOutputError("response JSON root must be an object")
    return payload


def validate_structured_output(output: str | dict[str, Any], schema: type[JsonSchema]) -> JsonSchema:
    """Validate output against a dataclass schema and return a schema instance."""

    if not is_dataclass(schema) or not issubclass(schema, JsonSchema):
        raise StructuredOutputError(f"schema must be a JsonSchema dataclass: {schema!r}")
    payload = parse_json_object(output)
    names = {field.name for field in fields(schema)}
    extra = sorted(set(payload) - names)
    if extra:
        raise StructuredOutputError(f"unexpected field(s) for {schema.__name__}: {', '.join(extra)}")
    missing = [f.name for f in fields(schema) if f.default is MISSING and f.default_factory is MISSING and f.name not in payload]
    if missing:
        raise StructuredOutputError(f"missing field(s) for {schema.__name__}: {', '.join(missing)}")
    type_hints = get_type_hints(schema)
    for field in fields(schema):
        expected_type = type_hints.get(field.name, field.type)
        if field.name in payload and not _matches(payload[field.name], expected_type):
            raise StructuredOutputError(f"field '{field.name}' does not match expected schema type")
    try:
        return schema.from_dict(payload)
    except Exception as exc:
        raise StructuredOutputError(f"failed to construct {schema.__name__}: {exc}") from exc


def _matches(value: Any, annotation: Any) -> bool:
    if annotation is Any or annotation is object:
        return True
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (UnionType, getattr(__import__("typing"), "Union")):
        return any(_matches(value, arg) for arg in args)
    if annotation is NoneType:
        return value is None
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return value in {member.value for member in annotation} or isinstance(value, annotation)
    if origin is list:
        return isinstance(value, list) and (not args or all(_matches(item, args[0]) for item in value))
    if origin is dict:
        return isinstance(value, dict)
    if annotation is bool:
        return isinstance(value, bool)
    if annotation is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if annotation is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if annotation in (str, bytes):
        return isinstance(value, annotation)
    return True
