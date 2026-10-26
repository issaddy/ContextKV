"""Explicit JSON serialization for immutable contracts; no executable persistence."""
from __future__ import annotations

import dataclasses
from functools import cache
import json
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

from .schema import TaskFrame


@cache
def _schema_hints(cls):
    # Contract classes have fixed annotations; no session data is cached.
    return get_type_hints(cls)


def to_jsonable(value: Any) -> Any:
    if value is None or type(value) in (str, int, float, bool):
        return value
    if dataclasses.is_dataclass(value):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, (set, frozenset)):
        return [to_jsonable(v) for v in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    return value


def dumps(value: Any) -> str:
    # Let the native JSON encoder traverse primitive containers (especially
    # token IDs). Expanding the entire tree in Python repeated that traversal
    # at every durable state transition and event write.
    return json.dumps(value, default=_json_default, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {f.name: getattr(value, f.name) for f in dataclasses.fields(value)}
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def decode(cls: Any, value: Any) -> Any:
    if value is None or cls is Any or cls in (str, int, float, bool):
        return value
    origin, args = get_origin(cls), get_args(cls)
    if origin in (Union, types.UnionType):
        candidates = [t for t in args if t is not type(None)]
        if len(candidates) != 1:
            raise TypeError(f"Unsupported ambiguous JSON union: {cls}")
        return decode(candidates[0], value)
    if origin in (tuple, list, frozenset, set):
        if origin is tuple and args and args[-1] is not Ellipsis:
            if len(args) != len(value):
                raise ValueError("Invalid tuple length")
            return tuple(decode(t, v) for t, v in zip(args, value))
        element = args[0] if args else Any
        # Primitive decoding is intentionally non-coercing. Reconstruct the
        # container directly instead of dispatching once per stored token ID.
        if element is Any or element in (str, int, float, bool):
            return origin(value)
        return origin(decode(element, v) for v in value)
    if dataclasses.is_dataclass(cls):
        if cls is TaskFrame and isinstance(value.get("conditions"), str):
            # v9 stored one composite condition. Keep its exact content until
            # the normal interpreter can split it; never guess attribute names.
            legacy = value["conditions"]
            value = {**value, "conditions": (("legacy", legacy),) if legacy else ()}
        hints = _schema_hints(cls)
        unknown = set(value) - set(hints)
        if unknown:
            raise ValueError(f"Unknown fields for {cls.__name__}: {unknown}")
        return cls(**{k: decode(hints[k], v) for k, v in value.items()})
    return value


def loads(cls: Any, payload: str) -> Any:
    return decode(cls, json.loads(payload))
