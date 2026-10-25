from __future__ import annotations

from dataclasses import dataclass
import math

import pytest

from controller.schema import EvidenceKey, Message, SessionSnapshot, TurnRecord
from controller.serialization import decode, dumps, loads, to_jsonable


@dataclass(frozen=True)
class ScalarContract:
    text: str
    count: int
    ratio: float
    enabled: bool
    optional: str | None
    ids: tuple[int, ...]
    pair: tuple[str, int]


def test_canonical_json_bytes_and_scalar_contract_values_are_unchanged():
    value = ScalarContract("中文\nquote\"", 3, .25, False, None, (1, 2), ("x", 4))
    expected = ('{"count":3,"enabled":false,"ids":[1,2],"optional":null,'
                '"pair":["x",4],"ratio":0.25,"text":"中文\\nquote\\\""}')
    assert dumps(value) == expected
    assert loads(ScalarContract, expected) == value
    assert to_jsonable({"set": frozenset({3, 1}), "list": [False, None, "x"]}) == {
        "set": [1, 3], "list": [False, None, "x"]}


def test_nested_session_roundtrip_preserves_immutable_containers_and_references():
    key = EvidenceKey("corpus", "doc", "passage", "v1", "body-hash")
    history = (TurnRecord(1, "question", "answer", (1, 5, 8), evidence_keys=(key,)),)
    snapshot = SessionSnapshot("session", raw_history=history, reference_map=(("E_ref", key),))
    reconstructed = loads(SessionSnapshot, dumps(snapshot))
    assert reconstructed == snapshot
    assert type(reconstructed.raw_history) is tuple
    assert type(reconstructed.raw_history[0].assistant_output_ids) is tuple
    assert type(reconstructed.reference_map[0][1]) is EvidenceKey


@pytest.mark.parametrize("cls", [str, int, float, bool])
def test_decode_retains_existing_noncoercing_primitive_semantics(cls):
    value = {"already": "not coerced or validated"}
    assert decode(cls, value) is value
    assert decode(cls, None) is None


@pytest.mark.parametrize("container", [tuple,list,set,frozenset])
def test_primitive_containers_preserve_noncoercing_semantics(container):
    values = [1, "not coerced", None, False]
    annotation = tuple[int,...] if container is tuple else container[int]
    assert decode(annotation, values) == container(values)
    assert type(decode(annotation, values)) is container


def test_unknown_fields_tuple_arity_and_ambiguous_union_still_fail():
    assert loads(Message, '{"role":"user","content":"first"}') == Message("user", "first")
    with pytest.raises(ValueError, match="Unknown fields for Message"):
        loads(Message, '{"role":"user","content":"second","extra":1}')
    with pytest.raises(ValueError, match="Invalid tuple length"):
        decode(tuple[str, int], ["x"])
    with pytest.raises(TypeError, match="Unsupported ambiguous JSON union"):
        decode(str | int, "x")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_output_is_rejected_even_through_scalar_fast_path(value):
    with pytest.raises(ValueError, match="Out of range float values"):
        dumps({"nested": (value,)})
    # Loading keeps the pre-existing stdlib JSON behavior; this optimization is
    # not a schema validation change.
    restored = loads(float, "NaN")
    assert math.isnan(restored)
