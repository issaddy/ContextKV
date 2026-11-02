import asyncio
import json

import numpy as np
import pytest

from adapters.history_state import (HistoryIndex, TaskInterpreter, TeacherTaskModel, history_text,
                                    boundary_ref_value, decode_boundary_task_refs, owned_text_ref_definitions)
from adapters.query_selector import CompletionResult
from adapters.retrieval import run_retrieval_worker
from controller.config import RunConfig, create_controller
from controller.context_budget import resolve_text
from controller.context_schema import Need, RetrievalState, TaskState, TextRef
from controller.schema import OnlineTurn, SessionSnapshot, TurnRecord
from dataclasses import replace
from test_task_state_sources import Characters


def actual_memory():
    rows = (TurnRecord(1, "Spouse identity documents and proof of my identity. "+"x"*90, "Actual first answer"),
            TurnRecord(2, "Our last names differ", "Actual second answer"))
    return RetrievalState(2, rows, (TaskState("original-task", rows[0].user_text,
        needs=(Need("prior-need"),), current_turn_id=1),))


class WireFixture:
    manifest = {"source": "specified-native-wire-unit-test"}
    def __init__(self, *, invalid=False, nullable=False):
        self.invalid, self.nullable = invalid, nullable
    def complete(self, messages, *, response_format):
        self.body = json.loads(messages[-1]["content"])
        self.schema = response_format["json_schema"]["schema"]
        current = {"turn_id": 3, "speaker": "user", "boundary_chars": [0, len(self.body["question"])-1]}
        anchors = [current]
        if self.invalid:
            # The absent owner from pilot-v2 remains invalid in the new wire.
            anchors.append({"turn_id": 0, "speaker": "user", "boundary_chars": [0, 114]})
        conditions = ({"c0": {"key_span": None, "predicate_span": None, "value_spans": [current],
            "relation_type": "PREDICATE", "target_refs": [], "supersedes": [], "update": "ADD"}}
            if self.nullable else {})
        self.output = {"relation": "revisit", "resume_task_id": "S0", "history_spans": [],
            "conditions": conditions, "condition_removals": [], "needs": {"n0": {
                "anchor_spans": anchors, "subject_refs": [], "operation": "lookup",
                "condition_ids": ["c0"] if self.nullable else [], "predecessor_need_ids": [],
                "scope": "selected_sources"}}, "version_intent": {"mode": "current", "temporal_relation": "at",
                "document_families": [], "version_ids": [], "resolved_version_ids": [], "time_refs": [], "source_turns": []}}
        return CompletionResult("completed", json.dumps(self.output), "stop", 20,
                                {"completion_tokens": 30}, elapsed_ms=5.0)


def interpreter(client, record_path=None):
    return TaskInterpreter(TeacherTaskModel(client, record_path=record_path),
        HistoryIndex(Characters(), lambda texts: np.ones((len(texts), 2))))


def test_native_ref_schema_owns_actual_messages_and_optional_spans_use_null():
    client = WireFixture(nullable=True)
    interpreted = interpreter(client).interpret(OnlineTurn("s", "q3", 3, "manual", "Can I use these documents?", (), (), "P3"),
                                               actual_memory())
    references = client.schema["$defs"]["TextRef"]["anyOf"]
    owners = {(branch["properties"]["turn_id"]["const"], branch["properties"]["speaker"]["const"])
              for branch in references}
    assert owners == {(1, "user"), (1, "assistant"), (2, "user"), (2, "assistant"), (3, "user")}
    assert (0, "user") not in owners and (3, "assistant") not in owners
    lengths = {(item["turn_id"], item["speaker"]): item["characters"] for item in client.body["text_ref_sources"]}
    for branch in references:
        bounds = branch["properties"]["boundary_chars"]
        assert bounds["minItems"] == bounds["maxItems"] == 2
        assert bounds["items"] == {"type": "integer", "minimum": 0,
            "maximum": lengths[branch["properties"]["turn_id"]["const"], branch["properties"]["speaker"]["const"]]-1}
        assert branch["required"] == ["turn_id", "speaker", "boundary_chars"] and branch["additionalProperties"] is False
    assert interpreted.conditions[0].key_span is None and interpreted.conditions[0].predicate_span is None
    assert client.schema["$defs"]["Condition_1"]["properties"]["key_span"]["anyOf"][-1] == {"type": "null"}


def ref_matches(schema, value, definitions):
    """Evaluate the JSON Schema keywords used by the independent ref fragment."""
    if isinstance(schema, bool):
        return schema
    if "$ref" in schema:
        return ref_matches(definitions[schema["$ref"].rsplit("/", 1)[1]], value, definitions)
    kind = schema.get("type")
    if kind == "object":
        if not isinstance(value, dict) or any(name not in value for name in schema.get("required", ())):
            return False
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False and set(value)-set(properties):
            return False
        if any(not ref_matches(item, value[name], definitions) for name, item in properties.items() if name in value):
            return False
    if kind == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
        return False
    if kind == "string" and not isinstance(value, str):
        return False
    if kind == "array":
        if not isinstance(value, list) or not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", len(value)):
            return False
        if any(not ref_matches(schema["items"], item, definitions) for item in value):
            return False
    if "const" in schema and value != schema["const"]:
        return False
    if "minimum" in schema and value < schema["minimum"]:
        return False
    if "maximum" in schema and value > schema["maximum"]:
        return False
    return "anyOf" not in schema or any(ref_matches(branch, value, definitions) for branch in schema["anyOf"])


@pytest.mark.parametrize("length", range(13))
def test_included_boundaries_cover_every_valid_nonempty_unicode_substring(length):
    definitions = owned_text_ref_definitions({(4, "user"): "x"*length})
    intervals = set()
    for first in range(-1, length+2):
        for second in range(-1, length+2):
            ref = {"turn_id": 4, "speaker": "user", "boundary_chars": [first, second]}
            accepted = ref_matches(definitions["TextRef"], ref, definitions)
            assert accepted == (0 <= first < length and 0 <= second < length)
            if accepted:
                internal = boundary_ref_value(ref)
                assert 0 <= internal["start"] < internal["end"] <= length
                intervals.add((internal["start"], internal["end"]))
    assert intervals == {(start, end) for start in range(length) for end in range(start+1, length+1)}
    assert set(definitions) == {"TextRef"}


def test_included_anchors_have_declared_nonempty_order_independent_semantics_and_unicode_bounds():
    definitions = owned_text_ref_definitions({(4, "user"): "x"*62, (5, "user"): "😀e\u0301中"})
    old_wire = {"turn_id": 4, "speaker": "user", "start": 21, "end": 21}
    assert not ref_matches(definitions["TextRef"], old_wire, definitions)
    single = {"turn_id": 4, "speaker": "user", "boundary_chars": [21, 21]}
    assert ref_matches(definitions["TextRef"], single, definitions)
    assert boundary_ref_value(single) == {**old_wire, "end": 22}
    unicode = {"turn_id": 5, "speaker": "user", "boundary_chars": [2, 1]}
    internal = boundary_ref_value(unicode)
    assert "😀e\u0301中"[internal["start"]:internal["end"]] == "e\u0301"
    assert ref_matches(definitions["TextRef"], unicode, definitions)
    assert not ref_matches(definitions["TextRef"], {**unicode, "boundary_chars": [1, 4]}, definitions)
    assert not ref_matches(definitions["TextRef"], {**unicode, "speaker": "assistant"}, definitions)
    for invalid in ([1.5, 2], [True, 2], [], [1], [1, 2, 3]):
        assert not ref_matches(definitions["TextRef"], {**unicode, "boundary_chars": invalid}, definitions)


def test_boundary_schema_size_depends_on_messages_and_integer_digits_not_message_characters():
    one = owned_text_ref_definitions({(1, "assistant"): "x"*10000})
    shared = owned_text_ref_definitions({(1, "assistant"): "x"*10000, (2, "user"): "z"*10000})
    huge = owned_text_ref_definitions({(1, "assistant"): "x"*1000000})
    assert len(one) == len(shared) == len(huge) == 1
    assert len(one["TextRef"]["anyOf"]) == 1 and len(shared["TextRef"]["anyOf"]) == 2
    assert len(json.dumps(one)) < 500 and len(json.dumps(shared)) < 1000
    assert len(json.dumps(huge))-len(json.dumps(one)) == 2


def test_boundary_wire_decodes_all_ref_fields_before_the_unchanged_internal_parser():
    from copy import deepcopy
    from controller.serialization import decode, to_jsonable
    ref = {"turn_id": 4, "speaker": "user", "boundary_chars": [3, 1]}
    value = {"history_spans": [ref], "conditions": {"c0": {"key_span": ref,
        "predicate_span": None, "value_spans": [ref]}},
        "condition_removals": [{"predicate_span": ref}],
        "needs": {"n0": {"anchor_spans": [ref], "subject_refs": [ref]}},
        "version_intent": {"time_refs": [ref]}}
    original = deepcopy(value)
    result = decode_boundary_task_refs(deepcopy(value))
    expected = {"turn_id": 4, "speaker": "user", "start": 1, "end": 4}
    assert result["history_spans"] == result["conditions"]["c0"]["value_spans"] == [expected]
    assert result["conditions"]["c0"]["key_span"] == result["condition_removals"][0]["predicate_span"] == expected
    assert result["conditions"]["c0"]["predicate_span"] is None
    assert result["needs"]["n0"]["anchor_spans"] == result["needs"]["n0"]["subject_refs"] == [expected]
    assert result["version_intent"]["time_refs"] == [expected]
    assert to_jsonable(decode(TextRef, expected)) == expected
    assert value == original


def test_exact_turn_zero_anchor_is_rejected_and_raw_teacher_record_is_not_repaired(tmp_path):
    client = WireFixture(invalid=True)
    calls = tmp_path/"teacher.jsonl"
    with pytest.raises(ValueError, match=r"needs\[0\].anchor_spans\[1\].*absent actual-QA message.*0"):
        interpreter(client, calls).interpret(OnlineTurn("s", "q3", 3, "manual", "Can I use these documents?", (), (), "P3"),
                                             actual_memory())
    raw = json.loads(calls.read_text())
    assert json.loads(raw["raw_output"])["needs"]["n0"]["anchor_spans"][1]["turn_id"] == 0
    assert raw["input_tokens"] == 20 and raw["usage"]["completion_tokens"] == 30
    assert raw["wire_schema"] == "owned_task_graph_v6_included_boundary_chars"
    assert "boundary_chars" in json.loads(raw["raw_output"])["needs"]["n0"]["anchor_spans"][0]
    assert all(owner["turn_id"] != 0 for owner in raw["text_ref_sources"])


@pytest.mark.parametrize("ref", (TextRef(0, "user", 0, 115), TextRef(3, "assistant", 0, 1),
                                  TextRef(1, "user", 0, 10000), TextRef(1, "user", 4, 2)))
def test_resolvers_raise_domain_errors_instead_of_iteration_completion(ref):
    memory = actual_memory()
    turn = OnlineTurn("s", "q3", 3, "manual", "Current user only", (), (), "P3")
    with pytest.raises(ValueError):
        history_text(ref, memory, turn)
    with pytest.raises(ValueError):
        resolve_text(ref, TaskState("s", turn.query, current_turn_id=3), SessionSnapshot("s", raw_history=memory.committed_history))


async def test_invalid_task_ref_finishes_worker_with_failed_attempt_and_no_qa(tmp_path):
    service = create_controller(replace(RunConfig(), state_dir=str(tmp_path)))
    memory = actual_memory()
    service.store.read_snapshot("s")
    service.store.sync_history("s", memory.committed_history, 0)
    model = interpreter(WireFixture(invalid=True))
    async def retrieve(snapshot):
        turn = OnlineTurn("s", "q3", 3, "manual", "Can I use these documents?", (), (), "P3")
        return await run_retrieval_worker(model.interpret, turn, memory)
    try:
        with pytest.raises(ValueError, match="absent actual-QA message"):
            await service.serve_turn("s", "Can I use these documents?", request_id="q3", evidence_provider=retrieve)
        assert service.store.find_request("s", "q3").stage == "FAILED"
        assert len(service.store.get_snapshot("s").raw_history) == 2
        assert not service.store.outbox()
    finally:
        await service.aclose()
        service.store.close()


async def test_stopiteration_transport_behavior_is_observed_without_hanging_the_test():
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    errors = []
    loop.set_exception_handler(lambda loop, context: errors.append(context))
    def old_missing_turn_lookup():
        return next(row for row in actual_memory().committed_history if row.turn_id == 0)
    worker = asyncio.create_task(asyncio.to_thread(old_missing_turn_lookup))
    try:
        finished, pending = await asyncio.wait({worker}, timeout=.1)
        if finished:
            # Current local CPython converts this to a delivered RuntimeError.
            assert isinstance(worker.exception(), RuntimeError) and "StopIteration" in str(worker.exception())
        else:
            # The deployed runtime's observed callback failure leaves it pending.
            assert pending == {worker}
            assert any(isinstance(context.get("exception"), TypeError) and "StopIteration" in str(context["exception"])
                       for context in errors)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        loop.set_exception_handler(previous)
