import asyncio
import json

import numpy as np
import pytest

from adapters.history_state import HistoryIndex, TaskInterpreter, TeacherTaskModel, history_text
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
        current = {"turn_id": 3, "speaker": "user", "start": 0, "end": len(self.body["question"])}
        anchors = [current]
        if self.invalid:
            # Exact offending field shape from the real pilot-v2 teacher record.
            anchors.append({"turn_id": 0, "speaker": "user", "start": 0, "end": 115})
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
    assert all(branch["properties"]["start"]["minimum"] == 0 and
               branch["properties"]["end"]["minimum"] == 1 for branch in references)
    assert interpreted.conditions[0].key_span is None and interpreted.conditions[0].predicate_span is None
    assert client.schema["$defs"]["Condition_1"]["properties"]["key_span"]["anyOf"][-1] == {"type": "null"}


def test_exact_turn_zero_anchor_is_rejected_and_raw_teacher_record_is_not_repaired(tmp_path):
    client = WireFixture(invalid=True)
    calls = tmp_path/"teacher.jsonl"
    with pytest.raises(ValueError, match=r"needs\[0\].anchor_spans\[1\].*absent actual-QA message.*0"):
        interpreter(client, calls).interpret(OnlineTurn("s", "q3", 3, "manual", "Can I use these documents?", (), (), "P3"),
                                             actual_memory())
    raw = json.loads(calls.read_text())
    assert json.loads(raw["raw_output"])["needs"]["n0"]["anchor_spans"][1]["turn_id"] == 0
    assert raw["input_tokens"] == 20 and raw["usage"]["completion_tokens"] == 30
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
