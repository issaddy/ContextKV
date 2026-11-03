"""Operation/scope semantics are visible and checked on immutable TaskStates."""
import json

import pytest

from adapters.history_state import TASK_NEED_SEMANTICS, TASK_NEED_SEMANTICS_VERSION
from adapters.query_selector import CompletionResult
from controller.context_schema import Need, RetrievalState, TaskState, TextRef
from controller.schema import OnlineTurn, TurnRecord
from controller.serialization import to_jsonable
from training.task_labeling import verify_grounding
from test_task_ref_ownership import WireFixture, interpreter, actual_memory


def test_task_teacher_sees_operation_scope_definitions_in_its_actual_system_message():
    class Client(WireFixture):
        def complete(self, messages, *, response_format):
            self.system = messages[0]["content"]
            return super().complete(messages, response_format=response_format)

    client = Client()
    current = OnlineTurn("s", "q", 3, "snapshot", "What steps do I have to take?", (), (), "P3")
    interpreter(client).interpret(current, actual_memory())
    assert TASK_NEED_SEMANTICS in client.system
    assert "procedure: identify the actions" in client.system
    assert "enumerate: list the requested entities" in client.system
    assert "infer: connect multiple factual relations" in client.system
    assert "Multiple available documents do not establish all_supplied_documents" in client.system
    assert "Asking for all required steps" in client.system
    assert client.body["need_semantics_version"] == TASK_NEED_SEMANTICS_VERSION
    # It remains a model decision; the CPU decoder never changes the operation.
    assert client.output["needs"]["n0"]["operation"] == "lookup"


@pytest.mark.parametrize("operation_ok,scope_ok,history_ok", [(False, True, True), (True, False, True), (True, True, False)])
def test_independent_review_separately_rejects_wrong_operation_scope_or_history_without_rewriting(operation_ok, scope_ok, history_ok):
    history = (TurnRecord(1, "Replace my title certificate.", "Actual prior assistant answer"),)
    question = "What steps do I have to take?"
    state = TaskState("task", question, needs=(Need("need", (TextRef(2, "user", 0, len(question)),),
        operation="lookup", scope="all_supplied_documents"),), current_turn_id=2)
    original = to_jsonable(state)
    row = {"task_id": "task", "question": question, "actual_history": to_jsonable(history),
        "grounding": [], "cohort_manifest_sha256": "registered", "original_sources": [],
        "grounding_original_sources": [], "reference_history": [{"text": "Authored history must not bind the task"}]}

    class Client:
        def complete(self, messages, *, response_format):
            body = json.loads(messages[-1]["content"])
            self.schema = response_format["json_schema"]["schema"]
            assert body["task_state"] == original
            assert body["actual_history"][0]["assistant_text"] == "Actual prior assistant answer"
            assert TASK_NEED_SEMANTICS in messages[0]["content"]
            assert "half-open [start,end)" in messages[0]["content"]
            assert "Author reference history never establishes actual intent" in messages[0]["content"]
            assert self.schema["properties"]["need_checks"]["required"] == ["need"]
            return CompletionResult("completed", json.dumps({"script_history_mismatch": False,
                "applicable_grounding_indices": [], "correct_task_binding": True, "source_support_verified": True,
                "need_checks": {"need": {"operation_correct": operation_ok, "scope_correct": scope_ok,
                    "reason": "Independent fixed-label assessment."}},
                "history_references_correct": history_ok, "history_reference_reason": "Review actual owners and objects."}),
                "stop", 10, {"completion_tokens": 10})

    reviewed = verify_grounding(row, state, RetrievalState(1, history), Client())
    assert not reviewed["need_semantics_verified"]
    assert reviewed["correct_task_binding"]  # A generic true cannot override a failed specific check.
    assert reviewed["task_label_review_protocol"] == "fixed_task_state_semantics_v2"
    assert to_jsonable(state) == original
    assert "task_state" not in reviewed  # The verifier returns assessments, never a rewritten target.
