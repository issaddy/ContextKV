"""One upstream dialogue path: source slices, paid calls, P3 and one QA commit."""
from dataclasses import replace

import pytest

pytest.importorskip("llama_index.core")

from bench.mock_engine import MockEngine
from controller.candidates import CandidatePolicy
from controller.context_schema import ContextBudget
from controller.evidence import make_evidence
from controller.qa import DocumentQA
from controller.service import Controller, EvidenceSelection
from controller.session_store import SessionStore


class DialogueEngine(MockEngine):
    def __init__(self):
        super().__init__()
        self.inputs = []

    async def generate(self, **kwargs):
        text = bytes(kwargs["input_ids"]).decode()
        self.inputs.append(text)
        self.response_text = ("Supported final answer." if "<llamaindex_context>" in text
                              else "Standalone followup question")
        return await super().generate(**kwargs)


@pytest.mark.asyncio
async def test_real_compact_refine_shares_qa_ledger_and_committed_history(tmp_path):
    body = " ".join(f"Entry {index}: Eligible applicants use the online form."
                    for index in range(140))
    block = make_evidence(body, "a")
    other = make_evidence("After submitting the form, retain the confirmation.", "b")

    class Provider:
        corpus_snapshot_id = "corpus"
        budget = ContextBudget(window_tokens=4096, output_tokens=128, history_tokens=1024)

        def __init__(self):
            self.queries = []

        async def retrieve(self, snapshot, query, task_id):
            self.queries.append((query, snapshot.raw_history))
            return EvidenceSelection((block, other), (block.key, other.key), auxiliary_usage={
                "input_tokens": 7, "output_tokens": 0, "compute_ms": 1.0, "models": []})

        def finish_request(self, task_id):
            pass

    engine, provider = DialogueEngine(), Provider()
    store = SessionStore(tmp_path/"state.db")
    controller = Controller(store, engine, dialogue_mode="llamaindex",
        policy=CandidatePolicy(max_total_model_tokens=4096, max_new_tokens=128))
    qa = DocumentQA(controller, provider)
    response = await qa.ask("session", "How do I apply?", "first")
    steps = store.execution_steps(response.attempt_id)
    assert len(steps) > 1
    assert steps[0].task.kind == "LLAMAINDEX_COMPACT"
    assert {span["key"]["document_id"] for span in steps[0].metrics["source_view"]["evidence_spans"]} == {"a"}
    assert all(step.task.kind == "LLAMAINDEX_REFINE" for step in steps[1:])
    assert len(engine.inputs) == len(steps)  # no additional final ANSWER
    assert response.reader_input_tokens == sum(len(step.input_ids) for step in steps)
    assert response.reader_output_tokens == sum(len(step.output_ids) for step in steps)
    assert response.total_input_tokens == response.reader_input_tokens+7
    assert len(response.step_metrics) == len(steps)-1  # outer result accounts for the final call
    attempt = store.get_attempt(response.attempt_id)
    assert attempt.context_plan.online_tokens_spent == response.total_input_tokens+response.total_output_tokens
    assert attempt.result.receipt.attempt_id.endswith(":"+attempt.final_step_id)
    for step in steps:
        source_view = step.metrics["source_view"]
        for span in source_view["evidence_spans"]:
            message = step.input_messages[span["message_index"]].content
            source = body if span["key"]["document_id"] == "a" else other.text
            assert message[span["start"]:span["end"]] == source[span["source_start"]:span["source_end"]]
            if source == body:
                assert span["source_end"]-span["source_start"] < len(body)
    assert len(store.get_snapshot("session").raw_history) == 1
    assert store.get_snapshot("session").execution.evidence_spans
    assert store.get_snapshot("session").current_epoch.state == "RETIRED"
    assert store.get_snapshot("session").must_rebuild
    calls = len(engine.inputs)
    replay = await qa.ask("session", "How do I apply?", "first")
    assert replay.idempotent_replay and len(engine.inputs) == calls
    assert len(provider.queries) == 1

    provider.budget = replace(provider.budget, max_steps=64)
    second = await qa.ask("session", "And then?", "second")
    second_steps = store.execution_steps(second.attempt_id)
    assert second_steps[0].task.kind == "LLAMAINDEX_CONDENSE"
    assert "Supported final answer." in engine.inputs[calls]
    assert provider.queries[-1][0] == "Standalone followup question"
    assert provider.queries[-1][1][0].assistant_text == response.assistant_text
    assert store.get_snapshot("session").version == 2
    await controller.aclose()
    assert not store.outbox()
    store.close()
