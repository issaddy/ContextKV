"""Offline reader stage audit, immutable review and native cohort boundaries."""


from dataclasses import dataclass


import json


from types import SimpleNamespace


import pytest






from training.reader_data import (history_digest, reader_review_body, review_reader_trace)




def trace(identity="answer", phase="ANSWER"):
    return {"example_id": identity, "task_id": identity, "attempt_id": identity,
        "conversation_id": identity, "turn": 1, "split": "train", "phase": phase,
        "step_id": identity, "question": "Read the mass.", "messages": [
            {"role": "user", "content": "Observed mass: 655 kg."}],
        "actual_history": [], "actual_history_sha256": history_digest([]),
        "prompt_token_hash": identity, "reader_identity": "R0", "task_state": None,
        "read_protocol_version": 2, "read_task": {"output_budget": 1024},
        "output_schema": {"type": "object"}, "actual_output": "655 kg",
        "original_sources": [{"source_key": {"document_id": identity}, "start": 0,
            "end": 31, "text": "Observed mass: 655 kg. Total: 666 kg."}]}


@dataclass
class Completion:
    text: str
    input_tokens: int = 10
    usage: dict = None
    status: str = "completed"
    finish_reason: str = "stop"
    elapsed_ms: float = 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["READ", "REDUCE", "ANSWER"])
async def test_fixed_old_targets_only_get_independent_review_with_complete_original(phase):
    item = trace(phase, phase)
    target = "655 kg" if phase == "ANSWER" else json.dumps({"claims": [{"text": "655 kg"}]})
    calls, ledger = [], []

    class Client:
        config = SimpleNamespace(max_new_tokens=1024)
        tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text.encode()))

        def complete(self, messages, *, response_format, max_new_tokens, chat_template_kwargs):
            calls.append(response_format["json_schema"]["name"])
            body = json.loads(messages[-1]["content"])
            assert body["target_output"] == (target if phase == "ANSWER" else json.loads(target))
            assert "666 kg" in body["original_sources"][0]["text"]
            assert "target" not in response_format["json_schema"]["schema"]["properties"]
            return Completion(json.dumps({"quality": .2, "source_support_verified": False,
                "correct_task_binding": True, "visible_prompt_support": True, "upstream_errors": [], "reason": "incorrect total"}),
                usage={"completion_tokens": 4})

    result = await review_reader_trace(item, Client(), minimum_quality=.85,
        reviewer_identity="teacher", review_input_budget=4096, record_call=ledger.append,
        fixed_target_text=target)
    assert calls == ["reader_review"]
    assert result["target_text"] == target and result["status"] == "rejected"
    assert result["review_protocol"] == "visible_target_independent_review_v6"
    assert ledger[0]["stage"] == "review"
    assert "666 kg" in reader_review_body(item)["original_sources"][0]["text"]
