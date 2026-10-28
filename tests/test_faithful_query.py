import json
from types import SimpleNamespace

import pytest
import yaml

from adapters.faithful_query import FaithfulQueryResolver
from adapters.mtrag import OnlineMessage, OnlineTurn
from adapters.query_resolution import PriorContext
from adapters.query_selector import CompletionResult
from adapters.retriever_factory import create_retriever, load_retriever_config
from controller.schema import QueryContext, TaskFrame
from test_retriever_factory import retriever_config as retriever_config


class Client:
    manifest = {"model": "fixture"}

    def __init__(self, config=None, *, value=None, lengths=None, finish="stop"):
        self.config = config or SimpleNamespace(max_input_tokens=200)
        self.value = value or {"query": "Aurora eligibility without membership?",
                               "relation": "followup", "history_turns": [1]}
        self.lengths, self.finish, self.inputs, self.closed = lengths, finish, [], False

    def catalog_token_lengths(self, messages, answers):
        return 100, self.lengths if self.lengths is not None else [10] * len(answers)

    def complete(self, messages, **kwargs):
        self.inputs.append((messages, kwargs))
        return CompletionResult("completed", json.dumps(self.value), self.finish, 120)

    def close(self):
        self.closed = True


def turn():
    return OnlineTurn("s", "q", 3, "c", "What if I am not a member?", (
        OnlineMessage("user", "How do I apply to Aurora?"), OnlineMessage("assistant", "Original first answer."),
        OnlineMessage("user", "I live in West."), OnlineMessage("assistant", "Original second answer.")), (), "P3")


def test_actual_dialogue_is_used_without_reinterpreted_state_or_reference_answers():
    client = Client()
    context = PriorContext(interpretations=(QueryContext("Invented condition", "standalone",
        task_frame=TaskFrame(task="wrong task", conditions=(("age", "62"),))),))
    view = FaithfulQueryResolver(client).resolve(turn(), context)
    body = json.loads(client.inputs[0][0][-1]["content"])
    assert body["current_question"] == turn().query
    assert [r["question"] for r in body["history"]] == list(turn().user_questions[:-1])
    assert [r["answer"] for r in body["history"]] == [m.text for m in turn().history[1::2]]
    assert "Invented condition" not in json.dumps(body) and "wrong task" not in json.dumps(body)
    assert view["context"] == {"query": client.value["query"], "relation": "followup",
                               "history_turns": (1,), "task_frame": None}


def test_budget_keeps_all_user_clauses_and_whole_recent_answers():
    client = Client(lengths=[500, 10])
    view = FaithfulQueryResolver(client).resolve(turn(), PriorContext())
    body = json.loads(client.inputs[0][0][-1]["content"])
    assert len(body["history"]) == 2 and "answer" not in body["history"][0]
    assert body["history"][1]["answer"] == turn().history[-1].text
    assert view["history"] == {"user_turns": [1, 2], "answer_turns": [2]}


@pytest.mark.parametrize("value,finish", [
    ({"query": "wrong", "relation": "revisit", "history_turns": [99]}, "stop"),
    ({"query": "", "relation": "followup", "history_turns": []}, "stop"),
    ({"query": "unfinished", "relation": "followup", "history_turns": []}, "length"),
])
def test_invalid_or_incomplete_resolution_falls_back_to_literal_history_once(value, finish):
    client = Client(value=value, finish=finish)
    view = FaithfulQueryResolver(client).resolve(turn(), PriorContext())
    assert view["status"] == "failed" and view["context"] is None
    assert view["text"] == "\n".join(turn().user_questions) and len(client.inputs) == 1


def test_ambiguity_keeps_original_question_in_the_generation_context():
    client = Client(value={"query": "guessed referent", "relation": "ambiguous", "history_turns": [1, 2]})
    view = FaithfulQueryResolver(client).resolve(turn(), PriorContext())
    assert view["context"]["query"] == turn().query and view["context"]["relation"] == "ambiguous"
    assert "guessed referent" not in view["text"]


def test_factory_wires_the_resolver_to_shared_retrieval_and_closes_it(retriever_config, monkeypatch):
    from adapters import query_selector
    monkeypatch.setattr(query_selector, "OpenAICompletionClient", Client)
    value = yaml.safe_load(retriever_config.read_text())
    value.update(query_mode="faithful", selector={"base_url": "http://fixture", "model": "fixture", "revision": "v1"})
    retriever_config.write_text(yaml.safe_dump(value))
    retriever = create_retriever("P3", config_path=retriever_config)
    try:
        assert retriever.requires_history_context and not retriever.requires_source_context
        result = retriever.retrieve_turn(turn(), prior_context=PriorContext())
        assert result.query == turn().query and result.history == turn().history
        assert result.query_context.query == retriever.resolver.client.value["query"]
        assert retriever.encoder.inputs == [result.query_context.query]
        assert retriever.reranker.inputs == retriever.encoder.inputs
        assert len(result.contexts) == 5
    finally:
        retriever.close()
    assert retriever.resolver.client.closed
    value.pop("selector")
    retriever_config.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="requires a selector"):
        load_retriever_config(retriever_config)
