from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from adapters.mtrag import OnlineMessage, OnlineTurn
from adapters.query_resolution import (PriorContext, PriorSource, TopicResolver, candidates,
    controller_context_provider, native_context_provider, snapshot_context)
from adapters.query_selector import CompletionResult
from controller.evidence import make_evidence
from controller.mock_engine import MockEngine
from controller.planner import PlannerKnobs
from controller.schema import SessionSnapshot, TurnRecord
from controller.service import Controller
from controller.session_store import SessionStore


def fixture():
    evidence = make_evidence("IBM Cloudant is a JSON document database. Cloudant accepts attachments.", "cloudant")
    history = (OnlineMessage("user", "Does IBM offer document databases?"),
               OnlineMessage("assistant", "IBM Cloudant supports documents; the limit is 50 MB. PhantomStore has 90 MB."),
               OnlineMessage("user", "Can I store an image or pdf with a document?"),
               OnlineMessage("assistant", "Use Cloudant. The invented limit is 70 MB. `FakeSDK` is a code name."))
    turn = OnlineTurn("s", "task", 3, "cloud", "Is there a file size limit? ", history, (), "P3")
    context = PriorContext((PriorSource(evidence, (1, 2)),), snapshot_version=2)
    return turn, context


class SelectingClient:
    manifest = {"model": "fake-selector", "revision": "fixture"}
    def __init__(self, output=None, finish="stop", status="completed"):
        self.inputs, self.output, self.finish, self.status = [], output, finish, status
    def complete(self, messages, **kwargs):
        self.inputs.append((messages, kwargs))
        body = json.loads(messages[-1]["content"])
        result = self.output
        if result is None:
            match = next(e for e in body["entity_candidates"] if e["text"] == "IBM Cloudant")
            result = {"status": "resolved", "entity_ids": [match["id"]], "user_turn_ids": ["u2"]}
        text = result if isinstance(result, str) else json.dumps(result)
        return CompletionResult(self.status, text, self.finish, 100, raw_response={"raw": text})
    def close(self):
        pass


def test_source_qualified_entity_and_user_object_exclude_assistant_numbers_and_code():
    turn, context = fixture()
    client = SelectingClient()
    result = TopicResolver(client).resolve(turn, context)
    assert result["status"] == "resolved" and result["text"].startswith(turn.query + "\n")
    assert "IBM Cloudant" in result["text"] and "image or pdf" in result["text"]
    assert all(s not in result["text"] for s in ("50", "70", "90", "PhantomStore", "FakeSDK"))
    entity = result["selected_entities"][0]
    assert entity["supports"]
    for support in entity["supports"]:
        span = support["span"]
        assert context.sources[0].block.text[span["start"]:span["end"]] == span["text"]
    assert "task" not in json.loads(client.inputs[0][0][-1]["content"])
    assert result["text"] != turn.query and turn.contexts == ()


def test_unverified_baseline_uses_same_selector_but_includes_unsupported_candidates():
    turn, context = fixture()
    supported, _, rejected = candidates(turn, context, "source_topic")
    unverified, _, _ = candidates(turn, context, "assistant_topic_unverified")
    assert "PhantomStore" in [e["text"] for e in rejected]
    assert "PhantomStore" not in [e["text"] for e in supported]
    phantom = next(e for e in unverified if e["text"] == "PhantomStore")
    client = SelectingClient({"status": "resolved", "entity_ids": [phantom["id"]], "user_turn_ids": []})
    result = TopicResolver(client, "assistant_topic_unverified").resolve(turn, context)
    assert "PhantomStore" in result["text"] and not result["source_qualification_required"]
    assert not result["selected_entities"][0]["supports"]


def test_later_source_does_not_retroactively_ground_an_earlier_assistant_name():
    turn, _ = fixture()
    later = make_evidence("PhantomStore is a database.", "late")
    entities, _, rejected = candidates(turn, PriorContext((PriorSource(later, (2,)),)), "source_topic")
    assert "PhantomStore" in [e["text"] for e in rejected]
    assert "PhantomStore" not in [e["text"] for e in entities]


def test_empty_or_current_turn_sources_cannot_ground_history_entities():
    turn, context = fixture()
    for prior in (PriorContext(), PriorContext((PriorSource(context.sources[0].block, (turn.turn,)),))):
        entities, users, rejected = candidates(turn, prior, "source_topic")
        assert entities == [] and users and any("Cloudant" in e["text"] for e in rejected)


def test_entity_occurrences_are_merged_and_model_gets_one_declared_source_window():
    turn, context = fixture()
    repeated = replace(turn, history=(turn.history[0], OnlineMessage("assistant", "Cloudant. Cloudant. Cloudant."),
                                     *turn.history[2:]))
    entities, _, _ = candidates(repeated, context, "source_topic")
    cloudant = next(e for e in entities if e["text"] == "Cloudant")
    assert len(cloudant["history_occurrences"]) == 4
    assert len([e for e in entities if e["text"] == "Cloudant"]) == 1
    client = SelectingClient({"status": "unchanged", "entity_ids": [], "user_turn_ids": []})
    trace = TopicResolver(client).resolve(repeated, context)
    model_entities = json.loads(client.inputs[0][0][-1]["content"])["entity_candidates"]
    compact = next(e for e in model_entities if e["text"] == "Cloudant")
    assert compact["source_example"]["text"] in context.sources[0].block.text
    assert "content_hash" not in json.dumps(compact) and "history_occurrences" not in compact
    assert trace["entity_candidates"][0]["supports"]


def test_declared_history_window_preserves_original_qa_and_can_use_older_visible_source():
    turn, context = fixture()
    client = SelectingClient({"status": "unchanged", "entity_ids": [], "user_turn_ids": []})
    trace = TopicResolver(client, history_turns=1).resolve(turn, context)
    body = json.loads(client.inputs[0][0][-1]["content"])
    assert len(body["actual_history"]) == 2 and body["first_history_turn"] == 2
    assert trace["omitted_earlier_history_messages"] == 2 and len(turn.history) == 4
    cloudant = next(e for e in trace["entity_candidates"] if e["text"] == "Cloudant")
    assert 1 in cloudant["supports"][0]["visible_turns"]
    assert [u["id"] for u in body["prior_user_questions"]] == ["u2"]


@pytest.mark.parametrize("first", [True, False])
def test_first_turn_and_explicit_topic_shift_do_not_call_selector(first):
    turn, context = fixture()
    turn = replace(turn, history=(), turn=1) if first else replace(turn, query="Switch the topic to DNS.")
    client = SelectingClient()
    result = TopicResolver(client).resolve(turn, context)
    assert result["text"] == turn.query and not result["model_called"] and client.inputs == []


@pytest.mark.parametrize("output", [
    {"status": "resolved", "entity_ids": ["unknown"], "user_turn_ids": []},
    {"status": "resolved", "entity_ids": [["nested"]], "user_turn_ids": []},
    {"status": "unchanged", "entity_ids": [], "user_turn_ids": ["u1"]},
    {"status": "resolved", "entity_ids": [], "user_turn_ids": [], "rewrite": "invented"},
    {"status": [], "entity_ids": [], "user_turn_ids": []}, "```json\n{}\n```",
])
def test_unknown_ids_or_invalid_model_structure_remain_raw_and_use_original_query(output):
    turn, context = fixture()
    result = TopicResolver(SelectingClient(output)).resolve(turn, context)
    assert result["status"] == "failed" and result["text"] == turn.query
    assert result["completion"]["raw_response"] and "parse_error" in result


@pytest.mark.parametrize("status,finish", [("completed", "length"), ("completed", None),
                                           ("failed", None), ("input_over_budget", None)])
def test_truncated_failed_and_overbudget_selector_do_not_change_query(status, finish):
    turn, context = fixture()
    result = TopicResolver(SelectingClient("{}", finish=finish, status=status)).resolve(turn, context)
    assert result["text"] == turn.query
    assert result["model_called"] == (status != "input_over_budget")


def test_projection_uses_only_committed_evidence_keys_and_keeps_old_sources_after_rebuild():
    seen, unseen = make_evidence("seen", "seen"), make_evidence("never displayed", "unseen")
    history = (TurnRecord(1, "q1", "a1", evidence_keys=(seen.key,)), TurnRecord(2, "q2", "a2"))
    snapshot = SessionSnapshot("s", version=2, raw_history=history, archive=(seen, unseen), execution=None)
    turn = OnlineTurn("s", "t", 3, "cloud", "q3", tuple(OnlineMessage(role, text) for r in history
                      for role, text in (("user", r.user_text), ("assistant", r.assistant_text))), (), "P3")
    projection = snapshot_context(turn, snapshot)
    assert [source.block.key for source in projection.sources] == [seen.key]
    assert projection.sources[0].visible_turns == (1,)
    assert native_context_provider(SimpleNamespace(snapshots={"s": snapshot}))(turn) == projection
    with pytest.raises(ValueError, match="differs"):
        snapshot_context(replace(turn, history=turn.history[:-2]), snapshot)
    missing = snapshot_context(turn, replace(snapshot, archive=(unseen,)))
    assert missing.missing_keys == (seen.key,) and missing.sources == ()


@pytest.mark.asyncio
async def test_controller_provider_reads_real_committed_state_without_creating_session(tmp_path):
    store = SessionStore(tmp_path / "state.sqlite3")
    service = Controller(store, MockEngine(response_text="Cloudant stores JSON."),
                         knobs=PlannerKnobs(strategy="always_rebuild"))
    provider = controller_context_provider(store, "run")
    first = OnlineTurn("s", "t1", 1, "cloud", "q1", (), (), "P3")
    assert provider(first) == PriorContext()
    assert store.db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0
    source = make_evidence("Cloudant stores JSON.", "cloudant")
    answer = await service.serve_turn("run:s", "q1", (source,))
    next_turn = replace(first, task_id="t2", turn=2, query="And attachments?", history=(
        OnlineMessage("user", "q1"), OnlineMessage("assistant", answer.assistant_text)))
    before = store.get_snapshot("run:s")
    context = provider(next_turn)
    assert context.sources == (PriorSource(source, (1,)),)
    assert store.get_snapshot("run:s") == before
    await service.aclose()
