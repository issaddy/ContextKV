from dataclasses import asdict, replace
import json

import pytest

from adapters.conversation import ConversationResolver
from adapters.mtrag import OnlineMessage, OnlineTurn
from adapters.query_resolution import PriorContext, PriorSource, snapshot_context
from adapters.query_selector import CompletionResult
from controller.candidates import build_candidates
from controller.evidence import make_evidence
from controller.mock_engine import MockEngine
from controller.planner import PlannerKnobs
from controller.schema import QueryContext, SessionSnapshot
from controller.service import Controller, EvidenceSelection
from controller.session_store import SessionStore
from test_candidates import committed


class Client:
    manifest = {"model": "test"}
    def __init__(self, value, *, finish="stop"):
        self.value, self.finish, self.inputs = value, finish, []
    def complete(self, messages, **kwargs):
        self.inputs.append((messages, kwargs))
        return CompletionResult("completed", json.dumps(self.value), self.finish, 100)


def turn(current, questions=("How does Atlas Cloud store documents?",)):
    history = tuple(m for q in questions for m in (OnlineMessage("user", q),
                      OnlineMessage("assistant", "PhantomStore allows 999 invented items.")))
    return OnlineTurn("s", "opaque-id", len(questions)+1, "corpus", current, history, (), "P3")


def test_clear_topic_and_definition_skip_rewriting_and_discard_last_three_bag():
    client = Client(None)
    resolver = ConversationResolver(client)
    for query in ("Use IBM Cloudant Query", "What is a data lake?"):
        view = resolver.resolve(turn(query), PriorContext())
        assert view["text"] == query and not view["model_called"]
        assert view["context"] == asdict(QueryContext(query, "standalone"))
    assert resolver.resolve(turn("First question", ()), PriorContext())["context"] is None
    assert not client.inputs


@pytest.mark.parametrize("query,relation,resolved", [
    ("I meant the warning when installing it with CocoaPods.", "correction", "Atlas Cloud SDK CocoaPods installation warning"),
    ("换到地区 B 呢？", "condition_change", "Atlas Cloud v2 在地区 B 的申请条件是什么？"),
    ("回到最初那个项目，条件是什么？", "revisit", "Atlas Cloud v2 在地区 A 的申请条件是什么？"),
])
def test_contextual_rewrite_preserves_current_question_and_has_bounded_history_provenance(query, relation, resolved):
    client = Client({"query": resolved, "relation": relation, "history_turns": [1]})
    original = turn(query, ("Atlas Cloud v2 在地区 A 如何申请？", "What about support?"))
    context = PriorContext(interpretations=(QueryContext("first accepted interpretation", "standalone"), None))
    view = ConversationResolver(client).resolve(original, context)
    assert view["text"] == resolved and original.query == query
    assert view["context"]["history_turns"] == (1,)
    messages, options = client.inputs[0]
    body = json.loads(messages[-1]["content"])
    assert body["current_question"] == query and len(body["prior_user_turns"]) == 2
    assert body["prior_user_turns"][0]["prior_interpretation"] == "first accepted interpretation"
    assert "PhantomStore" not in messages[-1]["content"] and "999" not in messages[-1]["content"]
    assert "opaque-id" not in messages[-1]["content"]
    assert options["response_format"]["json_schema"]["schema"]["properties"]["history_turns"]["items"]["enum"] == [1, 2]


def test_qualified_assistant_name_is_context_not_an_answer_fact():
    original = turn("What about its limits?")
    history = (original.history[0], OnlineMessage("assistant", "AtlasCloud holds 999 units. PhantomStore is better."))
    source = make_evidence("AtlasCloud is a document database.", "a")
    client = Client({"query": "AtlasCloud document limits?", "relation": "followup", "history_turns": [1]})
    view = ConversationResolver(client).resolve(replace(original, history=history), PriorContext((PriorSource(source, (1,)),)))
    body = json.loads(client.inputs[0][0][-1]["content"])
    assert [e["text"] for e in body["qualified_topic_names"]] == ["AtlasCloud"]
    assert "999" not in json.dumps(body) and "PhantomStore" not in json.dumps(body)
    assert "PhantomStore" in view["excluded_unqualified_names"]


@pytest.mark.parametrize("value,finish", [
    ({"query":"answer", "relation":"followup", "history_turns":[2]}, "stop"),
    ({"query":"answer", "relation":"followup", "history_turns":[]}, "stop"),
    ({"query":"answer", "relation":"unknown", "history_turns":[1]}, "stop"),
    ({"query":"answer", "relation":"standalone", "history_turns":[1]}, "stop"),
    ({"query":"answer", "relation":"followup", "history_turns":[1]}, "length"),
])
def test_invalid_or_truncated_interpretation_uses_current_query_once(value, finish):
    original = turn("What about its limits?")
    client = Client(value, finish=finish)
    view = ConversationResolver(client).resolve(original, PriorContext())
    assert view["status"] == "failed" and view["text"] == original.query and view["context"] is None
    assert len(client.inputs) == 1


@pytest.mark.parametrize("relation", ("condition_change", "correction", "standalone", "followup"))
def test_intent_label_does_not_disable_cache_reuse_but_current_source_scope_excludes_old_condition(relation):
    shared = make_evidence("Atlas project common rules.", "base")
    a = make_evidence("Region A condition: students may apply.", "a")
    b = make_evidence("Region B condition: workers may apply.", "b")
    snap = SessionSnapshot("s")
    snap = committed(build_candidates(snap, "Region A conditions?", (shared, a))[0], snap,
                     "Region A conditions?", (shared, a))
    intent = QueryContext("Atlas project Region B conditions?", relation, (1,))
    append, rebuild = build_candidates(snap, "What about B?", (shared, b), query_context=intent)
    assert append.feasible and append.required_blocks == rebuild.required_blocks
    assert rebuild.required_blocks == {shared.key, b.key}
    assert snap.raw_history[0].assistant_text in [m.content for m in rebuild.messages]
    assert rebuild.messages[-1].content.endswith("\nQuestion:\nWhat about B?\n")
    assert intent.query in rebuild.messages[-1].content
    assert a.key not in append.required_blocks and a.key in append.visible_blocks
    followup = QueryContext("Atlas project Region A conditions in detail?", "followup", (1,))
    assert build_candidates(snap, "More details?", (shared, a), query_context=followup)[0].feasible


def test_explicit_revision_replaces_retrieval_hit_without_mutating_archive():
    v1 = make_evidence("The original v1 limit is 10.", "Atlas", version="v1")
    v2 = make_evidence("The v2 limit is 20.", "Atlas", version="v2")
    snap = SessionSnapshot("s", archive=(v1, v2))
    for hits in ((v2,), (v1, v2)):
        plan = build_candidates(snap, "Use Atlas v1", hits)[0]
        assert plan.required_blocks == {v1.key}
        assert v2.text not in plan.messages[-1].content
    compare = build_candidates(snap, "Compare Atlas v1 and v2", (v2,))[0]
    assert compare.required_blocks == {v1.key, v2.key}
    assert snap.archive == (v1, v2)


@pytest.mark.parametrize("revision", ("v1.1", "v1-rc2", "v1+build1"))
def test_revision_prefix_does_not_select_the_wrong_archived_version(revision):
    old = make_evidence("Original revision.", "Atlas", version="v1")
    new = make_evidence("Requested revision.", "Atlas", version=revision)
    snap = SessionSnapshot("s", archive=(old, new))
    candidate = build_candidates(snap, f"Use Atlas {revision}.", (old, new))[0]
    assert candidate.required_blocks == {new.key}


def test_selected_revisit_restores_older_source_beyond_recent_history_window():
    old = make_evidence("Atlas eligibility condition: students may apply.", "atlas")
    snap = SessionSnapshot("s")
    snap = committed(build_candidates(snap, "Atlas conditions?", (old,))[0], snap,
                     "Atlas conditions?", (old,))
    for index in range(4):
        latest = make_evidence(f"Different project {index}, unrelated condition.", str(index))
        query = f"Describe project {index}."
        snap = committed(build_candidates(snap, query, (latest,))[-1], snap, query, (latest,))
    context = QueryContext("Atlas eligibility conditions?", "revisit", (1,))
    for candidate in build_candidates(snap, "Back to the first one, its conditions?", (latest,), query_context=context):
        assert old.key in candidate.required_blocks <= candidate.visible_blocks
        assert old.text in candidate.messages[-1].content


def test_multiple_selected_antecedents_do_not_restore_the_last_unrelated_source():
    snap = SessionSnapshot("s")
    for name in ("Atlas", "Nova"):
        block = make_evidence(f"{name} application material.", name)
        query = f"{name} requirements?"
        snap = committed(build_candidates(snap, query, (block,))[-1], snap, query, (block,))
    current = make_evidence("Atlas revised material.", "new")
    intent = QueryContext("Atlas application material?", "revisit", (1, 2))
    for plan in build_candidates(snap, "Back to the first one?", (current,), query_context=intent):
        assert plan.required_blocks == {current.key}


@pytest.mark.asyncio
@pytest.mark.parametrize("lost_response", (False, True))
async def test_interpretation_commits_with_answer_and_idempotent_replay_does_not_rewrite(tmp_path, lost_response):
    from controller.engine_client import UnknownOutcomeError
    store, engine = SessionStore(tmp_path/'state.sqlite3'), MockEngine()
    service = Controller(store, engine, knobs=PlannerKnobs(strategy="always_append"))
    evidence = make_evidence("A condition with its full original source.", "a")
    calls = []
    intent = QueryContext("Atlas v2 region B condition?", "condition_change", (1,))
    async def provider(snapshot):
        calls.append(snapshot.version)
        return EvidenceSelection((evidence,), (evidence.key,), intent)
    try:
        await service.serve_turn("s", "Original user question", (evidence,), request_id="one")
        if lost_response:
            engine.drop_next_generate_response = True
            with pytest.raises(UnknownOutcomeError):
                await service.serve_turn("s", "And region B?", request_id="two", evidence_provider=provider)
            assert store.get_snapshot("s").raw_history[-1].query_context is None
        answer = await service.serve_turn("s", "And region B?", request_id="two", evidence_provider=provider)
        attempt = store.find_request("s", "two")
        snapshot = store.get_snapshot("s")
        assert attempt.query_context == snapshot.raw_history[-1].query_context == intent
        assert snapshot.raw_history[-1].user_text == "And region B?"
        again = await service.serve_turn("s", "And region B?", request_id="two", evidence_provider=provider)
        assert again.output_ids == answer.output_ids and again.idempotent_replay and calls == [1]
        online = turn("Next", tuple(t.user_text for t in snapshot.raw_history))
        online = replace(online, history=tuple(m for t in snapshot.raw_history for m in (
            OnlineMessage("user", t.user_text), OnlineMessage("assistant", t.assistant_text))))
        assert snapshot_context(online, snapshot).interpretations[-1] == intent
    finally:
        await service.aclose()
        store.close()
