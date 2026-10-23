"""Explicit user-only correction routing, independent of answers and labels."""
from copy import deepcopy
import json

import pytest

from adapters.hybrid_retrieval import query_text
from adapters.query_views import correction_cue, correction_query_mode, named_user_spans, user_topic_view
from adapters.retriever_factory import load_retriever_config
from bench.query_selector import select_rows


@pytest.mark.parametrize("question", ["Not documents, I was asking for attachments?", " No, the warning.",
                                      "I meant the install warning.", "Actually I mean attachments.",
                                      "I WAS REFERRING to the previous limit."])
def test_explicit_correction_cues_route_to_history(question):
    assert correction_query_mode(question) == "last3"
    assert correction_cue(question)


@pytest.mark.parametrize("question", ["Does NoSQL support attachments?", "Notice the error.",
                                      "How can I use Cloudant?", "What if this is not supported?"])
def test_plain_questions_and_substrings_stay_current(question):
    assert correction_query_mode(question) == "current"
    assert correction_cue(question) is None


def test_current_cue_controls_view_without_assistant_or_earlier_cues():
    users = ["I meant older context", "What about images?", "Is there a size limit?", "Not documents; attachments?"]
    assert query_text(users, "correction_aware") == "\n".join(users[-3:])
    assert query_text(users[:-1] + ["What is Cloudant?"], "correction_aware") == "What is Cloudant?"
    assert query_text(["No, attachments."], "correction_aware") == "No, attachments."
    assert query_text(users, "correction_aware", generated_history=({"speaker": "assistant", "text": "No, different answer"},)) == "\n".join(users[-3:])


def examples():
    tasks = [{"task_id": "arbitrary-id", "conversation_id": "s", "turn": 2,
              "user_questions": ["File size limit?", "I meant attachments."]}]
    rows = [{"task_id": tasks[0]["task_id"], "conversation_id": "s", "turn": 2,
             "stage": "hybrid_rerank", "query_mode": mode, "method": mode + "/hybrid_rerank",
             "query_text": query_text(tasks[0]["user_questions"], mode), "status": "complete",
             "candidate_ids": [mode + "-document"], "contexts": [], "elapsed_ms": 1}
            for mode in ("current", "last3")]
    return rows, tasks


def test_offline_selection_preserves_frozen_ranking_and_ignores_quality_metadata():
    rows, tasks = examples()
    before = deepcopy(rows)
    selected, routes = select_rows(rows, tasks)
    assert selected[0]["candidate_ids"] == ["last3-document"]
    assert selected[0]["source_method"] == "last3/hybrid_rerank"
    assert routes[0]["cue"] == "I meant" and rows == before
    for row in rows:
        row.update(task_id="different-id", scores={"recall": 0 if row["query_mode"] == "last3" else 1})
    tasks[0]["task_id"] = "different-id"
    again, _ = select_rows(rows, tasks)
    assert again[0]["candidate_ids"] == selected[0]["candidate_ids"]


def test_offline_replay_rejects_missing_or_mismatched_saved_query():
    rows, tasks = examples()
    with pytest.raises(ValueError, match="complete current and last3"):
        select_rows(rows[:1], tasks)
    rows[1]["query_text"] = "future question must not be replayed"
    with pytest.raises(ValueError, match="query text differs"):
        select_rows(rows, tasks)


def test_factory_accepts_optional_mode_without_changing_default(tmp_path):
    config = {"sparse_index": "sparse", "dense_index": "dense", "encoder_path": "encoder",
              "reranker_path": "reranker"}
    path = tmp_path / "retriever.json"
    path.write_text(json.dumps(config))
    assert load_retriever_config(path).query_mode == "current"
    path.write_text(json.dumps(config | {"query_mode": "correction_aware"}))
    assert load_retriever_config(path).query_mode == "correction_aware"
    path.write_text(json.dumps(config | {"query_mode": "user_topic"}))
    assert load_retriever_config(path).query_mode == "user_topic"


def test_user_topic_carries_literal_older_user_anchor_without_assistant_facts():
    users = ["How do I start with Acme Store?", "What are differences between plans?",
             "Is the free plan enough?", "One last thing, where is the python SDK?"]
    original = deepcopy(users)
    view = user_topic_view(users)
    assert users == original
    assert view["text"] == users[-1]+"\nUser topic context: Acme Store"
    assert view["source_user_turns"] == [1, 4]
    span = view["selected_history_spans"][0]
    assert users[span["user_turn"]-1][span["start"]:span["end"]] == "Acme Store"
    assert query_text(users, "user_topic", generated_history=({"speaker": "assistant", "text": "Invented Other Database"},)) == view["text"]
    assert "Acme Store" not in query_text(users, "last3")
    assert "Acme Store" in query_text(users, "all_user")


@pytest.mark.parametrize("current, reason", [
    ("What about Rival Database instead?", "current_has_named_wording"),
    ("How does Carthage work?", "current_has_named_wording"),
    ("Different topic: how does it work?", "explicit_topic_shift"),
    ("Now let's discuss how it works.", "explicit_topic_shift"),
    ("Not Acme Store, what about Rival Database?", "current_has_named_wording"),
    ("How can I secure my phone connection?", "no_followup_cue"),
    ("How can I use the Python SDK?", "current_has_named_wording"),
    ("How can I do this without the previous product?", "unresolved_current_negation"),
])
def test_current_wording_and_topic_switches_do_not_receive_old_topic(current, reason):
    view = user_topic_view(["I use Acme Store.", current])
    assert view["text"] == current
    assert view["reason"] == reason
    assert view["selected_history_spans"] == []


def test_correction_reuses_existing_last_three_user_view():
    users = ["Earlier unused question", "How do records work?", "Is there a size limit?", "Not records, I meant attachments."]
    view = user_topic_view(users)
    assert view["text"] == query_text(users, "last3")
    assert view["source_user_turns"] == [2, 3, 4]
    assert view["reason"] == "explicit_correction_reuses_last3"


@pytest.mark.parametrize("boundary", ["I no longer use Acme Store.", "Let's change the topic to cooking."])
def test_past_explicit_boundaries_stop_carrying_older_topic(boundary):
    view = user_topic_view(["How does Acme Store work?", boundary, "How do I use it?"])
    assert view["route"] == "current"
    assert view["reason"] == "history_topic_or_negation_boundary"
    assert view["boundary_user_turn"] == 2


def test_multiple_literal_names_preserved_without_invented_disambiguation():
    users = ["Compare Acme Store and Rival Database.", "How do they differ?"]
    view = user_topic_view(users)
    assert [s["text"] for s in view["selected_history_spans"]] == ["Acme Store", "Rival Database"]
    assert view["text"].endswith("Acme Store; Rival Database")


def test_lexical_cues_preserve_case_typos_and_do_not_claim_ner():
    assert named_user_spans("How does AtlasCloud v2 work?")[0]["text"] == "AtlasCloud"
    assert named_user_spans("How does Acme Clod work?")[0]["text"] == "Acme Clod"
    assert named_user_spans("how does acme store work?") == []
    assert named_user_spans("Where are the SDK and API?") == []
    assert user_topic_view(["how does acme store work?", "How does it work?"])["reason"] == "no_named_user_history"
    with pytest.raises(ValueError, match="current user question"):
        user_topic_view([])
