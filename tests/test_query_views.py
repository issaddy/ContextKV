"""Explicit user-only correction routing, independent of answers and labels."""
from copy import deepcopy
import json

import pytest

from adapters.hybrid_retrieval import query_text
from adapters.query_views import correction_cue, correction_query_mode
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
