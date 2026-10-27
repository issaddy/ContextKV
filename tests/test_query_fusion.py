"""Fixed synthetic scores check fusion contracts, never retrieval quality."""
from dataclasses import asdict, replace
import inspect

import numpy as np
import pytest

from adapters.hybrid_retrieval import reciprocal_rank_fusion, search_current_topic, search_variants
from adapters.mtrag import OnlineContext
from bench.retrieval_eval import RetrievalQuery, evaluate_queries


class Encoder:
    def __init__(self):
        self.calls = []

    def encode(self, queries):
        self.calls.append(queries)


class Sparse:
    corpus_snapshot_id = "synthetic-snapshot"
    manifest = {"corpus_sha256": "synthetic-shared-corpus"}

    def __init__(self, contexts):
        self.contexts, self.calls = contexts, []

    def retrieve(self, queries, *, top_k):
        self.calls.append((queries, top_k))
        # Distinct raw/topic pools force previously unscored documents into RRF.
        return self.contexts[:top_k] if queries[0] == "raw" else self.contexts[5:5+top_k]


class Dense:
    manifest = Sparse.manifest

    def __init__(self, contexts):
        self.by_id = {c.document_id: c for c in contexts}
        self.contexts, self.calls = contexts, []

    def search(self, query, encoder, *, top_k):
        return self.search_many([query], encoder, top_k=top_k)[0]

    def search_many(self, queries, encoder, *, top_k):
        encoder.encode(list(queries))
        self.calls.extend((query, top_k) for query in queries)
        return [((self.contexts[:5] if query == "raw" else self.contexts[5:])[::-1][:top_k])
                for query in queries]


class Reranker:
    def __init__(self):
        self.calls = []

    def score(self, query, contexts):
        self.calls.append((query, contexts))
        return np.arange(len(contexts))


def fixture():
    contexts = tuple(OnlineContext(str(i), f"Original body {i}\nPreserve text.",
                                  f"https://example.test/{i}", f"Title {i}") for i in range(10))
    return Sparse(contexts), Dense(contexts), Encoder(), Reranker()


def test_same_query_exact_ranking_and_single_model_call_path():
    expected = search_variants("raw", *fixture(), candidate_k=5, rerank_k=5)
    sparse, dense, encoder, reranker = fixture()
    actual = search_current_topic("raw", "raw", sparse, dense, encoder, reranker,
                                  candidate_k=5, rerank_k=5)
    assert set(actual) == {"hybrid", "hybrid_rerank"}
    for stage in actual:
        assert actual[stage]["contexts"] == expected[stage]["contexts"]
    assert sparse.calls == [(["raw"], 5)] and dense.calls == [("raw", 5)]
    assert encoder.calls == [["raw"]] and len(reranker.calls) == 1
    assert actual["hybrid_rerank"]["rerank_candidate_ids"] == expected["hybrid_rerank"]["rerank_candidate_ids"]
    trace = actual["hybrid_rerank"]["query_fusion_trace"]
    assert trace["duplicate_query_skipped"] and trace["retrieval_query_count"] == 1
    assert trace["additional_current_retrieval_ms"] == trace["cross_query_fusion_ms"] == 0
    assert trace["rerank_pairs"] == 5


def test_distinct_queries_one_four_list_rrf_and_all_new_pairs_use_topic():
    sparse, dense, encoder, reranker = fixture()
    actual = search_current_topic("raw", "topic", sparse, dense, encoder, reranker,
                                  candidate_k=5, rerank_k=5)
    raw_ids, topic_ids = list("01234"), list("56789")
    expected = reciprocal_rank_fusion([raw_ids, raw_ids[::-1], topic_ids, topic_ids[::-1]])
    assert [(c.document_id, c.score) for c in actual["hybrid"]["contexts"]] == expected
    pool = actual["hybrid"]["contexts"][:5]
    assert len(pool) == len({c.document_id for c in pool}) == 5
    assert set(c.document_id for c in pool).intersection(raw_ids)
    assert set(c.document_id for c in pool).intersection(topic_ids)
    assert encoder.calls == [["raw", "topic"]]
    assert sparse.calls == [(["raw"], 5), (["topic"], 5)]
    assert reranker.calls == [("topic", pool)]  # No raw logits or unscored new documents.
    assert actual["hybrid_rerank"]["contexts"] == tuple(replace(c, score=float(i)) for i, c in reversed(list(enumerate(pool))))
    assert all(replace(c, score=None) == dense.by_id[c.document_id] for c in pool)
    trace = actual["hybrid_rerank"]["query_fusion_trace"]
    assert not trace["duplicate_query_skipped"] and trace["retrieval_query_count"] == 2
    assert trace["rerank_k_total"] == trace["rerank_pairs"] == 5
    assert trace["fused_union_candidates"] == 10 and trace["rerank_query"] == "topic"
    assert trace["source_queries"][0]["bm25_candidate_ids"] == raw_ids
    assert trace["source_queries"][1]["dense_candidate_ids"] == topic_ids[::-1]
    assert trace["additional_current_retrieval_ms"] is None and trace["rerank_ms"] >= 0
    assert not {"qrels", "labels", "targets", "answers"}.intersection(inspect.signature(search_current_topic).parameters)


def test_hybrid_only_does_not_score_and_requires_dense():
    sparse, dense, encoder, reranker = fixture()
    stages = search_current_topic("raw", "topic", sparse, dense, encoder, candidate_k=5, rerank_k=5)
    assert set(stages) == {"hybrid"} and not reranker.calls
    assert stages["hybrid"]["query_fusion_trace"]["rerank_pairs"] == 0
    with pytest.raises(ValueError, match="dense index"):
        search_current_topic("raw", "topic", sparse, None, None)


@pytest.mark.parametrize("topic", ["raw", "topic"])
def test_explicit_rerank_query_changes_only_scoring_not_raw_or_topic_recall(topic):
    original = search_current_topic("raw", topic, *fixture(), candidate_k=5, rerank_k=5)
    sparse, dense, encoder, reranker = fixture()
    actual = search_current_topic("raw", topic, sparse, dense, encoder, reranker,
        candidate_k=5, rerank_k=5, rerank_query="short scoring query")
    assert actual["hybrid"]["contexts"] == original["hybrid"]["contexts"]
    assert actual["hybrid_rerank"]["rerank_candidate_ids"] == original["hybrid_rerank"]["rerank_candidate_ids"]
    assert encoder.calls == ([["raw"]] if topic == "raw" else [["raw", "topic"]])
    assert len(reranker.calls) == 1 and reranker.calls[0][0] == "short scoring query"
    assert len(reranker.calls[0][1]) == 5
    trace = actual["hybrid_rerank"]["query_fusion_trace"]
    assert trace["topic_query"] == topic and trace["rerank_query"] == "short scoring query"
    assert actual["hybrid_rerank"]["rerank_query"] == "short scoring query"


def test_explicit_single_route_rerank_query_preserves_original_union():
    sparse, dense, encoder, reranker = fixture()
    actual = search_variants("topic", sparse, dense, encoder, reranker,
        candidate_k=5, rerank_k=5, rerank_query="raw")
    assert sparse.calls == [(["topic"], 5)] and encoder.calls == [["topic"]]
    assert reranker.calls == [("raw", actual["hybrid"]["contexts"][:5])]
    assert actual["hybrid_rerank"]["rerank_query"] == "raw"


def test_offline_dual_stages_share_trace_and_keep_original_question():
    sparse, dense, encoder, reranker = fixture()
    query = RetrievalQuery("unrelated-task-id", "arbitrary", 2,
                           ("How do I use Acme Store?", "One last thing, where is the python SDK?"))
    before = asdict(query)
    rows = list(evaluate_queries([query], sparse, dense, encoder, reranker, modes=("user_topic",),
                                query_fusion="current_topic_rrf", candidate_k=5, rerank_k=5))
    assert asdict(query) == before
    assert {r["method"] for r in rows} == {"user_topic+current_topic_rrf/hybrid", "user_topic+current_topic_rrf/hybrid_rerank"}
    assert all(r["query_text"].endswith("User topic context: Acme Store") for r in rows)
    assert rows[-1]["query_fusion_trace"]["current_query"] == query.user_questions[-1]
    assert rows[-1]["query_fusion_trace"]["rerank_query"] == rows[-1]["query_text"]
    assert rows[-1]["rerank_candidate_ids"] == rows[0]["candidate_ids"][:5]
    assert len(reranker.calls) == 1 and len(encoder.calls) == 1
    assert rows[-1]["elapsed_ms"] >= rows[0]["elapsed_ms"] >= rows[0]["query_view_ms"]
    for modes in (("current",), ("user_topic", "current")):
        with pytest.raises(ValueError, match="only user_topic"):
            list(evaluate_queries([query], sparse, dense, encoder, modes=modes, query_fusion="current_topic_rrf"))


def test_dialogue_views_preserve_original_need_and_bound_the_derived_family_vote():
    from adapters.hybrid_retrieval import search_dialogue_views
    sparse, dense, encoder, reranker = fixture()
    pairs = []
    def score_many(queries, contexts):
        pairs.append((tuple(queries), tuple(contexts)))
        # The literal question and each derived view favor opposite documents.
        return [[-int(c.document_id) if q == "raw" else int(c.document_id) for c in contexts]
                for q in queries]
    reranker.score_many = score_many
    ranked, trace = search_dialogue_views("raw", "topic", ("need one", "need two", "topic"),
        sparse, dense, encoder, reranker, candidate_k=5, rerank_k=10)
    assert encoder.calls == [["raw", "topic", "need one", "need two"]]
    assert len(pairs) == 1 and len(pairs[0][1]) == 10
    assert trace["query_weights"] == [1, 1/3, 1/3, 1/3]
    assert trace["rerank_pairs"] == 40 and trace["encoder_batches"] == 1
    assert {ranked[0].document_id, ranked[1].document_id} == {"0", "9"}
    assert all(replace(c, score=None) == dense.by_id[c.document_id] for c in ranked)
    # More identical derived views do not outvote the user's original question.
    repeated, repeated_trace = search_dialogue_views("raw", "topic", ("topic", "topic"),
        *fixture(), candidate_k=5, rerank_k=10)
    assert repeated_trace["rerank_queries"] == ["raw", "topic"]
    assert len(repeated) == 10


def test_dialogue_identical_view_deduplicates_work_and_preserves_single_view_order():
    from adapters.hybrid_retrieval import search_dialogue_views
    sparse, dense, encoder, reranker = fixture()
    ranked, trace = search_dialogue_views("raw", "raw", (), sparse, dense, encoder, reranker,
                                         candidate_k=5, rerank_k=5)
    expected = search_variants("raw", *fixture(), candidate_k=5, rerank_k=5)["hybrid_rerank"]["contexts"]
    assert [c.document_id for c in ranked] == [c.document_id for c in expected]
    assert encoder.calls == [["raw"]] and len(reranker.calls) == 1
    assert trace["query_weights"] == [1] and trace["rerank_pairs"] == 5


def test_dialogue_route_uses_literal_history_without_changing_committed_intent():
    from types import SimpleNamespace
    from adapters.hybrid_retrieval import HybridRetriever
    from adapters.mtrag import OnlineMessage, OnlineTurn
    from adapters.query_resolution import PriorContext
    from controller.schema import QueryContext, TaskFrame
    history = (OnlineMessage("user", "Define the API."), OnlineMessage("assistant", "Actual answer."))
    turn = OnlineTurn("conversation", "task", 2, "corpus", "How do I create it?", history, (), "P3")
    intent = QueryContext("definition of API", "followup", (1,), TaskFrame("API", "", "definition"))
    resolver = SimpleNamespace(resolve=lambda *args: {"text": intent.query, "context": asdict(intent)})
    sparse, dense, encoder, reranker = fixture()
    retriever = HybridRetriever(sparse, dense, encoder, reranker, query_mode="conversational",
                               query_fusion="dialogue_rrf", resolver=resolver, candidate_k=5, rerank_k=5)
    result = retriever.retrieve_turn(turn, prior_context=PriorContext())
    assert encoder.calls == [["Define the API.\nHow do I create it?", "definition of API"]]
    assert result.query == turn.query and result.history == history and result.query_context == intent
    trace = retriever.query_view_traces[-1]
    assert trace["rerank_query"] is None and trace["rerank_query_mode"] == "dialogue_views"
    assert trace["query_fusion_trace"]["rerank_queries"] == encoder.calls[0]
    assert len(result.contexts) == 5
