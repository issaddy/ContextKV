from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from adapters.hybrid_retrieval import DenseIndex, query_text, reciprocal_rank_fusion, search_variants
from adapters.mtrag import OnlineContext, json_hash, load_rows
from adapters.retrieval import BM25SRetriever, load_beir_corpus
from bench.retrieval_eval import evaluate_queries, prepare_queries, summarize


FIXTURE = Path(__file__).parent/"fixtures/mtrag_small.jsonl"


class Encoder:
    manifest = {"model_id": "fake", "revision": "fixture", "max_length": 32, "pooling": "fixture"}
    token_limit_hits = 0

    def __init__(self):
        self.inputs = []

    def encode(self, texts):
        self.inputs.extend(texts)
        # These scores are test fixtures, not measurements of a pretrained model.
        return np.array([[1., 0.] if "cat" in text else [0., 1.] for text in texts], dtype=np.float32)


class Sparse:
    def __init__(self, contexts):
        self.contexts = contexts
        self.manifest = {"corpus_sha256": json_hash([asdict(c) for c in contexts])}
        self.calls = []

    def retrieve(self, questions, *, top_k=5):
        self.calls.append((questions, top_k))
        return tuple(self.contexts[::-1][:top_k])


class Reranker:
    def __init__(self):
        self.calls = []

    def score(self, query, contexts):
        self.calls.append((query, contexts))
        return np.arange(len(contexts))


def contexts():
    return tuple(OnlineContext(str(i), f"Original {'cat' if i % 2 else 'dog'} text {i}.\nDo not rewrite.",
                               "https://example.test", f"Title {i}") for i in range(8))


def test_rrf_duplicates_ties_and_formula():
    fused = reciprocal_rank_fusion([["b", "b", "a"], ["a", "b"]])
    assert [pid for pid, _ in fused] == ["a", "b"]
    assert fused[0][1] == pytest.approx(1/61+1/62)
    with pytest.raises(ValueError, match="nonnegative"):
        reciprocal_rank_fusion([], k=-1)


def test_query_views_and_explicit_generated_history():
    questions = ["first", "second", "third", "current"]
    assert query_text(questions, "current") == "current"
    assert query_text(questions, "last3") == "second\nthird\ncurrent"
    assert query_text(questions, "all_user") == "first\nsecond\nthird\ncurrent"
    with pytest.raises(ValueError, match="saved actual"):
        query_text(questions, "generated_history")
    text = query_text(questions, "generated_history", generated_history=[{"speaker": "assistant", "text": "actual"}])
    assert text.startswith("Current question: current") and "actual" in text


def test_dense_persistence_exact_scores_and_original_text(tmp_path):
    original = contexts()
    encoder = Encoder()
    index = DenseIndex.build(reversed(original), encoder, tmp_path/"dense", corpus_snapshot_id="fixture", batch_size=3)
    restored = DenseIndex.load(tmp_path/"dense")
    assert restored.contexts == original
    assert restored.manifest["corpus_sha256"] == Sparse(original).manifest["corpus_sha256"]
    assert restored.manifest["dimensions"] == 2
    assert np.array_equal(index.vectors, restored.vectors)
    ranked = restored.search("cat query", encoder, top_k=5)
    assert [c.document_id for c in ranked] == ["1", "3", "5", "7", "0"]
    assert [c.score for c in ranked] == [1., 1., 1., 1., 0.]
    assert all(c.text == original[int(c.document_id)].text for c in ranked)
    encoder.manifest = {**encoder.manifest, "revision": "changed"}
    with pytest.raises(ValueError, match="query encoder"):
        restored.search("cat", encoder)


def test_hybrid_rerank_uses_original_union_only(tmp_path):
    original = contexts()
    encoder, reranker, sparse = Encoder(), Reranker(), Sparse(original)
    dense = DenseIndex.build(original, encoder, tmp_path/"dense", corpus_snapshot_id="fixture")
    stages = search_variants("cat", sparse, dense, encoder, reranker, candidate_k=5, rerank_k=5)
    assert set(stages) == {"bm25", "dense", "hybrid", "hybrid_rerank"}
    assert sparse.calls == [(["cat"], 5)]
    candidates = stages["hybrid"]["contexts"][:5]
    assert reranker.calls[0] == ("cat", candidates)
    assert [c.document_id for c in stages["hybrid_rerank"]["contexts"]] == [c.document_id for c in candidates[::-1]]
    assert all(c.text == original[int(c.document_id)].text for c in stages["hybrid_rerank"]["contexts"])
    sparse.manifest["corpus_sha256"] = "different"
    with pytest.raises(ValueError, match="same original corpus"):
        search_variants("cat", sparse, dense, encoder)


def test_optional_bm25_budget_preserves_default_baseline():
    retriever = BM25SRetriever(load_beir_corpus(FIXTURE.with_name("corpus_small.jsonl")), corpus_snapshot_id="fixture")
    default = retriever.retrieve(["passage"])
    more = retriever.retrieve(["passage"], top_k=7)
    assert len(default) == 5 and len(more) == 6
    assert default == more[:5]
    with pytest.raises(ValueError, match="positive"):
        retriever.retrieve(["passage"], top_k=0)


def test_query_boundary_never_copies_labels_or_official_assistant():
    rows = load_rows(FIXTURE)
    queries = prepare_queries(rows, {"dev": ["session-a"]})
    assert len(queries) == 3
    serialized = json.dumps([asdict(q) for q in queries])
    assert "GOLD" not in serialized and "LABEL" not in serialized
    assert queries[1].user_questions == ("session-a question 1", "session-a question 2")
    assert queries[0].user_questions == ("session-a question 1",)
    assert all(q.generated_history is None for q in queries)


def test_explicit_history_checks_protocol_current_query_and_prior_turns():
    rows = load_rows(FIXTURE)
    queries = prepare_queries(rows, {"dev": ["session-a"]})
    audits = [{"task_id": q.task_id, "conversation_id": q.conversation_id, "protocol": "P2",
        "query": q.user_questions[-1], "actual_history": [message for question in q.user_questions[:-1]
        for message in ({"speaker": "user", "text": question}, {"speaker": "assistant", "text": "actual generated"})]}
        for q in queries]
    result = prepare_queries(rows, {"dev": ["session-a"]}, history_audits=audits)
    assert result[-1].generated_history[-1]["text"] == "actual generated"
    audits[-1]["actual_history"].append({"speaker": "assistant", "text": "future"})
    with pytest.raises(ValueError, match="prior complete"):
        prepare_queries(rows, {"dev": ["session-a"]}, history_audits=audits)
    audits[-1]["protocol"] = "P1"
    with pytest.raises(ValueError, match="actual P2/P3"):
        prepare_queries(rows, {"dev": ["session-a"]}, history_audits=audits)


def test_offline_analysis_has_missing_coverage_and_session_metrics():
    queries = prepare_queries(load_rows(FIXTURE), {"dev": ["session-a"]})
    sparse = Sparse(contexts())
    rows = list(evaluate_queries(queries, sparse, modes=("current", "last3"), candidate_k=7))
    assert len(rows) == 6
    assert all(len(r["contexts"]) == 5 and len(r["candidate_ids"]) == 7 for r in rows)
    before = json.dumps(rows, sort_keys=True)
    qrels = {queries[0].task_id: {"7": 2, "1": 1}, queries[1].task_id: {"6": 0}}
    report = summarize(rows, qrels, candidate_k=7)
    for method in report["methods"].values():
        assert method["top5"]["mean_recall"] == .5
        assert method["candidate_pool"]["mean_recall"] == 1.
        assert method["top5"]["missing_qrels_task_ids"] == [queries[2].task_id]
        assert method["top5"]["no_positive_qrels_task_ids"] == [queries[1].task_id]
        assert method["sessions"]["session-a"]["turns"] == 1
    assert not report["answer_quality_evaluated"]
    assert before == json.dumps(rows, sort_keys=True)


def test_offline_user_topic_preserves_query_source_trace_and_cost():
    from bench.retrieval_eval import RetrievalQuery
    item = RetrievalQuery("arbitrary", "session", 2,
        ("How does Acme Store work?", "One last thing, where is the python SDK?"))
    sparse = Sparse(contexts())
    rows = list(evaluate_queries([item], sparse, modes=("current", "all_user", "user_topic"), candidate_k=7))
    assert len(rows) == 3
    result = rows[-1]
    assert result["query_text"].endswith("User topic context: Acme Store")
    assert result["query_view_trace"]["selected_history_spans"][0]["user_turn"] == 1
    assert result["elapsed_ms"] >= result["query_view_ms"] >= 0
    assert sparse.calls[-1][0] == [result["query_text"]]
    assert rows[1]["query_mode"] == "all_user"


def test_actual_cloud_dev_query_cohort_has_41_turns_and_5_sessions():
    root = Path(__file__).parents[1]
    path = root/"data/mtrag/RAG.jsonl"
    if not path.exists():
        pytest.skip("full optional dataset is not installed")
    queries = prepare_queries(load_rows(path), json.loads((root/"data/mtrag/split_manifest.json").read_text()))
    assert len(queries) == 41
    assert len({q.conversation_id for q in queries}) == 5
    assert all(len(q.user_questions) == q.turn for q in queries)


@pytest.mark.parametrize("mode", ["last3", "user_topic"])
def test_cli_reads_qrels_after_all_rankings_and_keeps_artifacts(tmp_path, monkeypatch, mode):
    from bench import retrieval_eval
    index = tmp_path/"sparse"
    retriever = BM25SRetriever(load_beir_corpus(FIXTURE.with_name("corpus_small.jsonl")), corpus_snapshot_id="fixture")
    retriever.save(index)
    split = tmp_path/"split.json"
    split.write_text(json.dumps({"dev": ["session-a"], "test": ["session-b"]}))
    qrels = tmp_path/"qrels.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\nsession-a<::>1\ta\t1\n")
    calls = []
    original_retrieve = BM25SRetriever.retrieve
    original_qrels = retrieval_eval.load_qrels

    def track_retrieve(self, questions, **kwargs):
        calls.append(questions)
        return original_retrieve(self, questions, **kwargs)

    def track_qrels(path):
        assert len(calls) == 3
        return original_qrels(path)

    monkeypatch.setattr(BM25SRetriever, "retrieve", track_retrieve)
    monkeypatch.setattr(retrieval_eval, "load_qrels", track_qrels)
    output = tmp_path/"evaluation"
    retrieval_eval.main(["evaluate", "--input", str(FIXTURE), "--split-manifest", str(split),
        "--sparse-index", str(index), "--qrels", str(qrels), "--output", str(output), "--query-modes", mode])
    manifest = json.loads((output/"manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["encoder"] is None
    assert len(manifest["tasks"]) == 3
    assert len((output/"retrieval.jsonl").read_text().splitlines()) == 3
    assert json.loads((output/"summary.json").read_text())["methods"][mode+"/bm25"]["top5"]["evaluated_tasks"] == 1
    if mode == "user_topic":
        assert manifest["query_view"]["extra_model_calls"] == 0
        assert all(r["query_view_trace"] for r in map(json.loads, (output/"retrieval.jsonl").read_text().splitlines()))
