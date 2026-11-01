from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from adapters.dialogue_query import QUERY_POOLING, query_view, load_query_adapter
from adapters.hybrid_retrieval import DenseIndex, HybridRetriever
from controller.schema import OnlineContext, OnlineMessage, OnlineTurn, QueryContext, TaskFrame
from adapters.conversation import PriorContext


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": list(text)}


def test_literal_view_preserves_current_request_and_whole_recent_exchanges():
    history = [
        {"speaker": "user", "text": "Use v2 in Paris."},
        {"speaker": "agent", "text": "cat database context"},
        {"speaker": "user", "text": "No, use v3."},
        {"speaker": "agent", "text": "actual answer, no invented state"},
    ]
    full, trace = query_view("And London?", history, Tokenizer(), limit=1000)
    assert full.index("And London?") < full.index("No, use v3.") < full.index("Use v2 in Paris.")
    assert trace["history_turns"] == [2, 1] and trace["omitted_history_turns"] == []
    shorter, trace = query_view("And London?", history, Tokenizer(), limit=len(full) - 1)
    assert trace["history_turns"] == [2] and trace["omitted_history_turns"] == [1]
    assert history[-1]["text"] in shorter and history[0]["text"] not in shorter
    assert query_view("And London?", history, Tokenizer(), mode="current")[0] == "And London?"


@pytest.mark.parametrize(
    "history",
    [
        [{"speaker": "user", "text": "incomplete"}],
        [
            {"speaker": "assistant", "text": "wrong first"},
            {"speaker": "user", "text": "wrong second"},
        ],
    ],
)
def test_incomplete_or_wrong_role_history_is_rejected(history):
    with pytest.raises(ValueError, match="completed"):
        query_view("current", history, Tokenizer())


def test_question_is_not_truncated_to_fit_embedding_budget():
    with pytest.raises(ValueError, match="not truncated"):
        query_view("current question", [], Tokenizer(), limit=4)


def test_dialogue_is_retrieval_input_not_a_derived_task_or_source():
    class Encoder:
        manifest = {
            "model_id": "fixture",
            "revision": "v1",
            "max_length": 512,
            "pooling": "fixture",
        }
        max_length = 512
        tokenizer = Tokenizer()
        texts_processed = 0
        token_limit_hits = 0
        inputs = []

        def encode(self, texts):
            self.inputs.extend(texts)
            self.texts_processed += len(texts)
            return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

    encoder = Encoder()
    contexts = tuple(OnlineContext(str(i), f"original source {i}") for i in range(5))
    sparse = SimpleNamespace(manifest={"corpus_snapshot_id": "same"}, corpus_snapshot_id="v1")
    dense = DenseIndex(
        contexts,
        np.array([[1.0, 0.0]] * 5, dtype=np.float32),
        {"corpus_snapshot_id": "same", "encoder": encoder.manifest},
    )
    retriever = HybridRetriever(sparse, dense, encoder, query_mode="dialogue", stage="dense")
    turn = OnlineTurn(
        "s",
        "t2",
        2,
        "c",
        "Where is it?",
        (
            OnlineMessage("user", "Tell me about cat database."),
            OnlineMessage("assistant", "actual cat database answer"),
        ),
        (),
        "P3",
    )
    with pytest.raises(ValueError, match="committed"):
        retriever.retrieve_turn(turn)
    result = retriever.retrieve_turn(turn, prior_context=PriorContext(interpretations=(None,)))
    assert (
        result.query == turn.query
        and result.history == turn.history
        and result.query_context is None
    )
    assert tuple(replace(c, score=None) for c in result.contexts) == contexts
    assert "actual cat database answer" in encoder.inputs[0]
    trace = retriever.manifest["query_view_traces"][0]
    assert trace["model_called"] is False and trace["context"] is None
    from adapters.serving_retrieval import ServingRetriever
    from controller.schema import SessionSnapshot, TurnRecord
    import time

    wrapper = ServingRetriever(retriever, corpus_id="c")
    snapshot = SessionSnapshot(
        "s",
        raw_history=(TurnRecord(1, "Tell me about cat database.", "actual cat database answer"),),
    )
    selection = wrapper._retrieve(snapshot, turn.query, "request", time.monotonic())
    assert selection.query_context is None and len(selection.evidence) == 5
    assert "actual cat database answer" in encoder.inputs[-1]
    with pytest.raises(ValueError, match="P3"):
        retriever.retrieve_turn(replace(turn, protocol="P1"), prior_context=PriorContext())


@pytest.mark.parametrize("rerank", [False, True])
@pytest.mark.parametrize("query", ["And London?", "And London? " * 25])
@pytest.mark.parametrize("with_state", [False, True])
def test_dialogue_hybrid_combines_actual_qa_encoding_with_user_only_bm25(rerank, query, with_state):
    class Encoder:
        manifest = {"model_id": "fixture", "revision": "v1", "max_length": 512, "pooling": "fixture"}
        max_length = 512
        tokenizer = Tokenizer()
        texts_processed = 0
        token_limit_hits = 0

        def __init__(self):
            self.inputs = []

        def encode(self, texts):
            self.inputs.extend(texts)
            self.texts_processed += len(texts)
            return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

    contexts = tuple(OnlineContext(str(i), f"original source {i}") for i in range(5))

    class Sparse:
        manifest = {"corpus_snapshot_id": "same"}
        corpus_snapshot_id = "same"

        def __init__(self):
            self.queries = []

        def retrieve(self, queries, *, top_k):
            self.queries.append(tuple(queries))
            return tuple(reversed(contexts))

    sparse, encoder = Sparse(), Encoder()
    dense = DenseIndex(contexts, np.array([[1.0, 0.0]] * 5, dtype=np.float32),
                       {"corpus_snapshot_id": "same", "encoder": encoder.manifest})
    scored = []
    def score(query, contexts):
        scored.append(query)
        return list(range(len(contexts)))
    reranker = SimpleNamespace(score=score, tokenizer=Tokenizer(), max_length=512)
    intent = QueryContext("Resolved archival task", "revisit", (1,),
                          TaskFrame(task="Compare", requirements=("limits", "retention")))
    calls = []
    def resolve(turn, context):
        calls.append((turn, context))
        return {"text": intent.query, "context": asdict(intent), "status": "resolved", "model_called": True}
    resolver = SimpleNamespace(resolve=resolve, manifest={"version": "fixture"}) if with_state else None
    retriever = HybridRetriever(sparse, dense, encoder, reranker if rerank else None,
        query_mode="dialogue", stage="hybrid_rerank" if rerank else "hybrid", resolver=resolver)
    history = (OnlineMessage("user", "old question"), OnlineMessage("assistant", "old answer"),
               OnlineMessage("user", "v2 in Paris"), OnlineMessage("assistant", "actual prior answer"),
               OnlineMessage("user", "No, v3"), OnlineMessage("assistant", "actual latest answer"))
    turn = OnlineTurn("s", "t4", 4, "c", query, history, (), "P3")
    result = retriever.retrieve_turn(turn, prior_context=PriorContext())
    assert sparse.queries == [("v2 in Paris\nNo, v3\n" + query,)]
    assert "actual latest answer" in encoder.inputs[0] and "And London?" in encoder.inputs[0]
    assert [context.document_id for context in result.contexts] == (
        ["2", "3", "1", "4", "0"] if rerank else ["0", "4", "1", "3", "2"])
    if rerank:
        assert query in scored[0] and "old answer" not in scored[0]
        if len(query) > 256:
            assert scored[0] == query
        else:
            assert "actual latest answer" in scored[0] and len(scored[0]) <= 256
    assert result.query_context == (intent if with_state else None) and result.history == history
    assert len(calls) == int(with_state)
    assert intent.query not in encoder.inputs[0]
    trace = retriever.query_view_traces[-1]
    assert trace["sparse_query"] == sparse.queries[0][0] and trace["model_called"] == with_state
    assert trace["resolved_query"] == (intent.query if with_state else None)


def test_dialogue_adapter_configuration_accepts_hybrid_without_a_selector(tmp_path):
    from adapters.retriever_factory import load_retriever_config

    config = {"sparse_index": "sparse", "dense_index": "dense", "encoder_path": "base",
              "query_mode": "dialogue", "query_adapter_path": "adapter", "stage": "hybrid"}
    path = tmp_path / "retriever.json"
    path.write_text(json.dumps(config))
    assert load_retriever_config(path).stage == "hybrid"
    path.write_text(json.dumps({**config, "stage": "hybrid_rerank", "reranker_path": "reranker"}))
    assert load_retriever_config(path).stage == "hybrid_rerank"
    path.write_text(json.dumps({**config, "selector": {
        "base_url": "http://localhost:30000", "model": "Qwen3-32B", "revision": "pinned"}}))
    assert load_retriever_config(path).selector["model"] == "Qwen3-32B"


def test_adapter_uses_its_declared_document_space_and_query_view(tmp_path):
    root = tmp_path / "adapter"
    root.mkdir()
    (root / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "task_type": "FEATURE_EXTRACTION"}))
    base = {"model_id": "base", "revision": "pinned", "max_length": 512, "pooling": "CLS"}
    metadata = {"schema_version": 1, "status": "completed", "query_mode": "dialogue",
                "document_encoder": base, "query_pooling": QUERY_POOLING}
    path = tmp_path / "query-adapter.json"
    path.write_text(json.dumps(metadata))
    assert load_query_adapter(tmp_path, base) == metadata
    with pytest.raises(ValueError, match="different document encoder"):
        load_query_adapter(tmp_path, {**base, "revision": "different"})
    path.write_text(json.dumps({**metadata, "query_view_version": "other"}))
    with pytest.raises(ValueError, match="input recipe"):
        load_query_adapter(tmp_path, base)


def test_query_adapter_uses_base_index_but_cannot_reencode_documents(tmp_path):
    base = {"model_id": "base", "revision": "v1", "max_length": 512, "pooling": "CLS"}
    encoder = SimpleNamespace(
        manifest={"model_id": "adapter", "document_encoder": base},
        encode=lambda texts: np.array([[1.0, 0.0] for _ in texts]),
    )
    contexts = [OnlineContext("a", "original A"), OnlineContext("b", "original B")]
    index = DenseIndex(contexts, np.array([[1.0, 0.0], [0.0, 1.0]]), {"encoder": base})
    assert index.search("question", encoder)[0].document_id == "a"
    with pytest.raises(ValueError, match="base document encoder"):
        DenseIndex.build(contexts, encoder, tmp_path / "index", corpus_snapshot_id="v1")
    assert not (tmp_path / "index").exists()
    encoder.manifest["document_encoder"] = {**base, "revision": "v2"}
    with pytest.raises(ValueError, match="differs"):
        index.search("question", encoder)
