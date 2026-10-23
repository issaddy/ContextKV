import asyncio
from dataclasses import replace
import json
from pathlib import Path
import time

import httpx
import numpy as np
import pytest
import yaml

from adapters.hybrid_retrieval import DenseIndex
from adapters.mtrag import OnlineContext, OnlineSession, load_rows
from adapters.retrieval import BM25SRetriever, load_beir_corpus
from adapters.retriever_factory import create_retriever, load_retriever_config
from bench.run_sessions import run_sessions


FIXTURE = Path(__file__).parent/"fixtures/mtrag_small.jsonl"


class LocalModelFixture:
    instances = []

    def __init__(self, path, *, reranker=False, **kwargs):
        self.reranker = reranker
        self.inputs, self.texts_processed, self.token_limit_hits = [], 0, 0
        self.manifest = {"model_id": "fixture", "revision": "fixture-v1", "pooling": "fixture",
                         "max_length": kwargs.get("max_length", 8192), "path": str(path)}
        self.active = self.maximum_active = 0
        self.closed = False
        self.instances.append(self)

    def encode(self, texts):
        self.inputs.extend(texts)
        self.texts_processed += len(texts)
        return np.ones((len(texts), 2), dtype=np.float32)/np.sqrt(2)

    def score(self, query, contexts):
        self.inputs.append(query)
        self.texts_processed += len(contexts)
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        time.sleep(.01)
        self.active -= 1
        return np.arange(len(contexts))

    def close(self):
        self.closed = True


@pytest.fixture
def retriever_config(tmp_path, monkeypatch):
    from adapters import hybrid_retrieval
    corpus = load_beir_corpus(FIXTURE.with_name("corpus_small.jsonl"))
    sparse = BM25SRetriever(corpus, corpus_snapshot_id="fixture-v1")
    sparse.save(tmp_path/"sparse")
    DenseIndex.build([OnlineContext(**c) for c in corpus.values()], LocalModelFixture("encoder"),
                     tmp_path/"dense", corpus_snapshot_id="fixture-v1")
    LocalModelFixture.instances.clear()
    monkeypatch.setattr(hybrid_retrieval, "LocalBGE", LocalModelFixture)
    value = {"sparse_index": str(tmp_path/"sparse"), "dense_index": str(tmp_path/"dense"),
             "encoder_path": "encoder", "reranker_path": "reranker", "query_mode": "current"}
    path = tmp_path/"retriever.yaml"
    path.write_text(yaml.safe_dump(value))
    return path


def test_factory_loads_models_once_and_retrieves_only_user_questions(retriever_config):
    retriever = create_retriever("P3", config_path=retriever_config)
    gate = OnlineSession(load_rows(FIXTURE)[:3], "P3")
    outputs = []
    while not gate.done:
        turn = retriever.retrieve_turn(gate.current())
        outputs.append(turn)
        gate.commit(turn.task_id, "actual generated assistant; never a retrieval query")
    assert len(LocalModelFixture.instances) == 2
    assert [c.document_id for c in outputs[0].contexts] == list("fedcb")
    assert retriever.encoder.inputs == [f"session-a question {i}" for i in (1, 2, 3)]
    assert retriever.reranker.inputs == retriever.encoder.inputs
    assert all("REFERENCE" not in json.dumps(turn.history, default=lambda m: m.text) for turn in outputs)
    assert retriever.manifest["completed_retrievals"] == 3
    assert retriever.manifest["configuration"]["resolved"]["query_mode"] == "current"
    assert retriever.manifest["model_usage"]["reranker"]["texts_processed"] == 18
    with pytest.raises(ValueError, match="P3"):
        retriever.retrieve_turn(replace(outputs[0], protocol="P1"))


def test_legacy_bm25_default_and_fixed_protocol_need_no_model(retriever_config):
    config = load_retriever_config(retriever_config)
    retriever = create_retriever("P3", index_path=config.sparse_index)
    turn = OnlineSession(load_rows(FIXTURE)[:3], "P3").current()
    assert [c.document_id for c in retriever.retrieve_turn(turn).contexts] == list("abcde")
    assert create_retriever("P2", config_path="absent.yaml") is None
    assert LocalModelFixture.instances == []
    with pytest.raises(ValueError, match="choose"):
        create_retriever("P3", config_path=retriever_config, index_path=config.sparse_index)
    value = yaml.safe_load(retriever_config.read_text())
    value["query_mode"] = "generated_history"
    retriever_config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="user questions"):
        create_retriever("P3", config_path=retriever_config)


def test_dense_stage_does_not_load_or_call_reranker(retriever_config, monkeypatch):
    value = yaml.safe_load(retriever_config.read_text())
    value.update(stage="dense", query_mode="all_user")
    retriever_config.write_text(json.dumps(value))
    retriever = create_retriever("P3", config_path=retriever_config)
    monkeypatch.setattr(retriever.sparse, "retrieve", lambda *a, **k: pytest.fail("dense stage must not run BM25"))
    gate = OnlineSession(load_rows(FIXTURE)[:3], "P3")
    first = retriever.retrieve_turn(gate.current())
    gate.commit(first.task_id, "generated assistant")
    retriever.retrieve_turn(gate.current())
    assert len(LocalModelFixture.instances) == 1 and retriever.reranker is None
    assert retriever.encoder.inputs[-1] == "session-a question 1\nsession-a question 2"


def test_optional_user_topic_query_is_shared_with_encoder_reranker_and_manifest(retriever_config):
    from adapters.mtrag import OnlineMessage
    value = yaml.safe_load(retriever_config.read_text())
    value["query_mode"] = "user_topic"
    retriever_config.write_text(yaml.safe_dump(value))
    retriever = create_retriever("P3", config_path=retriever_config)
    original = OnlineSession(load_rows(FIXTURE)[:3], "P3").current()
    turn = replace(original, query="One last thing, where is the python SDK?", turn=3,
        history=(OnlineMessage("user", "How do I use Acme Store?"),
                 OnlineMessage("assistant", "Misleading Other Database SDK"),
                 OnlineMessage("user", "What about storage?"),
                 OnlineMessage("assistant", "Reference answer must not enter retrieval")))
    output = retriever.retrieve_turn(turn)
    expected = turn.query+"\nUser topic context: Acme Store"
    assert retriever.encoder.inputs == retriever.reranker.inputs == [expected]
    assert output.history == turn.history and output.query == turn.query
    assert retriever.manifest["query_view"]["extra_model_calls"] == 0
    trace = retriever.manifest["query_view_traces"][0]
    assert trace["task_id"] == turn.task_id and trace["source_user_turns"] == [1, 3]
    assert trace["selected_history_spans"][0]["text"] == "Acme Store"


def test_concurrent_sessions_include_retrieval_queue_in_pipeline_time(retriever_config):
    retriever = create_retriever("P3", config_path=retriever_config)
    async def answer(turn):
        return "actual answer"
    report = asyncio.run(run_sessions(load_rows(FIXTURE), answer, protocol="P3", concurrency=2, retriever=retriever))
    assert len(report.turns) == 6 and all(r.status == "success" for r in report.turns)
    assert report.time_origin == "pipeline_arrival"
    assert retriever.reranker.maximum_active == 1
    assert all(r.t_done >= r.t_retrieval_done > r.t_arrive for r in report.turns)
    assert all(r.t_retrieval_done-r.t_arrive >= .009 for r in report.turns)
    assert all("actual answer" not in query for query in retriever.encoder.inputs)


@pytest.mark.parametrize("query_mode", ["current", "user_topic"])
def test_both_real_cli_paths_share_retriever_and_persist_generated_history(retriever_config, tmp_path, monkeypatch, query_mode):
    import bench.native_baseline as native_module
    import controller.cli as cli
    from controller.templates import ByteTemplate

    value = yaml.safe_load(retriever_config.read_text())
    value["query_mode"] = query_mode
    retriever_config.write_text(yaml.safe_dump(value))

    def respond(request):
        if request.url.path == "/server_info":
            return httpx.Response(200, json={"version": "fake-http-correctness"})
        return httpx.Response(200, json={"text": "native actual", "output_ids": list(b"native actual"),
            "meta_info": {"finish_reason": {"type": "stop"}, "completion_tokens": len(b"native actual")}})
    http = httpx.AsyncClient(base_url="http://fake.test", transport=httpx.MockTransport(respond))
    client_class, baseline_class = native_module.NativeSGLangClient, native_module.NativeBaseline
    monkeypatch.setattr(native_module, "HuggingFaceTemplate", lambda *a, **k: ByteTemplate())
    monkeypatch.setattr(native_module, "NativeSGLangClient", lambda *a, **k:
                        client_class("http://fake.test", http_client=http, stream=False))
    monkeypatch.setattr(native_module, "NativeBaseline", lambda *a, **k:
                        baseline_class(*a, **k, test_mode=True, run_id="hybrid-test"))
    native_dir, controller_dir = tmp_path/"native", tmp_path/"controller"
    engine = tmp_path/"engine.json"
    engine.write_text("{}")
    common = [str(FIXTURE), "--protocol", "P3", "--domain", "all", "--limit-sessions", "1",
              "--retriever-config", str(retriever_config)]
    try:
        native_module.main(common+["--baseline", "B0", "--engine-manifest", str(engine), "--output", str(native_dir)])
        cli.main(["--state-dir", str(controller_dir), "bench"]+common)
    finally:
        asyncio.run(http.aclose())
    native = json.loads((native_dir/"native-B0-hybrid-test.json").read_text())
    controller = json.loads(next(controller_dir.glob("benchmark-*.json")).read_text())
    controller_manifest = json.loads(next(controller_dir.glob("manifest-*.json")).read_text())
    assert len(LocalModelFixture.instances) == 4  # One encoder/reranker per CLI run, not per turn.
    assert all(model.closed for model in LocalModelFixture.instances)
    assert native["retrieval"]["configuration"] == controller_manifest["retrieval"]["configuration"]
    for data in (native, controller):
        assert data["report"]["time_origin"] == "pipeline_arrival"
        turns = data["report"]["turns"]
        assert len(turns) == 3 and all(t["status"] == "success" for t in turns)
        assert all([c["document_id"] for c in t["online_turn"]["contexts"]] == list("fedcb") for t in turns)
        assert turns[1]["online_turn"]["history"][-1]["text"] == turns[0]["assistant_text"]
        assert "REFERENCE ANSWER" not in json.dumps(turns)
    assert native["retrieval"]["completed_retrievals"] == controller_manifest["retrieval"]["completed_retrievals"] == 3
    for manifest in (native["retrieval"], controller_manifest["retrieval"]):
        assert manifest["query_mode"] == query_mode
        if query_mode == "user_topic":
            assert len(manifest["query_view_traces"]) == 3
            assert all("actual" not in v["text"] for v in manifest["query_view_traces"])
