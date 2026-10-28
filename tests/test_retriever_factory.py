import asyncio
from dataclasses import replace
import json
from pathlib import Path
import time

import httpx
import numpy as np
import pytest
import yaml
from test_model_scope import local_scope as local_scope

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


@pytest.mark.parametrize("stage", ["dense", "hybrid", "hybrid_rerank"])
@pytest.mark.parametrize("window", [False, True])
def test_semantic_factory_shares_the_interpreter_client_and_requires_bound_sources(retriever_config, monkeypatch, stage, window):
    from adapters import query_selector
    from adapters.serving_retrieval import ServingRetriever
    clients=[]
    class Client:
        manifest={"model":"fixture"}
        def __init__(self,config):
            self.config=config
            self.closes=0
            clients.append(self)
        def close(self):
            self.closes+=1
    monkeypatch.setattr(query_selector,"OpenAICompletionClient",Client)
    value=yaml.safe_load(retriever_config.read_text())
    value.update(stage=stage,query_mode="conversational",evidence_selection="semantic",source_window=window,
                 selector={"model":"fixture","revision":"fixed","base_url":"http://fixture"})
    retriever_config.write_text(yaml.safe_dump(value))
    retriever=create_retriever("P3",config_path=retriever_config)
    try:
        assert retriever.evidence_selector.resolver is retriever.resolver
        assert retriever.resolver.client is clients[0]
        assert (retriever.reranker is None) == (stage != "hybrid_rerank")
        with pytest.raises(ValueError,match="evidence-corpus"):
            ServingRetriever(retriever,corpus_id="c")
    finally:
        retriever.close()
    assert clients[0].closes==1


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['hybrid', 'hybrid_rerank'])
async def test_complete_reading_choices_and_conditions_reach_both_candidates_without_redundant_scores(
        retriever_config,tmp_path,monkeypatch,stage):
    from adapters import query_selector
    from adapters.corpus import CorpusIndex
    from adapters.query_selector import CompletionResult
    from adapters.serving_retrieval import ServingRetriever
    from controller.candidates import build_candidates
    from controller.mock_engine import MockEngine
    from controller.service import Controller
    from controller.session_store import SessionStore
    catalogs=[]
    class Client:
        manifest={'model':'fixture'}
        def __init__(self,config):
            self.config=config
        def catalog_token_lengths(self,messages,documents):
            return 40,[len(d) for d in documents]
        def complete(self,messages,**kwargs):
            body=json.loads(messages[-1]['content'])
            catalogs.append(body)
            ids=[d['id'] for d in body['documents'] if d.get('required_context')]
            ids.append(next(d['id'] for d in body['documents'] if not d.get('required_context')))
            value={'relation':'standalone','history_turns':[],'resume_turn':0,
                'frame':{'topic':'Atlas','target':'','task':'download and install',
                    'requirements':['download package','installation procedure'],
                    'conditions':{'version':'v4','platform':'Windows'}}, 'selected_ids':ids}
            return CompletionResult('completed',json.dumps(value),'stop')
        def close(self):
            pass
    monkeypatch.setattr(query_selector,'OpenAICompletionClient',Client)
    source=load_beir_corpus(FIXTURE.with_name('corpus_small.jsonl'))
    rows=[dict(row,passage_id=pid,document_id=pid,span=[0,len(row['text'])],
               **({'dependencies':['condition']} if pid=='f' else {})) for pid,row in source.items()]
    rows.append({'passage_id':'condition','document_id':'conditions','text':'Only for Windows v4.'})
    corpus=CorpusIndex.from_records(rows,corpus_id='fixture',snapshot_id='fixture-v1')
    original=tmp_path/'originals.jsonl'
    original.write_text(''.join(json.dumps({'snapshot_id':'fixture-v1','source':row['source'],'document_id':pid,'text':'Context before.\n\n'+row['text']+'\n\nContext after.'})+'\n'
                               for pid,row in source.items()))
    config=yaml.safe_load(retriever_config.read_text())
    config.update(stage=stage,query_mode='conversational',evidence_selection='semantic',source_window=True,
                  source_documents=str(original),selector={'base_url':'http://fixture','model':'fixture','revision':'fixed'})
    retriever_config.write_text(yaml.safe_dump(config))
    retriever=create_retriever('P3',config_path=retriever_config)
    wrapper=ServingRetriever(retriever,corpus_id='fixture',corpus_index=corpus)
    store=SessionStore(tmp_path/'reading.sqlite3')
    service=Controller(store,MockEngine())
    selected=[]
    async def provider(snapshot):
        selection=await wrapper.retrieve(snapshot,'Atlas Windows v4: download and install','one')
        selected.append(selection)
        return selection
    try:
        response=await service.serve_turn('s','Atlas Windows v4: download and install',evidence_provider=provider,
            request_id='one',corpus_snapshot_id='fixture-v1')
        replay=await service.serve_turn('s','Atlas Windows v4: download and install',evidence_provider=provider,
            request_id='one',corpus_snapshot_id='fixture-v1')
        assert replay.idempotent_replay and response.output_ids==replay.output_ids
        assert len(catalogs)==len(selected)==1
        documents=catalogs[0]['documents']
        assert any('Context before.' in d['text'] for d in documents)
        assert any(d.get('required_context',[{}])[0].get('text')=='Only for Windows v4.' for d in documents)
        visible={d['text'] for d in documents}|{s['text'] for d in documents for s in d.get('required_context',[])}
        blocks=selected[0].evidence
        assert {b.text for b in blocks}<=visible and any(b.text=='Only for Windows v4.' for b in blocks)
        assert any(b.key.passage_id.startswith('source-window:v2:') for b in blocks)
        for candidate in build_candidates(store.get_snapshot('s'),'Continue',blocks):
            assert candidate.required_blocks=={b.key for b in blocks}
        trace=retriever.query_view_traces[-1]
        assert 'requirement_evidence' not in trace
        assert len(retriever.encoder.inputs)==1
        assert trace['semantic_evidence']['status']=='selected'
        assert trace['context']['task_frame']['requirements']==('download package','installation procedure')
        if retriever.reranker:
            assert len(retriever.reranker.inputs)==1
        await service.close_session('s')
        assert not store.outbox()
    finally:
        await service.aclose()
        store.close()
        wrapper.close()


@pytest.mark.asyncio
async def test_distinct_content_selection_flows_through_serving_and_durable_replay(retriever_config, tmp_path):
    from adapters.serving_retrieval import ServingRetriever
    from controller.mock_engine import MockEngine
    from controller.service import Controller
    from controller.session_store import SessionStore

    retriever = create_retriever("P3",config_path=retriever_config)
    original = retriever.dense.by_id["f"]
    retriever.dense.by_id["e"] = replace(original,document_id="e")
    wrapper = ServingRetriever(retriever,corpus_id="fixture")
    store = SessionStore(tmp_path/"distinct.sqlite3")
    service = Controller(store,MockEngine())
    try:
        async def provider(snapshot):
            return await wrapper.retrieve(snapshot,"Question", "request")
        first = await service.serve_turn("s","Question",evidence_provider=provider,
                    corpus_snapshot_id=wrapper.corpus_snapshot_id,request_id="request")
        state = store.get_snapshot("s")
        # Five bodies fit, while both retrieved identities of the shared body remain.
        assert {key.passage_id for key in state.raw_history[0].evidence_keys} == set("fedcba")
        trace = retriever.manifest["query_view_traces"][0]["evidence_selection"]
        assert trace["selected_ids"] == list("fedcba")
        assert len(retriever.encoder.inputs) == len(retriever.reranker.inputs) == 1
        replay = await service.serve_turn("s","Question",evidence_provider=provider,
                    corpus_snapshot_id=wrapper.corpus_snapshot_id,request_id="request")
        assert replay.idempotent_replay and replay.output_ids == first.output_ids
        assert len(retriever.reranker.inputs) == 1
    finally:
        await service.aclose()
        wrapper.close()
        store.close()


@pytest.mark.parametrize("relation", ("correction", "ambiguous"))
def test_conversational_factory_carries_resolution_or_ambiguity_into_serving(retriever_config, monkeypatch, relation):
    from adapters import query_selector
    from adapters.query_selector import CompletionResult
    from adapters.serving_retrieval import ServingRetriever
    from controller.schema import QueryContext, SessionSnapshot, TaskFrame, TurnRecord
    value = yaml.safe_load(retriever_config.read_text())
    value.update(query_mode="conversational", query_fusion="single",
                 selector={"base_url":"http://fake","model":"fake","revision":"fixed"})
    retriever_config.write_text(yaml.safe_dump(value))
    calls = []
    class Client:
        manifest = {"model":"fixture"}
        def complete(self, messages, **kwargs):
            calls.append(messages)
            return CompletionResult("completed", json.dumps({"frame":{"task":"Atlas attachment limits?","topic":"","target":"","conditions":{},"requirements":[]},
                "relation":relation,"history_turns":[1],"resume_turn":0}), "stop")
        def close(self):
            pass
    monkeypatch.setattr(query_selector, "OpenAICompletionClient", lambda _:Client())
    retriever = create_retriever("P3", config_path=retriever_config)
    assert retriever.requires_history_context
    wrapper = ServingRetriever(retriever, corpus_id="fixture")
    snapshot = SessionSnapshot("s", raw_history=(TurnRecord(1,"Atlas document limits?","An actual response."),))
    try:
        selection = wrapper._retrieve(snapshot, "No, I meant attachments.", "request", time.monotonic())
        expected_query = "No, I meant attachments." if relation == "ambiguous" else "Atlas attachment limits?"
        frame = None if relation == "ambiguous" else TaskFrame(task=expected_query)
        assert selection.query_context == QueryContext(expected_query, relation, (1,), frame)
        search_query = "Atlas document limits?\nNo, I meant attachments." if relation == "ambiguous" else expected_query
        assert retriever.encoder.inputs == retriever.reranker.inputs == [search_query]
        assert len(selection.evidence) == 5 and len(calls) == 1
        assert retriever.manifest["query_view_traces"][0]["context"]["relation"] == relation
    finally:
        wrapper.close()


def test_serving_shares_one_metadata_index_with_ranking_and_generation(retriever_config):
    from adapters.corpus import CorpusIndex
    from adapters.serving_retrieval import ServingRetriever
    from controller.schema import SessionSnapshot

    retriever = create_retriever("P3",config_path=retriever_config)
    metadata = CorpusIndex.from_records([
        {"_id":c.document_id,"text":c.text,"title":c.title,"url":f"https://example.org/guide/{c.document_id}"}
        for c in retriever.dense.contexts],corpus_id="fixture",snapshot_id=retriever.corpus_snapshot_id)
    wrapper = ServingRetriever(retriever,corpus_id="fixture",corpus_index=metadata)
    scored = []
    original_score = retriever.reranker.score
    def score(query, contexts):
        scored.extend(contexts)
        return original_score(query,contexts)
    retriever.reranker.score = score
    try:
        selected = wrapper._retrieve(SessionSnapshot("s"),"question","one",time.monotonic())
        assert len(scored) == 6 and all(c.source.startswith("https://example.org/guide/") for c in scored)
        assert all(b.source_uri == f"https://example.org/guide/{b.key.passage_id}" for b in selected.evidence)
        assert retriever.manifest["pre_rerank_metadata"]
        assert len(retriever.encoder.inputs) == len(retriever.reranker.inputs) == 1
    finally:
        wrapper.close()


@pytest.mark.parametrize("fusion", ["single", "current_topic_rrf"])
def test_conversational_factory_uses_the_same_pipeline_without_a_model_client(retriever_config, monkeypatch, fusion):
    from adapters import query_selector
    from adapters.serving_retrieval import ServingRetriever
    from controller.schema import SessionSnapshot, TurnRecord
    value = yaml.safe_load(retriever_config.read_text())
    value.update(query_mode="conversational", query_fusion=fusion)
    retriever_config.write_text(yaml.safe_dump(value))
    def unexpected_client(*args, **kwargs):
        raise AssertionError("No auxiliary model was configured")
    monkeypatch.setattr(query_selector, "OpenAICompletionClient", unexpected_client)
    retriever = create_retriever("P3", config_path=retriever_config)
    wrapper = ServingRetriever(retriever, corpus_id="fixture")
    snapshot = SessionSnapshot("s", raw_history=(TurnRecord(1,"How do I create tables?","Actual response."),))
    try:
        selected = wrapper._retrieve(snapshot, "Use AtlasCloud", "request", time.monotonic())
        assert selected.query_context.query == "How do I create tables?\nUse AtlasCloud"
        assert retriever.encoder.inputs == ([selected.query_context.query] if fusion == "single" else
                                             ["Use AtlasCloud", selected.query_context.query])
        assert retriever.reranker.inputs == [selected.query_context.query, "Use AtlasCloud"]
        trace = retriever.manifest["query_view_traces"][0]["condition_evidence"]
        assert trace["extra_rerank_pairs"] == 6
        assert [key.document_id for key in selected.retrieval_priority] == trace["selected_ids"]
        assert retriever.manifest["query_view"]["transport"]["kind"] == "no_auxiliary_model"
    finally:
        wrapper.close()


@pytest.mark.asyncio
async def test_condition_selection_commits_once_through_the_serving_pipeline(retriever_config, tmp_path):
    from adapters.serving_retrieval import ServingRetriever
    from controller.evidence import make_evidence
    from controller.mock_engine import MockEngine
    from controller.service import Controller
    from controller.session_store import SessionStore
    value = yaml.safe_load(retriever_config.read_text())
    value.update(query_mode="conversational", query_fusion="single")
    retriever_config.write_text(yaml.safe_dump(value))
    retriever = create_retriever("P3", config_path=retriever_config)
    wrapper = ServingRetriever(retriever, corpus_id="fixture")
    store = SessionStore(tmp_path/"sessions.sqlite3")
    service = Controller(store, MockEngine())
    try:
        await service.serve_turn("session", "How do I create tables?", (make_evidence("Prior evidence", "prior"),),
                                 corpus_snapshot_id=wrapper.corpus_snapshot_id, request_id="first")
        async def provider(snapshot):
            return await wrapper.retrieve(snapshot, "Use AtlasCloud", "second")
        response = await service.serve_turn("session", "Use AtlasCloud", evidence_provider=provider,
                                            corpus_snapshot_id=wrapper.corpus_snapshot_id, request_id="second")
        state = store.get_snapshot("session")
        trace = retriever.manifest["query_view_traces"][0]["condition_evidence"]
        assert {key.document_id for key in state.raw_history[-1].evidence_keys} == set(trace["selected_ids"])
        assert state.raw_history[-1].query_context.query == "How do I create tables?\nUse AtlasCloud"
        assert state.raw_history[-1].assistant_text == response.assistant_text
        calls = list(retriever.reranker.inputs)
        replay = await service.serve_turn("session", "Use AtlasCloud", evidence_provider=provider,
                                          corpus_snapshot_id=wrapper.corpus_snapshot_id, request_id="second")
        assert replay.assistant_text == response.assistant_text and retriever.reranker.inputs == calls
        await service.close_session("session")
        assert not store.outbox()
    finally:
        await service.aclose()
        store.close()
        wrapper.close()


@pytest.mark.asyncio
async def test_source_window_reaches_both_candidates_and_commits_without_extra_model_calls(retriever_config, tmp_path):
    from adapters.corpus import CorpusIndex
    from adapters.serving_retrieval import ServingRetriever
    from controller.candidates import build_candidates
    from controller.mock_engine import MockEngine
    from controller.service import Controller
    from controller.session_store import SessionStore
    value = yaml.safe_load(retriever_config.read_text())
    value["source_window"] = True
    retriever_config.write_text(yaml.safe_dump(value))
    retriever = create_retriever("P3", config_path=retriever_config)
    rows = [{"_id": c.document_id, "text": c.text, "title": c.title,
             "document_id": "manual", "span": [i*100, (i+1)*100]}
            for i, c in enumerate(retriever.dense.contexts)]
    index = CorpusIndex.from_records(rows, corpus_id="fixture", snapshot_id="fixture-v1")
    wrapper = ServingRetriever(retriever, corpus_id="fixture", corpus_index=index)
    store = SessionStore(tmp_path/"windows.sqlite3")
    service = Controller(store, MockEngine())
    captured = []
    async def provider(snapshot):
        selection = await wrapper.retrieve(snapshot, "question", "second" if snapshot.raw_history else "first")
        captured.append(selection)
        return selection
    try:
        for request in ("first", "second"):
            response = await service.serve_turn("s", "question", evidence_provider=provider,
                corpus_snapshot_id="fixture-v1", request_id=request)
            selected = captured[-1]
            assert [key.passage_id for key in selected.retrieval_priority] == list("fedcb")
            assert len(selected.evidence) == 6
            snapshot = store.get_snapshot("s")
            candidates = build_candidates(snapshot, "follow up", selected.evidence,
                retrieval_priority=selected.retrieval_priority)
            assert {c.action for c in candidates} == {"APPEND", "REBUILD"}
            assert all(len(c.required_blocks) == 6 and c.required_blocks <= c.visible_blocks for c in candidates)
            replay = await service.serve_turn("s", "question", evidence_provider=provider,
                corpus_snapshot_id="fixture-v1", request_id=request)
            assert replay.idempotent_replay and replay.output_ids == response.output_ids
        assert len(captured) == 2 and retriever.reranker.texts_processed == 12
        assert len(retriever.encoder.inputs) == 2
        assert len(snapshot.raw_history) == 2 and len(snapshot.archive) == 6
        assert all(t["source_window"]["supporting_ids"] == ["a"] for t in retriever.query_view_traces)
        await service.close_session("s")
        assert not store.outbox()
    finally:
        wrapper.close()
        await service.aclose()
        store.close()


@pytest.mark.asyncio
async def test_compound_requirements_drive_shared_selection_and_survive_restore(retriever_config, tmp_path, monkeypatch):
    from adapters import query_selector
    from adapters.query_selector import CompletionResult
    from adapters.serving_retrieval import ServingRetriever
    from controller.candidates import build_candidates
    from controller.mock_engine import MockEngine
    from controller.service import Controller
    from controller.session_store import SessionStore
    completions = []
    class Client:
        manifest = {"kind":"fixture"}
        def __init__(self, config):
            pass
        def complete(self, messages, **kwargs):
            body = json.loads(messages[-1]["content"])
            completions.append(body)
            if body["previous_turn"] is None:
                value = {"relation":"standalone", "history_turns":[], "resume_turn":0,
                    "frame":{"topic":"Atlas", "target":"", "task":"download and install",
                        "requirements":["download package", "installation procedure"],
                        "conditions":{"version":"v4", "platform":"Linux"}}}
            else:
                value = {"relation":"condition_change", "history_turns":[1], "resume_turn":0,
                    "task":{"operation":"keep", "value":"", "requirements":[]},
                    "updates":{"conditions":{"platform":"Windows"}}}
            return CompletionResult("completed",json.dumps(value),"stop")
        def close(self):
            pass
    monkeypatch.setattr(query_selector,"OpenAICompletionClient",Client)
    value = yaml.safe_load(retriever_config.read_text())
    value.update(query_mode="conversational",selector={"base_url":"http://fixture","model":"fixture","revision":"fixed"})
    retriever_config.write_text(yaml.safe_dump(value))
    retriever = create_retriever("P3",config_path=retriever_config)
    scoring=[]
    def score(query, contexts):
        scoring.append(query)
        wanted = ("a" if "download package" in query and "installation procedure" not in query else
                  "b" if "installation procedure" in query and "download package" not in query else "f")
        return [1000 if c.document_id==wanted else ord(c.document_id) for c in contexts]
    retriever.reranker.score = score
    wrapper = ServingRetriever(retriever,corpus_id="fixture")
    path=tmp_path/"compound.sqlite3"
    store=SessionStore(path)
    service=Controller(store,MockEngine())
    selections=[]
    try:
        for request,query in (("one","Atlas v4 Linux: download package and installation procedure"),
                              ("two","Use Windows instead.")):
            async def provider(snapshot):
                selection=await wrapper.retrieve(snapshot,query,request)
                selections.append(selection)
                return selection
            result=await service.serve_turn("s",query,evidence_provider=provider,
                                           request_id=request,corpus_snapshot_id=wrapper.corpus_snapshot_id)
            replay=await service.serve_turn("s",query,evidence_provider=provider,
                                           request_id=request,corpus_snapshot_id=wrapper.corpus_snapshot_id)
            assert replay.idempotent_replay and result.output_ids==replay.output_ids
        assert len(completions)==len(selections)==2 and len(scoring)==6
        assert len(retriever.encoder.inputs)==6  # Three views per turn, encoded as one batch.
        assert all(len(s.evidence)==5 and {"a","b","f"}<={k.passage_id for k in s.retrieval_priority} for s in selections)
        assert all("Windows" in q and "Linux" not in q for q in scoring[3:])
        state=store.get_snapshot("s")
        for candidate in build_candidates(state,"More detail?",selections[-1].evidence,
                retrieval_priority=selections[-1].retrieval_priority,query_context=selections[-1].query_context):
            assert {b.key for b in selections[-1].evidence} == candidate.required_blocks <= candidate.visible_blocks
        await service.close_session("s")
        reopened=SessionStore(path)
        try:
            restored=reopened.get_snapshot("s")
        finally:
            reopened.close()
        assert restored.raw_history==state.raw_history and restored.archive==state.archive
        frame=restored.raw_history[-1].query_context.task_frame
        assert frame.requirements == ("download package","installation procedure")
        assert dict(frame.conditions)=={"version":"v4","platform":"Windows"}
    finally:
        wrapper.close()
        await service.aclose()
        store.close()


def test_serving_retrieval_projects_committed_task_into_encoder_and_reranker(retriever_config):
    from adapters.serving_retrieval import ServingRetriever
    from controller.schema import QueryContext, SessionSnapshot, TurnRecord
    value = yaml.safe_load(retriever_config.read_text())
    value.update(query_mode="conversational", query_fusion="single")
    retriever_config.write_text(yaml.safe_dump(value))
    retriever = create_retriever("P3", config_path=retriever_config)
    wrapper = ServingRetriever(retriever, corpus_id="fixture")
    task = QueryContext("Atlas v2 region B graduate application?", "correction", (1,))
    snapshot = SessionSnapshot("s", raw_history=(
        TurnRecord(1, "Atlas v1 region A student application?", "Actual first answer."),
        TurnRecord(2, "No, graduates in region B using v2.", "Actual corrected answer.", query_context=task),
        TurnRecord(3, "What documents?", "Actual third answer.")))
    try:
        selected = wrapper._retrieve(snapshot, "And the deadline?", "next-request", time.monotonic())
        query = task.query + "\nWhat documents?\nAnd the deadline?"
        assert retriever.encoder.inputs == retriever.reranker.inputs == [query]
        assert len(selected.evidence) == 5 and selected.query_context is None
        assert retriever.manifest["query_view_traces"][0]["history_view_turns"] == (2, 3)
        assert snapshot.raw_history[1].query_context == task
    finally:
        wrapper.close()


def test_external_trace_sink_does_not_retain_service_history_or_change_ranking(retriever_config):
    value=yaml.safe_load(retriever_config.read_text())
    value['query_mode']='user_topic'
    retriever_config.write_text(yaml.safe_dump(value))
    traces=[]
    service=create_retriever('P3',config_path=retriever_config,trace_sink=traces.append)
    benchmark=create_retriever('P3',config_path=retriever_config)
    gate=OnlineSession(load_rows(FIXTURE)[:3],'P3')
    while not gate.done:
        turn=gate.current()
        assert service.retrieve_turn(turn).contexts==benchmark.retrieve_turn(turn).contexts
        gate.commit(turn.task_id,'actual previous answer')
    assert len(traces)==3 and service.query_view_traces==[]
    assert 'query_view_traces' not in service.manifest
    assert service.manifest['query_view_trace_storage']=='external_sink'
    assert len(benchmark.manifest['query_view_traces'])==3
    assert 'query_view_trace_storage' not in benchmark.manifest
    service.close()
    benchmark.close()


@pytest.mark.parametrize('failure',['reranker','hybrid'])
def test_failed_factory_closes_only_successfully_constructed_owned_models(retriever_config,monkeypatch,failure):
    from adapters import hybrid_retrieval
    def model(path,*,reranker=False,**kwargs):
        if failure=='reranker' and reranker:
            raise RuntimeError('injected reranker initialization failure')
        return LocalModelFixture(path,reranker=reranker,**kwargs)
    monkeypatch.setattr(hybrid_retrieval,'LocalBGE',model)
    if failure=='hybrid':
        def fail(*args,**kwargs):
            raise RuntimeError('injected hybrid initialization failure')
        monkeypatch.setattr(hybrid_retrieval,'HybridRetriever',fail)
    with pytest.raises(RuntimeError,match='initialization failure'):
        create_retriever('P3',config_path=retriever_config)
    assert len(LocalModelFixture.instances)==(1 if failure=='reranker' else 2)
    assert all(model.closed for model in LocalModelFixture.instances)


def test_factory_closes_selector_client_if_resolver_initialization_fails(retriever_config,monkeypatch):
    from adapters import query_resolution, query_selector
    value=yaml.safe_load(retriever_config.read_text())
    value.update(query_mode='source_topic',query_fusion='current_topic_rrf',
                 selector={'base_url':'http://fake','model':'fake','revision':'fixed'})
    retriever_config.write_text(yaml.safe_dump(value))
    closed=[]
    class Client:
        def __init__(self,config):
            pass
        def close(self):
            closed.append(True)
    def fail(*args,**kwargs):
        raise RuntimeError('resolver construction failed')
    monkeypatch.setattr(query_selector,'OpenAICompletionClient',Client)
    monkeypatch.setattr(query_resolution,'TopicResolver',fail)
    with pytest.raises(RuntimeError,match='resolver construction failed'):
        create_retriever('P3',config_path=retriever_config)
    assert closed==[True] and all(model.closed for model in LocalModelFixture.instances)


def test_external_trace_sink_retains_retrieval_failure_without_swallowing_it(retriever_config,monkeypatch):
    value=yaml.safe_load(retriever_config.read_text())
    value['query_mode']='user_topic'
    retriever_config.write_text(yaml.safe_dump(value))
    traces=[]
    retriever=create_retriever('P3',config_path=retriever_config,trace_sink=traces.append)
    def fail(*args,**kwargs):
        raise RuntimeError('ranker failed')
    monkeypatch.setattr(retriever.reranker,'score',fail)
    turn=OnlineSession(load_rows(FIXTURE)[:3],'P3').current()
    with pytest.raises(RuntimeError,match='ranker failed'):
        retriever.retrieve_turn(turn)
    assert traces[0]['retrieval_status']=='failed' and 'ranker failed' in traces[0]['retrieval_error']
    assert retriever.query_view_traces==[]
    retriever.close()


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
    with pytest.raises(ValueError, match="query_mode"):
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


def test_optional_enum_schema_is_wired_once_and_rejected_on_unused_modes(retriever_config, monkeypatch):
    from adapters import query_selector
    class NoNetworkClient:
        manifest = {"model": "fake"}
        def __init__(self, config):
            self.config = config
        def close(self):
            pass
    monkeypatch.setattr(query_selector, "OpenAICompletionClient", NoNetworkClient)
    value = yaml.safe_load(retriever_config.read_text())
    value.update(query_mode="source_topic", query_fusion="current_topic_rrf", selector_schema_mode="enum_v2",
                 selector={"base_url": "http://fake", "model": "fake", "revision": "fixed"})
    retriever_config.write_text(yaml.safe_dump(value))
    retriever = create_retriever("P3", config_path=retriever_config)
    assert retriever.resolver.schema_mode == "enum_v2"
    assert retriever.resolver.manifest["version"] == "source-topic-selector-enum-v2"
    value.update(query_mode="current", query_fusion="single", selector=None)
    retriever_config.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="only used"):
        load_retriever_config(retriever_config)


@pytest.mark.parametrize("selector_failed", [False, True])
@pytest.mark.parametrize("failed_stage", ["encode", "score"])
def test_query_resolution_trace_survives_downstream_retrieval_failure(
        retriever_config, monkeypatch, selector_failed, failed_stage):
    from adapters import query_selector
    from adapters.mtrag import OnlineMessage
    from adapters.query_resolution import PriorContext
    from adapters.query_selector import CompletionResult

    raw = {"error": "selector unavailable"} if selector_failed else {"raw": "completed selection"}

    class Selector:
        manifest = {"model": "fake-selector"}

        def __init__(self, config):
            pass

        def complete(self, messages, **kwargs):
            return CompletionResult("failed" if selector_failed else "completed",
                json.dumps({"status": "unchanged", "entity_ids": [], "user_turn_ids": []}),
                None if selector_failed else "stop", raw_response=raw, elapsed_ms=7.5)

        def close(self):
            pass

    monkeypatch.setattr(query_selector, "OpenAICompletionClient", Selector)
    value = yaml.safe_load(retriever_config.read_text())
    value.update(query_mode="source_topic", query_fusion="current_topic_rrf",
                 selector={"base_url": "http://fake", "model": "fake", "revision": "fixed"})
    retriever_config.write_text(yaml.safe_dump(value))
    retriever = create_retriever("P3", config_path=retriever_config)
    original = OnlineSession(load_rows(FIXTURE)[:3], "P3").current()
    turn = replace(original, turn=2, history=(OnlineMessage("user", "Earlier question"),
                                            OnlineMessage("assistant", "Earlier actual answer")))
    error = RuntimeError("original model failure")

    def fail(*args, **kwargs):
        raise error

    model = retriever.encoder if failed_stage == "encode" else retriever.reranker
    monkeypatch.setattr(model, failed_stage, fail)
    try:
        with pytest.raises(RuntimeError) as observed:
            retriever.retrieve_turn(turn, prior_context=PriorContext())
        assert observed.value is error
        assert retriever.requests == 0
        trace, = retriever.manifest["query_view_traces"]
        assert trace["completion"]["raw_response"] == raw
        assert trace["completion"]["elapsed_ms"] == 7.5
        assert trace["status"] == ("failed" if selector_failed else "unchanged")
        assert trace["text"] == turn.query
        assert trace["retrieval_status"] == "failed"
        assert trace["retrieval_error"] == "RuntimeError: original model failure"
        assert trace["retrieval_elapsed_ms"] >= trace["elapsed_ms"] >= 0
        assert "query_fusion_trace" not in trace
    finally:
        retriever.close()


@pytest.mark.parametrize("fusion", ["single", "current_topic_rrf"])
def test_optional_user_topic_query_is_shared_with_encoder_reranker_and_manifest(retriever_config, fusion):
    from adapters.mtrag import OnlineMessage
    value = yaml.safe_load(retriever_config.read_text())
    value["query_mode"] = "user_topic"
    value["query_fusion"] = fusion
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
    assert retriever.encoder.inputs == ([turn.query, expected] if fusion == "current_topic_rrf" else [expected])
    assert retriever.reranker.inputs == [expected]
    assert output.history == turn.history and output.query == turn.query
    assert retriever.manifest["query_view"]["extra_model_calls"] == 0
    trace = retriever.manifest["query_view_traces"][0]
    assert trace["task_id"] == turn.task_id and trace["source_user_turns"] == [1, 3]
    assert trace["retrieval_status"] == "completed" and trace["retrieval_error"] is None
    assert trace["retrieval_elapsed_ms"] > 0
    assert trace["selected_history_spans"][0]["text"] == "Acme Store"
    assert retriever.manifest["query_fusion"] == fusion
    assert retriever.manifest["configuration"]["resolved"]["query_fusion"] == fusion
    if fusion == "current_topic_rrf":
        assert trace["query_fusion_trace"]["retrieval_query_count"] == 2
        assert trace["query_fusion_trace"]["rerank_pairs"] == 6
        assert "Misleading" not in json.dumps(trace["query_fusion_trace"])


@pytest.mark.parametrize("query_mode,fusion,rerank_mode", [
    ("user_topic", "single", "current"), ("user_topic", "current_topic_rrf", "current"),
    ("current", "single", "user_topic"), ("user_topic", "single", "resolved")])
def test_factory_separates_recall_and_rerank_views_without_changing_generation_input(
        retriever_config, query_mode, fusion, rerank_mode):
    from adapters.mtrag import OnlineMessage
    value = yaml.safe_load(retriever_config.read_text())
    value.update(query_mode=query_mode, query_fusion=fusion, rerank_query_mode=rerank_mode)
    retriever_config.write_text(yaml.safe_dump(value))
    retriever = create_retriever("P3", config_path=retriever_config)
    original = OnlineSession(load_rows(FIXTURE)[:3], "P3").current()
    turn = replace(original, query="One last thing, where is the python SDK?", turn=2,
        history=(OnlineMessage("user", "How do I use Acme Store?"),
                 OnlineMessage("assistant", "WrongProduct has an invented 100 MB limit")))
    topic = turn.query+"\nUser topic context: Acme Store"
    try:
        output = retriever.retrieve_turn(turn)
        assert output.query == turn.query and output.history == turn.history
        expected_recall = topic if query_mode == "user_topic" else turn.query
        expected_scoring = turn.query if rerank_mode == "current" else topic
        assert retriever.encoder.inputs == ([turn.query, topic] if fusion == "current_topic_rrf" else [expected_recall])
        assert retriever.reranker.inputs == [expected_scoring]
        trace, = retriever.manifest["query_view_traces"]
        assert trace["text"] == expected_recall and trace["rerank_query"] == expected_scoring
        assert trace["rerank_query_mode"] == retriever.manifest["rerank_query_mode"] == rerank_mode
        assert "WrongProduct" not in expected_scoring
        assert len(output.contexts) == 5
    finally:
        retriever.close()


@pytest.mark.parametrize("stage,mode", [("dense", "current"), ("hybrid", "user_topic"),
                                       ("hybrid_rerank", "not-a-mode")])
def test_invalid_rerank_query_policy_fails_before_loading_models(retriever_config, stage, mode):
    value = yaml.safe_load(retriever_config.read_text())
    value.update(stage=stage, rerank_query_mode=mode)
    retriever_config.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="rerank"):
        create_retriever("P3", config_path=retriever_config)
    assert LocalModelFixture.instances == []


@pytest.mark.parametrize("mode,stage", [("current", "hybrid_rerank"), ("user_topic", "dense")])
def test_query_fusion_configuration_rejects_ambiguous_stages_before_model_loading(retriever_config, mode, stage):
    value = yaml.safe_load(retriever_config.read_text())
    value.update(query_fusion="current_topic_rrf", query_mode=mode, stage=stage)
    retriever_config.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="requires user_topic.* and hybrid/hybrid_rerank"):
        create_retriever("P3", config_path=retriever_config)
    assert not LocalModelFixture.instances


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


def test_offline_cli_dual_uses_shared_models_then_opens_qrels(retriever_config, tmp_path, monkeypatch):
    from bench import retrieval_eval
    monkeypatch.setattr(retrieval_eval, "LocalBGE", LocalModelFixture)
    config = load_retriever_config(retriever_config)
    split = tmp_path/"split.json"
    split.write_text(json.dumps({"dev": ["session-a"]}))
    qrels = tmp_path/"qrels.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\nsession-a<::>1\ta\t1\n")
    original_qrels = retrieval_eval.load_qrels

    def after_ranking(path):
        assert LocalModelFixture.instances[0].texts_processed == 3
        assert LocalModelFixture.instances[1].texts_processed == 18
        return original_qrels(path)

    monkeypatch.setattr(retrieval_eval, "load_qrels", after_ranking)
    output = tmp_path/"offline-dual"
    retrieval_eval.main(["evaluate", "--input", str(FIXTURE), "--split-manifest", str(split),
        "--sparse-index", config.sparse_index, "--dense-index", config.dense_index,
        "--encoder-path", "encoder", "--reranker-path", "reranker", "--qrels", str(qrels),
        "--query-modes", "user_topic", "--query-fusion", "current_topic_rrf", "--output", str(output)])
    manifest = json.loads((output/"manifest.json").read_text())
    rows = [json.loads(line) for line in (output/"retrieval.jsonl").read_text().splitlines()]
    assert manifest["status"] == "complete" and manifest["config"]["query_fusion"] == "current_topic_rrf"
    assert len(rows) == 6 and {r["stage"] for r in rows} == {"hybrid", "hybrid_rerank"}
    assert all(r["query_fusion_trace"]["duplicate_query_skipped"] for r in rows)


@pytest.mark.parametrize("query_mode,fusion", [("current", "single"), ("user_topic", "single"),
                                               ("user_topic", "current_topic_rrf"),
                                               ("source_topic", "current_topic_rrf"),
                                               ("assistant_topic_unverified", "current_topic_rrf")])
def test_both_real_cli_paths_share_retriever_and_persist_generated_history(retriever_config, tmp_path, monkeypatch, query_mode, fusion, local_scope):
    import bench.native_baseline as native_module
    import controller.cli as cli
    from controller.templates import ByteTemplate

    value = yaml.safe_load(retriever_config.read_text())
    value["query_mode"] = query_mode
    value["query_fusion"] = fusion
    window = query_mode == "current"
    value["source_window"] = window
    selectors = []
    if query_mode in {"source_topic", "assistant_topic_unverified"}:
        from adapters import query_selector
        from adapters.query_selector import CompletionResult
        class Selector:
            manifest = {"model": "fake-selector", "revision": "fixture"}
            def __init__(self, config):
                self.inputs, self.closed = [], False
                selectors.append(self)
            def complete(self, messages, **kwargs):
                self.inputs.append(json.loads(messages[-1]["content"]))
                return CompletionResult("completed", json.dumps({"status": "unchanged", "entity_ids": [],
                                                                  "user_turn_ids": []}), "stop")
            def close(self):
                self.closed = True
        monkeypatch.setattr(query_selector, "OpenAICompletionClient", Selector)
        value["selector"] = {"base_url": "http://fake", "model": "fake", "revision": "fixed"}
    retriever_config.write_text(yaml.safe_dump(value))

    def respond(request):
        if request.url.path == "/server_info":
            return httpx.Response(200, json={"version": "fake-http-correctness"})
        return httpx.Response(200, json={"text": "native actual", "output_ids": list(b"native actual"),
            "meta_info": {"finish_reason": {"type": "stop"}, "completion_tokens": len(b"native actual")}})
    http = httpx.AsyncClient(base_url="http://fake.test", transport=httpx.MockTransport(respond))
    client_class, baseline_class = native_module.NativeSGLangClient, native_module.NativeBaseline
    monkeypatch.setattr("controller.config.create_model_template", lambda *a, **k: ByteTemplate())
    monkeypatch.setattr(native_module, "NativeSGLangClient", lambda *a, **k:
                        client_class("http://fake.test", http_client=http, stream=False))
    monkeypatch.setattr(native_module, "NativeBaseline", lambda *a, **k:
                        baseline_class(*a, **k, test_mode=True, run_id="hybrid-test"))
    native_dir, controller_dir = tmp_path/"native", tmp_path/"controller"
    engine = tmp_path/"engine.json"
    engine.write_text("{}")
    run_config = tmp_path/"native-config.yaml"
    run_config.write_text(yaml.safe_dump({"model":{"scope_path":str(local_scope.path)}}))
    common = [str(FIXTURE), "--protocol", "P3", "--domain", "all", "--limit-sessions", "1",
              "--retriever-config", str(retriever_config)]
    if window:
        corpus = tmp_path/"metadata.jsonl"
        rows = [dict(row, document_id="manual", passage_id=pid, span=[i*100, (i+1)*100],
                     source="https://example.org/manual")
                for i, (pid, row) in enumerate(sorted(load_beir_corpus(FIXTURE.with_name("corpus_small.jsonl")).items()))]
        corpus.write_text("".join(json.dumps(row)+"\n" for row in rows))
        common += ["--evidence-corpus", str(corpus)]
    try:
        native_module.main(common+["--baseline", "B0", "--run-config",str(run_config),"--engine-manifest", str(engine), "--output", str(native_dir)])
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
        if window:
            assert all([c["document_id"] for c in t["online_turn"]["supporting_contexts"]] == ["a"] for t in turns)
        assert turns[1]["online_turn"]["history"][-1]["text"] == turns[0]["assistant_text"]
        assert "REFERENCE ANSWER" not in json.dumps(turns)
    assert native["retrieval"]["completed_retrievals"] == controller_manifest["retrieval"]["completed_retrievals"] == 3
    if selectors:
        assert len(selectors) == 2 and all(s.closed and len(s.inputs) == 2 for s in selectors)
        for selector in selectors:
            assert [len(body["actual_history"]) for body in selector.inputs] == [2, 4]
            assert "REFERENCE ANSWER" not in json.dumps(selector.inputs)
    for manifest in (native["retrieval"], controller_manifest["retrieval"]):
        assert manifest["query_mode"] == query_mode
        assert manifest["query_fusion"] == fusion
        if query_mode == "user_topic":
            assert len(manifest["query_view_traces"]) == 3
            assert all("actual" not in v["text"] for v in manifest["query_view_traces"])
            if fusion == "current_topic_rrf":
                assert all(v["query_fusion_trace"]["duplicate_query_skipped"] for v in manifest["query_view_traces"])
                assert manifest["model_usage"]["encoder"]["texts_processed"] == 3
        if query_mode in {"source_topic", "assistant_topic_unverified"}:
            traces = manifest["query_view_traces"]
            assert [t["snapshot_version"] for t in traces] == [None, 1, 2]
            assert [t["model_called"] for t in traces] == [False, True, True]
            assert all(t["missing_source_keys"] == [] for t in traces)
            assert all(t["query_fusion_trace"]["duplicate_query_skipped"] for t in traces)


def test_factory_uses_original_document_windows_with_existing_serving_selection(retriever_config, tmp_path):
    from adapters.corpus import CorpusIndex
    from adapters.serving_retrieval import ServingRetriever
    from controller.schema import SessionSnapshot
    import asyncio
    document=tmp_path/'original.jsonl'
    config=yaml.safe_load(retriever_config.read_text())
    config.update(source_window=True,source_documents=str(document))
    retriever_config.write_text(yaml.safe_dump(config))
    originals=load_beir_corpus(FIXTURE.with_name("corpus_small.jsonl"))
    documents=[{'document_id':pid,'text':f'Only for source {pid}.\n\n'+row['text']+'\n\nEnd.',
                'url':row['source'],'title':row['title'],'snapshot_id':'fixture-v1'} for pid,row in originals.items()]
    document.write_text(''.join(json.dumps(row)+'\n' for row in documents))
    retriever=create_retriever('P3',config_path=retriever_config)
    rows=[{'passage_id':pid,'document_id':pid,'text':row['text'],'title':row['title'],
           'source':row['source'],'span':[0,len(row['text'])]} for pid,row in originals.items()]
    document_text={row['document_id']:row['text'] for row in documents}
    corpus=CorpusIndex.from_records(rows,corpus_id='c',snapshot_id=retriever.corpus_snapshot_id)
    wrapper=ServingRetriever(retriever,corpus_id='c',corpus_index=corpus)
    try:
        selected=asyncio.run(wrapper.retrieve(SessionSnapshot('s'),'query','q'))
        assert len(selected.retrieval_priority)==5
        windows=[b for b in selected.evidence if b.key.passage_id.startswith('source-window:')]
        assert windows and all(b.text==document_text[b.key.document_id][b.span[0]:b.span[1]] for b in windows)
        assert retriever.manifest['source_window']['mode']=='original_document_paragraphs'
    finally:
        wrapper.close()


@pytest.mark.asyncio
async def test_joint_conversational_turn_keeps_state_and_uses_one_call_per_turn(retriever_config,tmp_path,monkeypatch):
    from adapters import query_selector
    from adapters.conversation import frame_query
    from adapters.corpus import CorpusIndex
    from adapters.query_selector import CompletionResult
    from adapters.serving_retrieval import ServingRetriever
    from controller.mock_engine import MockEngine
    from controller.schema import QueryContext,TaskFrame
    from controller.service import Controller
    from controller.session_store import SessionStore
    calls=[]
    class Client:
        manifest={'model':'fixture'}
        def __init__(self,config):
            self.config=config
        def catalog_token_lengths(self,messages,documents):
            return 20,[len(d) for d in documents]
        def complete(self,messages,**kwargs):
            body=json.loads(messages[-1]['content'])
            calls.append(body)
            social=body['current_question']=="Thanks, that's all."
            value={'relation':'conversational' if social else 'condition_change',
                'history_turns':[] if social else [2],'resume_turn':0,
                'task':{'operation':'keep','value':'','requirements':[]},
                'updates':{} if social else {'conditions':{'platform':'Windows'}},
                'selected_ids':[] if social else [body['documents'][0]['id']]}
            return CompletionResult('completed',json.dumps(value),'stop')
        def close(self):
            pass
    monkeypatch.setattr(query_selector,'OpenAICompletionClient',Client)
    config=yaml.safe_load(retriever_config.read_text())
    config.update(query_mode='conversational',evidence_selection='semantic',source_window=True,
                  selector={'base_url':'http://fixture','model':'fixture','revision':'fixed'})
    retriever_config.write_text(yaml.safe_dump(config))
    retriever=create_retriever('P3',config_path=retriever_config)
    rows=load_beir_corpus(FIXTURE.with_name('corpus_small.jsonl'))
    corpus=CorpusIndex.from_records(({'_id':pid,**r} for pid,r in rows.items()),corpus_id='fixture',snapshot_id='fixture-v1')
    serving=ServingRetriever(retriever,corpus_id='fixture',corpus_index=corpus)
    store=SessionStore(tmp_path/'dialogue.sqlite3')
    service=Controller(store,MockEngine())
    frame=TaskFrame('Atlas SDK','installer','installation',(('version','v4'),('platform','Linux')))
    intent=QueryContext(frame_query(frame),'standalone',task_frame=frame)
    try:
        await service.serve_turn('s','Install Atlas SDK v4 on Linux.',corpus.resolve(('a',)),
                                 query_context=intent,request_id='first',corpus_snapshot_id='fixture-v1')
        initial=store.get_snapshot('s')
        async def acknowledge(snapshot):
            return await serving.retrieve(snapshot,"Thanks, that's all.",'thanks')
        response=await service.serve_turn('s',"Thanks, that's all.",evidence_provider=acknowledge,
                                         request_id='thanks',corpus_snapshot_id='fixture-v1')
        replay=await service.serve_turn('s',"Thanks, that's all.",evidence_provider=acknowledge,
                                       request_id='thanks',corpus_snapshot_id='fixture-v1')
        social=store.get_snapshot('s')
        assert replay.idempotent_replay and replay.output_ids==response.output_ids and len(calls)==1
        assert retriever.encoder.texts_processed>0  # Source recall precedes the joint decision.
        assert social.raw_history[:-1]==initial.raw_history and social.raw_history[-1].query_context.task_frame==frame
        assert social.raw_history[-1].evidence_keys==() and response.citation_scope==()
        assert retriever.query_view_traces[-1]['retrieval_status']=='completed_conversational'
        current=next(m.content for m in reversed(social.execution.messages) if m.role=='user')
        assert 'not a new factual lookup' in current
        async def resume(snapshot):
            return await serving.retrieve(snapshot,'What changes on Windows?','resume')
        await service.serve_turn('s','What changes on Windows?',evidence_provider=resume,
                                 request_id='resume',corpus_snapshot_id='fixture-v1')
        final=store.get_snapshot('s')
        assert dict(final.raw_history[-1].query_context.task_frame.conditions)=={'version':'v4','platform':'Windows'}
        assert calls[1]['previous_turn']['answer']==social.raw_history[-1].assistant_text
        assert calls[1]['previous_turn']['state']['topic']=='Atlas SDK'
        assert retriever.encoder.texts_processed>0 and retriever.requests==2
        assert len(calls)==2  # One joint interpretation/selection per newly committed turn.
        await service.close_session('s')
        assert not store.outbox()
    finally:
        await service.aclose()
        store.close()
        serving.close()
