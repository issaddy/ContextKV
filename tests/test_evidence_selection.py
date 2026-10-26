from dataclasses import replace
import json
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from adapters.corpus import CorpusIndex
from adapters.evidence_selection import EvidenceSelector
from adapters.mtrag import OnlineContext, OnlineTurn
from adapters.query_resolution import PriorContext
from adapters.query_selector import CompletionResult
from controller.candidates import build_candidates
from controller.mock_engine import MockEngine
from controller.schema import QueryContext, TaskFrame
from controller.service import Controller
from controller.session_store import SessionStore


class Client:
    def __init__(self, *, budget=10000, invalid=False):
        self.config=SimpleNamespace(max_input_tokens=budget)
        self.calls=[]
        self.invalid=invalid
    def catalog_token_lengths(self, messages, documents):
        return 20,[len(d) for d in documents]
    def complete(self, messages, **kwargs):
        body=json.loads(messages[-1]['content'])
        self.calls.append((body,kwargs))
        uses=[]
        for doc in body['documents']:
            for line,text in doc['lines']:
                if 'Windows v4 private installer' in text:
                    uses.append({'source':doc['id'],'needs':[0],'line':line})
                if text=='Run the installer.':
                    uses.append({'source':doc['id'],'needs':[1],'line':line})
        if self.invalid:
            uses[0]['source']=0  # A valid line ID attributed to a different source.
        return CompletionResult('completed',json.dumps({'uses':uses}),'stop',100)


def fixture_data():
    rows=[{'_id':'latest','text':'Public latest version only.'},
          {'_id':'package','text':'Linux package\nWindows v4 private installer: ORIGINAL_URL\nOther package'},
          {'_id':'steps','text':'Installer steps\nRun the installer.\nSeparate binary procedure.'}]
    corpus=CorpusIndex.from_records(rows,corpus_id='c',snapshot_id='v4')
    contexts=tuple(OnlineContext(r['_id'],r['text']) for r in rows)
    intent=QueryContext('Windows v4 private package and installation','standalone',task_frame=TaskFrame(
        'Atlas','','package and installation',(('version','v4'),('platform','Windows'),('source','private')),
        ('download package','installation procedure')))
    turn=OnlineTurn('s','q',1,'c',intent.query,(),(),'P3')
    return corpus,contexts,intent,turn


def select(client):
    corpus,contexts,intent,turn=fixture_data()
    result=EvidenceSelector(client).select(turn,intent,contexts,contexts[:1],neighbors=lambda hits:(),
        indexed={c.document_id:c for c in contexts},contextualize=corpus.contextualize,keys=corpus.retrieved_keys)
    return corpus,contexts,intent,result


@pytest.mark.asyncio
async def test_semantic_sources_and_exact_reading_guides_reach_both_candidates_and_replay(tmp_path):
    client=Client()
    corpus,original,intent,(contexts,planned,trace)=select(client)
    assert [c.document_id for c in contexts]==['package','steps']
    assert trace['status']=='selected' and len(client.calls)==1
    assert planned.task_frame==intent.task_frame and len(planned.evidence_uses)==2
    blocks=corpus.enrich(contexts)
    for use,block in zip(planned.evidence_uses,blocks):
        assert use.key==block.key and block.text[use.start:use.end] in ('Windows v4 private installer: ORIGINAL_URL','Run the installer.')
    store=SessionStore(tmp_path/'state.sqlite3')
    service=Controller(store,MockEngine())
    try:
        response=await service.serve_turn('s','Question',blocks,query_context=planned,request_id='one',corpus_snapshot_id='v4')
        snapshot=store.get_snapshot('s')
        assert snapshot.raw_history[-1].query_context==planned
        for candidate in build_candidates(snapshot,'Continue',blocks,query_context=planned,
                                           retrieval_priority=tuple(b.key for b in blocks)):
            assert candidate.required_blocks=={b.key for b in blocks} and candidate.required_blocks<=candidate.visible_blocks
            guide=json.loads(candidate.messages[-1].content.split('Current source guide:',1)[1].split('\n',1)[1].split('\n',1)[0])
            assert [row['for_needs'] for row in guide]==[['download package'],['installation procedure']]
            assert [row['excerpt'] for row in guide]==['Windows v4 private installer: ORIGINAL_URL','Run the installer.']
        replay=await service.serve_turn('s','Question',blocks,query_context=planned,request_id='one',corpus_snapshot_id='v4')
        assert replay.idempotent_replay and replay.output_ids==response.output_ids
        restored=SessionStore(tmp_path/'state.sqlite3')
        try:
            assert restored.get_snapshot('s').raw_history==snapshot.raw_history
        finally:
            restored.close()
        await service.close_session('s')
        assert not store.outbox()
    finally:
        await service.aclose()
        store.close()


def test_wrong_source_line_assignment_falls_back_without_inventing_a_quote():
    client=Client(invalid=True)
    _,original,intent,(contexts,planned,trace)=select(client)
    assert trace['status']=='failed' and contexts==original[:1] and planned==intent
    assert not planned.evidence_uses and len(client.calls)==1


def test_catalog_budget_does_not_cut_original_bodies_or_retry():
    client=Client(budget=21)
    _,original,intent,(contexts,planned,trace)=select(client)
    assert contexts==original[:1] and planned==intent
    assert trace['catalog_count']==0 and not client.calls and not trace['model_called']


def test_semantic_http_wait_does_not_hold_the_local_ranking_lock():
    from adapters.hybrid_retrieval import HybridRetriever
    entered,release=Event(),Event()
    corpus,contexts,intent,turn=fixture_data()
    class Resolver:
        def resolve(self,turn,context):
            from dataclasses import asdict
            return {'text':turn.query,'context':asdict(replace(intent,task_frame=TaskFrame(task=turn.query)))}
    class Selector:
        def select(self,turn,intent,ranked,fallback,**kwargs):
            if turn.query=='slow':
                entered.set()
                assert release.wait(5)
            return fallback,intent,{'status':'selected'}
    dense=SimpleNamespace(manifest={'corpus_sha256':'same'},by_id={c.document_id:c for c in contexts},
        search=lambda *args,**kwargs:contexts)
    sparse=SimpleNamespace(manifest=dense.manifest,corpus_snapshot_id='v4')
    retriever=HybridRetriever(sparse,dense,object(),stage='dense',query_mode='conversational',
                              resolver=Resolver(),evidence_selector=Selector())
    retriever.bind_corpus(corpus)
    with ThreadPoolExecutor(max_workers=2) as pool:
        slow=pool.submit(retriever.retrieve_turn,replace(turn,query='slow'),prior_context=PriorContext())
        try:
            assert entered.wait(5)
            fast=pool.submit(retriever.retrieve_turn,replace(turn,query='fast'),prior_context=PriorContext())
            assert fast.result(timeout=2).contexts==contexts
        finally:
            release.set()
        assert slow.result(timeout=2).contexts==contexts
