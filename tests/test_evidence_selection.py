from dataclasses import replace
import json
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from adapters.corpus import CorpusIndex
from adapters.conversation import ConversationResolver
from adapters.evidence_selection import EvidenceSelector, reading_catalog
from controller.schema import OnlineContext, OnlineTurn
from adapters.conversation import PriorContext
from adapters.query_selector import CompletionResult
from controller.candidates import build_candidates
from bench.mock_engine import MockEngine
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
        selected = [doc['id'] for doc in body['documents']
                    if 'Windows v4 private installer' in doc['text'] or 'Run the installer.' in doc['text']]
        if self.invalid:
            selected.append(len(body['documents']))
        value={'relation':'standalone','history_turns':[],'resume_turn':0,
            'frame':{'topic':'Atlas','target':'','task':'package and installation',
                     'conditions':{'version':'v4','platform':'Windows','source':'private'},
                     'requirements':['download package','installation procedure']},
            'selected_ids':selected}
        return CompletionResult('completed',json.dumps(value),'stop',100)


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
    catalog,_=reading_catalog(contexts,neighbors=lambda hits:(),
        indexed={c.document_id:c for c in contexts},contextualize=corpus.contextualize)
    result=EvidenceSelector(ConversationResolver(client)).select(turn,PriorContext(),catalog,contexts[:1],
        required_context=corpus.selection_support(catalog))
    return corpus,contexts,intent,result


@pytest.mark.asyncio
async def test_selected_original_sources_reach_both_candidates_and_replay(tmp_path):
    client=Client()
    corpus,original,intent,(contexts,view,trace)=select(client)
    assert [c.document_id for c in contexts]==['package','steps']
    assert trace['status']=='selected' and len(client.calls)==1
    supplied=client.calls[0][0]['documents']
    assert [d['text'] for d in supplied]==[c.text for c in original]
    blocks=corpus.enrich(contexts)
    store=SessionStore(tmp_path/'state.sqlite3')
    service=Controller(store,MockEngine())
    try:
        response=await service.serve_turn('s','Question',blocks,query_context=intent,request_id='one',corpus_snapshot_id='v4')
        snapshot=store.get_snapshot('s')
        assert snapshot.raw_history[-1].query_context==intent
        for candidate in build_candidates(snapshot,'Continue',blocks,query_context=intent,
                                           retrieval_priority=tuple(b.key for b in blocks)):
            assert candidate.required_blocks=={b.key for b in blocks} and candidate.required_blocks<=candidate.visible_blocks
            guide=json.loads(candidate.messages[-1].content.split('Current source guide:',1)[1].split('\n',1)[1].split('\n',1)[0])
            assert len(guide)==2 and all('excerpt' not in row for row in guide)
            for span in candidate.evidence_spans:
                block=next(b for b in blocks if b.key==span.key)
                assert candidate.messages[span.message_index].content[span.start:span.end]==block.text
        replay=await service.serve_turn('s','Question',blocks,query_context=intent,request_id='one',corpus_snapshot_id='v4')
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


def test_out_of_catalog_selection_falls_back_without_retry():
    client=Client(invalid=True)
    _,original,intent,(contexts,view,trace)=select(client)
    assert trace['status']=='failed' and contexts==original[:1]
    assert len(client.calls)==1


def test_catalog_budget_does_not_cut_original_bodies_or_retry():
    client=Client(budget=21)
    _,original,intent,(contexts,view,trace)=select(client)
    assert contexts==original[:1]
    assert trace['catalog_count']==0 and len(client.calls)==1
    assert client.calls[0][0]['documents']==[]  # Task interpretation still happens once.


def test_large_first_unit_does_not_hide_later_complete_applicable_material():
    _,contexts,intent,turn=fixture_data()
    catalog=(OnlineContext('large','Long unrelated source.'*1000),contexts[1])
    client=Client(budget=600)
    result,view,trace=EvidenceSelector(ConversationResolver(client)).select(turn,PriorContext(),catalog,(),required_context={})
    assert result==catalog[1:] and trace['catalog_ids']==['package']
    assert trace['budget_skipped_ids']==['large']
    body,schema=client.calls[0]
    assert body['documents'][0]['id']==1 and body['documents'][0]['text']==contexts[1].text
    assert schema['response_format']['json_schema']['schema']['properties']['selected_ids']['items']['enum']==[1]
    assert body['current_question']==turn.query and body['previous_turn'] is None
    assert dict(view['context']['task_frame']['conditions'])=={'version':'v4','platform':'Windows','source':'private'}


def test_selection_receives_declared_version_and_section_for_sources_and_dependencies():
    from adapters.hybrid_retrieval import HybridRetriever
    rows = [
        {'_id':'package','text':'Run the installer.','title':'Installation','version':'v4',
         'parent_section':'Windows offline','dependencies':['condition']},
        {'_id':'new-package','text':'Run the installer.','title':'Installation','version':'v5',
         'parent_section':'Windows online'},
        {'_id':'condition','text':'Private package required.','version':'v4',
         'parent_section':'Private distribution'},
        {'_id':'general','text':'General guidance.'},
    ]
    corpus = CorpusIndex.from_records(rows,corpus_id='manual',snapshot_id='index-snapshot')
    contexts = corpus.contextualize(tuple(OnlineContext(r['_id'],r['text']) for r in rows if r['_id']!='condition'))
    _,_,intent,turn = fixture_data()
    client = Client()
    dense = SimpleNamespace(manifest={'corpus_snapshot_id':'same'},by_id={c.document_id:c for c in contexts},
        search=lambda *args,**kwargs:contexts)
    sparse = SimpleNamespace(manifest=dense.manifest,corpus_snapshot_id=corpus.snapshot_id)
    resolver = ConversationResolver(client)
    retriever = HybridRetriever(sparse,dense,object(),stage='dense',query_mode='conversational',
        resolver=resolver,evidence_selector=EvidenceSelector(resolver))
    retriever.bind_corpus(corpus)
    retriever.retrieve_turn(turn,prior_context=PriorContext())
    documents = client.calls[0][0]['documents']
    assert [(d.get('version'),d.get('section')) for d in documents] == [
        ('v4','Windows offline'),('v5','Windows online'),(None,None)]
    assert documents[0]['text'] == documents[1]['text'] == 'Run the installer.'
    dependency, = documents[0]['required_context']
    assert dependency['version']=='v4' and dependency['section']=='Private distribution'
    assert dependency['text']=='Private package required.'
    assert all('index-snapshot' != d.get('version') for d in documents)
    assert all('content_sha256' not in d and 'provenance' not in d for d in documents)


def test_catalog_coalesces_identical_windows_without_chain_merging_other_choices():
    from adapters.source_documents import SourceDocuments
    text='\n\n'.join(f'Paragraph {i} original text.' for i in range(12))
    raw=tuple(OnlineContext(f'p{i}',f'Paragraph {i} original text.') for i in (2,4,6,8))
    raw+= (OnlineContext('p2-short','Paragraph 2'),)
    corpus=CorpusIndex.from_records([{'passage_id':c.document_id,'document_id':'doc',
        'span':[i,i+1],'text':c.text} for i,c in enumerate(raw)],corpus_id='c',snapshot_id='v1')
    documents=SourceDocuments([{'document_id':'doc','text':text,'snapshot_id':'v1','corpus_id':'c'}])
    catalog,origins=reading_catalog((*raw,raw[0]),neighbors=lambda _:(),
        indexed={c.document_id:c for c in raw},contextualize=corpus.contextualize,
        read=lambda contexts:corpus.reading_contexts(contexts,documents))
    assert len(catalog)==4
    assert all(len(c.text)<len(text) for c in catalog)
    assert set(origins)=={c.document_id for c in catalog}
    assert {hit.document_id for hits in origins.values() for hit in hits}=={c.document_id for c in raw}
    assert max(map(len,origins.values()))==2


def test_semantic_http_wait_does_not_hold_the_local_ranking_lock():
    from adapters.hybrid_retrieval import HybridRetriever
    entered,release=Event(),Event()
    corpus,contexts,intent,turn=fixture_data()
    class Resolver:
        def resolve(self,*args,**kwargs):
            raise AssertionError('joint source planning must not run a preliminary interpretation')
    class Selector:
        def select(self,turn,prior_context,ranked,fallback,**kwargs):
            if turn.query=='slow':
                entered.set()
                assert release.wait(5)
            from dataclasses import asdict
            view={'text':turn.query,'context':asdict(replace(intent,task_frame=TaskFrame(task=turn.query)))}
            return fallback,view,{'status':'selected'}
    dense=SimpleNamespace(manifest={'corpus_snapshot_id':'same'},by_id={c.document_id:c for c in contexts},
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


def test_joint_selection_reads_actual_dialogue_before_deciding_scope_and_accepts_partial_support():
    from controller.schema import OnlineMessage
    _,contexts,intent,request=fixture_data()
    history=tuple(m for n in range(1,8) for m in
        (OnlineMessage('user',f'Question {n}'),OnlineMessage('assistant',f'Actual answer {n}\nWith object Atlas.')))
    request=replace(request,turn=8,history=history,query='Back to the original Windows v4 package and installation.')
    class PartialClient(Client):
        def complete(self,messages,**kwargs):
            result=super().complete(messages,**kwargs)
            value=json.loads(result.text)
            value['relation']='revisit'
            value['resume_turn']=1
            value['history_turns']=[1]
            value['selected_ids']=[1]
            return replace(result,text=json.dumps(value))
    client=PartialClient()
    result,view,trace=EvidenceSelector(ConversationResolver(client)).select(
        request,PriorContext(interpretations=(None,)*7),contexts,contexts[:1],required_context={})
    assert result==(contexts[1],) and trace['status']=='selected' and len(client.calls)==1
    body=client.calls[0][0]
    supplied=[*body['earlier_turns'],body['previous_turn']]
    assert {1,5,6,7} <= {r['turn'] for r in supplied}
    assert all(r['question']==history[2*r['turn']-2].text and r['answer']==history[2*r['turn']-1].text for r in supplied)
    assert view['context']['relation']=='revisit' and view['context']['history_turns']==(1,)
    assert body['current_question']==request.query
