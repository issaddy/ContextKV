from dataclasses import replace
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from adapters.corpus import CorpusIndex
from adapters.mtrag import OnlineContext
from adapters.retrieval import BM25SRetriever
from adapters.retriever_factory import create_retriever
from adapters.serving_retrieval import ServingRetriever, ServingRetrievalError
from controller.events import EventLog
from controller.evidence import make_evidence
from controller.schema import SessionSnapshot, TurnRecord


def event_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


@pytest.fixture
def real_index(tmp_path):
    rows=[{'_id':f'p{i}', 'title':f'ORCHID rule {i}', 'text':f'ORCHID request timeout rule {i}.',
           'metadata':{'document_id':'orchid-manual','version':'r2','dependencies':['scope'] if i==0 else []}}
          for i in range(5)]
    rows.append({'_id':'scope','title':'Applicable scope','text':'Only unattended batch operations in release r2.',
                 'metadata':{'document_id':'orchid-manual','version':'r2'}})
    corpus={r['_id']:{'document_id':r['_id'],'text':r['text'],'title':r['title']} for r in rows}
    sparse=BM25SRetriever(corpus,corpus_snapshot_id='http-v1')
    sparse.save(tmp_path/'index')
    loaded=create_retriever('P3',index_path=tmp_path/'index')
    metadata=CorpusIndex.from_records(rows,corpus_id='manual',snapshot_id='http-v1')
    return loaded,metadata


@pytest.mark.asyncio
async def test_real_saved_bm25_ranking_keeps_five_originals_and_metadata_closure(real_index,tmp_path):
    retriever,index=real_index
    events=tmp_path/'events.jsonl'
    serving=ServingRetriever(retriever,corpus_id='manual',corpus_index=index,events=EventLog(events))
    snapshot=SessionSnapshot('s',corpus_snapshot_id='http-v1')
    selection=await serving.retrieve(snapshot,'ORCHID request timeout','r1')
    assert [k.passage_id for k in selection.retrieval_priority]==[f'p{i}' for i in range(5)]
    assert len(selection.evidence)==6 and selection.evidence[-1].key.passage_id=='scope'
    assert all(b.key.document_id=='orchid-manual' and b.key.version=='r2' for b in selection.evidence)
    entry=event_rows(events)[0]
    assert entry['event']=='retrieval_backend_completed' and len(entry['ranking'])==5
    assert len(entry['supplied_evidence_keys'])==6 and entry['retrieval_elapsed_ms']>=0
    assert serving.manifest['corpus_snapshot_id']=='http-v1' and serving.manifest['metadata_enabled']
    assert snapshot.raw_history==() and snapshot.archive==()
    serving.close()  # BM25 has no model resource close method.


@pytest.mark.parametrize('corpus,snapshot',[('other','http-v1'),('manual','another-snapshot')])
def test_metadata_identity_mismatch_rejected_at_configuration(real_index,corpus,snapshot):
    retriever,index=real_index
    other=CorpusIndex(index.blocks,corpus_id=corpus,snapshot_id=snapshot)
    with pytest.raises(ValueError,match='metadata corpus/snapshot'):
        ServingRetriever(retriever,corpus_id='manual',corpus_index=other)


@pytest.mark.asyncio
async def test_worker_uses_complete_committed_history_and_only_committed_source_inventory():
    source=make_evidence('ORCHID supports requests.','visible')
    unseen=make_evidence('ARCHIVE_ONLY must not qualify a topic.','unseen')
    snapshot=SessionSnapshot('real-session',version=9,corpus_snapshot_id='snapshot',
        raw_history=(TurnRecord(7,'Original user?','Actual assistant with \n Unicode 文本.',evidence_keys=(source.key,)),),
        archive=(source,unseen))
    observations=[]
    class Retriever:
        corpus_snapshot_id='snapshot'
        requires_history_context=True
        manifest={'kind':'cpu-fixture'}
        closed=False
        def retrieve_turn(self,turn,*,prior_context):
            observations.append((threading.get_ident(),turn,prior_context))
            return replace(turn,contexts=(OnlineContext('current','Current original body.'),))
        def close(self):
            self.closed=True
    inner=Retriever()
    serving=ServingRetriever(inner,corpus_id='manual')
    selected=await serving.retrieve(snapshot,'Follow-up?','actual-request')
    thread,turn,prior=observations[0]
    assert thread!=threading.get_ident() and turn.turn==2 and turn.task_id=='actual-request'
    assert turn.contexts==() and [m.text for m in turn.history]==['Original user?','Actual assistant with \n Unicode 文本.']
    assert turn.user_questions==('Original user?','Follow-up?')
    assert [s.block.key for s in prior.sources]==[source.key] and prior.sources[0].visible_turns==(1,)
    assert selected.evidence[0].key.corpus_id=='manual' and selected.evidence[0].key.version=='snapshot'
    serving.close()
    assert inner.closed and snapshot.raw_history[0].turn_id==7


@pytest.mark.asyncio
async def test_failure_keeps_original_cause_and_logs_request_without_generation(tmp_path):
    class Retriever:
        corpus_snapshot_id='snapshot'
        def retrieve_turn(self,turn):
            raise OSError('index unavailable')
    path=tmp_path/'events.jsonl'
    serving=ServingRetriever(Retriever(),corpus_id='manual',events=EventLog(path))
    with pytest.raises(ServingRetrievalError,match='index unavailable') as error:
        await serving.retrieve(SessionSnapshot('s'),'question','r2')
    assert isinstance(error.value.__cause__,OSError)
    assert error.value.code=='RETRIEVAL_FAILED'
    assert event_rows(path)[0]['event']=='retrieval_backend_failed'
    assert event_rows(path)[0]['request_id']=='r2'


@pytest.mark.asyncio
async def test_cancellation_is_not_relabelled_as_retrieval_failure():
    class Retriever:
        corpus_snapshot_id='snapshot'
        def retrieve_turn(self,turn):
            raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await ServingRetriever(Retriever(),corpus_id='manual').retrieve(SessionSnapshot('s'),'q','r')


@pytest.mark.asyncio
async def test_cancelled_worker_before_hybrid_lock_finishes_before_model_close(monkeypatch):
    from adapters import hybrid_retrieval

    entered, release = threading.Event(), threading.Event()
    actions = []

    class Encoder:
        closed = False

        def close(self):
            self.closed = True
            actions.append('close')

    class Dense:
        manifest = {'corpus_sha256': 'same-original-corpus'}

        def search(self, query, encoder, *, top_k):
            assert not encoder.closed, 'cancelled worker used a closed model'
            actions.append('search')
            return (OnlineContext('p', 'Original body.'),)

    def paused_query_view(questions):
        # This real Hybrid seam precedes its model lock; close cannot drain it
        # merely by acquiring that lock once.
        entered.set()
        assert release.wait(5), 'test did not release retrieval worker'
        return {'text': questions[-1]}

    monkeypatch.setattr(hybrid_retrieval, 'user_topic_view', paused_query_view)
    encoder = Encoder()
    sparse = SimpleNamespace(manifest=Dense.manifest, corpus_snapshot_id='snapshot')
    hybrid = hybrid_retrieval.HybridRetriever(sparse, Dense(), encoder,
                                             query_mode='user_topic', stage='dense')
    serving = ServingRetriever(hybrid, corpus_id='manual')
    task = asyncio.create_task(serving.retrieve(SessionSnapshot('s'), 'q', 'r'))

    async def cancel_then_shutdown():
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.to_thread(serving.close)

    shutdown = None
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        shutdown = asyncio.create_task(cancel_then_shutdown())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not task.done() and not shutdown.done() and not encoder.closed
    finally:
        release.set()
        if shutdown is not None:
            await shutdown
        else:
            await task
            serving.close()
    assert actions == ['search', 'close']


@pytest.mark.asyncio
async def test_metadata_cannot_replace_retrieved_original_body(real_index):
    retriever,index=real_index
    original=retriever.retrieve_turn
    def changed(turn):
        result=original(turn)
        return replace(result,contexts=(replace(result.contexts[0],text='rewritten body'),*result.contexts[1:]))
    retriever.retrieve_turn=changed
    serving=ServingRetriever(retriever,corpus_id='manual',corpus_index=index)
    with pytest.raises(ServingRetrievalError,match='exact retrieved body'):
        await serving.retrieve(SessionSnapshot('s'),'ORCHID request timeout','r')
