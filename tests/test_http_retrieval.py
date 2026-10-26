"""HTTP query delivery using real BM25, the controller, and CPU protocol engine."""
import asyncio
from dataclasses import replace
import json
import sqlite3
from threading import Event

from fastapi.testclient import TestClient
import pytest

from adapters.retrieval import BM25SRetriever
from adapters.serving_retrieval import ServingRetriever
from controller import api, cli
from controller.config import RunConfig, create_controller
from controller.engine_client import EngineError
from controller.serialization import to_jsonable


CORPUS = {
    'a': {'text': 'Request timeout is thirty seconds. Retry only idempotent client calls.', 'source': 'docs://timeout'},
    'b': {'text': 'Retry at most three times with exponential backoff.', 'source': 'docs://retry'},
    'c': {'text': 'Authentication failures must not be retried.', 'source': 'docs://auth'},
    'd': {'text': 'Connection timeout is five seconds.', 'source': 'docs://connection'},
    'e': {'text': 'Logging uses standard output.', 'source': 'docs://logging'},
    'z': {'text': 'Error code E42 means the request was cancelled.', 'source': 'docs://errors'},
}


class ObservedRetriever:
    def __init__(self, inner):
        self.inner, self.seen, self.close_calls = inner, [], 0
        self.corpus_snapshot_id = inner.corpus_snapshot_id
        self.fail = False

    @property
    def manifest(self):
        return self.inner.manifest

    def retrieve_turn(self, turn):
        self.seen.append(turn)
        if self.fail:
            raise ValueError('controlled backend retrieval failure')
        return self.inner.retrieve_turn(turn)

    def close(self):
        self.close_calls += 1


@pytest.fixture
def service(tmp_path):
    config = replace(RunConfig(), state_dir=str(tmp_path/'state'))
    config = replace(config, planner=replace(config.planner, strategy='always_rebuild'))
    result = create_controller(config)
    result.engine.response_text = 'The actual first answer.'
    return result


@pytest.fixture
def observed():
    return ObservedRetriever(BM25SRetriever(CORPUS, corpus_snapshot_id='manuals-v1'))


def generations(service):
    return sum(e['event'] == 'generate_complete' for e in service.engine.events)


def test_real_bm25_multiturn_idempotency_and_original_source_manifest(service, observed):
    retrieval = ServingRetriever(observed, corpus_id='manuals', events=service.events)
    with TestClient(api.create_app(service, retrieval=retrieval)) as client:
        info = client.get('/health').json()
        assert info['retrieval']['corpus_snapshot_id'] == 'manuals-v1'
        first = client.post('/sessions/s/query', json={'query': 'Request timeout?', 'request_id': 'q1'})
        assert first.status_code == 200
        answer1 = first.json()
        assert len(observed.seen) == generations(service) == 1
        assert observed.seen[0].history == ()
        assert answer1['retrieval_ms'] >= 0 and answer1['complete_ms'] >= answer1['retrieval_ms']
        assert answer1['visible_evidence_manifest']
        attempt1 = service.store.find_request('s', 'q1')
        assert answer1['visible_evidence_manifest'] == to_jsonable(attempt1.plan.candidate.evidence_spans)
        service.engine.response_text = 'The actual second answer.'
        second = client.post('/sessions/s/query', json={'query': 'Error code E42?', 'request_id': 'q2'})
        assert second.status_code == 200
        assert [m.text for m in observed.seen[1].history] == ['Request timeout?', answer1['assistant_text']]
        snapshot = service.store.get_snapshot('s')
        assert snapshot.version == 2 and len(snapshot.raw_history) == 2
        original = {b.key: b for b in snapshot.archive}
        for span in attempt1.plan.candidate.evidence_spans:
            assert attempt1.plan.candidate.messages[span.message_index].content[span.start:span.end] == original[span.key].text
        assert second.json()['visible_evidence_manifest'] != answer1['visible_evidence_manifest']
        replay = client.post('/sessions/s/query', json={'query': 'Request timeout?', 'request_id': 'q1'}).json()
        assert replay['idempotent_replay']
        for field in ('assistant_text', 'output_ids', 'visible_evidence_manifest', 'retrieval_ms', 'attempt_id'):
            assert replay[field] == answer1[field]
        assert replay['complete_ms'] is None and replay['timing'] is None
        assert len(observed.seen) == generations(service) == 2
        conflict = client.post('/sessions/s/query', json={'query': 'Different query', 'request_id': 'q1'})
        assert conflict.status_code == 409
        supplied = client.post('/sessions/s/turns', json={'query': 'Request timeout?', 'request_id': 'q1',
                                                        'corpus_snapshot_id': 'manuals-v1'})
        assert supplied.status_code == 409
        assert len(observed.seen) == generations(service) == 2
    assert observed.close_calls == 1


@pytest.mark.parametrize('extra', [{'history': []}, {'targets': 'reference'}, {'evidence': []},
                                   {'corpus_snapshot_id': 'caller-chosen'}])
def test_query_whitelist_rejects_history_labels_and_snapshot(service, observed, extra):
    with TestClient(api.create_app(service, retrieval=ServingRetriever(observed, corpus_id='manuals'))) as client:
        response = client.post('/sessions/s/query', json={'query': 'q', 'request_id': 'q1', **extra})
        assert response.status_code == 422
        assert not observed.seen and generations(service) == 0


def test_unconfigured_query_is_503_and_supplied_evidence_remains_available(service):
    with TestClient(api.create_app(service)) as client:
        unavailable = client.post('/sessions/s/query', json={'query': 'q', 'request_id': 'q1'})
        assert unavailable.status_code == 503 and unavailable.json()['code'] == 'RETRIEVAL_NOT_READY'
        assert client.get('/health').json()['retrieval'] is None
        direct = client.post('/sessions/s/turns', json={'query': 'q', 'request_id': 'direct',
            'evidence': [{'document_id': 'manual', 'text': 'Original content.'}]})
        assert direct.status_code == 200


def test_startup_event_binds_actual_config_and_engine_once(tmp_path):
    config = replace(RunConfig(), state_dir=str(tmp_path/'configured'))
    with TestClient(api.create_app(config=config)) as client:
        actual_engine = client.get('/health').json()['engine']
        client.get('/health')
        events = [json.loads(line) for line in (tmp_path/'configured/events.jsonl').read_text().splitlines()]
        started, = [event for event in events if event['event'] == 'service_started']
        assert started['configuration_source'] == 'run_config'
        assert started['config'] == to_jsonable(config)
        assert started['engine'] == actual_engine
        assert started['fingerprint'] == actual_engine['fingerprint']


def test_injected_controller_does_not_claim_unused_config(service, tmp_path):
    unused = replace(RunConfig(), state_dir=str(tmp_path/'unused'))
    with TestClient(api.create_app(service, config=unused)):
        events = [json.loads(line) for line in service.events.path.read_text().splitlines()]
        started, = [event for event in events if event['event'] == 'service_started']
        assert started['configuration_source'] == 'injected_controller'
        assert started['config'] is None
        assert started['fingerprint'] == to_jsonable(service.template.fingerprint)
        assert not (tmp_path/'unused').exists()


def test_retrieval_error_is_503_and_failed_attempt_requires_new_id(service, observed):
    observed.fail = True
    with TestClient(api.create_app(service, retrieval=ServingRetriever(observed, corpus_id='manuals'))) as client:
        response = client.post('/sessions/s/query', json={'query': 'timeout', 'request_id': 'failed'})
        assert response.status_code == 503 and response.json()['code'] == 'RETRIEVAL_FAILED'
        assert 'controlled backend retrieval failure' in response.json()['detail']
        snapshot = service.store.get_snapshot('s')
        assert snapshot.version == 0 and snapshot.raw_history == () and snapshot.current_epoch is None
        assert service.store.find_request('s', 'failed').stage == 'FAILED'
        assert generations(service) == 0
        assert not any(e['event'] == 'prepare' for e in service.engine.events)
        observed.fail = False
        assert client.post('/sessions/s/query', json={'query': 'timeout', 'request_id': 'failed'}).status_code == 409
        assert len(observed.seen) == 1
        assert client.post('/sessions/s/query', json={'query': 'timeout', 'request_id': 'retry'}).status_code == 200
        assert generations(service) == 1


def test_query_recovers_lost_engine_response_without_retrieval_or_generation(service, observed):
    with TestClient(api.create_app(service, retrieval=ServingRetriever(observed, corpus_id='manuals'))) as client:
        service.engine.drop_next_generate_response = True
        body = {'query': 'timeout', 'request_id': 'lost'}
        assert client.post('/sessions/s/query', json=body).status_code == 503
        assert service.store.get_snapshot('s').version == 0
        restored = client.post('/sessions/s/query', json=body)
        assert restored.status_code == 200 and restored.json()['idempotent_replay']
        assert service.store.get_snapshot('s').version == 1
        assert len(observed.seen) == generations(service) == 1
        assert restored.json()['visible_evidence_manifest']


def test_index_and_metadata_loaded_once_at_startup(service, observed, tmp_path, monkeypatch):
    index_dir = tmp_path/'index'
    observed.inner.save(index_dir)
    rows = [{'passage_id': key, **value, 'version': 'v2',
             **({'dependencies': ['header']} if key == 'a' else {})} for key, value in CORPUS.items()]
    rows.append({'passage_id': 'header', 'text': 'These settings apply to client version two.', 'version': 'v2'})
    for row in rows:
        row.pop('source', None)
        row['url'] = f"https://example.org/manual/{row['passage_id']}"
    metadata = tmp_path/'metadata.jsonl'
    metadata.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    from adapters import retriever_factory
    original = retriever_factory.create_retriever
    calls = []
    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(retriever_factory, 'create_retriever', counted)
    app = api.create_app(service, retriever_index=str(index_dir), evidence_corpus=str(metadata), corpus_id='manuals')
    assert calls == []
    with TestClient(app) as client:
        assert len(calls) == 1 and callable(calls[0]['trace_sink'])
        for number in range(2):
            response = client.post('/sessions/s/query', json={'query': 'Request timeout?', 'request_id': str(number)})
            assert response.status_code == 200
        assert len(calls) == 1
        manifest = response.json()['visible_evidence_manifest']
        assert any(s['key']['passage_id'] == 'header' for s in manifest)
        assert all(s['key']['corpus_id'] == 'manuals' and s['key']['version'] == 'v2' for s in manifest)
        assert all(s['source_uri'] == f"https://example.org/manual/{s['key']['passage_id']}" for s in manifest)
        attempt = service.store.find_request('s', '1')
        assert len(attempt.retrieval_priority) == 5
        assert all(k.passage_id != 'header' for k in attempt.retrieval_priority)
        assert client.get('/health').json()['retrieval']['metadata_enabled']


def test_startup_engine_failure_closes_injected_resources(service, observed):
    closed = []
    original = service.aclose
    async def close():
        closed.append(True)
        await original()
    async def fail():
        raise EngineError('CONTROLLED_STARTUP_FAILURE')
    service.aclose, service.engine.info = close, fail
    with pytest.raises(EngineError, match='CONTROLLED_STARTUP_FAILURE'):
        with TestClient(api.create_app(service, retrieval=ServingRetriever(observed, corpus_id='manuals'))):
            pass
    assert closed == [True] and observed.close_calls == 1
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        service.store.db.execute('SELECT 1')


def test_metadata_startup_failure_closes_new_retriever_and_store(service, observed, tmp_path, monkeypatch):
    monkeypatch.setattr('adapters.retriever_factory.create_retriever', lambda *a, **kw: observed)
    metadata = tmp_path/'bad.jsonl'
    metadata.write_text('{"text":"no identity"}\n')
    with pytest.raises(ValueError, match='require passage'):
        with TestClient(api.create_app(service, retriever_index='configured', evidence_corpus=str(metadata))):
            pass
    assert observed.close_calls == 1
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        service.store.db.execute('SELECT 1')


@pytest.mark.asyncio
async def test_cancelled_startup_waits_for_loader_and_closes_its_adapter(service, observed, monkeypatch):
    entered, release = Event(), Event()
    adapter = ServingRetriever(observed, corpus_id='manuals')
    def load(*args, **kwargs):
        entered.set()
        release.wait(2)
        return adapter
    monkeypatch.setattr(api, '_load_retrieval', load)
    app = api.create_app(service, retriever_index='configured')
    context = app.router.lifespan_context(app)
    task = asyncio.create_task(context.__aenter__())
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and observed.close_calls == 0
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert observed.close_calls == 1 and not app.state.ready
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        service.store.db.execute('SELECT 1')


def test_serve_cli_forwards_explicit_retrieval_and_rejects_conflicts(tmp_path, monkeypatch):
    received = {}
    app = object()
    def create(**kwargs):
        received.update(kwargs)
        return app
    monkeypatch.setattr(api, 'create_app', create)
    monkeypatch.setattr('uvicorn.run', lambda supplied, **kw: received.update(app=supplied, **kw))
    cli.main(['--state-dir', str(tmp_path), 'serve', '--retriever-index', 'fixed-index',
              '--evidence-corpus', 'metadata.jsonl', '--corpus-id', 'manuals'])
    assert received['app'] is app
    assert received['retriever_index'] == 'fixed-index' and received['retriever_config'] is None
    assert received['evidence_corpus'] == 'metadata.jsonl' and received['corpus_id'] == 'manuals'
    with pytest.raises(SystemExit) as exc:
        cli.main(['serve', '--retriever-index', 'a', '--retriever-config', 'b'])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        cli.main(['serve', '--evidence-corpus', 'metadata.jsonl'])
    assert exc.value.code == 2
