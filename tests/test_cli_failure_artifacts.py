"""Real CLI workload failures retain reports before session cleanup can obscure them."""
import json
from pathlib import Path
import sqlite3

import pytest

from controller import cli
from controller.config import create_controller

FIXTURE = Path(__file__).parent / 'fixtures/mtrag_small.jsonl'


class FailedRetriever:
    corpus_snapshot_id = 'synthetic-retrieval-failure'
    manifest = {'kind': 'synthetic_failure_fixture'}

    def __init__(self, fail_turn=1):
        self.fail_turn, self.closed = fail_turn, False

    def retrieve_turn(self, turn):
        if turn.turn == self.fail_turn:
            raise RuntimeError('controlled retrieval failure before serving')
        # Use the fixture's public contexts, not its reference answer.
        from dataclasses import replace
        from adapters.mtrag import clean_context
        row = next(json.loads(line) for line in FIXTURE.read_text().splitlines()
                   if json.loads(line)['task_id'] == turn.task_id)
        return replace(turn, contexts=tuple(clean_context(c) for c in row['contexts']))

    def close(self):
        self.closed = True


def raw_output(tmp_path):
    path, = tmp_path.glob('benchmark-*.json')
    summary, = tmp_path.glob('summary-*.json')
    return json.loads(path.read_text()), json.loads(summary.read_text())


def arguments(tmp_path, protocol):
    return ['--state-dir', str(tmp_path), 'bench', str(FIXTURE), '--domain', 'all',
            '--limit-sessions', '1', '--protocol', protocol]


def test_p3_first_retrieval_failure_saves_report_without_creating_a_session(tmp_path, monkeypatch):
    retriever = FailedRetriever()
    monkeypatch.setattr('adapters.retriever_factory.create_retriever', lambda *a, **kw: retriever)
    with pytest.raises(SystemExit) as exc:
        cli.main(arguments(tmp_path, 'P3'))
    assert exc.value.code == 1
    payload, summary = raw_output(tmp_path)
    turn, = payload['report']['turns']
    assert turn['status'] == 'failed'
    assert turn['error'] == 'RuntimeError: controlled retrieval failure before serving'
    assert turn['result'] is None and payload['report']['unfinished_turns'] > 0
    assert payload['cleanup']['status'] == 'completed'
    assert payload['cleanup']['skipped_uncreated_sessions'] == [turn['conversation_id']]
    assert summary['timing']['statuses'] == {'failed': 1}
    with sqlite3.connect(tmp_path/'sessions.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM sessions').fetchone()[0] == 0
    assert retriever.closed


def test_cleanup_failure_keeps_successful_turns_and_still_raises(tmp_path, monkeypatch):
    def controller(config):
        service = create_controller(config)
        async def fail_close(session_id):
            raise RuntimeError('controlled close failure')
        service.close_session = fail_close
        return service
    monkeypatch.setattr(cli, 'create_controller', controller)
    with pytest.raises(RuntimeError, match='controlled close failure'):
        cli.main(arguments(tmp_path, 'P2'))
    payload, summary = raw_output(tmp_path)
    assert payload['report']['turns'] and all(t['status'] == 'success' for t in payload['report']['turns'])
    assert payload['cleanup']['status'] == 'failed'
    assert payload['cleanup']['failed_stage'] == 'close_sessions'
    assert payload['cleanup']['error'] == 'RuntimeError: controlled close failure'
    assert payload['telemetry']['after'] is None
    assert summary['timing']['statuses']['success'] == len(payload['report']['turns'])


def test_later_p3_retrieval_failure_closes_the_existing_session(tmp_path, monkeypatch):
    retriever = FailedRetriever(fail_turn=2)
    monkeypatch.setattr('adapters.retriever_factory.create_retriever', lambda *a, **kw: retriever)
    with pytest.raises(SystemExit) as exc:
        cli.main(arguments(tmp_path, 'P3'))
    assert exc.value.code == 1
    payload, summary = raw_output(tmp_path)
    assert [t['status'] for t in payload['report']['turns']] == ['success', 'failed']
    assert summary['server_metrics']['timing_sources'] == {'unavailable': 2}
    assert payload['cleanup']['status'] == 'completed'
    assert payload['cleanup']['skipped_uncreated_sessions'] == []
    with sqlite3.connect(tmp_path/'sessions.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM sessions WHERE closed=1').fetchone()[0] == 1
    assert retriever.closed


def test_normal_p2_cleanup_and_cli_success_are_unchanged(tmp_path):
    cli.main(arguments(tmp_path, 'P2'))
    payload, summary = raw_output(tmp_path)
    assert payload['cleanup']['status'] == 'completed'
    assert payload['cleanup']['skipped_uncreated_sessions'] == []
    assert payload['cleanup']['pending_retire_operations'] == 0
    assert payload['telemetry']['after'] is not None
    assert all(t['status'] == 'success' for t in payload['report']['turns'])
    assert summary['decisions']['observed_turns'] == len(payload['report']['turns'])
    assert summary['decisions']['cost_based_selection_turns'] == 0
    assert summary['decisions']['fallback_turns'] > 0
    with sqlite3.connect(tmp_path/'sessions.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM sessions WHERE closed=1').fetchone()[0] == 1


def test_pending_retirement_is_not_reported_as_completed_cleanup(tmp_path, monkeypatch):
    from controller.engine_client import EngineError

    def controller(config):
        service = create_controller(config)

        async def pending_retire(*args, **kwargs):
            raise EngineError('RETIRE_PENDING')

        service.engine.retire = pending_retire
        return service

    monkeypatch.setattr(cli, 'create_controller', controller)
    with pytest.raises(RuntimeError, match='retirement operation'):
        cli.main(arguments(tmp_path, 'P2'))
    payload, summary = raw_output(tmp_path)
    assert all(turn['status'] == 'success' for turn in payload['report']['turns'])
    assert payload['cleanup']['status'] == 'failed'
    assert payload['cleanup']['failed_stage'] == 'drain'
    assert payload['cleanup']['pending_retire_operations'] > 0
    assert payload['telemetry']['after'] is not None
    assert summary['cleanup'] == payload['cleanup']
    with sqlite3.connect(tmp_path/'sessions.sqlite3') as db:
        assert db.execute("SELECT count(*) FROM attempts WHERE stage='COMMITTED'").fetchone()[0] == 3


@pytest.mark.parametrize('fail_every_export', [False, True])
@pytest.mark.parametrize('concurrency', [1, 2])
def test_evaluation_export_failure_keeps_committed_history_and_fails_artifact_delivery(
        tmp_path, monkeypatch, fail_every_export, concurrency):
    from bench import quality
    original_export = quality.export_evaluation
    exported = []
    def failing_export(turn, answer, **kwargs):
        exported.append((turn, answer))
        if len(exported) == 1 or fail_every_export:
            raise OSError('controlled evaluation sink failure')
        return original_export(turn, answer, **kwargs)
    monkeypatch.setattr(quality, 'export_evaluation', failing_export)
    args = arguments(tmp_path, 'P2')
    args[args.index('--limit-sessions') + 1] = str(concurrency)
    args += ['--concurrency', str(concurrency)]
    with pytest.raises(RuntimeError, match='evaluation export failed for') as caught:
        cli.main(args)
    assert isinstance(caught.value.__cause__, OSError)
    payload, summary = raw_output(tmp_path)
    turns = payload['report']['turns']
    assert len(turns) == len(exported) == 3*concurrency
    assert all(turn['status'] == 'success' for turn in turns)
    assert payload['report']['unfinished_turns'] == 0
    assert summary['timing']['statuses'] == {'success': 3*concurrency}
    assert payload['cleanup']['status'] == 'completed'
    failures = payload['evaluation_export']['failures']
    assert payload['evaluation_export']['status'] == 'failed'
    assert len(failures) == (3*concurrency if fail_every_export else 1)
    assert failures[0] == {'conversation_id': exported[0][0].conversation_id,
                           'task_id': exported[0][0].task_id, 'turn': 1,
                           'error': 'OSError: controlled evaluation sink failure'}
    for following in turns:
        if following['turn'] > 1:
            previous = next(turn for turn in turns if turn['conversation_id'] == following['conversation_id']
                            and turn['turn'] == following['turn'] - 1)
            assert following['online_turn']['history'][-1]['text'] == previous['assistant_text']
    with sqlite3.connect(tmp_path/'sessions.sqlite3') as db:
        snapshots = db.execute('SELECT session_id, snapshot, closed FROM sessions').fetchall()
        assert len(snapshots) == concurrency
        for session_id, snapshot, closed in snapshots:
            assert closed == 1
            assert [turn['assistant_text'] for turn in json.loads(snapshot)['raw_history']] == [
                turn['assistant_text'] for turn in turns if session_id.endswith(':' + turn['conversation_id'])]
        assert db.execute("SELECT count(*) FROM attempts WHERE stage='COMMITTED'").fetchone()[0] == 3*concurrency


def test_export_snapshot_read_failure_is_recorded_after_successful_serving(tmp_path, monkeypatch):
    import sys
    export_reads = []
    def controller(config):
        service = create_controller(config)
        original_read = service.store.read_snapshot
        def read_snapshot(*args, **kwargs):
            # Fail only the post-commit export read, never serving or recovery.
            if sys._getframe(1).f_code.co_name == 'export_result':
                snapshot = original_read(*args, **kwargs)
                export_reads.append(len(snapshot.raw_history))
                if len(export_reads) == 1:
                    assert len(snapshot.raw_history) == 1
                    raise OSError('controlled export snapshot read failure')
                return snapshot
            return original_read(*args, **kwargs)
        service.store.read_snapshot = read_snapshot
        return service
    monkeypatch.setattr(cli, 'create_controller', controller)
    with pytest.raises(RuntimeError, match='evaluation export failed for 1') as caught:
        cli.main(arguments(tmp_path, 'P2'))
    assert isinstance(caught.value.__cause__, OSError)
    payload, summary = raw_output(tmp_path)
    turns = payload['report']['turns']
    assert export_reads == [1, 2, 3]
    assert [turn['status'] for turn in turns] == ['success'] * 3
    assert turns[1]['online_turn']['history'][-1]['text'] == turns[0]['assistant_text']
    assert payload['report']['unfinished_turns'] == 0
    assert summary['timing']['statuses'] == {'success': 3}
    assert payload['cleanup']['status'] == 'completed'
    assert payload['evaluation_export'] == {'status': 'failed', 'failures': [{
        'conversation_id': turns[0]['conversation_id'], 'task_id': turns[0]['task_id'],
        'turn': 1, 'error': 'OSError: controlled export snapshot read failure'}]}
