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
    payload, _ = raw_output(tmp_path)
    assert payload['cleanup']['status'] == 'completed'
    assert payload['cleanup']['skipped_uncreated_sessions'] == []
    assert payload['cleanup']['pending_retire_operations'] == 0
    assert payload['telemetry']['after'] is not None
    assert all(t['status'] == 'success' for t in payload['report']['turns'])
    with sqlite3.connect(tmp_path/'sessions.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM sessions WHERE closed=1').fetchone()[0] == 1
