import asyncio
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import pytest

from adapters.mtrag import load_rows
from bench.run_sessions import load_think_time_plan, run_sessions


ROWS = load_rows(Path(__file__).parent / 'fixtures/mtrag_small.jsonl')[:3]


def test_planned_gaps_follow_committed_turns_and_are_bound_to_file(tmp_path):
    plan = {'schema_version': 1, 'gaps_after_task': {ROWS[0]['task_id']: .01, ROWS[1]['task_id']: .02}}
    path = tmp_path / 'plan.json'
    payload = json.dumps(plan).encode()
    path.write_bytes(payload)
    gaps, provenance = load_think_time_plan(path)
    assert provenance['sha256'] == hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match='changed after matrix provenance'):
        load_think_time_plan(path, expected_sha256='0' * 64)

    async def run():
        async def handler(turn):
            return {'assistant_text': f'answer {turn.turn}'}
        return await run_sessions(ROWS, handler, think_time_by_task=gaps)

    report = asyncio.run(run())
    turns = sorted(report.turns, key=lambda turn: turn.turn)
    assert [turn.status for turn in turns] == ['success'] * 3
    assert turns[1].t_arrive - turns[0].t_done >= .008
    assert turns[2].t_arrive - turns[1].t_done >= .018


@pytest.mark.parametrize('gaps', [
    {},
    {ROWS[0]['task_id']: -1, ROWS[1]['task_id']: 0},
    {ROWS[0]['task_id']: float('nan'), ROWS[1]['task_id']: 0},
    {ROWS[0]['task_id']: 0, ROWS[1]['task_id']: 0, ROWS[2]['task_id']: 0},
])
def test_planned_gaps_reject_incomplete_or_invalid_schedules(gaps):
    async def handler(turn):
        return {'assistant_text': 'answer'}

    with pytest.raises(ValueError, match='think-time plan'):
        asyncio.run(run_sessions(ROWS, handler, think_time_by_task=gaps))


def test_fixed_and_per_task_pacing_cannot_both_be_requested():
    async def handler(turn):
        return {'assistant_text': 'answer'}

    gaps = {ROWS[0]['task_id']: 0, ROWS[1]['task_id']: 0}
    with pytest.raises(ValueError, match='choose a fixed think_time'):
        asyncio.run(run_sessions(ROWS, handler, think_time=1, think_time_by_task=gaps))


def test_export_time_is_not_added_to_a_planned_gap():
    calls = []
    async def fake_sleep(seconds):
        calls.append(seconds)
    async def handler(turn):
        return {'assistant_text': 'answer'}

    gaps = {ROWS[0]['task_id']: .005, ROWS[1]['task_id']: .005}
    with patch('bench.run_sessions.asyncio.sleep', fake_sleep):
        report = asyncio.run(run_sessions(ROWS, handler, think_time_by_task=gaps,
                                          on_result=lambda _: time.sleep(.01)))
    assert len(report.turns) == 3
    assert calls == [0.0, 0.0, 0.0]


@pytest.mark.parametrize('planned', [False, True])
def test_next_turn_arrival_reflects_response_plus_gap_even_if_export_is_slow(planned):
    async def handler(turn):
        return {'assistant_text': 'answer'}

    gap = .005
    options = {'think_time_by_task': {ROWS[0]['task_id']: gap}} if planned else {'think_time': gap}
    report = asyncio.run(run_sessions(ROWS[:2], handler, on_result=lambda _: time.sleep(.02), **options))
    first, second = sorted(report.turns, key=lambda row: row.turn)
    assert second.t_arrive == pytest.approx(first.t_done + gap, abs=1e-6)
    assert second.t_done - second.t_arrive >= .01


def test_running_session_uses_frozen_plan_values():
    gaps = {ROWS[0]['task_id']: .02, ROWS[1]['task_id']: 0}

    async def handler(turn):
        if turn.turn == 1:
            gaps[turn.task_id] = 0
        return {'assistant_text': 'answer'}

    report = asyncio.run(run_sessions(ROWS, handler, think_time_by_task=gaps))
    turns = sorted(report.turns, key=lambda turn: turn.turn)
    assert turns[1].t_arrive - turns[0].t_done >= .018
