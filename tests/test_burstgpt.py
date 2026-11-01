import csv
import json

import pytest

from bench.burstgpt import sample_plan, summarize_trace


def test_trace_gaps_preserve_failures_ties_and_overlaps(tmp_path):
    path = tmp_path / 'trace.csv'
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Timestamp', 'Session ID', 'Elapsed time', 'Model',
                         'Response tokens', 'Log Type'])
        for timestamp, session, elapsed, output, model in (
            (0, 'a', 2, 10, 'ChatGPT'), (10, 'a', 1, 10, 'ChatGPT'),
            (11, 'a', 1, 0, 'ChatGPT'), (12, 'a', 2, 10, 'ChatGPT'),
            (15, 'a', 1, 10, 'ChatGPT'), (20, 'b', 1, 10, 'GPT-4'),
            (20, 'b', 1, 10, 'GPT-4'), (22, 'b', 1, 10, 'GPT-4'),
            (25, 'b', 1, 10, 'GPT-4'), (26, 'c', 10, 10, 'ChatGPT'),
            (29, 'c', 1, 10, 'ChatGPT'),
        ):
            writer.writerow([timestamp, session, elapsed, model, output, 'Conversation log'])
        writer.writerow([30, '', 3, 'GPT-4', 10, 'API log'])
    result = summarize_trace(path, short_gap_seconds=2)
    assert result['sessions_retained'] == 3
    assert result['counts']['nonnegative_pairs'] == 3
    assert result['counts']['pairs_touching_failure'] == 2
    assert result['counts']['tied_timestamp_pairs'] == 1
    assert result['counts']['pairs_after_tied_timestamp'] == 1
    assert result['counts']['overlapping_pairs'] == 1
    assert result['counts']['nonconversation_or_missing_session'] == 1
    assert result['gap_seconds']['mean_seconds'] == pytest.approx(11 / 3)
    assert result['gap_seconds']['quantiles_seconds']['0.5'] == 2
    assert result['gap_seconds']['short_gap_fraction'] == pytest.approx(2 / 3)


def test_trace_rejects_out_of_order_timestamps(tmp_path):
    path = tmp_path / 'trace.csv'
    path.write_text('Timestamp,Session ID,Elapsed time,Model,Response tokens,Log Type\n'
                    '2,a,1,ChatGPT,1,Conversation log\n'
                    '1,a,1,ChatGPT,1,Conversation log\n')
    with pytest.raises(ValueError, match='not globally ordered'):
        summarize_trace(path)


def test_sampled_plan_is_deterministic_and_reports_capping():
    rows = [{'conversation_id': 'a', 'turn': n, 'task_id': f'a{n}', 'target': 'not used'}
            for n in (1, 2, 3)]
    first = sample_plan(rows, [5, 100], trace_sha256='trace-hash', seed=3, max_gap_seconds=6)
    assert first == sample_plan(rows, [5, 100], trace_sha256='trace-hash', seed=3, max_gap_seconds=6)
    assert set(first['gaps_after_task']) == {'a1', 'a2'}
    assert all(gap <= 6 for gap in first['gaps_after_task'].values())
    assert first['source']['trace_sha256'] == 'trace-hash'
    assert first['source']['capped_gaps'] == sum(gap == 6 for gap in first['gaps_after_task'].values())
    assert 'not used' not in json.dumps(first)
