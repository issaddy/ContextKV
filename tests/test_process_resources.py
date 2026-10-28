from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from bench import process_resources as resources


def test_process_tree_ownership_excludes_parent_and_catches_root_reuse():
    table = {1: {'ppid': 0, 'start_ticks': 1}, 8: {'ppid': 1, 'start_ticks': 20},
             9: {'ppid': 8, 'start_ticks': 21}, 10: {'ppid': 9, 'start_ticks': 22},
             11: {'ppid': 1, 'start_ticks': 23}}
    assert resources.owned_pids(table, 8, 20) == {8, 9, 10}
    with pytest.raises(ValueError, match='reused'):
        resources.owned_pids(table, 8, 19)


@pytest.mark.parametrize('value', ['[N/A]', 'nan', 'inf', '-1'])
def test_unavailable_gpu_memory_is_never_zero(value):
    with pytest.raises(ValueError):
        resources.parse_gpu_processes(f'GPU-x, 123, {value}')


def test_multiple_gpu_contexts_are_summed_without_losing_process_identity():
    rows = resources.parse_gpu_processes('GPU-a, 12, 1.5\nGPU-b, 12, 2\nGPU-a, 13, 3')
    assert sum(r['bytes'] for r in rows) == 6.5*1024**2
    assert resources.parse_gpu_processes('') == []
    with pytest.raises(ValueError, match='duplicate'):
        resources.parse_gpu_processes('GPU-a, 12, 1\nGPU-a, 12, 1')


def test_pss_uses_proportional_ram_not_double_counted_rss(tmp_path):
    (tmp_path/'12').mkdir()
    (tmp_path/'12/smaps_rollup').write_text('Rss: 100 kB\nPss: 37 kB\n')
    assert resources.host_pss(12, tmp_path) == 37*1024
    (tmp_path/'12/smaps_rollup').write_text('Rss: 100 kB\n')
    with pytest.raises(ValueError, match='missing'):
        resources.host_pss(12, tmp_path)


def test_proc_names_with_spaces_and_parentheses_do_not_shift_identity(tmp_path):
    path = tmp_path/'12'
    path.mkdir()
    fields = ['S', '8'] + ['0']*17 + ['9921']
    (path/'stat').write_text('12 (worker (GPU) process) '+' '.join(fields))
    assert resources.process_table(tmp_path) == ({12: {'ppid': 8, 'start_ticks': 9921}}, [])


def test_integral_clips_startup_and_shutdown_and_keeps_missing_intervals():
    samples = [{'timestamp': t, 'gpu_bytes': gpu, 'host_pss_bytes': host}
               for t,gpu,host in [(0, 100, 10), (2, 200, 20), (4, None, 30), (6, 300, 40)]]
    result = resources.memory_summary({'samples': samples}, 1, 5)
    assert result['host_pss_bytes']['byte_seconds'] == 10+40+30
    assert result['gpu_bytes']['byte_seconds'] is None
    assert result['gpu_bytes']['covered_seconds'] == 3
    assert result['gpu_bytes']['observed_byte_seconds'] == 100+400
    assert result['gpu_bytes']['sampled_peak_bytes'] == 200
    for start,end in [(-1, 3), (1, 7)]:
        assert resources.memory_summary({'samples': samples}, start, end)['host_pss_bytes']['byte_seconds'] is None


@pytest.mark.parametrize('start,end', [(None, 5), (1, None), (5, 1), (1, float('nan'))])
def test_missing_workload_clock_does_not_become_full_sampler_window(start, end):
    assert not resources.memory_summary({'samples': []}, start, end)['available']


def test_unknown_gpu_pid_does_not_look_like_empty_owned_gpu(monkeypatch):
    sampler = resources.ProcessResourceSampler()
    table = {sampler.root_pid: {'ppid': 1, 'start_ticks': 10}}
    monkeypatch.setattr(resources, 'process_table', lambda: (table, []))
    monkeypatch.setattr(resources, 'host_pss', lambda pid: 1000)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout='GPU-a, 999999, 400'))
    row = sampler._collect()
    assert row['host_pss_bytes'] == 1000
    assert row['gpu_bytes'] is None
    assert row['gpu_processes'][0]['ownership'] == 'unknown'


def test_sampler_error_remains_observable_without_interrupting_workload(monkeypatch):
    sampler = resources.ProcessResourceSampler()
    def unavailable():
        raise FileNotFoundError(Path('/proc'))
    monkeypatch.setattr(resources, 'process_table', unavailable)
    row = sampler._collect()
    assert row['gpu_bytes'] is None and row['host_pss_bytes'] is None
    assert row['errors'] and row['scrape_seconds'] >= 0
