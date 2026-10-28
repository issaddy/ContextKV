"""Sample an owned Linux process tree's GPU allocation and proportional host RAM.

This complements native KV gauges: weights, retrieval and auxiliary models are
included. Memory residence is not energy, utilization or compute cost. Unknown
ownership/counters remain missing; samples never authorize lifecycle actions.
"""
from __future__ import annotations

import asyncio
import csv
import math
import os
from pathlib import Path
import subprocess
import time


def process_table(proc=Path('/proc')):
    rows, errors = {}, []
    for path in proc.iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path/'stat').read_text().rsplit(') ', 1)[1].split()
            rows[int(path.name)] = {'ppid': int(fields[1]), 'start_ticks': int(fields[19])}
        except FileNotFoundError:
            continue  # A process that exited during enumeration has no live RAM.
        except (OSError, ValueError, IndexError) as error:
            errors.append({'pid': int(path.name), 'error': str(error)})
    return rows, errors


def owned_pids(table, root_pid, root_start_ticks):
    if table.get(root_pid, {}).get('start_ticks') != root_start_ticks:
        raise ValueError('owned root exited or its PID was reused')
    owned = {root_pid}
    while True:
        found = {pid for pid, row in table.items() if row['ppid'] in owned}
        if found <= owned:
            return owned
        owned |= found


def parse_gpu_processes(text):
    rows = []
    for cells in csv.reader(text.splitlines()):
        if not cells:
            continue
        if len(cells) != 3:
            raise ValueError('unexpected nvidia-smi process fields')
        gpu, pid, mib = [value.strip() for value in cells]
        value = float(mib)
        if not gpu or not value >= 0 or value == float('inf'):
            raise ValueError('invalid GPU process memory')
        rows.append({'gpu_uuid': gpu, 'pid': int(pid), 'bytes': int(value*1024**2)})
    identities = [(r['gpu_uuid'], r['pid']) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate GPU process counter')
    return rows


def host_pss(pid, proc=Path('/proc')):
    for line in (proc/str(pid)/'smaps_rollup').read_text().splitlines():
        cells = line.split()
        if cells and cells[0] == 'Pss:':
            if len(cells) != 3 or cells[2] != 'kB' or int(cells[1]) < 0:
                raise ValueError('invalid Linux PSS counter')
            return int(cells[1])*1024
    raise ValueError('Linux PSS counter missing')


class ProcessResourceSampler:
    def __init__(self, *, interval=1.0):
        self.interval = interval
        self.root_pid = os.getpid()
        self.root_start_ticks = None
        self.samples = []
        self._stop = asyncio.Event()
        self._task = None

    def _collect(self):
        start = time.monotonic()
        row = {'gpu_bytes': None, 'host_pss_bytes': None, 'errors': []}
        try:
            table, errors = process_table()
            row['errors'].extend(errors)
            if self.root_start_ticks is None:
                self.root_start_ticks = table[self.root_pid]['start_ticks']
            owned = owned_pids(table, self.root_pid, self.root_start_ticks)
            row['owned_pids'] = sorted(owned)
            pss = []
            for pid in sorted(owned):
                try:
                    pss.append({'pid': pid, 'bytes': host_pss(pid)})
                except (OSError, ValueError) as error:
                    row['errors'].append({'pid': pid, 'error': str(error)})
            row['host_processes'] = pss
            if not row['errors']:
                row['host_pss_bytes'] = sum(r['bytes'] for r in pss)
            output = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=5)
            gpu = parse_gpu_processes(output.stdout)
            row['gpu_processes'] = [{**r, 'ownership': 'owned' if r['pid'] in owned else
                                    'foreign' if r['pid'] in table else 'unknown'} for r in gpu]
            if not errors and all(r['ownership'] != 'unknown' for r in row['gpu_processes']):
                row['gpu_bytes'] = sum(r['bytes'] for r in row['gpu_processes'] if r['ownership'] == 'owned')
            for r in row['gpu_processes']:
                if r['ownership'] == 'unknown':
                    row['errors'].append({'pid': r['pid'], 'error': 'GPU PID absent from process table; ownership unknown'})
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            row['errors'].append({'error': f'{type(error).__name__}: {error}'})
        end = time.monotonic()
        return {**row, 'timestamp': (start+end)/2, 'scrape_seconds': end-start}

    async def start(self):
        self.samples.append(await asyncio.to_thread(self._collect))
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        while True:
            try:
                await asyncio.wait_for(self._stop.wait(), self.interval)
                return
            except TimeoutError:
                self.samples.append(await asyncio.to_thread(self._collect))

    async def stop(self):
        self._stop.set()
        if self._task is not None:
            await self._task
        self.samples.append(await asyncio.to_thread(self._collect))

    def observation(self):
        return {'source': 'nvidia-smi compute processes and Linux smaps_rollup Pss',
                'clock': 'host_monotonic', 'root_pid': self.root_pid,
                'root_start_ticks': self.root_start_ticks, 'interval_seconds': self.interval,
                'scope': 'Owned executor tree, including generator, helper, retrieval and observer; '
                         'GPU allocation and proportional host RAM, not energy or compute utilization. '
                         'PID namespace mismatch or missing counters remain unavailable.',
                'samples': self.samples}


def memory_summary(observation, start, end):
    """Left-hold sampled bytes over the actual workload window, without extrapolation."""
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end))
            or end < start):
        return {'available': False, 'reason': 'workload window missing or invalid; raw samples retained'}
    samples = sorted(observation['samples'], key=lambda r: r['timestamp'])
    result = {'available': True, 'window_seconds': end-start, 'integration': 'left-hold, bounded by actual samples'}
    for field in ('gpu_bytes', 'host_pss_bytes'):
        area, covered, peak = 0.0, 0.0, None
        for a, b in zip(samples, samples[1:]):
            width = min(end, b['timestamp'])-max(start, a['timestamp'])
            if width <= 0 or a.get(field) is None:
                continue
            value = a[field]
            area += value*width
            covered += width
            peak = value if peak is None else max(peak, value)
        complete = covered >= end-start and end > start
        result[field] = {'covered_seconds': covered, 'complete_window': complete,
                         'byte_seconds': area if complete else None,
                         'observed_byte_seconds': area, 'sampled_peak_bytes': peak}
    relevant = [r for r in samples if start <= r['timestamp'] <= end]
    owned_gpus = {g['gpu_uuid'] for r in relevant for g in r.get('gpu_processes', []) if g['ownership']=='owned'}
    result['foreign_processes_on_owned_gpus'] = [g for r in relevant for g in r.get('gpu_processes', [])
                                                if g['ownership']=='foreign' and g['gpu_uuid'] in owned_gpus]
    result['max_sample_gap_seconds'] = max((b['timestamp']-a['timestamp']
        for a,b in zip(samples,samples[1:]) if a['timestamp'] < end and b['timestamp'] > start), default=None)
    result['error_samples'] = sum(bool(r.get('errors')) for r in relevant)
    return result
