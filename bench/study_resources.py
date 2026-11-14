"""Physical GPU memory and process PSS on the experiment's common time axis."""
from __future__ import annotations

import asyncio
import csv
import io
import os
from pathlib import Path
import subprocess
import time


def listening_processes(ports):
    import psutil
    return tuple(sorted({connection.pid for connection in psutil.net_connections(kind="tcp")
        if connection.status == psutil.CONN_LISTEN and connection.laddr.port in ports
        and connection.pid is not None}))


def descendants(process_ids):
    parents = {}
    for line in subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True).splitlines():
        pid, parent = map(int, line.split())
        parents.setdefault(parent, []).append(pid)
    found, pending = set(), list(process_ids)
    while pending:
        pid = pending.pop()
        if pid not in found:
            found.add(pid)
            pending.extend(parents.get(pid, ()))
    return found


def physical_sample(gpu_uuids, process_ids):
    before = time.monotonic()
    owners = descendants((*process_ids, os.getpid()))
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits"], text=True, start_new_session=True)
    gpu_gib = sum(float(row[2]) for row in csv.reader(io.StringIO(output))
                  if int(row[0]) in owners and row[1].strip() in gpu_uuids) / 1024
    pss_kib = 0
    for pid in owners:
        try:
            contents = Path(f"/proc/{pid}/smaps_rollup").read_text()
        except (FileNotFoundError, ProcessLookupError):
            continue
        pss_kib += next(int(line.split()[1]) for line in contents.splitlines() if line.startswith("Pss:"))
    return {"timestamp": (before+time.monotonic())/2,
            "gpu_gib": gpu_gib, "host_gib": pss_kib/(1024**2)}


def integrate_physical(samples, start, end):
    before = [row for row in samples if row["timestamp"] <= start]
    after = [row for row in samples if row["timestamp"] >= end]
    if not before or not after:
        return {"complete_window_coverage": False, "gpu_gib_seconds": None,
                "host_gib_seconds": None}
    rows = [{**before[-1], "timestamp": start},
            *(row for row in samples if start < row["timestamp"] < end),
            {**after[0], "timestamp": end}]
    return {"complete_window_coverage": True,
        **{tier+"_gib_seconds": sum(row[tier+"_gib"]*(nxt["timestamp"]-row["timestamp"])
            for row, nxt in zip(rows, rows[1:])) for tier in ("gpu", "host")},
        **{tier+"_peak_gib": max(row[tier+"_gib"] for row in rows)
           for tier in ("gpu", "host")}}


def attribute_physical(samples, windows):
    """Share concurrent occupancy once across active requests, retaining idle cost."""
    costs = {key: {"gpu_gib_seconds": 0.0, "host_gib_seconds": 0.0} for key in windows}
    boundaries = sorted({*(row["timestamp"] for row in samples),
                         *(value for window in windows.values() for value in window)})
    for start, end in zip(boundaries, boundaries[1:]):
        middle = (start+end)/2
        active = [key for key, (left, right) in windows.items() if left <= middle < right]
        observed = [row for row in samples if row["timestamp"] <= start]
        if active and observed:
            for key in active:
                for tier in ("gpu", "host"):
                    costs[key][tier+"_gib_seconds"] += observed[-1][tier+"_gib"]*(end-start)/len(active)
    return costs


class StudyResourceSampler:
    def __init__(self, gpu_devices, process_ids, *, interval=.5, sample=physical_sample):
        self.gpu_devices, self.process_ids, self.interval = gpu_devices, process_ids, interval
        self.sample = sample
        self.samples = []
        self._stop = asyncio.Event()
        self._task = None

    async def start(self):
        devices = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],text=True)
        self.gpu_uuids = {row[1].strip() for row in csv.reader(io.StringIO(devices)) if int(row[0]) in self.gpu_devices}
        self.samples.append(await asyncio.to_thread(self.sample, self.gpu_uuids, self.process_ids))
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        while True:
            try:
                await asyncio.wait_for(self._stop.wait(), self.interval)
                return
            except TimeoutError:
                self.samples.append(await asyncio.to_thread(self.sample, self.gpu_uuids, self.process_ids))

    async def stop(self):
        self._stop.set()
        await self._task
        self.samples.append(await asyncio.to_thread(self.sample, self.gpu_uuids, self.process_ids))
