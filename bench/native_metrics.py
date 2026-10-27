"""Sample existing SGLang pool gauges for the shared owned-engine comparison.

The pinned FULL-attention engine reports free GPU slots and used Host slots.
Occupied GPU slots are capacity minus free slots, including evictable cache.
This is a sampled estimate from published gauges, not allocator-event tracing.
"""
from __future__ import annotations

import asyncio
import math
import time

import httpx


def pool_snapshot(text, *, gpu_capacity, host_enabled, model):
    from prometheus_client.parser import text_string_to_metric_families

    names = {"sglang:kv_available_tokens", "sglang:kv_evictable_tokens", "sglang:kv_used_tokens",
             "sglang:hicache_host_used_tokens"}
    values = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name not in names or sample.labels.get("model_name") != model:
                continue
            if any(sample.labels.get(rank, "0") != "0"
                   for rank in ("tp_rank", "pp_rank", "dp_rank", "moe_ep_rank")):
                continue
            if sample.name in values:
                raise ValueError("ambiguous native pool gauge for the selected model/rank")
            values[sample.name] = sample.value
    free = values.get("sglang:kv_available_tokens")
    host = values.get("sglang:hicache_host_used_tokens") if host_enabled else 0.0
    if free is None or host is None:
        raise ValueError("native pool gauges not published yet")
    if free == 0 and values.get("sglang:kv_evictable_tokens") == 0 and values.get("sglang:kv_used_tokens") == 0:
        raise ValueError("native pool gauges still contain their uninitialized zeros")
    if not (math.isfinite(free) and 0 <= free <= gpu_capacity and math.isfinite(host) and host >= 0):
        raise ValueError("native pool gauges outside the configured capacity")
    return {"gpu_slots": gpu_capacity-free, "host_slots": host}


class NativePoolSampler:
    def __init__(self, base_url, *, gpu_capacity, host_enabled, model, initial_empty=False,
                 interval=.5, transport=None):
        self.base_url = base_url.rstrip("/")
        self.options = dict(gpu_capacity=gpu_capacity, host_enabled=host_enabled, model=model)
        self.interval = interval
        self.initial_empty = initial_empty
        self.samples, self.errors = [], []
        self._client = httpx.AsyncClient(timeout=2, transport=transport)
        self._stop = asyncio.Event()
        self._task = None

    async def _sample(self):
        start = time.monotonic()
        try:
            response = await self._client.get(self.base_url+"/metrics")
            response.raise_for_status()
            end = time.monotonic()
            values = pool_snapshot(response.text, **self.options)
            self.samples.append({"timestamp": (start+end)/2, "scrape_seconds": end-start, **values})
        except (httpx.HTTPError, ValueError) as error:
            self.errors.append({"timestamp": time.monotonic(), "error": f"{type(error).__name__}: {error}"})

    async def start(self):
        import prometheus_client.parser  # noqa: F401
        # The owned executor supplies this only without a warmup/before-job hook.
        # Fresh gauges can be uninitialized; record the known state explicitly.
        if self.initial_empty:
            self.samples.append({"timestamp": time.monotonic(), "gpu_slots": 0, "host_slots": 0,
                                 "basis": "fresh_owned_engine_no_generation_warmup"})
        await self._sample()
        self._task = asyncio.create_task(self._run())

    async def _run(self):
        while True:
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
                return
            except TimeoutError:
                await self._sample()

    async def stop(self):
        self._stop.set()
        try:
            if self._task is not None:
                await self._task
                await self._sample()
        finally:
            await self._client.aclose()

    def observation(self):
        return {"source": "native_prometheus", "clock": "client_monotonic",
                "measurement": "sampled_unique_pool_slots", "interval_seconds": self.interval,
                "scope": "owned single-GPU FULL-attention engine; TP0/PP0/DP0; published gauges may lag allocator changes",
                "gpu_capacity": self.options["gpu_capacity"], "host_enabled": self.options["host_enabled"],
                "samples": self.samples, "errors": self.errors}


def pool_series(observation, start, end):
    """Clip samples to the measured request window; do not invent missing boundaries."""
    samples = sorted(observation["samples"], key=lambda row: row["timestamp"])
    before = [row for row in samples if row["timestamp"] <= start]
    after = [row for row in samples if row["timestamp"] >= end]
    series = [row for row in samples if start < row["timestamp"] < end]
    if before:
        series.insert(0, {**before[-1], "timestamp": start})
    if after and series:
        series.append({**series[-1], "timestamp": end})
    gaps = [b["timestamp"]-a["timestamp"] for a,b in zip(samples,samples[1:])]
    return series, {"available": len(series) >= 2, "source": "native_prometheus",
                    "clock": "client_monotonic", "measurement": observation["measurement"],
                    "interval_seconds": observation["interval_seconds"],
                    "max_scrape_gap_seconds": max(gaps, default=None),
                    "complete_window_coverage": bool(before and after),
                    "errors": observation["errors"], "scope": observation["scope"]}
