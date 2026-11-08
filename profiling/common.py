"""Calibration records and run manifests, usable without NumPy or CUDA."""
from __future__ import annotations
import asyncio
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable

from bench.metrics import distribution
from controller.engine_client import EngineError

async def prepare_cache_fixture(engine, mode, **kwargs):
    """Wait for the profiling engine's idle boundary after the last receipt."""
    for attempt in range(20):
        try:
            return await engine.profile_prepare(mode, **kwargs)
        except EngineError as error:
            if error.code != "ENGINE_BUSY" or attempt == 19:
                raise
            await asyncio.sleep(.05)


@dataclass(frozen=True)
class Measurement:
    phase: str
    prompt_tokens: int
    prefix_tokens: int
    output_tokens: int
    concurrency: int
    transfer_bytes: int
    elapsed_ms: float
    source: str
    run_id: str = ""
    repetition: int = 0
    cache_state: str = "unspecified"
    host_load_tokens: int = 0
    timing_scope: str = "server_phase"

    def __post_init__(self):
        if self.phase not in {"prefill", "restore_prefill", "first_token", "decode", "H2D", "D2H", "control"}:
            raise ValueError("unknown measured phase")
        if not math.isfinite(self.elapsed_ms) or self.elapsed_ms < 0:
            raise ValueError("elapsed_ms must be finite and nonnegative")
        if self.concurrency < 1 or min(self.prompt_tokens, self.prefix_tokens, self.output_tokens, self.transfer_bytes) < 0:
            raise ValueError("invalid calibration sizes")
        if self.prefix_tokens > self.prompt_tokens:
            raise ValueError("prefix cannot exceed prompt")
        if not 0 <= self.host_load_tokens <= self.prefix_tokens:
            raise ValueError("host-loaded tokens must be part of the reusable prefix")
        if self.phase == "decode" and self.output_tokens <= 1:
            raise ValueError("decode requires at least two output tokens")
        if self.source not in {"hardware", "mock"}:
            raise ValueError("source must explicitly identify hardware or mock")


def load_measurements(path: str | Path) -> list[Measurement]:
    return [Measurement(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_measurements(rows: Iterable[Measurement], *, allow_mock: bool = False) -> list[dict]:
    buckets: dict[tuple, list[float]] = {}
    for row in rows:
        if row.source != "hardware" and not allow_mock:
            raise ValueError("mock samples cannot calibrate a hardware cost model")
        key = (row.phase, row.prompt_tokens, row.prefix_tokens, row.output_tokens, row.concurrency,
               row.transfer_bytes, row.source, row.host_load_tokens, row.timing_scope)
        buckets.setdefault(key, []).append(row.elapsed_ms)
    names = ("phase", "prompt_tokens", "prefix_tokens", "output_tokens", "concurrency", "transfer_bytes",
             "source", "host_load_tokens", "timing_scope")
    return [{**dict(zip(names, key)), "elapsed_ms": distribution(values)} for key, values in sorted(buckets.items())]


def write_measurements(path: str | Path, rows: Iterable[Measurement]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(asdict(row), sort_keys=True) + "\n")
