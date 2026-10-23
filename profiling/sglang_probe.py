"""Collect real server phase/DMA observations against an isolated patched SGLang.

The server must opt in with CONTEXTKV_ENABLE_PROFILING=1. Cache preparation is
explicit and destructive only on that dedicated server. No client wall time or
CPU kernel enqueue duration is reported as GPU execution time.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import random
import uuid

from controller.engine_client import EngineError, SGLangEngineClient
from controller.schema import PromptCandidate
from controller.templates import HuggingFaceTemplate, token_hash
from profiling.common import Measurement, write_measurements
from profiling.prefill import PrefillCase, calibration_grid
from profiling.transfers import from_completed_events


class SGLangProbe:
    def __init__(self, engine: SGLangEngineClient, token_ids, *, seed=42, telemetry_path=None):
        self.engine = engine
        self.tokens = tuple(token_ids)
        if len(set(self.tokens)) < 2:
            raise ValueError("supply at least two distinct valid non-special model token IDs")
        self.seed = seed
        self.telemetry_path = Path(telemetry_path) if telemetry_path else None
        self.run_id = uuid.uuid4().hex
        self.repetition = 0
        self.warmup = False
        self._inputs = ()
        self._before = None

    def _write_telemetry(self, record):
        if self.telemetry_path:
            self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
            with self.telemetry_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"run_id": self.run_id, "warmup": self.warmup, **record}) + "\n")

    async def _fixture(self, mode):
        # HTTP completion can precede the last overlapped housekeeping step.
        # Retry only ENGINE_BUSY, never suppress configuration/admission failures.
        for attempt in range(20):
            try:
                return await self.engine.profile_prepare(mode)
            except EngineError as error:
                if error.code != "ENGINE_BUSY" or attempt == 19:
                    raise
                await asyncio.sleep(.05)

    async def _generate(self, ids, output_tokens):
        info = await self.engine.info()
        sid, attempt = uuid.uuid4().hex, uuid.uuid4().hex
        prepared = await self.engine.prepare(sid, uuid.uuid4().hex, info.engine_restart_id)
        try:
            return await self.engine.generate(input_ids=ids, session_id=sid,
                expected_generation=prepared.session_generation,
                expected_restart_id=info.engine_restart_id, attempt_id=attempt,
                fingerprint=self.engine.fingerprint, max_new_tokens=output_tokens, ignore_eos=True)
        finally:
            # A retirement closes admission even when the request failed.
            await self.engine.retire(sid, prepared.session_generation,
                                     uuid.uuid4().hex, info.engine_restart_id, "profiling")

    def _candidate(self, ids, index):
        return PromptCandidate(str(index), "APPEND", 0, (), tuple(ids), token_hash(ids),
                               frozenset(), frozenset())

    async def prepare(self, case: PrefillCase):
        info = await self.engine.info()
        if info.engine_kind != "sglang" or not info.profiling_enabled:
            raise ValueError("profiling requires a dedicated patched SGLang with CONTEXTKV_ENABLE_PROFILING=1")
        if case.cache_state not in {"cold", "gpu_ready", "host_only"}:
            raise ValueError("unknown cache state")
        if case.cache_state == "cold" and case.prefix_tokens:
            raise ValueError("cold cases must request zero reusable prefix")
        if case.cache_state != "cold" and case.prefix_tokens <= 0:
            raise ValueError("warm cases need a nonempty reusable prefix")
        await self._fixture("cold")
        self.repetition += 1
        rng = random.Random(self.seed + self.repetition)
        # Distinct leading tokens prevent cross-request cache hits in the same
        # concurrency bucket; suffix/prefix samples are otherwise reproducible.
        if case.concurrency > len(set(self.tokens)):
            raise ValueError("token vocabulary too small for distinct concurrent prefixes")
        leaders = rng.sample(list(dict.fromkeys(self.tokens)), case.concurrency)
        self._inputs = tuple(tuple([leader] + [rng.choice(self.tokens)
                              for _ in range(case.prompt_tokens-1)]) for leader in leaders)
        if case.prefix_tokens:
            warms = []
            for ids in self._inputs:
                # Cache prefix+sentinel; sentinel differs from the target suffix.
                sentinel = next(t for t in self.tokens if t != ids[case.prefix_tokens])
                warms.append(self._generate(ids[:case.prefix_tokens]+(sentinel,), 1))
            await asyncio.gather(*warms)
            await self._fixture("host_only" if case.cache_state == "host_only" else "settle")
        quotes = await self.engine.quote_batch(
            [self._candidate(ids, index) for index, ids in enumerate(self._inputs)], self.engine.fingerprint)
        for quote in quotes.quotes:
            observed = (quote.ready_gpu_prefix_tokens if case.cache_state != "host_only"
                        else quote.host_only_load_tokens)
            if observed != case.prefix_tokens:
                raise ValueError(f"cache fixture {case.cache_state} expected {case.prefix_tokens}, got {observed}; "
                                 "check capacity, page alignment, host write policy and native admission threshold")
        self._before = await self.engine.metrics()
        self._write_telemetry({"event": "prepared", "case": asdict(case),
                               "quote": asdict(quotes), "metrics": self._before})

    async def probe(self, case: PrefillCase):
        results = await asyncio.gather(*(self._generate(ids, case.output_tokens) for ids in self._inputs))
        # Include generation D2H ACKs, but exclude warm-up writes settled before
        # the start cursor. This happens outside the server request phase times.
        await self._fixture("settle")
        after = await self.engine.metrics(after_seq=self._before["next_seq"])
        rows = []
        for index, result in enumerate(results):
            if not result.generation_complete or result.server_ttft_ms is None:
                raise ValueError("engine did not provide a complete request with native server TTFT")
            receipt = result.receipt
            prefix = receipt.actual_device_hit_tokens + receipt.actual_host_loaded_tokens
            common = dict(prompt_tokens=case.prompt_tokens, prefix_tokens=prefix,
                output_tokens=len(result.output_ids), concurrency=case.concurrency,
                transfer_bytes=0, source="hardware", run_id=self.run_id,
                repetition=self.repetition, cache_state=case.cache_state,
                host_load_tokens=receipt.actual_host_loaded_tokens,
                timing_scope="server_request")
            rows.append(Measurement(phase="first_token", elapsed_ms=result.server_ttft_ms, **common))
            if len(result.output_ids) > 1 and result.decode_ms is not None:
                rows.append(Measurement(phase="decode", elapsed_ms=result.decode_ms, **common))
            self._write_telemetry({"event": "request", "case": asdict(case), "index": index,
                                   "result": asdict(result)})
        if after["events_truncated"]:
            raise ValueError("telemetry buffer overflow; reduce calibration concurrency or poll metrics more often")
        dma = [e for e in after["events"] if e["event"] == "transfer" and e.get("duration_ms") is not None]
        rows.extend(replace(row, concurrency=case.concurrency, run_id=self.run_id,
                            repetition=self.repetition, timing_scope="cuda_event")
                    for row in from_completed_events(dma))
        self._write_telemetry({"event": "completed", "case": asdict(case), "metrics": after})
        return tuple(rows)


async def collect_case(probe, case, *, warmup, repetitions, output_path):
    """Warm every cache/shape/concurrency case without training on warmup runs."""
    probe.warmup = True
    for _ in range(warmup):
        await probe.prepare(case)
        await probe.probe(case)
    probe.warmup = False
    count = 0
    for _ in range(repetitions):
        await probe.prepare(case)
        measured = await probe.probe(case)
        write_measurements(output_path, measured)
        count += len(measured)
    return count


async def _run(args):
    from controller.config import load_config
    config = load_config(args.config)
    runtime_manifest = (json.loads(Path(args.runtime_manifest).read_text())
                        if args.runtime_manifest else None)
    if config.engine.kind != "sglang":
        raise ValueError("use an engine.kind=sglang config")
    template = HuggingFaceTemplate(args.model_path or config.model.name, config.model.revision,
        namespace_hash=config.model.namespace_hash, local_files_only=True)
    special = set(template.tokenizer.all_special_ids)
    tokens = sorted(set(template.tokenizer.get_vocab().values()) - special)
    engine = SGLangEngineClient(args.base_url or config.engine.base_url, template.fingerprint,
                                timeout=config.engine.timeout_seconds, stream=False)
    output = Path(args.output)
    probe = SGLangProbe(engine, tokens, seed=args.seed,
                       telemetry_path=output.with_suffix(".telemetry.jsonl"))
    try:
        info = await engine.info()
        snapshot = await engine.metrics()
        cases = calibration_grid(args.lengths, args.prefix_fractions, args.outputs,
            args.concurrency, page_size=snapshot["pool"]["page_size"],
            context_limit=config.context.max_total_model_tokens)
        expanded = [replace(case, cache_state=state) for case in cases for state in args.cache_states
                    if (state == "cold") == (case.prefix_tokens == 0)]
        if not expanded:
            raise ValueError("no compatible calibration cases")
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"run_id": probe.run_id, "status": "running", "source": "hardware",
                    "engine": asdict(info), "pool": snapshot["pool"],
                    "cache_config": snapshot.get("cache_config"),
                    "runtime_manifest": runtime_manifest,
                    "fingerprint": asdict(template.fingerprint), "seed": args.seed,
                    "timing_scope": "server_request; transfer CUDA ACK events",
                    "cases": [asdict(case) for case in expanded], "repetitions": args.repetitions,
                    "warmup_per_case": args.warmup,
                    "warmup_request_count": sum(case.concurrency for case in expanded)*args.warmup,
                    "warmup_in_training_measurements": False}
        manifest_path = output.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        count = 0
        for case in expanded:
            count += await collect_case(probe,case,warmup=args.warmup,
                                        repetitions=args.repetitions,output_path=output)
        manifest.update(status="complete", measurement_count=count)
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        print(json.dumps({"status": "complete", "measurement_count": count, "output": str(output)}))
    finally:
        await engine.aclose()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sglang.yaml")
    parser.add_argument("--model-path", default=os.environ.get("CONTEXTKV_MODEL_PATH"))
    parser.add_argument("--base-url", help="Override config URL for the isolated profiler server")
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime-manifest", help="Bind the JSON emitted by profiling.runtime_manifest unchanged")
    parser.add_argument("--lengths", type=lambda s: tuple(map(int, s.split(","))), default=(1024,4096,8192))
    parser.add_argument("--prefix-fractions", type=lambda s: tuple(map(float, s.split(","))), default=(0,.5,.99))
    parser.add_argument("--outputs", type=lambda s: tuple(map(int, s.split(","))), default=(128,512))
    parser.add_argument("--concurrency", type=lambda s: tuple(map(int, s.split(","))), default=(1,4))
    parser.add_argument("--cache-states", type=lambda s: tuple(s.split(",")), default=("cold","gpu_ready","host_only"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1, help="Per-case warmup runs excluded from training JSONL")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.repetitions < 1 or args.warmup < 0:
        parser.error("--repetitions must be positive and --warmup nonnegative")
    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
