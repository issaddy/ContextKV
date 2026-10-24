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
from controller.sampling import request_sampling
from controller.templates import token_hash
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
                stream.write(json.dumps({"run_id": self.run_id, "repetition": self.repetition,
                                         "warmup": self.warmup, **record}) + "\n")

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
        device_prefix, host_prefix = case.prefix_split()
        reset = await self._fixture("cold")
        if case.cache_state == "mixed":
            page = reset["pool"]["page_size"]
            if case.prefix_tokens % page or device_prefix % page:
                raise ValueError("mixed prefixes must align to the actual native page size")
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
            await self._fixture("host_only" if case.cache_state in {"host_only", "mixed"} else "settle")
            if case.cache_state == "mixed":
                # Reuse native matching/splitting: restore only the shorter GPU
                # prefix, while a divergent sentinel leaves the target tail on host.
                warms = []
                for ids in self._inputs:
                    sentinel = next(t for t in self.tokens if t != ids[device_prefix])
                    warms.append(self._generate(ids[:device_prefix]+(sentinel,), 1))
                await asyncio.gather(*warms)
                await self._fixture("settle")
        quotes = await self.engine.quote_batch(
            [self._candidate(ids, index) for index, ids in enumerate(self._inputs)], self.engine.fingerprint)
        for quote in quotes.quotes:
            if case.cache_state == "mixed":
                observed = (quote.ready_gpu_prefix_tokens, quote.host_only_load_tokens)
                if (observed != (device_prefix, host_prefix)
                        or quote.recoverable_prefix_tokens != case.prefix_tokens
                        or quote.inflight_h2d_tokens or quote.inflight_d2h_tokens):
                    self._write_telemetry({"event": "fixture_failed", "stage": "prepare", "case": case.record(),
                                           "quote": asdict(quotes), "reason": "mixed_prefix_split_or_pending"})
                    raise ValueError(f"cache fixture mixed expected device/host "
                        f"{(device_prefix, host_prefix)}, got {observed}; check capacity, page alignment, "
                        "host write policy, pending transfers and native admission threshold")
            else:
                observed = (quote.ready_gpu_prefix_tokens if case.cache_state != "host_only"
                            else quote.host_only_load_tokens)
                if observed != case.prefix_tokens:
                    raise ValueError(f"cache fixture {case.cache_state} expected {case.prefix_tokens}, got {observed}; "
                                     "check capacity, page alignment, host write policy and native admission threshold")
        self._before = await self.engine.metrics()
        self._write_telemetry({"event": "prepared", "case": case.record(),
                               "quote": asdict(quotes), "metrics": self._before})

    async def probe(self, case: PrefillCase):
        results = await asyncio.gather(*(self._generate(ids, case.output_tokens) for ids in self._inputs))
        # Include generation D2H ACKs, but exclude warm-up writes settled before
        # the start cursor. This happens outside the server request phase times.
        await self._fixture("settle")
        after = await self.engine.metrics(after_seq=self._before["next_seq"])
        rows = []
        if case.cache_state == "mixed":
            expected = case.prefix_split()
            mismatches = []
            for index, result in enumerate(results):
                receipt = result.receipt
                valid = ((receipt.actual_device_hit_tokens, receipt.actual_host_loaded_tokens) == expected
                         and receipt.actual_prefill_tokens == case.prompt_tokens-case.prefix_tokens)
                self._write_telemetry({"event": "request", "case": case.record(), "index": index,
                                       "result": asdict(result), "exact_prefix_split": valid})
                if not valid:
                    mismatches.append(index)
            if mismatches:
                self._write_telemetry({"event": "fixture_failed", "stage": "execution", "case": case.record(),
                                       "request_indices": mismatches, "metrics": after,
                                       "reason": "mixed_execution_prefix_split_or_prefill"})
                raise ValueError(f"mixed execution device/host split or prefill changed for requests {mismatches}")
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
            if case.cache_state != "mixed":
                self._write_telemetry({"event": "request", "case": case.record(), "index": index,
                                       "result": asdict(result)})
        if after["events_truncated"]:
            raise ValueError("telemetry buffer overflow; reduce calibration concurrency or poll metrics more often")
        dma = [e for e in after["events"] if e["event"] == "transfer" and e.get("duration_ms") is not None]
        rows.extend(replace(row, concurrency=case.concurrency, run_id=self.run_id,
                            repetition=self.repetition, timing_scope="cuda_event")
                    for row in from_completed_events(dma))
        self._write_telemetry({"event": "completed", "case": case.record(), "metrics": after})
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
    from controller.config import create_model_template, load_config
    output = Path(args.output)
    telemetry_path = output.with_suffix(".telemetry.jsonl")
    manifest_path = output.with_suffix(".manifest.json")
    if any(path.exists() for path in (output, telemetry_path, manifest_path)):
        raise FileExistsError("profile output already exists; choose a new output prefix")
    config = load_config(args.config)
    runtime_manifest = (json.loads(Path(args.runtime_manifest).read_text())
                        if args.runtime_manifest else None)
    if config.engine.kind != "sglang":
        raise ValueError("use an engine.kind=sglang config")
    template = create_model_template(config.model, model_path=args.model_path, local_files_only=True)
    special = set(template.tokenizer.all_special_ids)
    tokens = sorted(set(template.tokenizer.get_vocab().values()) - special)
    engine = SGLangEngineClient(args.base_url or config.engine.base_url, template.fingerprint,
                                timeout=config.engine.timeout_seconds, sampling=config.sampling, stream=False,
                                **({"model_scope": config.model.scope.manifest()} if config.model.scope else {}))
    probe = SGLangProbe(engine, tokens, seed=args.seed,
                       telemetry_path=telemetry_path)
    try:
        info = await engine.info()
        snapshot = await engine.metrics()
        cases = calibration_grid(args.lengths, args.prefix_fractions, args.outputs,
            args.concurrency, page_size=snapshot["pool"]["page_size"],
            context_limit=config.context.max_total_model_tokens)
        expanded = [replace(case, cache_state=state,
                            device_prefix_tokens=args.device_prefix_tokens if state == "mixed" else None)
                    for case in cases for state in args.cache_states
                    if (state == "cold") == (case.prefix_tokens == 0)]
        for case in expanded:
            device, _ = case.prefix_split()
            if case.cache_state == "mixed" and device % snapshot["pool"]["page_size"]:
                raise ValueError("--device-prefix-tokens must align to the actual native page size")
        if not expanded:
            raise ValueError("no compatible calibration cases")
        output.parent.mkdir(parents=True, exist_ok=True)
        manifest = {"run_id": probe.run_id, "status": "running", "source": "hardware",
                    "engine": asdict(info), "pool": snapshot["pool"],
                    "cache_config": snapshot.get("cache_config"),
                    "runtime_manifest": runtime_manifest,
                    "fingerprint": asdict(template.fingerprint), "seed": args.seed,
                    "sampling_recipe": request_sampling(config.sampling),
                    "generation_protocol": {"ignore_eos": True, "output_budget": "per_case",
                                            "seed_scope": "input_fixture_only_not_request_sampler"},
                    "timing_scope": "server_request; transfer CUDA ACK events",
                    "cases": [case.record() for case in expanded], "repetitions": args.repetitions,
                    "warmup_per_case": args.warmup,
                    "warmup_request_count": sum(case.concurrency for case in expanded)*args.warmup,
                    "warmup_in_training_measurements": False,
                    "prefix_tier_binding": {"mixed": "prepared_and_execution_exact",
                                            "cold_gpu_ready_host_only": "prepared_split_execution_drift_audited"}}
        if config.model.scope is not None:
            manifest["model_scope"] = config.model.scope.manifest()
        # Claim this run before any destructive cache preparation or generation.
        # In particular, concurrent invocations cannot replace the same manifest
        # while appending measurements with different run IDs to one file.
        with manifest_path.open("x") as stream:
            stream.write(json.dumps(manifest, indent=2)+"\n")
        count = 0
        try:
            for case in expanded:
                count += await collect_case(probe,case,warmup=args.warmup,
                                            repetitions=args.repetitions,output_path=output)
        except Exception as error:
            written = sum(bool(line.strip()) for line in output.read_text().splitlines()) if output.exists() else 0
            manifest.update(status="failed", measurement_count=written, completed_case_measurement_count=count,
                            error=f"{type(error).__name__}: {error}")
            manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
            raise
        manifest.update(status="complete", measurement_count=count)
        manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
        print(json.dumps({"status": "complete", "measurement_count": count, "output": str(output)}))
    finally:
        await engine.aclose()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qwen3_8b_scope.example.yaml")
    parser.add_argument("--model-path", default=os.environ.get("CONTEXTKV_MODEL_PATH"))
    parser.add_argument("--base-url", help="Override config URL for the isolated profiler server")
    parser.add_argument("--output", required=True)
    parser.add_argument("--runtime-manifest", help="Bind the JSON emitted by profiling.runtime_manifest unchanged")
    parser.add_argument("--lengths", type=lambda s: tuple(map(int, s.split(","))), default=(1024,4096,8192))
    parser.add_argument("--prefix-fractions", type=lambda s: tuple(map(float, s.split(","))), default=(0,.5,.99))
    parser.add_argument("--outputs", type=lambda s: tuple(map(int, s.split(","))), default=(128,512))
    parser.add_argument("--concurrency", type=lambda s: tuple(map(int, s.split(","))), default=(1,4))
    parser.add_argument("--cache-states", type=lambda s: tuple(s.split(",")), default=("cold","gpu_ready","host_only"),
                        help="cold,gpu_ready,host_only,mixed; mixed needs --device-prefix-tokens")
    parser.add_argument("--device-prefix-tokens", type=int,
                        help="prepared GPU part of a mixed prefix; total prefix still uses --prefix-fractions")
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
