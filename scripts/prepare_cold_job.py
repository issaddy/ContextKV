"""Reset an explicitly selected, dedicated research server before a matrix job.

Uses the pinned model/tokenizer/template identity from --config. The server must
enable CONTEXTKV_ENABLE_PROFILING=1. This command generates no model answers and
does not start, restart, or reconfigure a server.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path

from controller.config import load_config
from controller.engine_client import EngineError, SGLangEngineClient
from controller.templates import HuggingFaceTemplate


def _state(metrics):
    return {key: metrics[key] for key in ("engine_restart_id", "observed_seq", "observed_time_ns",
            "clock", "pool", "cache_config", "active_requests", "queued_requests")}


async def prepare_cold(engine):
    info = await engine.info()
    if info.engine_kind != "sglang" or not info.profiling_enabled:
        raise ValueError("cold preparation requires a dedicated SGLang with CONTEXTKV_ENABLE_PROFILING=1")
    before = _state(await engine.metrics())
    # Match the profiler's bounded retry for final scheduler housekeeping.
    # Other errors, including a lost response, are not blindly retried.
    for attempt in range(20):
        try:
            reset = await engine.profile_prepare("cold")
            break
        except EngineError as error:
            if error.code != "ENGINE_BUSY" or attempt == 19:
                raise
            await asyncio.sleep(.05)
    after = _state(await engine.metrics())
    cold_verified = (after["pool"]["unique_gpu_tokens"] == 0
                     and after["pool"]["unique_host_tokens"] == 0
                     and after["active_requests"] == 0 and after["queued_requests"] == 0)
    return {"status": "ready" if cold_verified else "cold_not_verified",
            "cache_state_protocol": "dedicated_research_cold_reset_no_generation_warmup",
            "answer_generation": False, "cold_verified": cold_verified,
            "prepare_attempts": attempt + 1, "engine_before": asdict(info),
            "before": before, "reset": reset, "after": after}


async def _run(args):
    config = load_config(args.config)
    if config.engine.kind != "sglang":
        raise ValueError("cold preparation requires engine.kind=sglang in --config")
    template = HuggingFaceTemplate(config.model.name, config.model.revision,
        namespace_hash=config.model.namespace_hash, local_files_only=config.model.local_files_only)
    engine = SGLangEngineClient(args.base_url, template.fingerprint,
                               timeout=config.engine.timeout_seconds, stream=False)
    try:
        report = await prepare_cold(engine)
    finally:
        await engine.aclose()
    report.update(base_url=args.base_url, config=str(Path(args.config).resolve()),
                  job={key.lower(): os.environ.get(f"CONTEXTKV_JOB_{key}")
                       for key in ("ID", "SPLIT", "PROTOCOL", "METHOD")})
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0 if report["cold_verified"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="explicit dedicated research server URL")
    parser.add_argument("--config", required=True, help="run config containing the pinned model/tokenizer identity")
    parser.add_argument("--output", help="optional JSON copy; stdout is captured by before_job")
    return asyncio.run(_run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
