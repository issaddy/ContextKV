"""Export native fixed-reader traces and independently audit their outputs."""
from __future__ import annotations

import argparse
import asyncio
from functools import partial
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse

import yaml

from training.reader_data import export_reader_traces, review_reader_traces


async def measured_command(config, command, function):
    from bench.study_resources import StudyResourceSampler, integrate_physical, listening_processes
    settings = config["reader_audit"]
    invocation_id = str(time.time_ns())
    processes, devices = [os.getpid()], []
    if command == "review":
        devices = settings["review"]["gpu_devices"]
        processes.extend(listening_processes([urlparse(url).port for url in settings["review"]["base_urls"]]))
    sampler = StudyResourceSampler(devices, processes,
        interval=settings.get("resource_sample_seconds", .5))
    await sampler.start()
    started = time.monotonic()
    completed = False
    try:
        result = await function() if command == "review" else await asyncio.to_thread(function)
        completed = True
    finally:
        ended = time.monotonic()
        await sampler.stop()
        output = Path(settings["data"]["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        (output/f"{command}_physical_resources.{invocation_id}.json").write_text(
            json.dumps(sampler.observation(), indent=2)+"\n")
        cost = {"command": command, "start": started, "end": ended,
            "invocation_id": invocation_id, "status": "completed" if completed else "failed",
            "complete_ms": (ended-started)*1000, "physical": integrate_physical(sampler.samples, started, ended),
            "gpu_devices": devices,
            "physical_counts": "each physical GPU counted once; PSS summed across declared process roots and descendants"}
        (output/f"{command}_cost.{invocation_id}.json").write_text(json.dumps(cost, indent=2)+"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "review"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit-turns", type=int)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    operation = (partial(export_reader_traces, config) if args.command == "export" else
        partial(review_reader_traces, config, limit_turns=args.limit_turns, concurrency=args.concurrency))
    result = asyncio.run(measured_command(config, args.command, operation))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
