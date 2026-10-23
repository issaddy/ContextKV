"""Measure candidate construction, quote RPC and planning without executing candidates."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import time

from bench.metrics import distribution
from controller.candidates import build_candidates
from controller.config import create_controller, load_config
from controller.evidence import make_evidence
from controller.planner import choose_plan
from controller.schema import SessionSnapshot
from controller.serialization import to_jsonable


async def measure(config, *, samples: int = 100, warmup: int = 10, session_id: str | None = None):
    if samples < 2 or warmup < 0:
        raise ValueError("at least two samples and nonnegative warmup are required")
    controller = create_controller(config)
    try:
        info = await controller.engine.info()
        snapshot = controller.store.get_snapshot(session_id) if session_id else SessionSnapshot("control-profile")
        blocks = (make_evidence("Request timeout is 30 seconds; connection timeout is 5 seconds.", "timeout"),)
        rows = []
        for index in range(warmup + samples):
            t0 = time.perf_counter()
            candidates = build_candidates(snapshot, "Compare the two timeout settings.", blocks,
                                          controller.policy, controller.template)
            t1 = time.perf_counter()
            quotes = await controller.engine.quote_batch(tuple(c for c in candidates if c.feasible), controller.template.fingerprint)
            t2 = time.perf_counter()
            choose_plan(candidates, quotes, controller.cost_model, snapshot, controller.knobs)
            t3 = time.perf_counter()
            if index >= warmup:
                rows.append({"build_ms": (t1-t0)*1000, "quote_ms": (t2-t1)*1000,
                             "plan_ms": (t3-t2)*1000, "total_ms": (t3-t0)*1000})
        summary = {key: distribution(row[key] for row in rows) for key in rows[0]}
        return {"measurement_kind": "control_path_wall_time", "engine_kind": info.engine_kind,
                "clock": "client_monotonic", "samples": samples, "warmup": warmup,
                "configuration": to_jsonable(config), "fingerprint": to_jsonable(controller.template.fingerprint),
                "snapshot_version": snapshot.version, "summary": summary, "observations": rows,
                "delta_abs_2x_p95_ms": 2 * summary["total_ms"]["p95"],
                "scope": "candidate/template + quote + decision; excludes retrieval and persistence",
                "hardware_generation_measured": False,
                "note": "Mock engine quote timing cannot stand in for a remote SGLang RPC. Use representative saved sessions on the target system."}
    finally:
        if hasattr(controller.engine, "aclose"):
            await controller.engine.aclose()
        # Profiling must not flush retired epochs or mutate the cache lifecycle.
        controller.store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--session-id")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", default="runs/control-profile.json")
    args = parser.parse_args(argv)
    report = asyncio.run(measure(load_config(args.config), samples=args.samples,
                                 warmup=args.warmup, session_id=args.session_id))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "engine_kind": report["engine_kind"],
                      "delta_abs_2x_p95_ms": report["delta_abs_2x_p95_ms"]}))


if __name__ == "__main__":
    main()
