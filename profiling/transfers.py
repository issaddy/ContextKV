"""Transfer calibration specification. Only completed DMA events are samples."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence
from profiling.common import Measurement


@dataclass(frozen=True)
class TransferCase:
    direction: str
    transfer_bytes: int
    concurrency: int


def transfer_grid(sizes: Sequence[int] = (1 << 20, 16 << 20, 64 << 20, 256 << 20),
                  concurrency: Sequence[int] = (1, 4, 8, 16)) -> tuple[TransferCase, ...]:
    if any(v <= 0 for v in (*sizes, *concurrency)):
        raise ValueError("transfer sizes/concurrency must be positive")
    return tuple(TransferCase(direction, size, count) for direction in ("H2D", "D2H")
                 for size in sizes for count in concurrency)


def from_completed_events(events: Sequence[Mapping]) -> tuple[Measurement, ...]:
    results, seen = [], set()
    for event in events:
        if event.get("status") != "completed":
            continue
        key = str(event["transfer_id"])
        if key in seen:
            continue
        seen.add(key)
        if event.get("clock") not in {"cuda_event", "server_monotonic"}:
            raise ValueError("transfer duration needs one aligned server/CUDA clock")
        if event["direction"] not in {"H2D", "D2H"}:
            raise ValueError("unknown transfer direction")
        results.append(Measurement(event["direction"], 0, 0, 0, int(event.get("concurrency", 1)),
                                   int(event["bytes"]), float(event["duration_ms"]), str(event["source"]),
                                   run_id=str(event.get("run_id", ""))))
    return tuple(results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"status": "not_run", "hardware_measurements": None,
                                 "cases": [asdict(c) for c in transfer_grid()]}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
