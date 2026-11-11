"""Metrics use actual timestamps/counters; absent engine telemetry stays absent."""
from __future__ import annotations
import math
from statistics import mean, median
from typing import Iterable, Mapping, Sequence


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    if not 0 <= q <= 1:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values: Iterable[float]) -> dict:
    data = list(values)
    return {"count": len(data), "mean": mean(data) if data else None,
            "median": median(data) if data else None, "p90": percentile(data, .9),
            "p95": percentile(data, .95), "p99": percentile(data, .99)}


def occupancy_integral(samples: Sequence[Mapping[str, float]], *, end_timestamp: float | None = None) -> dict:
    """Integrate actual global unique-slot samples, not per-epoch summed residency."""
    rows = sorted(samples, key=lambda r: r["timestamp"])
    if not rows:
        return {"gpu_peak_slots": None, "host_peak_slots": None, "gpu_slot_seconds": None, "host_slot_seconds": None}
    if any(r[tier] < 0 for r in rows for tier in ("gpu_slots", "host_slots")):
        raise ValueError("occupancy cannot be negative")
    end = rows[-1]["timestamp"] if end_timestamp is None else end_timestamp
    if end < rows[-1]["timestamp"]:
        raise ValueError("end precedes last sample")
    result = {}
    for tier in ("gpu", "host"):
        key = tier + "_slots"
        result[tier + "_peak_slots"] = max(r[key] for r in rows)
        result[tier + "_slot_seconds"] = sum(r[key] * ((rows[i + 1]["timestamp"] if i + 1 < len(rows) else end) - r["timestamp"])
                                                       for i, r in enumerate(rows))
    result["measurement"] = "global_unique_slots"
    return result


def merge_ranges(ranges):
    merged = []
    for left, right in sorted(ranges):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return merged
