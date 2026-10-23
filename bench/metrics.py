"""Metrics use actual timestamps/counters; absent engine telemetry stays absent."""
from __future__ import annotations
from dataclasses import dataclass
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


@dataclass(frozen=True)
class TurnTiming:
    conversation_id: str
    task_id: str
    status: str
    output_tokens: int
    t_arrive: float
    t_first: float | None = None
    t_last_token: float | None = None
    t_response_done: float | None = None
    t_commit: float | None = None
    t_submit: float | None = None
    t_admit: float | None = None
    t_retire_ack: float | None = None
    token_observation: str = "unknown"  # token_events, chunk_events, or unknown

    @property
    def ttft(self) -> float | None:
        return None if self.t_first is None else self.t_first - self.t_arrive

    @property
    def tpot(self) -> float | None:
        if self.output_tokens <= 1 or self.t_first is None or self.t_last_token is None:
            return None
        return (self.t_last_token - self.t_first) / (self.output_tokens - 1)

    @property
    def completed_at(self) -> float | None:
        if self.t_response_done is None or self.t_commit is None:
            return None
        return max(self.t_response_done, self.t_commit)

    @property
    def turn_seconds(self) -> float | None:
        return None if self.completed_at is None else self.completed_at - self.t_arrive

    @property
    def queue_seconds(self) -> float | None:
        return None if self.t_submit is None or self.t_admit is None else self.t_admit - self.t_submit


def summarize_timings(turns: Sequence[TurnTiming], *, window_start: float, window_end: float,
                      ttft_slo: float, tpot_slo: float, single_token_policy: str = "ttft_only") -> dict:
    if window_end <= window_start or any(not math.isfinite(v) or v < 0 for v in (ttft_slo, tpot_slo)):
        raise ValueError("invalid observation window or SLO")
    if single_token_policy not in {"ttft_only", "exclude"}:
        raise ValueError("single-token policy must be ttft_only or exclude")
    duration = window_end - window_start
    success = [t for t in turns if t.status == "success"]
    in_window = [t for t in success if t.completed_at is not None and window_start <= t.completed_at < window_end]
    good = [t for t in in_window if t.ttft is not None and t.ttft <= ttft_slo
            and ((t.tpot is not None and t.tpot <= tpot_slo)
                 or (t.output_tokens == 1 and single_token_policy == "ttft_only"))]
    statuses = {s: sum(t.status == s for t in turns) for s in sorted({t.status for t in turns})}
    return {"observation_seconds": duration, "completed_in_window": len(in_window),
            "slo": {"ttft_seconds": ttft_slo, "tpot_seconds": tpot_slo},
            "throughput_requests_per_second": len(in_window) / duration,
            "slo_goodput_requests_per_second": len(good) / duration,
            "ttft_seconds": distribution(t.ttft for t in success if t.ttft is not None),
            "tpot_seconds": distribution(t.tpot for t in success if t.tpot is not None),
            "turn_seconds": distribution(t.turn_seconds for t in success if t.turn_seconds is not None),
            "queue_seconds": distribution(t.queue_seconds for t in turns if t.queue_seconds is not None),
            "undefined_tpot_count": sum(t.tpot is None for t in success),
            "single_token_policy": single_token_policy, "statuses": statuses,
            "failure_fraction": (len(turns) - len(success)) / len(turns) if turns else None,
            "token_observation": {kind: sum(t.token_observation == kind for t in turns)
                                  for kind in sorted({t.token_observation for t in turns})}}


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


def transfer_totals(events: Sequence[Mapping]) -> dict:
    """Count completed transfers once by transfer_id; pending stays separate."""
    completed = {}
    pending = {}
    for event in events:
        identity = str(event["transfer_id"])
        direction = event["direction"]
        if direction not in {"H2D", "D2H"} or int(event["bytes"]) < 0:
            raise ValueError("invalid transfer event")
        if event["status"] == "completed":
            if identity in completed and completed[identity] != event:
                raise ValueError("conflicting duplicate completion")
            completed[identity] = event
            pending.pop(identity, None)
        elif identity not in completed:
            pending[identity] = event
    return {"completed_bytes": {d: sum(int(e["bytes"]) for e in completed.values() if e["direction"] == d) for d in ("H2D", "D2H")},
            "pending_bytes": {d: sum(int(e["bytes"]) for e in pending.values() if e["direction"] == d) for d in ("H2D", "D2H")},
            "completed_transfers": len(completed), "pending_transfers": len(pending)}
