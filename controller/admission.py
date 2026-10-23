"""Stateless, analytical waiting scenarios from a native workload snapshot.

This is neither SGLang admission simulation nor a calibrated heterogeneous queue
model. Native queue order is treated as FIFO, rates are fixed when a virtual job
starts, and logical KV lengths only apportion the observed physical locked pool.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Callable, Sequence

from .schema import CacheQuoteBatch, NativeRequestSummary


ForwardFunction = Callable[[int, int, int], tuple[float, bool]]
DecodeFunction = Callable[[int, int, int], tuple[float, bool]]


@dataclass(frozen=True)
class WaitEstimate:
    mean_ms: float
    upper_ms: float
    supported: bool
    reason: str
    model_kind: str = "analytical_native_workload"


def estimate_wait(candidate_input_tokens: int, resident_gpu_tokens: int,
                  output_mean: int, output_p90: int, batch: CacheQuoteBatch | None,
                  forward_fn: ForwardFunction, decode_fn: DecodeFunction,
                  development_lengths: Sequence[int], idle_ms: float) -> WaitEstimate:
    """Estimate time before the candidate can start forward, excluding its forward.

    Callbacks return (milliseconds, measured-domain support). ``forward_fn`` is
    forward-entry to first token, not total TTFT; ``decode_fn`` prices exactly the
    requested number of future decode steps from its supplied context length.
    ``upper_ms`` is the greater of mean and P90-input scenarios, not a confidence
    bound. No future actual output, request text, or persistent state is used.
    """
    if batch is None:
        return WaitEstimate(0.0, 0.0, False, "missing_native_workload")
    capacity = batch.gpu_capacity_tokens
    if capacity is None or capacity <= 0:
        return WaitEstimate(0.0, 0.0, False, "missing_gpu_capacity")
    running, waiting = batch.native_running_requests, batch.native_waiting_requests
    if running is None or waiting is None:
        return WaitEstimate(0.0, 0.0, False, "missing_native_workload")

    available = min(capacity, batch.free_gpu_tokens + batch.evictable_gpu_tokens)
    locked = capacity - available
    logical_total = sum(req.kv_allocated_tokens for req in running)
    # These are proportions of ONE observed physical counter, not sums of
    # duplicated shared prefixes or promises about native completion releases.
    releases = [locked * req.kv_allocated_tokens / logical_total if logical_total else 0.0
                for req in running]
    resident = min(candidate_input_tokens, max(0, resident_gpu_tokens))
    prefix_reserve = min(resident, batch.evictable_gpu_tokens)
    flags = set()
    if waiting:
        flags.add("fifo_waiting_cold_prefix")

    def remaining(req: NativeRequestSummary, upper: bool) -> int:
        budget = max(0, req.max_new_tokens - req.generated_tokens)
        if not budget or req.ignore_eos:
            return budget
        residuals = sorted(min(length, req.max_new_tokens) - req.generated_tokens
                           for length in development_lengths
                           if min(length, req.max_new_tokens) > req.generated_tokens)
        if not residuals:
            flags.add("development_tail_budget_fallback" if development_lengths else
                      "missing_development_lengths_budget_fallback")
            return budget
        return (residuals[math.ceil(.9 * len(residuals))-1] if upper else
                math.ceil(sum(residuals) / len(residuals)))

    def service(req, tokens, concurrency, *, queued):
        if tokens == 0:
            return 0.0, True
        context = req.input_tokens + req.generated_tokens
        prefill = queued or req.generated_tokens == 0
        forward_ms, forward_supported = 0.0, True
        if prefill:
            # Queued/retracted requests have no per-request cache quote. Charge
            # the whole current input; running prefill uses observed committed KV.
            prefix = 0 if queued else min(max(0, context-1), req.kv_committed_tokens)
            forward_ms, forward_supported = forward_fn(context, prefix, concurrency)
            context += 1
            tokens -= 1  # this forward produces the next output token
        decode_ms, decode_supported = decode_fn(context, tokens, concurrency) if tokens else (0.0, True)
        return forward_ms + decode_ms, forward_supported and decode_supported

    def scenario(upper):
        reserve = output_p90 if upper else output_mean
        needed = candidate_input_tokens - resident + reserve + prefix_reserve
        if needed > capacity:
            return 0.0, False, "candidate_reservation_exceeds_capacity"
        running_tokens = [remaining(req, upper) for req in running]
        waiting_tokens = [remaining(req, upper) for req in waiting]
        waiting_reserves = [req.input_tokens + req.generated_tokens + tokens if tokens else 0
                            for req, tokens in zip(waiting, waiting_tokens)]
        if any(demand > capacity for demand in waiting_reserves):
            return 0.0, False, "queued_reservation_exceeds_capacity"
        # Admission needs future decode headroom as well as currently allocated
        # slots. A negative virtual free value records an overcommitted scenario;
        # it does not assert that native execution can progress without retraction.
        now, free = 0.0, float(available - sum(running_tokens))
        if needed + sum(waiting_reserves) <= free:
            # No completion is needed, so unrelated job timing support is irrelevant.
            return idle_ms, True, "ok"
        if locked and logical_total:
            flags.add("logical_release_proxy")
        if free < 0:
            flags.add("virtual_decode_headroom_overcommitted")
        # (finish time, unique local ordinal, proxy slots). No request identity.
        events = []
        running_concurrency = max(1, sum(tokens > 0 for tokens in running_tokens))
        for index, (req, release, tokens) in enumerate(zip(running, releases, running_tokens)):
            duration, supported = service(req, tokens, running_concurrency, queued=False)
            if not supported:
                return 0.0, False, "unsupported_running_timing"
            heapq.heappush(events, (duration, index, release + tokens))

        def advance_until(needed):
            nonlocal now, free
            while True:
                # All simultaneous completions change capacity and concurrency
                # before the next virtual admission is considered.
                while events and events[0][0] <= now:
                    _, _, release = heapq.heappop(events)
                    free = min(float(capacity), free + release)
                if needed <= free or math.isclose(needed, free, abs_tol=1e-9):
                    return True
                if not events:
                    return False
                now = events[0][0]

        for index, (req, tokens, demand) in enumerate(zip(waiting, waiting_tokens, waiting_reserves), len(running)):
            if not advance_until(demand):
                return now, False, "unattributed_locked_capacity"
            duration, supported = service(req, tokens, len(events)+1, queued=True)
            if not supported:
                return now, False, "unsupported_waiting_timing"
            free -= demand
            heapq.heappush(events, (now + duration, index, float(demand)))

        if not advance_until(needed):
            return now, False, "unattributed_locked_capacity"
        return idle_ms + now, True, "ok"

    mean, mean_supported, mean_reason = scenario(False)
    upper, upper_supported, upper_reason = scenario(True)
    supported = mean_supported and upper_supported
    if not supported:
        reason = mean_reason if not mean_supported else upper_reason
    else:
        reason = ";".join(sorted(flags)) or "idle_capacity_available"
    return WaitEstimate(mean, max(mean, upper), supported, reason)
