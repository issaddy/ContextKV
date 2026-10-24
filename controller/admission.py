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
QueuedForwardFunction = Callable[[int, int, int, int], tuple[float, bool]]


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


def estimate_native_budget_wait(candidate_input_tokens: int, resident_gpu_tokens: int,
                                max_new_tokens: int, batch: CacheQuoteBatch | None,
                                forward_fn: ForwardFunction, decode_fn: DecodeFunction,
                                queued_forward_fn: QueuedForwardFunction,
                                development_lengths: Sequence[int], idle_ms: float,
                                page_size: int | None, *,
                                initial_pinned_tokens: int | None = None) -> WaitEstimate:
    """Optional native-budget/empirical-service scenarios, not a scheduler clone.

    Initial running headroom is the native scalar and is released by existing
    request identity. Unmatched budget members remain held. Queued capacity is
    a full cold-request reservation proxy, independent of service length. The
    mean service scenario uses a saved match if its input shape still agrees;
    the P90 scenario recomputes cold. Neither claims that an old match resides
    now. Page/gate formulas mirror FULL native admission; chunks and subsequent
    ratio changes are not simulated. Callback four-arg order is (n, GPU, host, c).
    An observed pure-GPU incremental pin is used only for the initial fit with
    no queued requests. Later lock releases still use the logical proxy below.
    """
    kind = "analytical_native_budget"

    def unavailable(reason):
        return WaitEstimate(0.0, 0.0, False, reason, kind)

    if batch is None or batch.native_running_requests is None or batch.native_waiting_requests is None:
        return unavailable("missing_native_workload")
    capacity = batch.gpu_capacity_tokens
    if capacity is None or capacity <= 0:
        return unavailable("missing_gpu_capacity")
    if page_size is None or page_size <= 0:
        return unavailable("missing_native_page_size")
    ratio, clip = batch.native_new_token_ratio, batch.native_max_new_tokens_clip
    if (ratio is None or clip is None or batch.native_running_budget_tokens is None
            or batch.native_running_budget_requests is None):
        return unavailable("missing_native_budget")
    running, waiting = batch.native_running_requests, batch.native_waiting_requests
    budget_members = batch.native_running_budget_requests
    if any(req.request_id is None for req in (*running, *waiting, *budget_members)):
        return unavailable("missing_native_request_identity")

    def ceil_page(tokens):
        return -(-tokens // page_size) * page_size

    # Do not filter finished/retracted members out of the exact native budget.
    offsets = {req.request_id: min(req.max_new_tokens - req.generated_tokens, clip)
               * (1.0 if req.ignore_eos else ratio) for req in budget_members}
    available = min(capacity, batch.free_gpu_tokens + batch.evictable_gpu_tokens)
    locked = capacity - available
    logical_total = sum(req.kv_allocated_tokens for req in running)
    releases = [locked * req.kv_allocated_tokens / logical_total if logical_total else 0.0
                for req in running]
    resident = min(candidate_input_tokens, max(0, resident_gpu_tokens))
    pinned_proxy = min(resident, batch.evictable_gpu_tokens)
    flags = {"native_initial_budget", "frozen_ratio_full_request_proxy"}
    initial_pin = pinned_proxy
    if not waiting and initial_pinned_tokens is not None:
        initial_pin = initial_pinned_tokens
        flags.add("observed_initial_gpu_pin")
    running_ids = {req.request_id for req in running}
    if any(req.request_id not in running_ids for req in budget_members):
        flags.add("unmatched_native_budget_held")
    if any(req.request_id not in offsets for req in running):
        flags.add("service_member_outside_budget_batch")
    if waiting:
        flags.add("fifo_waiting_cold_capacity")
    queued_capacity = []
    for req in waiting:
        context = req.input_tokens + req.generated_tokens
        paged = ceil_page(context)
        gate = context + min(max(0, req.max_new_tokens-req.generated_tokens), clip) + page_size
        immediate = paged + page_size
        queued_capacity.append((gate, paged, immediate + min(req.max_new_tokens, clip), immediate))

    def remaining(req, upper):
        budget = max(0, req.max_new_tokens - req.generated_tokens)
        if not budget or req.ignore_eos:
            return budget
        values = sorted(min(length, req.max_new_tokens) - req.generated_tokens
                        for length in development_lengths
                        if min(length, req.max_new_tokens) > req.generated_tokens)
        if not values:
            flags.add("development_tail_budget_fallback" if development_lengths else
                      "missing_development_lengths_budget_fallback")
            return budget
        return values[math.ceil(.9 * len(values))-1] if upper else math.ceil(sum(values)/len(values))

    def service(req, steps, concurrency, upper, *, queued):
        if not steps:
            return 0.0, True
        context = req.input_tokens + req.generated_tokens
        first_ms, first_supported = 0.0, True
        if queued:
            device = host = 0
            match = req.last_native_match
            if (not upper and match is not None and match.input_tokens == context
                    and match.observed_time_ns <= batch.observed_time_ns):
                matched = min(context-1, match.matched_prefix_tokens)
                device = min(matched, match.device_prefix_tokens)
                host = min(matched-device, match.host_hit_tokens)
                flags.add("last_match_service_scenario_not_residency")
            else:
                flags.add("cold_service_scenario")
            first_ms, first_supported = queued_forward_fn(context, device, host, concurrency)
            context += 1
            steps -= 1
        elif req.generated_tokens == 0:
            prefix = min(max(0, context-1), req.kv_committed_tokens)
            first_ms, first_supported = forward_fn(context, prefix, concurrency)
            context += 1
            steps -= 1
        decode_ms, decode_supported = decode_fn(context, steps, concurrency) if steps else (0.0, True)
        return first_ms + decode_ms, first_supported and decode_supported

    def scenario(upper):
        # The two native books differ: running decode offset reduces total,
        # never current. Virtual full prefills debit page-ceiled input + a page.
        total = float(available - batch.native_running_budget_tokens)
        current = float(available)
        now = 0.0
        released_current = 0.0
        events = []
        candidate_gate = candidate_input_tokens-resident + min(max_new_tokens, clip) + page_size
        candidate_paged = ceil_page(candidate_input_tokens-resident)
        if candidate_gate + pinned_proxy >= capacity:
            return now, False, "candidate_native_gate_exceeds_capacity"

        def fits(gate, paged, total_available, current_available, pinned=0):
            # Native check_prefill uses strict <; can_allocate uses <=.
            return gate < total_available-pinned and paged <= min(current_available-pinned, total_available-pinned)

        if any(gate >= capacity for gate, *_ in queued_capacity):
            return now, False, "queued_native_gate_exceeds_capacity"
        # If all FIFO reservations and this candidate fit now, no completion
        # forecast is needed, even when an unrelated job's timing is unsupported.
        immediate_total, immediate_current = total, current
        for gate, paged, reserved_total, reserved_current in queued_capacity:
            if not fits(gate, paged, immediate_total, immediate_current):
                break
            immediate_total -= reserved_total
            immediate_current -= reserved_current
        else:
            if fits(candidate_gate, candidate_paged, immediate_total, immediate_current, initial_pin):
                return idle_ms, True, "ok"
        if locked and logical_total:
            flags.add("logical_release_proxy")
        if total < 0:
            flags.add("native_budget_overcommitted")
        running_steps = [remaining(req, upper) for req in running]
        concurrency = max(1, sum(steps > 0 for steps in running_steps))
        for ordinal, (req, release, steps) in enumerate(zip(running, releases, running_steps)):
            duration, supported = service(req, steps, concurrency, upper, queued=False)
            if not supported:
                return now, False, "unsupported_running_timing"
            heapq.heappush(events, (duration, ordinal, release + offsets.get(req.request_id, 0.0), release))

        def advance(gate, paged, *, protect_candidate=False):
            nonlocal now, total, current, released_current
            while True:
                while events and events[0][0] <= now:
                    _, _, total_release, current_release = heapq.heappop(events)
                    total = min(float(capacity), total + total_release)
                    current = min(float(capacity), current + current_release)
                    released_current += current_release
                # A completion may unlock the candidate's previously protected
                # prefix. Do not credit those slots as both reusable and free.
                pinned = min(resident, batch.evictable_gpu_tokens + released_current) if protect_candidate else 0
                if fits(gate, paged, total, current, pinned):
                    return True
                if not events:
                    return False
                now = events[0][0]

        for ordinal, (req, reservation) in enumerate(zip(waiting, queued_capacity), len(running)):
            steps = remaining(req, upper)
            gate, paged, reserved_total, immediate = reservation
            if not advance(gate, paged):
                return now, False, "unattributed_native_capacity_or_budget"
            duration, supported = service(req, steps, len(events)+1, upper, queued=True)
            if not supported:
                return now, False, "unsupported_waiting_timing"
            # Native final-prefill commit uses the full clipped output budget,
            # including for a retracted request. Our full-request proxy holds
            # this reservation until completion, instead of modeling each chunk.
            total -= reserved_total
            current -= immediate
            heapq.heappush(events, (now + duration, ordinal, float(reserved_total), float(immediate)))
        if not advance(candidate_gate, candidate_paged, protect_candidate=True):
            return now, False, "unattributed_native_capacity_or_budget"
        return idle_ms + now, True, "ok"

    mean, mean_supported, mean_reason = scenario(False)
    upper, upper_supported, upper_reason = scenario(True)
    supported = mean_supported and upper_supported
    reason = (";".join(sorted(flags)) if supported else
              mean_reason if not mean_supported else upper_reason)
    return WaitEstimate(mean, max(mean, upper), supported, reason, kind)
