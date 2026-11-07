"""Request-local receipts for paid auxiliary model calls."""
from contextlib import contextmanager
from contextvars import ContextVar
import time
import uuid


_capture = ContextVar("auxiliary_spend_capture", default=None)


@contextmanager
def capture_auxiliary_spend(store, attempt_id, phase):
    started = time.monotonic()
    group = {"call_id": uuid.uuid4().hex, "attempt_id": attempt_id, "phase": phase,
             "kind": "provider_usage", "status": "SUBMITTED", "started_monotonic": started,
             "input_tokens": None, "output_tokens": None, "elapsed_ms": None}
    store.save_auxiliary_call(group)
    observation = {"calls": [], "provider_usage": None, "group_id": group["call_id"]}
    token = _capture.set((store, attempt_id, phase, observation))
    try:
        yield observation
    finally:
        _capture.reset(token)
        usage = reconcile_auxiliary_usage(observation)
        store.save_auxiliary_call({**group, "status": "RECORDED",
            "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
            "compute_ms": usage["compute_ms"], "elapsed_ms": (time.monotonic()-started)*1000})


def record_provider_usage(usage):
    capture = _capture.get()
    if capture is not None:
        capture[3]["provider_usage"] = usage


@contextmanager
def auxiliary_call(kind):
    capture = _capture.get()
    receipt = {"status": "SUBMITTED", "input_tokens": None, "output_tokens": None}
    if capture is None:
        yield receipt
        return
    store, attempt_id, phase, observation = capture
    start = time.monotonic()
    receipt.update(call_id=uuid.uuid4().hex, attempt_id=attempt_id, phase=phase, group_id=observation["group_id"],
                   kind=kind, started_monotonic=start, elapsed_ms=None)
    store.save_auxiliary_call(receipt)
    try:
        yield receipt
    finally:
        if receipt["status"] == "SUBMITTED":
            receipt["status"] = "UNKNOWN"
        receipt["elapsed_ms"] = (time.monotonic()-start)*1000
        store.save_auxiliary_call(receipt)
        observation["calls"].append(receipt)


def auxiliary_call_usage(calls):
    totals = {name: (sum(call[name] for call in calls)
                    if calls and all(call[name] is not None for call in calls) else None)
              for name in ("input_tokens", "output_tokens")}
    return {**totals, "compute_ms": sum(call["elapsed_ms"] for call in calls) if calls else None,
            "models": list(calls)}


def reconcile_auxiliary_usage(observation, reported=None):
    calls = observation["calls"]
    if reported is None:
        reported = observation["provider_usage"]
    if reported is None:
        return auxiliary_call_usage(calls)
    return {**reported, **{name: None for name in ("input_tokens", "output_tokens")
                          if any(call[name] is None for call in calls)}}
