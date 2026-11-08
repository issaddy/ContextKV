"""Empirical time reserves for the mandatory remainder of a request."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .admission import quoted_native_work
from .serialization import dumps


def timing_point(task, reserve, batch, quote, paid_observation, timing_context, semantic_kind, *, gap_query_chars=0):
    if paid_observation is None or batch.active_requests is None or batch.queued_requests is None:
        return None
    prefill, host = quoted_native_work(quote) if quote is not None else (0, 0)
    kind = task.kind if task.kind != "REDUCE" else task.operation
    devices = sorted(row["uuid"] for row in paid_observation["samples"][-1]["devices"])
    scope = dumps((timing_context, batch.fingerprint, batch.cache_config_hash, devices, semantic_kind, kind))
    work = (reserve.action_input_tokens, prefill, host, reserve.action_output_tokens,
            len(task.input_claim_ids), reserve.future_input_tokens, reserve.future_output_tokens,
            reserve.future_steps, gap_query_chars, batch.active_requests, batch.queued_requests)
    return scope, work


class CompletionTiming:
    def __init__(self, observations):
        self.observations = observations

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text())["observations"])

    def reserve_ms(self, point):
        if point is None:
            return None
        scope, work = point
        by_session = {}
        for sample in self.observations.get(scope, ()):
            if all(current <= observed for current, observed in zip(work, sample["work"], strict=True)):
                session = sample["session_id"]
                by_session[session] = max(by_session.get(session, 0.0), sample["remaining_ms"])
        if len(by_session) < 2:
            return None
        ordered = sorted(by_session.values())
        return ordered[math.ceil(.95*len(ordered))-1]
