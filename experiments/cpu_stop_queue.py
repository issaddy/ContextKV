"""CPU reproduction of STOP time features and a labelled synthetic queue.

Run: python -m experiments.cpu_stop_queue
No model weights, generated answers or GPU latency are simulated.
"""
from __future__ import annotations

from collections import deque
import ast
import asyncio
import heapq
import json
import math
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

from adapters.document_compute import DocumentIndex
from adapters.shared_encoding import EvidenceEncoding, QueryEncoding
from adapters.shared_retrieval import SharedEvidenceState
from controller import cooperative
from controller.cooperative import CooperativeController
from controller.cooperative_request import CompiledRequest, EvidenceSelection
from controller.schema import EvidencePool, EvidenceUnit, Need, SessionSnapshot, TaskState


ROOT = Path(__file__).resolve().parents[1]
BASE = "8f9fd328458d7df23e63da7f99b01822c2236679"


def evidence_fixture():
    query = QueryEncoding(TaskState("synthetic", "Find the requested fact.", needs=(Need("n"),)),
                          None, None, None, (), {})
    current = EvidenceUnit("current", (), "Selected evidence", body_tokens=128)
    pending = EvidenceUnit("pending", (), "Unencoded evidence", body_tokens=256)
    evidence = EvidenceEncoding(("current",), None, None, (), None)
    state = SharedEvidenceState(query, evidence, EvidencePool((current,)), ("current",),
                                DocumentIndex(()), pending_units=(pending,))
    # The snapshot and quote remain fixed; the only experimental input is arrival.
    state.costs = SimpleNamespace(
        costs={"prefill_token_ms": 0.5, "transfer_token_ms": 0.1},
        snapshot={"queue_depth": 0},
        quote=lambda ids: {"milliseconds": 1000.0, "gpu_tokens": 64, "host_tokens": 0},
    )
    state.evidence.summary = lambda *args: None  # No embeddings or trained head are evaluated.
    return state


def baseline_serve_turn():
    source = subprocess.check_output(["git", "show", BASE + ":controller/cooperative.py"], cwd=ROOT, text=True)
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == "CooperativeController")
    method = next(node for node in cls.body
                  if isinstance(node, ast.AsyncFunctionDef) and node.name == "serve_turn")
    namespace = dict(vars(cooperative))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 "baseline/controller/cooperative.py", "exec"), namespace)
    return namespace["serve_turn"]


async def run_turn(method, wait):
    """Real serve_turn/executor/_value_state/_finish with timed provider/engine stubs."""
    clock, observed, committed = [1000.0], [], []
    state = evidence_fixture()
    store = SimpleNamespace(read_snapshot=lambda *args: SessionSnapshot("session"),
        committed_facts=lambda snapshot: (), commit=lambda *args, **kwargs: committed.append(kwargs),
        fail=lambda *args, **kwargs: None)

    async def retrieve(*args, **kwargs):
        clock[0] += 1.5
        return EvidenceSelection((), shared_state=state, auxiliary_usage={"compute_ms": 1500.0})

    async def prepare(*args):
        clock[0] += 0.5

    async def answer_request(*args):
        return CompiledRequest((1,), (), (), (), (), (), ())

    async def compute(*args, **kwargs):
        clock[0] += 3.0
        return {"text": "synthetic", "output_ids": (2,), "elapsed_ms": 3000.0,
                "meta": {"prompt_tokens": 1, "completion_tokens": 1, "finish_reason": {"type": "stop"},
                         "contextkv_compute": {"device_hit_tokens": 0, "host_loaded_tokens": 0,
                                               "prefill_tokens": 1, "new_compute_kv_pages": 1}}}

    def choose(summary, features, prices, **kwargs):
        observed.append(features)
        return "finish"

    # Avoid loading unavailable model assets; exercise the actual contextkv branch below.
    owner = CooperativeController(store, SimpleNamespace(compute=compute), None,
        settings={"max_total_tokens": 131072},
        policy=SimpleNamespace(max_total_model_tokens=32768, max_new_tokens=1024), dialogue_mode="rag")
    owner.dialogue_mode = "contextkv"
    owner.stop, owner.prices = SimpleNamespace(choose=choose), None
    owner.prepare_state, owner.answer_request = prepare, answer_request
    time_api = SimpleNamespace(monotonic=lambda: clock[0])
    with patch.object(cooperative, "time", time_api), patch.dict(method.__globals__, {"time": time_api}):
        response = await method(owner, "session", state.query.task.raw_question,
            request_id="request", task_id="synthetic", provider=SimpleNamespace(retrieve=retrieve),
            request_arrived=1000.0 - wait)
    assert len(observed) == len(committed) == 1
    assert response.complete_ms == committed[0]["metrics"]["complete_ms"] == (5.0 + wait) * 1000
    return observed[0], response.complete_ms / 1000


async def time_feature_reproduction():
    before = baseline_serve_turn()
    cases, reference = [], None
    for wait in (0.0, 10.0, 100.0, 500.0):
        old, old_complete = await run_turn(before, wait)
        new, new_complete = await run_turn(CooperativeController.serve_turn, wait)
        assert old["elapsed_seconds"] == 2.0 + wait
        assert new["elapsed_seconds"] == 2.0
        assert old_complete == new_complete == 5.0 + wait
        changed = [key for key in old if old[key] != new[key]]
        assert changed == (["elapsed_seconds"] if wait else [])
        if reference is None:
            reference = new
        assert new == reference
        cases.append({"external_queue_seconds": wait, "training_elapsed_seconds": 2.0,
                      "before_elapsed_seconds": old["elapsed_seconds"],
                      "after_elapsed_seconds": new["elapsed_seconds"], "changed_fields": changed,
                      "before_complete_seconds": old_complete, "after_complete_seconds": new_complete})
    return {"fixture": "Unmodified baseline serve_turn extracted from git AST versus working-tree serve_turn. Real executor, _value_state and _finish; synthetic provider/prepare 2 s and engine 3 s; STOP stub captures features then finishes.",
            "baseline_features": reference, "cases": cases,
            "stop_decision_evaluated": False,
            "reason": "Trained STOP weights are not available in this checkout."}


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    left, right = math.floor(position), math.ceil(position)
    return values[left] + (values[right] - values[left]) * (position - left)


def queue_simulation(turns_per_session, *, concurrency=4, service_seconds=2.0):
    """FIFO jobs, one outstanding request per session, zero think time."""
    pending = deque((session, 0, 0.0) for session in range(len(turns_per_session)))
    active = []
    sequence, now = 0, 0.0
    response, waiting = [], []
    while pending or active:
        while pending and len(active) < concurrency:
            session, turn, arrived = pending.popleft()
            heapq.heappush(active, (now + service_seconds, sequence, session, turn, arrived, now))
            sequence += 1
        now = active[0][0]
        finished = []
        while active and active[0][0] == now:
            finished.append(heapq.heappop(active))
        for ended, _, session, turn, arrived, started in finished:
            response.append(ended - arrived)
            waiting.append(started - arrived)
            if turn + 1 < turns_per_session[session]:
                pending.append((session, turn + 1, ended))
    return {"sessions": len(turns_per_session), "requests": len(response),
            "concurrency": concurrency, "service_seconds_per_request": service_seconds,
            "total_service_seconds": len(response) * service_seconds,
            "batch_makespan_seconds": now,
            "p95_response_seconds": percentile(response, 0.95),
            "p95_queue_seconds": percentile(waiting, 0.95)}


def main():
    total = 661 * 6
    batch = queue_simulation([6] * 661)
    active = queue_simulation([total // 4 + (i < total % 4) for i in range(4)])
    assert batch["requests"] == active["requests"] == total
    assert batch["total_service_seconds"] == active["total_service_seconds"]
    assert batch["batch_makespan_seconds"] == active["batch_makespan_seconds"]
    result = {
        "base_commit": BASE,
        "kind": "CPU source reproduction and synthetic discrete-event simulation",
        "time_feature": asyncio.run(time_feature_reproduction()),
        "queue": {"assumptions": "All first turns arrive at t=0; FIFO capacity 4; constant 2 s service; subsequent turns arrive immediately after prior completion. Equal total requests and service work. No GPU batching, cache, quality or real arrival trace is modelled.",
                  "synthetic_batch_661_sessions": batch,
                  "synthetic_four_active_sessions": active},
        "scope": "The queue values are illustrative, not a replay or estimate of any benchmark P95. The feature test demonstrates input mismatch, not a measured STOP decision or quality change.",
    }
    output = ROOT / "results/cpu_stop_queue_20261116.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
