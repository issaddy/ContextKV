"""Dependent session workloads: serial turns, concurrent sessions, explicit drain.

The scheduling layer owns raw scripted tasks. A serving handler receives only the
current whitelisted OnlineTurn. Public future questions and labels are inaccessible
through that request. Failed turns stop their session; dependent turns are counted
as unfinished, never executed with invented history.
"""
from __future__ import annotations
import asyncio
import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys
import uuid
from dataclasses import dataclass, replace
import inspect
import random
import time
from typing import Any, Awaitable, Callable, Mapping, Sequence

from adapters.mtrag import OnlineSession, Protocol, group_sessions
from controller.schema import OnlineTurn
from adapters.retrieval import run_retrieval_worker
from controller.config import create_controller, load_config
from controller.evidence import make_evidence
from controller.schema import TurnRecord
from controller.serialization import dumps, to_jsonable


def load_think_time_plan(path, *, expected_sha256=None):
    """Read a frozen per-task gap plan; its digest binds both system arms."""
    if path is None:
        if expected_sha256 is not None:
            raise ValueError("expected a frozen think-time plan")
        return None, None
    source = Path(path)
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("think-time plan changed after matrix provenance was fixed")
    plan = json.loads(payload)
    if not isinstance(plan, dict) or plan.get("schema_version") != 1 or not isinstance(plan.get("gaps_after_task"), dict):
        raise ValueError("think-time plan needs schema_version 1 and gaps_after_task")
    return plan["gaps_after_task"], {"path": str(source.resolve()), "sha256": digest}


@dataclass(frozen=True)
class TurnRunResult:
    conversation_id: str
    task_id: str
    turn: int
    protocol: str
    status: str
    t_arrive: float
    t_done: float
    assistant_text: str | None = None
    error: str | None = None
    result: Any = None
    online_turn: OnlineTurn | None = None
    t_retrieval_done: float | None = None


@dataclass(frozen=True)
class RunReport:
    mode: str
    seed: int
    started_at: float
    arrival_window_end: float
    completed_at: float
    turns: tuple[TurnRunResult, ...]
    scheduled_sessions: int
    arrived_sessions: int
    unfinished_turns: int
    timed_out_sessions: tuple[str, ...]
    session_makespans: tuple[tuple[str, float], ...]
    time_origin: str
    concurrency_scope: str = "session"

    @property
    def drain_seconds(self) -> float:
        return max(0.0, self.completed_at - self.arrival_window_end)


def _answer(result: Any) -> str:
    if isinstance(result, str):
        return result
    for name in ("assistant_text", "text", "answer"):
        value = result.get(name) if isinstance(result, Mapping) else getattr(result, name, None)
        if isinstance(value, str):
            return value
    raise ValueError("handler must return persisted assistant_text (or a string)")


async def run_sessions(sessions: Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]],
                       handler: Callable[[OnlineTurn], Awaitable[Any]], *, protocol: Protocol = "P2",
                       concurrency: int = 1, concurrency_scope: str = "session",
                       think_time: float = 0.0, arrival_rate: float | None = None,
                       think_time_by_task: Mapping[str, float] | None = None,
                       seed: int = 42, arrival_window: float | None = None,
                       drain_timeout: float | None = None, per_turn_timeout: float | None = None,
                       retriever: Any | None = None,
                       retrieval_context_provider: Callable[[OnlineTurn], Any] | None = None,
                       on_result: Callable[[TurnRunResult], Any] | None = None) -> RunReport:
    """Run a fixed cohort or Poisson session arrivals, then drain all arrived work.

    In session scope, concurrency bounds active sessions including think time;
    in turn scope it bounds active retrieval/generation pipelines and a thinking
    session releases its slot. A per-task plan specifies the gap after each
    committed non-final turn. Open-loop arrivals use their scheduled monotonic
    timestamps, including event-loop delay and queueing in response time. In
    closed-loop session scope, a session arrives when its slot becomes available.
    No future turn is submitted until the preceding result is committed. The
    per-turn deadline includes admission, retrieval and generation. Cancellation
    drains owned work before returning; publishing a committed result is outside
    that deadline.
    """
    if concurrency < 1 or think_time < 0:
        raise ValueError("invalid concurrency or think_time")
    if concurrency_scope not in {"session", "turn"}:
        raise ValueError("concurrency_scope must be session or turn")
    for name, value in (("arrival_rate", arrival_rate), ("arrival_window", arrival_window),
                        ("drain_timeout", drain_timeout), ("per_turn_timeout", per_turn_timeout)):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    if protocol == "P3" and retriever is None:
        raise ValueError("P3 requires a real retriever")
    grouped = dict(sessions) if isinstance(sessions, Mapping) else group_sessions(sessions)
    if any(str(sid) != str(turns[0]["conversation_id"]) for sid, turns in grouped.items() if turns):
        raise ValueError("session key must equal conversation_id")
    if think_time_by_task is not None:
        think_time_by_task = dict(think_time_by_task)
        if think_time != 0:
            raise ValueError("choose a fixed think_time or a per-task plan")
        expected = [str(row["task_id"]) for turns in grouped.values()
                    for row in sorted(turns, key=lambda value: int(value["turn"]))[:-1]]
        if len(expected) != len(set(expected)) or set(think_time_by_task) != set(expected):
            raise ValueError("think-time plan must cover every non-final task exactly")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
               for value in think_time_by_task.values()):
            raise ValueError("think-time plan gaps must be finite nonnegative numbers")
    ids = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(ids)
    offsets, offset = [], 0.0
    for sid in ids:
        if arrival_rate is not None:
            offset += rng.expovariate(arrival_rate)
        if arrival_window is not None and offset > arrival_window:
            break
        offsets.append((sid, offset))
    started = time.monotonic()
    semaphore = asyncio.Semaphore(concurrency)
    results: list[TurnRunResult] = []
    makespans: dict[str, float] = {}
    unfinished: dict[str, int] = {}
    arrived: set[str] = set()
    timed_out: set[str] = set()
    arrival_end = started + (arrival_window if arrival_window is not None else (offsets[-1][1] if offsets else 0))

    async def publish(item: TurnRunResult):
        results.append(item)
        if on_result is not None:
            callback_result = on_result(item)
            if inspect.isawaitable(callback_result):
                await callback_result

    async def one_session(sid: str, delay: float):
        gate = OnlineSession(grouped[sid], protocol)
        unfinished[sid] = gate.remaining
        await asyncio.sleep(max(0.0, started + delay - time.monotonic()))
        arrived.add(sid)
        session_arrive = (started + delay if arrival_rate is not None or concurrency_scope == "turn"
                          else time.monotonic())
        active: OnlineTurn | None = None
        turn_arrive = session_arrive
        session_slot = False
        try:
            if concurrency_scope == "session":
                # A closed-loop session has no arrival until admitted. In open
                # loop, however, the first request's deadline includes this wait.
                if arrival_rate is not None:
                    active = gate.current()
                    queue_deadline = (turn_arrive + per_turn_timeout if per_turn_timeout is not None else None)
                    try:
                        async with asyncio.timeout_at(queue_deadline):
                            await semaphore.acquire()
                        session_slot = True
                    except TimeoutError as exc:
                        failed = TurnRunResult(sid, active.task_id, active.turn, protocol, "timeout",
                                               turn_arrive, time.monotonic(), error=f"{type(exc).__name__}: {exc}",
                                               online_turn=active)
                        unfinished[sid] = max(0, gate.remaining - 1)
                        active = None
                        await publish(failed)
                        makespans[sid] = time.monotonic() - session_arrive
                        return
                    active = None
                else:
                    await semaphore.acquire()
                    session_slot = True
                    session_arrive = turn_arrive = time.monotonic()
            while not gate.done:
                active = gate.current()
                retrieval_done = None
                try:
                    deadline = turn_arrive + per_turn_timeout if per_turn_timeout is not None else None
                    # Admission, retrieval and generation share the same budget.
                    async with asyncio.timeout_at(deadline):
                        if deadline is not None and time.monotonic() >= deadline:
                            raise TimeoutError("turn deadline expired before execution")
                        turn_slot = False
                        try:
                            if concurrency_scope == "turn":
                                await semaphore.acquire()
                                turn_slot = True
                            if deadline is not None and time.monotonic() >= deadline:
                                raise TimeoutError("turn deadline expired before execution")
                            if protocol == "P3":
                                if retrieval_context_provider is None:
                                    active = await run_retrieval_worker(retriever.retrieve_turn, active)
                                else:
                                    def retrieve():
                                        context = retrieval_context_provider(active)
                                        return retriever.retrieve_turn(active, prior_context=context)
                                    active = await run_retrieval_worker(retrieve)
                            retrieval_done = time.monotonic()
                            result = await handler(active)
                        finally:
                            if turn_slot:
                                semaphore.release()
                    if (isinstance(result, Mapping) and result.get("success") is False) or getattr(result, "success", True) is False:
                        raise RuntimeError("handler reported unsuccessful generation")
                    answer = _answer(result)
                    gate.commit(active.task_id, answer)
                    unfinished[sid] = gate.remaining
                    completed = TurnRunResult(sid, active.task_id, active.turn, protocol, "success",
                                              turn_arrive, time.monotonic(), answer, result=result,
                                              online_turn=active, t_retrieval_done=retrieval_done)
                except Exception as exc:
                    status = "timeout" if isinstance(exc, TimeoutError) else "failed"
                    failed = TurnRunResult(sid, active.task_id, active.turn, protocol, status,
                                           turn_arrive, time.monotonic(), error=f"{type(exc).__name__}: {exc}",
                                           online_turn=active, t_retrieval_done=retrieval_done)
                    # The failed task is accounted separately from dependent unfinished work.
                    unfinished[sid] = max(0, gate.remaining - 1)
                    active = None
                    await publish(failed)
                    break
                # A sink failure must not reclassify an already committed generation.
                active = None
                await publish(completed)
                if not gate.done:
                    gap = think_time if think_time_by_task is None else think_time_by_task[completed.task_id]
                    turn_arrive = completed.t_done + gap
                    await asyncio.sleep(max(0.0, turn_arrive - time.monotonic()))
            makespans[sid] = time.monotonic() - session_arrive
        except asyncio.CancelledError:
            timed_out.add(sid)
            if active is not None:
                await publish(TurnRunResult(sid, active.task_id, active.turn, protocol, "cancelled",
                                            turn_arrive, time.monotonic(), error="cohort drain timeout", online_turn=active))
                unfinished[sid] = max(0, gate.remaining - 1)
            makespans[sid] = time.monotonic() - session_arrive
            raise
        finally:
            if session_slot:
                semaphore.release()

    tasks = [asyncio.create_task(one_session(sid, delay)) for sid, delay in offsets]
    try:
        if tasks:
            # Fixed-work closed-loop has no prescribed observation window: complete the cohort.
            if drain_timeout is None:
                await asyncio.gather(*tasks)
            else:
                deadline = max(0.0, arrival_end - time.monotonic()) + drain_timeout
                done, pending = await asyncio.wait(tasks, timeout=deadline)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                for task in done:
                    if task.exception():
                        raise task.exception()
            if arrival_window is not None:
                # A finite cohort may finish before the fixed observation window.
                # Keep the agreed denominator/window, without inventing more sessions.
                await asyncio.sleep(max(0.0, arrival_end - time.monotonic()))
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    completed = time.monotonic()
    if arrival_rate is None and arrival_window is None:
        arrival_end = completed
    return RunReport("session_open_loop" if arrival_rate is not None else "closed_loop", seed,
                     started, arrival_end, completed, tuple(sorted(results, key=lambda r: (r.conversation_id, r.turn))),
                     len(offsets), len(arrived), sum(unfinished.values()), tuple(sorted(timed_out)),
                     tuple(sorted(makespans.items())),
                     "pipeline_arrival" if protocol == "P3" else "fixed_retrieval_input_ready", concurrency_scope)


def _manifest(config, controller, info, protocol: str):
    payload = {"schema_version": 1, "run_id": controller.events.run_id,
               "measurement_kind": "simulation" if config.engine.kind == "mock" else "unvalidated_hardware_run",
               "protocol": protocol, "config": to_jsonable(config),
               "fingerprint": to_jsonable(controller.template.fingerprint), "engine": to_jsonable(info),
               "runtime": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
                    "logical_cpu_count": os.cpu_count(),
                    "requested_thread_environment": {key: os.environ.get(key) for key in
                        ("TOKENIZERS_PARALLELISM", "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}},
               "quality_evaluated": False}
    path = Path(config.state_dir) / f"manifest-{controller.events.run_id}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path

async def demo(config):
    service = create_controller(config)
    sid = f"demo-{uuid.uuid4().hex[:8]}"
    a = make_evidence("Client request timeout is 30 seconds in version 1. Retry only idempotent operations.", "client-timeout", anchors=("request_timeout",))
    b = make_evidence("Retry at most three times with exponential backoff. Authentication failures must not be retried.", "retry-policy")
    c = make_evidence("Connection timeout is 5 seconds. It is separate from the request timeout.", "connection-timeout")
    a2 = make_evidence("Client request timeout is 60 seconds in version 2. Retry only idempotent operations.", "client-timeout", version="v2")
    turns = [("What are the timeout and retry settings?", (a, b)),
             ("How does connection timeout differ?", (b, c)),
             ("What is the latest request timeout in version 2?", (a2,)),
             ("Revisit the retry policy and its exceptions.", (b,))]
    results = []
    try:
        info = await service.engine.info()
        simulated = info.engine_kind == "mock"
        manifest = _manifest(config, service, info, "synthetic-correctness-demo")
        hit_field = "simulated_gpu_hit_tokens" if simulated else "actual_device_hit_tokens"
        for index, (query, evidence) in enumerate(turns, 1):
            response = await service.serve_turn(sid, query, evidence, request_id=f"turn-{index}")
            results.append(response)
            print(json.dumps({"turn": index, "action": response.action, "reason": response.reason,
                              "epoch": response.epoch_no, "input_tokens": response.input_tokens,
                              hit_field: response.actual_device_hit_tokens,
                              "answer": response.assistant_text}, ensure_ascii=False))
        await service.close_session(sid)
        result_path = Path(config.state_dir) / f"demo-{service.events.run_id}.json"
        result_path.write_text(dumps({"measurement_kind": "simulation" if simulated else "hardware",
                                     "protocol": "synthetic-correctness-demo", "session_id": sid,
                                     "quality_evaluated": False,
                                     "turns": results}) + "\n", encoding="utf-8")
        note = ("模拟 token/缓存轨迹仅验证协议，不代表 GPU 性能或回答质量。" if simulated else
                "真实引擎上的合成材料协议演示，不建立回答质量或系统性能结论。")
        print(f"结果：{result_path.resolve()}\n运行清单：{manifest.resolve()}\n注意：{note}")
    finally:
        await service.aclose()
        service.store.close()

async def benchmark(args, config):
    from adapters.mtrag import CLOUD_COLLECTION, dataset_identity, load_rows
    from adapters.retrieval import bind_corpus_context, to_evidence
    from adapters.retriever_factory import create_retriever
    from bench.quality import export_evaluation
    from bench.report import summarize_run, markdown_report
    from adapters.corpus import CorpusIndex

    rows = load_rows(args.input)
    dataset = dataset_identity(rows)
    if args.domain == "cloud":
        rows = [r for r in rows if r["Collection"] == CLOUD_COLLECTION]
    if args.split_manifest:
        manifest = json.loads(Path(args.split_manifest).read_text())
        selected = set(manifest[args.split])
        rows = [r for r in rows if r["conversation_id"] in selected]
    if args.limit_sessions:
        ids = set(sorted({r["conversation_id"] for r in rows})[:args.limit_sessions])
        rows = [r for r in rows if r["conversation_id"] in ids]
    think_time_by_task, think_time_plan = load_think_time_plan(args.think_time_plan,
        expected_sha256=args.expected_think_time_plan_sha256)
    corpus_snapshot = f"{dataset['name']}-{dataset['revision']}"
    retriever = create_retriever(args.protocol, config_path=args.retriever_config,
                                index_path=args.retriever_index, corpus_path=args.corpus,
                                corpus_snapshot_id=corpus_snapshot)
    if retriever is not None:
        corpus_snapshot = retriever.corpus_snapshot_id
    corpus_index = (CorpusIndex.from_jsonl(args.evidence_corpus, corpus_id=dataset["name"], snapshot_id=corpus_snapshot)
                    if args.evidence_corpus else None)
    bind_corpus_context(retriever, corpus_index)
    service = create_controller(config)
    run_prefix = service.events.run_id
    retrieval_context_provider = None
    if getattr(retriever, "requires_history_context", False):
        from adapters.conversation import controller_context_provider
        retrieval_context_provider = controller_context_provider(service.store, run_prefix)
    export_failures = []
    first_export_error = None

    async def handle(turn):
        history = None
        if args.protocol == "P1":
            history = tuple(TurnRecord(i // 2 + 1, turn.history[i].text, turn.history[i+1].text)
                            for i in range(0, len(turn.history), 2))
        session_id = f"{run_prefix}:{turn.conversation_id}"
        if corpus_index:
            retrieval_priority = corpus_index.retrieved_keys(turn.contexts)
            supporting = corpus_index.retrieved_keys(turn.supporting_contexts)
            evidence = corpus_index.resolve((*retrieval_priority, *supporting))
        else:
            evidence = tuple(to_evidence(c, corpus_snapshot, corpus_id=dataset["name"]) for c in turn.contexts)
            retrieval_priority = tuple(block.key for block in evidence)
        return await service.serve_turn(session_id, turn.query,
                    evidence, request_id=turn.task_id, corpus_snapshot_id=corpus_snapshot,
                    reference_history=history, retrieval_priority=retrieval_priority,
                    query_context=turn.query_context)

    def export_result(item):
        nonlocal first_export_error
        if item.status != "success":
            return
        turn, result = item.online_turn, item.result
        try:
            snapshot = service.store.read_snapshot(f"{run_prefix}:{turn.conversation_id}")
            attempt = service.store.get_attempt(result.attempt_id)
            export_evaluation(turn, result.assistant_text,
                          official_path=Path(config.state_dir) / f"official-{run_prefix}.jsonl",
                          audit_path=Path(config.state_dir) / f"audit-{run_prefix}.jsonl",
                          visible_evidence_manifest=snapshot.execution.evidence_spans,
                          audit_evidence=snapshot.archive,
                          input_ids=snapshot.execution.last_input_ids, output_ids=snapshot.execution.last_output_ids,
                          branch_reason=result.branch_reason,
                          measurement_kind="simulation" if config.engine.kind == "mock" else "hardware",
                          truncated=result.truncated, citation_scope=result.citation_scope,
                          reading_outcome=attempt.reading_outcome,
                          reference_map=attempt.plan.candidate.reference_map)
        except Exception as exc:
            # Serving and the runner have committed this answer. Preserve the
            # actual next-turn history; fail artifact delivery after saving the
            # workload report instead of labelling generation as failed.
            if first_export_error is None:
                first_export_error = exc
            export_failures.append({"conversation_id": turn.conversation_id, "task_id": turn.task_id,
                                    "turn": turn.turn, "error": f"{type(exc).__name__}: {exc}"})

    try:
        manifest_path = _manifest(config, service, await service.engine.info(), args.protocol)
        manifest = json.loads(manifest_path.read_text())
        manifest.update(dataset={**dataset,
                                 "input": str(Path(args.input).resolve()),
                                 "split_manifest": args.split_manifest, "split": args.split,
                                 "domain": args.domain, "selected_sessions": sorted({r["conversation_id"] for r in rows})},
                        workload={"seed": args.seed, "concurrency": args.concurrency,
                                  "concurrency_scope": args.concurrency_scope,
                                  "think_time": args.think_time,
                                  "think_time_plan": think_time_plan,
                                  "arrival_rate": args.arrival_rate, "arrival_window": args.arrival_window,
                                  "per_turn_timeout": args.per_turn_timeout, "drain_timeout": args.drain_timeout,
                                  "retriever_config": args.retriever_config},
                        retrieval=retriever.manifest if retriever else {"kind": "dataset_provided_contexts"})
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        telemetry_before = await service.engine.metrics()
        report = await run_sessions(rows, handle, protocol=args.protocol, concurrency=args.concurrency,
                                    concurrency_scope=args.concurrency_scope,
                                    think_time=args.think_time, think_time_by_task=think_time_by_task,
                                    arrival_rate=args.arrival_rate,
                                    seed=args.seed, retriever=retriever, drain_timeout=args.drain_timeout,
                                    retrieval_context_provider=retrieval_context_provider,
                                    arrival_window=args.arrival_window, per_turn_timeout=args.per_turn_timeout,
                                    on_result=export_result)
        if retriever is not None:
            manifest["retrieval"] = retriever.manifest
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        cleanup_start = time.monotonic()
        cleanup_error = None
        cleanup_stage = "close_sessions"
        skipped_uncreated_sessions = []
        pending_retire_operations = telemetry_after = None
        try:
            for sid in {r.conversation_id for r in report.turns}:
                session_id = f"{run_prefix}:{sid}"
                try:
                    service.store.get_snapshot(session_id)
                except KeyError:
                    # P3 can fail in retrieval before serving creates a session.
                    skipped_uncreated_sessions.append(sid)
                    continue
                if not service.store.pending_attempt(session_id):
                    await service.close_session(session_id)
            cleanup_stage = "drain"
            await service.drain()
            pending_retire_operations = len(service.store.outbox())
            cleanup_stage = "metrics_after"
            telemetry_after = await service.engine.metrics()
            if pending_retire_operations:
                cleanup_stage = "drain"
                raise RuntimeError(f"{pending_retire_operations} retirement operation(s) remain pending; "
                                   "durable outbox retained for recovery")
        except Exception as exc:
            # Persist the already completed workload before propagating cleanup
            # failure; do not relabel successful turns or replace their errors.
            cleanup_error = exc
        cleanup_end = time.monotonic()
        output = Path(config.state_dir) / f"benchmark-{run_prefix}.json"
        from controller.sampling import request_sampling
        payload = {"sampling_recipe": request_sampling(config.sampling), "engine_kind": config.engine.kind, "quality_evaluated": False, "report": report,
                   "evaluation_export": {"status": "failed" if export_failures else "completed", "failures": export_failures},
                   "telemetry": {"before": telemetry_before, "after": telemetry_after},
                                "cleanup": {"start": cleanup_start, "end": cleanup_end,
                                            "pending_retire_operations": pending_retire_operations,
                                            "skipped_uncreated_sessions": sorted(skipped_uncreated_sessions),
                                            "status": "failed" if cleanup_error else "completed",
                                            **({"failed_stage": cleanup_stage,
                                                "error": f"{type(cleanup_error).__name__}: {cleanup_error}"}
                                               if cleanup_error else {})}}
        output.write_text(dumps(payload) + "\n", encoding="utf-8")
        summary = summarize_run(to_jsonable(payload))
        (Path(config.state_dir) / f"summary-{run_prefix}.json").write_text(dumps(summary) + "\n")
        (Path(config.state_dir) / f"summary-{run_prefix}.md").write_text(markdown_report(summary))
        success = sum(r.status == "success" for r in report.turns)
        print(json.dumps({"output": str(output.resolve()), "success": success,
                          "failed": len(report.turns)-success, "unfinished": report.unfinished_turns,
                          "evaluation_export_failed": len(export_failures),
                          "engine_kind": config.engine.kind}, ensure_ascii=False))
        if cleanup_error is not None:
            raise cleanup_error
        if first_export_error is not None:
            raise RuntimeError(f"evaluation export failed for {len(export_failures)} committed turn(s); "
                               f"workload and export failures saved to {output}") from first_export_error
    finally:
        await service.aclose()
        service.store.close()
        if retriever is not None and hasattr(retriever, "close"):
            retriever.close()
    return report

def main(argv=None):
    parser = argparse.ArgumentParser(description="ContextKV evaluation and CPU demonstration tools")
    parser.add_argument("--config")
    parser.add_argument("--state-dir")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--system-prompt")
    prompt.add_argument("--system-prompt-file")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="Run the CPU protocol demonstration")
    bench = sub.add_parser("run", help="Run P1/P2/P3 scripted sessions")
    bench.add_argument("input")
    bench.add_argument("--protocol", choices=["P1", "P2", "P3"], default="P2")
    bench.add_argument("--domain", choices=["cloud", "all"], default="cloud")
    bench.add_argument("--corpus")
    bench.add_argument("--retriever-index", help="Load a previously frozen BM25S index")
    bench.add_argument("--retriever-config", help="Optional shared P3 hybrid retriever YAML/JSON")
    bench.add_argument("--evidence-corpus", help="Original passages with version/condition/header metadata")
    bench.add_argument("--concurrency", type=int, default=1)
    bench.add_argument("--concurrency-scope", choices=["session", "turn"], default="session",
                       help="bound sessions including think time, or active turn pipelines")
    bench.add_argument("--think-time", type=float, default=0)
    bench.add_argument("--think-time-plan", help="JSON plan of seconds after each non-final task")
    bench.add_argument("--arrival-rate", type=float)
    bench.add_argument("--arrival-window", type=float)
    bench.add_argument("--per-turn-timeout", type=float)
    bench.add_argument("--drain-timeout", type=float)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--limit-sessions", type=int)
    bench.add_argument("--split-manifest")
    bench.add_argument("--split", choices=["dev", "test"], default="test")
    args = parser.parse_args(argv)
    config = load_config(args.config or ("configs/qwen3_32b_cost.yaml" if args.command == "run" else None))
    if args.state_dir:
        config = replace(config, state_dir=args.state_dir)
    system_prompt = args.system_prompt
    if args.system_prompt_file:
        with Path(args.system_prompt_file).open(encoding="utf-8", newline="") as stream:
            system_prompt = stream.read()
    if system_prompt is not None:
        config = replace(config, context=replace(config.context, system_prompt=system_prompt))
    if args.command == "demo":
        if config.engine.kind != "mock":
            parser.error("demo uses the CPU mock; run evaluates actual model configurations")
        asyncio.run(demo(config))
    else:
        report = asyncio.run(benchmark(args, config))
        if report.unfinished_turns or report.timed_out_sessions or any(t.status != "success" for t in report.turns):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
