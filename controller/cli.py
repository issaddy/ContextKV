"""Inspectable local demo, controller API and scripted evaluation workloads."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import time
import uuid

from .config import create_controller, load_config
from .evidence import make_evidence
from .schema import TurnRecord
from .serialization import dumps, to_jsonable


def _manifest(config, controller, info, protocol: str):
    files = sorted(Path(__file__).parent.glob("*.py"))
    source_hash = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in files)).hexdigest()
    payload = {"schema_version": 1, "run_id": controller.events.run_id,
               "measurement_kind": "simulation" if config.engine.kind == "mock" else "unvalidated_hardware_run",
               "protocol": protocol, "config": to_jsonable(config), "config_hash": config.config_hash,
               "fingerprint": to_jsonable(controller.template.fingerprint), "engine": to_jsonable(info),
               "controller_source_sha256": source_hash,
               "cost_model_artifact_sha256": hashlib.sha256(Path(config.cost_model.artifact).read_bytes()).hexdigest()
                    if config.cost_model.artifact else None,
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
    from bench.run_sessions import run_sessions
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
    corpus_snapshot = f"{dataset['name']}-{dataset['revision']}"
    retriever = create_retriever(args.protocol, config_path=getattr(args, "retriever_config", None),
                                index_path=getattr(args, "retriever_index", None), corpus_path=getattr(args, "corpus", None),
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
        from adapters.query_resolution import controller_context_provider
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
            export_evaluation(turn, result.assistant_text,
                          official_path=Path(config.state_dir) / f"official-{run_prefix}.jsonl",
                          audit_path=Path(config.state_dir) / f"audit-{run_prefix}.jsonl",
                          visible_evidence_manifest=snapshot.execution.evidence_spans,
                          audit_evidence=tuple(b for b in snapshot.archive if b.key in snapshot.execution.visible_blocks),
                          input_ids=snapshot.execution.last_input_ids, output_ids=snapshot.execution.last_output_ids,
                          branch_reason=result.branch_reason,
                          measurement_kind="simulation" if config.engine.kind == "mock" else "hardware",
                          truncated=result.truncated, citation_scope=result.citation_scope)
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
                                 "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                                 "split_manifest_sha256": hashlib.sha256(Path(args.split_manifest).read_bytes()).hexdigest() if args.split_manifest else None,
                                 "domain": args.domain, "selected_sessions": sorted({r["conversation_id"] for r in rows})},
                        workload={"seed": args.seed, "concurrency": args.concurrency, "think_time": args.think_time,
                                  "arrival_rate": args.arrival_rate, "arrival_window": args.arrival_window,
                                  "per_turn_timeout": args.per_turn_timeout, "drain_timeout": args.drain_timeout,
                                  "retriever_config": getattr(args, "retriever_config", None)},
                        retrieval=retriever.manifest if retriever else {"kind": "dataset_provided_contexts"})
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        telemetry_before = await service.engine.metrics()
        report = await run_sessions(rows, handle, protocol=args.protocol, concurrency=args.concurrency,
                                    think_time=args.think_time, arrival_rate=args.arrival_rate,
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
        from .sampling import request_sampling
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
    parser = argparse.ArgumentParser(prog="contextkv", description="ContextKV 研究控制器；mock仅用于正确性验证")
    parser.add_argument("--config", help="YAML configuration")
    parser.add_argument("--state-dir", help="Override persistent output directory")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--system-prompt", help="Override context.system_prompt for this run")
    prompt.add_argument("--system-prompt-file", help="UTF-8 system prompt; full content is recorded in run configuration")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="Run a four-turn offline correctness demo")
    serve = sub.add_parser("serve", help="Start the local controller API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    retrieval = serve.add_mutually_exclusive_group()
    retrieval.add_argument("--retriever-index", help="Frozen BM25S index for HTTP query retrieval")
    retrieval.add_argument("--retriever-config", help="Shared hybrid retriever YAML/JSON for HTTP queries")
    serve.add_argument("--evidence-corpus", help="Optional original version/condition metadata JSONL")
    serve.add_argument("--corpus-id", default="mtrag", help="Evidence corpus namespace; snapshot comes from the index")
    inspect = sub.add_parser("inspect", help="Read a persisted session")
    inspect.add_argument("session_id")
    recover = sub.add_parser("recover", help="Recover a pending engine result, or explicitly abandon preparation")
    recover.add_argument("session_id")
    recover.add_argument("--abandon", action="store_true")
    sub.add_parser("fingerprint", help="Print the model/template fingerprint for the engine patch")
    data = sub.add_parser("profile-data", help="Recompute MTRAG counts and write conversation splits")
    data.add_argument("input")
    data.add_argument("--output", default="data/mtrag")
    bench = sub.add_parser("bench", help="Run P1/P2/P3 scripted sessions")
    bench.add_argument("input")
    bench.add_argument("--protocol", choices=["P1", "P2", "P3"], default="P2")
    bench.add_argument("--domain", choices=["cloud", "all"], default="cloud")
    bench.add_argument("--corpus")
    bench.add_argument("--retriever-index", help="Load a previously frozen BM25S index")
    bench.add_argument("--retriever-config", help="Optional shared P3 hybrid retriever YAML/JSON")
    bench.add_argument("--evidence-corpus", help="Original passages with version/condition/header metadata")
    bench.add_argument("--concurrency", type=int, default=1)
    bench.add_argument("--think-time", type=float, default=0)
    bench.add_argument("--arrival-rate", type=float)
    bench.add_argument("--arrival-window", type=float)
    bench.add_argument("--per-turn-timeout", type=float)
    bench.add_argument("--drain-timeout", type=float)
    bench.add_argument("--seed", type=int, default=42)
    bench.add_argument("--limit-sessions", type=int)
    bench.add_argument("--split-manifest")
    bench.add_argument("--split", choices=["dev", "test"], default="test")
    args = parser.parse_args(argv)
    config = load_config(args.config)
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
            parser.error("demo requires engine.kind=mock; use bench or serve for SGLang")
        asyncio.run(demo(config))
    elif args.command == "serve":
        import uvicorn
        from .api import create_app
        if args.evidence_corpus and not (args.retriever_index or args.retriever_config):
            parser.error("--evidence-corpus requires --retriever-index or --retriever-config")
        uvicorn.run(create_app(config=config, retriever_index=args.retriever_index,
            retriever_config=args.retriever_config, evidence_corpus=args.evidence_corpus,
            corpus_id=args.corpus_id), host=args.host, port=args.port)
    elif args.command == "profile-data":
        from adapters.mtrag import dataset_profile, load_rows, stratified_split
        path = Path(args.input)
        rows = load_rows(path)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        profile = dataset_profile(rows)
        (output / "profile.json").write_text(dumps(profile) + "\n")
        split = stratified_split(rows, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        split_path = output / "split_manifest.json"
        if split_path.exists() and json.loads(split_path.read_text()) != to_jsonable(split):
            raise ValueError("Refusing to replace a different frozen split manifest; choose another output directory")
        split_path.write_text(dumps(split) + "\n")
        print(json.dumps(profile, indent=2, ensure_ascii=False))
    elif args.command == "bench":
        report = asyncio.run(benchmark(args, config))
        if report.unfinished_turns or report.timed_out_sessions or any(turn.status != "success" for turn in report.turns):
            raise SystemExit(1)
    else:
        service = create_controller(config)
        if args.command == "inspect":
            print(json.dumps(to_jsonable(service.store.get_snapshot(args.session_id)), indent=2, ensure_ascii=False))
        elif args.command == "fingerprint":
            print(dumps(service.template.fingerprint))
        else:
            async def recover_pending():
                try:
                    result = await service.recover(args.session_id, abandon=args.abandon)
                    print(dumps(result))
                finally:
                    await service.aclose()
            asyncio.run(recover_pending())
        service.store.close()


if __name__ == "__main__":
    main()
