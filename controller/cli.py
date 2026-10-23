"""Inspectable local demo, controller API and scripted evaluation workloads."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
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
               "output_length_artifact_sha256": hashlib.sha256(Path(config.cost_model.output_length_artifact).read_bytes()).hexdigest()
                    if config.cost_model.output_length_artifact else None,
               "runtime": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine()},
               "quality_evaluated": False, "gpu_performance_validated": False}
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
        manifest = _manifest(config, service, await service.engine.info(), "synthetic-correctness-demo")
        for index, (query, evidence) in enumerate(turns, 1):
            response = await service.serve_turn(sid, query, evidence, request_id=f"turn-{index}")
            results.append(response)
            print(json.dumps({"turn": index, "action": response.action, "reason": response.reason,
                              "epoch": response.epoch_no, "input_tokens": response.input_tokens,
                              "simulated_gpu_hit_tokens": response.actual_device_hit_tokens,
                              "answer": response.assistant_text}, ensure_ascii=False))
        await service.close_session(sid)
        result_path = Path(config.state_dir) / f"demo-{service.events.run_id}.json"
        result_path.write_text(dumps({"measurement_kind": "simulation", "session_id": sid, "turns": results}) + "\n", encoding="utf-8")
        print(f"结果：{result_path.resolve()}\n运行清单：{manifest.resolve()}\n注意：模拟 token/缓存轨迹仅验证协议，不代表 GPU 性能或回答质量。")
    finally:
        await service.aclose()
        service.store.close()


async def benchmark(args, config):
    from adapters.mtrag import CLOUD_COLLECTION, load_rows
    from adapters.retrieval import BM25SRetriever, load_beir_corpus, to_evidence
    from bench.run_sessions import run_sessions
    from bench.quality import export_evaluation
    from bench.report import summarize_run, markdown_report
    from adapters.corpus import CorpusIndex

    rows = load_rows(args.input)
    if args.domain == "cloud":
        rows = [r for r in rows if r["Collection"] == CLOUD_COLLECTION]
    if args.split_manifest:
        manifest = json.loads(Path(args.split_manifest).read_text())
        selected = set(manifest[args.split])
        rows = [r for r in rows if r["conversation_id"] in selected]
    if args.limit_sessions:
        ids = set(sorted({r["conversation_id"] for r in rows})[:args.limit_sessions])
        rows = [r for r in rows if r["conversation_id"] in ids]
    corpus_snapshot = "mtrag-53fef6f5917c7e10c436f697e69a955dbc795d81"
    retriever = None
    if args.protocol == "P3":
        if args.retriever_index:
            retriever = BM25SRetriever.load(args.retriever_index)
            corpus_snapshot = retriever.corpus_snapshot_id
        elif args.corpus:
            retriever = BM25SRetriever(load_beir_corpus(args.corpus), corpus_snapshot_id=corpus_snapshot)
        else:
            raise ValueError("P3 requires --corpus or --retriever-index")
    corpus_index = (CorpusIndex.from_jsonl(args.evidence_corpus, corpus_id="mtrag", snapshot_id=corpus_snapshot)
                    if args.evidence_corpus else None)
    service = create_controller(config)
    run_prefix = service.events.run_id

    async def handle(turn):
        history = None
        if args.protocol == "P1":
            history = tuple(TurnRecord(i // 2 + 1, turn.history[i].text, turn.history[i+1].text)
                            for i in range(0, len(turn.history), 2))
        session_id = f"{run_prefix}:{turn.conversation_id}"
        if corpus_index:
            retrieval_priority = corpus_index.retrieved_keys(turn.contexts)
            evidence = corpus_index.resolve(retrieval_priority)
        else:
            evidence = tuple(to_evidence(c, corpus_snapshot) for c in turn.contexts)
            retrieval_priority = tuple(block.key for block in evidence)
        result = await service.serve_turn(session_id, turn.query,
                    evidence, request_id=turn.task_id, corpus_snapshot_id=corpus_snapshot,
                    reference_history=history, retrieval_priority=retrieval_priority)
        snapshot = service.store.read_snapshot(session_id)
        export_evaluation(turn, result.assistant_text,
                          official_path=Path(config.state_dir) / f"official-{run_prefix}.jsonl",
                          audit_path=Path(config.state_dir) / f"audit-{run_prefix}.jsonl",
                          visible_evidence_manifest=snapshot.execution.evidence_spans,
                          audit_evidence=tuple(b for b in snapshot.archive if b.key in snapshot.execution.visible_blocks),
                          input_ids=snapshot.execution.last_input_ids, output_ids=snapshot.execution.last_output_ids,
                          branch_reason=result.branch_reason,
                          measurement_kind="simulation" if config.engine.kind == "mock" else "hardware",
                          truncated=result.truncated)
        return result

    try:
        manifest_path = _manifest(config, service, await service.engine.info(), args.protocol)
        manifest = json.loads(manifest_path.read_text())
        manifest.update(dataset={"commit": "53fef6f5917c7e10c436f697e69a955dbc795d81",
                                 "input_sha256": hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                                 "split_manifest_sha256": hashlib.sha256(Path(args.split_manifest).read_bytes()).hexdigest() if args.split_manifest else None,
                                 "domain": args.domain, "selected_sessions": sorted({r["conversation_id"] for r in rows})},
                        workload={"seed": args.seed, "concurrency": args.concurrency, "think_time": args.think_time,
                                  "arrival_rate": args.arrival_rate, "arrival_window": args.arrival_window,
                                  "per_turn_timeout": args.per_turn_timeout, "drain_timeout": args.drain_timeout},
                        retrieval=retriever.manifest if retriever else {"kind": "official_fixed_top5"})
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        telemetry_before = await service.engine.metrics()
        report = await run_sessions(rows, handle, protocol=args.protocol, concurrency=args.concurrency,
                                    think_time=args.think_time, arrival_rate=args.arrival_rate,
                                    seed=args.seed, retriever=retriever, drain_timeout=args.drain_timeout,
                                    arrival_window=args.arrival_window, per_turn_timeout=args.per_turn_timeout)
        cleanup_start = time.monotonic()
        for sid in {r.conversation_id for r in report.turns}:
            if not service.store.pending_attempt(f"{run_prefix}:{sid}"):
                await service.close_session(f"{run_prefix}:{sid}")
        await service.drain()
        cleanup_end = time.monotonic()
        telemetry_after = await service.engine.metrics()
        output = Path(config.state_dir) / f"benchmark-{run_prefix}.json"
        payload = {"engine_kind": config.engine.kind, "quality_evaluated": False, "report": report,
                   "telemetry": {"before": telemetry_before, "after": telemetry_after},
                                "cleanup": {"start": cleanup_start, "end": cleanup_end,
                                            "pending_retire_operations": len(service.store.outbox())}}
        output.write_text(dumps(payload) + "\n", encoding="utf-8")
        summary = summarize_run(to_jsonable(payload))
        (Path(config.state_dir) / f"summary-{run_prefix}.json").write_text(dumps(summary) + "\n")
        (Path(config.state_dir) / f"summary-{run_prefix}.md").write_text(markdown_report(summary))
        success = sum(r.status == "success" for r in report.turns)
        print(json.dumps({"output": str(output.resolve()), "success": success,
                          "failed": len(report.turns)-success, "unfinished": report.unfinished_turns,
                          "engine_kind": config.engine.kind}, ensure_ascii=False))
    finally:
        await service.aclose()
        service.store.close()


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
        uvicorn.run(create_app(config=config), host=args.host, port=args.port)
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
        asyncio.run(benchmark(args, config))
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
