"""Run document QA methods and report each benchmark's quality and cost."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import time
from urllib.parse import urlparse

import yaml

from adapters.mtrag import load_rows
from bench.result_records import retain_metrics, write_task_metrics
from bench.study_metrics import _track_specs, declared_queue_summary, split_metric_spec
from controller.config import apply_study_config, create_controller, load_config
from controller.schema import (Message, OnlineContext, OnlineMessage, OnlineTurn,
                               from_history_messages, history_messages)
from controller.serialization import dumps, to_jsonable


def official_record(turn: OnlineTurn, answer: str) -> dict:
    """Export the delivered answer and its actual reading material for scoring."""
    contexts = [{"document_id": c.document_id, "source": c.source, "score": c.score or 0.0,
                 "text": c.text, "title": c.title} for c in turn.contexts]
    messages = [{"speaker": "user" if m.speaker == "user" else "agent", "text": m.text} for m in turn.history]
    messages.append({"speaker": "user", "text": turn.query})
    record = {"task_id": turn.task_id, "conversation_id": turn.conversation_id,
            "Collection": turn.collection, "input": messages,
            "contexts": contexts, "predictions": [{"text": answer}]}
    return record


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def saved_study_rows(directory):
    """Read the dataset's single result journal."""
    export = Path(directory)/"results.jsonl"
    if export.exists():
        return read_jsonl(export)
    index = Path(directory)/"result_sets.json"
    if index.exists():
        return [row for source in json.loads(index.read_text())["datasets"] for row in saved_study_rows(source)]
    return []


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), ensure_ascii=False, indent=2)+"\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(dumps(row)+"\n" for row in rows))


def merge_config(base, overrides):
    result = dict(base)
    for name, value in overrides.items():
        result[name] = (merge_config(result[name] if isinstance(result.get(name), dict) else {}, value) if isinstance(value, dict)
                        else value)
    return result


def dataset_rows(spec, split, task_ids=None):
    path = spec["inputs"][split] if "inputs" in spec else spec["input"]
    rows = load_rows(path)
    if "split_manifest" in spec:
        allowed = set(json.loads(Path(spec["split_manifest"]).read_text())[split])
        rows = [row for row in rows if row["conversation_id"] in allowed]
    if "domain" in spec:
        rows = [row for row in rows if row["Collection"] == spec["domain"]]
    if task_ids is not None:
        by_id = {row["task_id"]: row for row in rows}
        return [by_id[task_id] for task_id in task_ids]
    if split == "dev" and "dev_conversations" in spec:
        identities = sorted({row["conversation_id"] for row in rows})
        selected = set(random.Random(spec["dev_seed"]).sample(identities, spec["dev_conversations"]))
        rows = [row for row in rows if row["conversation_id"] in selected]
    return rows


def method_config(config, method, dataset, split):
    retrieval = {key: dataset[key] for key in ("corpus_path", "corpus_id", "sparse_index", "dense_index")
                 if key in dataset}
    value = merge_config(config, {"retrieval": retrieval, "selection": dataset.get("selection", {})})
    if "max_passages" in dataset:
        value["selection"]["max_passages"] = dataset["max_passages"]
    if dataset.get("supplied_documents"):
        value["retrieval"] = {key: item for key, item in value["retrieval"].items()
                              if key not in {"corpus_path", "sparse_index", "dense_index"}}
        value["retrieval"]["corpus_id"] = dataset.get("corpus_id", "supplied")
    value = merge_config(value, {"retrieval": dataset.get("retrieval_by_split", {}).get(split, {})})
    return merge_config(value, method.get("overrides", {}))


@asynccontextmanager
async def _noop_slot():
    yield


async def run_dataset(config, matrix, method, dataset_name, split, directory, *, concurrency=1,
                      limit_conversations=None, limit_turns=20, cache_pair=False, record_answer=None,
                      arrival_rate=None, turn_think_ms=0.0, engine_url=None, aux_device=None,
                      gpu_devices=None):
    from adapters.providers import create_evidence_provider
    from bench.study_resources import StudyResourceSampler, integrate_physical, listening_processes
    from controller.evidence import make_evidence
    from controller.qa import DocumentQA

    spec = matrix["datasets"][dataset_name]
    source_rows = dataset_rows(spec, split, task_ids=matrix.get("cohorts", {}).get(split, {}).get(dataset_name))
    grouped = {}
    for row in source_rows:
        grouped.setdefault(row["conversation_id"], []).append(row)
    conversations = sorted(grouped)
    random.Random(matrix.get("seed", 42)).shuffle(conversations)
    conversations = conversations[:limit_conversations]
    ordered = [sorted(grouped[sid], key=lambda row: int(row["turn"])) for sid in conversations]
    # Spread a small queue across conversations while retaining adjacent turns.
    selected = [row for start in range(0, max(map(len, ordered), default=0), 2)
                for turns in ordered for row in turns[start:start+2]]
    selected = selected[:limit_turns]
    grouped = {}
    for row in selected:
        grouped.setdefault(row["conversation_id"], []).append(row)
    expected = [row["task_id"] for row in selected]
    protocol = spec.get("protocol", matrix.get("protocol", "P3"))
    config = method_config(config, method, spec, split)
    if aux_device and "shared" in config:
        config = merge_config(config, {
            "shared": {"query": {"device": aux_device}, "evidence": {"device": aux_device}},
        })
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    run = replace(apply_study_config(load_config(config["generation"]["run_config"]), config),
                  state_dir=str(directory/"state"))
    if engine_url:
        run = replace(run, engine=replace(run.engine, base_url=engine_url))
    if "answer_system_prompt" in spec:
        run = replace(run, context=replace(run.context, system_prompt=spec["answer_system_prompt"]))
    service = create_controller(run)
    trace = []
    if cache_pair and method["id"] == "contextkv":
        from controller.cooperative import blocks_for_units
        from controller.cooperative_request import source_address
        planned = {}
        answer_request = service.answer_request
        async def capture_plan(state, snapshot, complete):
            request = await answer_request(state, snapshot, complete)
            chosen = set(state.selected_ids)
            order = [identity for identity in state.evidence.frontier(state.pool.units)[-1] if identity in chosen]
            units = {unit.unit_id: unit for unit in state.pool.units}
            ranked = blocks_for_units((units[identity] for identity in order), state.pool.source_blocks)
            members = {source_address(block): block for block in request.evidence}
            ranked_ids = [source_address(block) for block in ranked if source_address(block) in members]
            ranked_set = set(ranked_ids)
            ranked_blocks = (*[block for address, block in members.items() if address not in ranked_set],
                             *[members[address] for address in ranked_ids])
            planned[id(request)] = {"ranked_evidence": ranked_blocks, "history": snapshot.raw_history}
            return request
        service.answer_request = capture_plan
        compute = service.engine.compute
        async def capture(request, **options):
            if not options["request_id"].endswith(":answer"):
                return await compute(request, **options)
            item = {"arrived": time.monotonic(), "request": request, "options": options,
                    **planned.pop(id(request))}
            trace.append(item)
            result = await compute(request, **options)
            item["finished"] = time.monotonic()
            item["receipt"] = result["meta"]["contextkv_compute"]
            return result
        service.engine.compute = capture
    if record_answer is not None:
        build_request = service.answer_request
        async def record_request(state, snapshot, complete):
            request = await build_request(state, snapshot, complete)
            record_answer(state.query.task.task_id, request)
            return request
        service.answer_request = record_request
    provider = await asyncio.to_thread(create_evidence_provider, config, service.template)
    qa = DocumentQA(service, provider)
    await qa.start()
    if config["execution"].get("cache_start", "cold") == "cold":
        await service.engine.profile_prepare()
    resource_settings = matrix["resources"]
    devices = tuple(gpu_devices) if gpu_devices is not None else tuple(resource_settings["gpu_devices"])
    process_ids = tuple(sorted({os.getpid(), *listening_processes((urlparse(run.engine.base_url).port,)),
                               *resource_settings.get("process_ids", ())}))
    write_json(directory/"generation_manifest.json", {"method": method, "dataset": dataset_name,
        "split": split, "protocol": protocol, "task_ids": expected,
        "concurrency": concurrency, "arrival_rate": arrival_rate, "turn_think_ms": turn_think_ms,
        "retrieval": provider.manifest,
        "measurement_kind": "hardware",
        "resource_scope": {"gpu_devices": devices, "process_roots": process_ids, "gpu_accounting": "device_memory"}})
    catalogue_path = spec.get("supplied_source_catalogue")
    catalogue = ({row["source_id"]: {**row, "document_id": row["source_id"], "source": row["source_uri"]}
                 for row in read_jsonl(catalogue_path)} if catalogue_path else None)
    for name in ("official.jsonl", "results.jsonl"):
        (directory/name).write_text("")
    results, references = {}, {}
    semaphore = None if arrival_rate is not None else asyncio.Semaphore(concurrency)
    session_offsets = {}
    if arrival_rate is not None:
        arrival_rng = random.Random(matrix.get("seed", 42))
        cursor = 0.0
        for index, sid in enumerate(grouped):
            if index > 0:
                cursor += arrival_rng.expovariate(arrival_rate)
            session_offsets[sid] = cursor
    physical = StudyResourceSampler(devices, process_ids)
    await physical.start()
    measured_start = time.monotonic()

    def append(name, row):
        with (directory/name).open("a") as stream:
            stream.write(dumps(row)+"\n")

    async def conversation(sid, turns):
        if arrival_rate is not None:
            delay = measured_start + session_offsets[sid] - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
        session_id = f"study:{method['id']}:{dataset_name}:{split}:{sid}"
        for turn_index, row in enumerate(turns):
            if arrival_rate is not None and turn_index > 0 and turn_think_ms > 0:
                await asyncio.sleep(turn_think_ms / 1000.0)
            task_id = row["task_id"]
            request_id = task_id
            snapshot = service.store.read_snapshot(session_id, provider.corpus_snapshot_id)
            if protocol == "P1":
                history = from_history_messages(tuple(Message(
                    "assistant" if item["speaker"] == "agent" else item["speaker"], item["text"])
                    for item in row["input"][:-1]))
                snapshot = service.store.sync_history(session_id, history)
            arrival = time.monotonic()
            query = row["input"][-1]["text"]
            history = tuple(OnlineMessage(message.role, message.content) for message in history_messages(snapshot.raw_history))
            reference = {key: value for key, value in row.items() if key != "contexts"}
            references[task_id] = reference
            record = {"method": method["id"], "dataset": dataset_name, "split": split, "protocol": protocol,
                "task_id": task_id, "conversation_id": sid, "turn": row["turn"], "request_id": request_id,
                "question": query, "history_origin": "author_reference_import" if protocol == "P1" else "model_actual",
                "measurement_kind": "hardware", "metrics": {}, "generation_dir": str(directory.resolve())}
            try:
                queued_at = time.monotonic()
                slot = semaphore if semaphore is not None else _noop_slot()
                async with slot:
                    slot_wait_ms = (time.monotonic() - queued_at) * 1000
                    supplied, metadata = None, None
                    if row.get("coverage_mode") or spec.get("supplied_documents", False):
                        contexts = [catalogue[key] for key in row["scope_source_ids"]] if catalogue else row["contexts"]
                        supplied = tuple(make_evidence(context["text"], context["document_id"],
                            corpus_id=dataset_name, passage_id=context["document_id"], version=context.get("version", "v1"),
                            source_uri=context.get("source") or f"{dataset_name}://{context['document_id']}",
                            title=context.get("title", "")) for context in contexts)
                        metadata = {block.key: {key: context[key] for key in (
                            "document_family", "valid_from", "valid_to", "published_at", "applicability",
                            "table_id", "heading", "current", "version_order") if key in context}
                            for block, context in zip(supplied, contexts)}
                    response = await qa.ask(session_id, query, request_id, task_id=task_id,
                        supplied_blocks=supplied, source_metadata=metadata, request_arrived=arrival,
                        coverage_scope=row.get("coverage_mode", spec.get("coverage_mode", "selected_sources")))
                turn = OnlineTurn(sid, task_id, row["turn"], row["Collection"], query, history,
                    tuple(OnlineContext(block.key.passage_id, block.text, block.source_uri, block.title)
                          for block in response.evaluation_evidence), protocol, query_context=response.query_context)
                append("official.jsonl", official_record(turn, response.assistant_text))
                decode_ms = max(0.0, response.engine_elapsed_ms - response.engine_ttft_ms)
                ttft_ms = max(0.0, response.complete_ms - decode_ms)
                cost = {"complete_ms": response.complete_ms, "slot_wait_ms": slot_wait_ms,
                    "ttft_ms": ttft_ms, "engine_ttft_ms": response.engine_ttft_ms,
                    "total_input_tokens": response.total_input_tokens,
                    "total_output_tokens": response.total_output_tokens, "auxiliary_ms": response.auxiliary_model_ms,
                    "online_compute_ms": response.engine_elapsed_ms + response.auxiliary_model_ms,
                    "actual_prefill_tokens": response.total_prefill_tokens,
                    "actual_device_hit_tokens": response.total_device_hit_tokens,
                    "actual_host_loaded_tokens": response.total_host_loaded_tokens}
                cost["new_compute_kv_pages"] = response.new_compute_kv_pages
                cost["kv_decision"] = {key: response.native[key] for key in (
                    "mode", "route_cost_ms", "cache_changed_members")
                    if key in response.native}
                record.update(status="completed", answer=response.assistant_text, cost=cost,
                    sources=to_jsonable(response.visible_evidence_manifest),
                    computation_sources=[item["sources"] for item in response.computations],
                    response={"output_tokens": response.output_tokens, "read_count": response.read_count,
                              "reduce_count": response.reduce_count, "truncated": response.truncated})
            except Exception as error:
                paid = service.store.result(session_id, request_id) or {}
                record.update(status="failed", error_type=type(error).__name__, error=str(error),
                    answer=None, sources=[], cost={"complete_ms": (time.monotonic()-arrival)*1000,
                        "total_input_tokens": paid.get("input_tokens", 0),
                        "total_output_tokens": paid.get("output_tokens", 0),
                        "auxiliary_ms": paid.get("auxiliary_ms", 0),
                        "online_compute_ms": paid.get("elapsed_ms", 0) + paid.get("auxiliary_ms", 0),
                        "actual_prefill_tokens": paid.get("prefill_tokens", 0),
                        "actual_device_hit_tokens": paid.get("device_hit_tokens", 0),
                        "actual_host_loaded_tokens": paid.get("host_loaded_tokens", 0)}, response={})
            results[task_id] = record
            append("results.jsonl", record)
        await service.close_session(session_id)

    try:
        await asyncio.gather(*(conversation(sid, turns) for sid, turns in grouped.items()))
    finally:
        measured_end = time.monotonic()
        try:
            await physical.stop()
            kv = await service.engine.metrics()
            write_json(directory/"batch_cost.json", {"duration_seconds": measured_end-measured_start,
                "gpu_kv_peak_tokens": kv["gpu_peak_used_tokens"],
                "gpu_kv_active_peak_tokens": kv["gpu_active_peak_tokens"],
                "gpu_kv_peak_gib": kv["gpu_peak_used_tokens"]*kv["pool"]["bytes_per_token"]/2**30,
                **integrate_physical(physical.samples, measured_start, measured_end)})
            write_jsonl(directory/"references.jsonl", [references[key] for key in expected if key in references])
        finally:
            provider.close()
            await service.aclose()
            service.store.close()
    ordered = [results[key] for key in expected]
    if trace:
        from bench.cache_reuse import replay_requests
        await replay_requests(run, trace, directory/"cache_pair.json", concurrency=concurrency,
                              references=list(references.values()), metric=spec["primary_metric"],
                              page_size=service.engine_info.page_size)
    return ordered


async def evaluate(config, matrix, output, *, split, methods=None, datasets=None,
                   concurrency=1, limit_conversations=None, limit_turns=20, score=True, cache_pair=False,
                   arrival_rate=None, turn_think_ms=0.0, engine_url=None, aux_device=None, gpu_devices=None):
    from bench.study_scoring import score_dataset

    output = Path(output)
    write_json(output/"effective_config.json", config)
    selected_methods = matrix["methods"]
    if methods is not None:
        selected_methods = [m for m in selected_methods if m["id"] in methods]
    generated = []
    for method in selected_methods:
        for name, dataset in _track_specs(matrix, split, datasets).items():
            directory = output/method["id"]/name
            rows = await run_dataset(config, matrix, method, name, split, directory,
                concurrency=concurrency, limit_conversations=limit_conversations, limit_turns=limit_turns,
                cache_pair=cache_pair, arrival_rate=arrival_rate, turn_think_ms=turn_think_ms,
                engine_url=engine_url, aux_device=aux_device, gpu_devices=gpu_devices)
            generated.append((name, dataset, directory, rows))

    results = []
    for name, dataset, directory, rows in generated:
        if score and any(row.get("scoring", {}).get("status") != "scored" for row in rows):
            rows = await score_dataset(config, dataset, rows, directory,
                expected_task_ids=[row["task_id"] for row in rows], split=split)
        if score:
            if all(row["status"] != "completed" or row["scoring"]["status"] == "scored" for row in rows):
                retain_metrics(directory)
        else:
            write_task_metrics(directory, rows)
        results.extend(rows)
    write_json(output/"result_sets.json", {"datasets": [str(directory.resolve())
        for _, _, directory, _ in generated]})
    return results


def report(matrix, rows, directory, *, split="test"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    methods = {method["id"]: method for method in matrix["methods"]}
    labels = {
        "ordinary_rag": "普通文档问答", "standard_rag": "普通文档问答",
        "engine_kv_only": "仅下层 KV", "ranked_selection": "排序选材",
        "context_only": "仅上层上下文", "context_kv_directory": "上层+KV目录截断",
        "contextkv": "ContextKV", "llamaindex": "LlamaIndex",
    }
    summaries, comparisons = [], []
    text = ["# ContextKV benchmark 结果", "", "每个 benchmark 独立报告。完整响应包含排队、检索、辅助编码、计算与最终生成。", ""]

    def number(value):
        return "—" if value is None else f"{value:.3f}"

    for dataset in dict.fromkeys(row["dataset"] for row in rows):
        spec = split_metric_spec(matrix["datasets"][dataset], split)
        text += [f"## {dataset}", "", f"质量指标：{spec['primary_metric']}。", "",
            "| 方法 | 完成交付 | 质量 | 完整 P95 (s) | Qwen 输入 token | Qwen 输出 token | GPU GiB·s | Host GiB·s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
        current = {}
        for method in methods:
            selected = [row for row in rows if row["dataset"] == dataset and row["method"] == method]
            if not selected:
                continue
            summary = declared_queue_summary(selected, spec, [row["task_id"] for row in selected])
            cost = summary["cost"]
            p95 = cost["p95_complete_ms"]
            p99 = cost.get("p99_complete_ms")
            p95_ttft = cost.get("p95_ttft_ms")
            p95_wait = cost.get("p95_slot_wait_ms")
            item = {"dataset": dataset, "method": method, "metric": spec["primary_metric"],
                "completed": summary["completed"], "requested": summary["declared"],
                "quality": summary["queue_primary_score"], "metrics": summary["benchmark_metrics"],
                "p95_seconds": p95 / 1000 if p95 is not None else None,
                "p99_seconds": p99 / 1000 if p99 is not None else None,
                "p95_ttft_seconds": p95_ttft / 1000 if p95_ttft is not None else None,
                "p95_slot_wait_seconds": p95_wait / 1000 if p95_wait is not None else None,
                "duration_seconds": cost.get("duration_seconds"),
                "throughput_turns_per_sec": cost.get("throughput_turns_per_sec"),
                **{key: cost[key] for key in ("total_input_tokens", "total_output_tokens", "gpu_gib_seconds",
                    "host_gib_seconds", "gpu_peak_gib", "host_peak_gib", "actual_device_hit_tokens",
                    "actual_host_loaded_tokens", "actual_prefill_tokens", "new_compute_kv_pages",
                    "gpu_kv_peak_tokens", "gpu_kv_peak_gib", "gpu_kv_active_peak_tokens")}}
            summaries.append(item)
            current[method] = item
            text.append("| " + labels.get(method, method) + f" | {item['completed']}/{item['requested']} | "
                + " | ".join(number(item[key]) for key in ("quality", "p95_seconds", "total_input_tokens",
                    "total_output_tokens", "gpu_gib_seconds", "host_gib_seconds")) + " |")
        if "contextkv" in current:
            candidate = current["contextkv"]
            for reference in ("ordinary_rag", "llamaindex"):
                if reference not in current:
                    continue
                base = current[reference]
                quality = (candidate["quality"] - base["quality"]
                           if candidate["quality"] is not None and base["quality"] is not None else None)
                ratio = (candidate["p95_seconds"] / base["p95_seconds"]
                         if candidate["p95_seconds"] is not None and base["p95_seconds"] else None)
                comparisons.append({"dataset": dataset, "method": "contextkv", "reference": reference,
                    "quality_difference": quality, "p95_ratio": ratio})
                text += ["", f"相对{labels[reference]}：质量差 {number(quality)}；P95 比值 {number(ratio)}。"]
        text.append("")
    write_json(directory/"summary.json", {"benchmarks": summaries, "comparisons": comparisons})
    (directory/"report.md").write_text("\n".join(text))
    return {"report": str((directory/"report.md").resolve()), "benchmarks": summaries, "comparisons": comparisons}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate and compare document QA methods by benchmark")
    parser.add_argument("command", choices=("evaluate", "report"))
    parser.add_argument("--config", default="configs/study.yaml")
    parser.add_argument("--matrix", default="configs/study_matrix.yaml")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--results")
    parser.add_argument("--output")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--arrival-rate", type=float, help="Open-loop Poisson session arrival rate (sessions/s)")
    parser.add_argument("--turn-think-ms", type=float, default=0.0, help="Delay between turns in a session for open-loop runs")
    parser.add_argument("--engine-url", help="Override generation SGLang base URL (for multi-GPU parallel runs)")
    parser.add_argument("--judge-url", help="Override scoring judge base URL")
    parser.add_argument("--aux-device", help="Override shared query/evidence encoder CUDA device (e.g. cuda:1)")
    parser.add_argument("--gpu-devices", type=int, nargs="+", help="Override physical GPU IDs sampled for batch_cost.json")
    parser.add_argument("--limit-conversations", type=int)
    parser.add_argument("--limit-turns", type=int, default=20, help="Questions per benchmark and method")
    parser.add_argument("--unscored", action="store_true")
    parser.add_argument("--cache-pair", action="store_true", help="Pair Native and cooperative cache on identical generated request inputs")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    if args.judge_url:
        config.setdefault("generation", {})["teacher_url"] = args.judge_url
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    root = Path(config["study"]["output_dir"])
    if args.command == "evaluate":
        output = Path(args.output or root/args.split)
        write_json(output/"matrix.json", matrix)
        rows = asyncio.run(evaluate(config, matrix, output, split=args.split,
            methods=args.methods, datasets=args.datasets, concurrency=args.concurrency,
            limit_conversations=args.limit_conversations, limit_turns=args.limit_turns,
            score=not args.unscored, cache_pair=args.cache_pair,
            arrival_rate=args.arrival_rate, turn_think_ms=args.turn_think_ms,
            engine_url=args.engine_url, aux_device=args.aux_device, gpu_devices=args.gpu_devices))
        result = report(matrix, rows, output, split=args.split)
    else:
        source = Path(args.results or root/args.split)
        rows = read_jsonl(source) if source.is_file() else saved_study_rows(source)
        result = report(matrix, rows, args.output or root/"report", split=args.split)
    print(dumps(result))


if __name__ == "__main__":
    main()
