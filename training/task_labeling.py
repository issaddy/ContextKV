"""Parallel frozen-reader task labels and actual-history grounding review."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path
from threading import Lock
import time

import yaml

from adapters.history_state import HistoryIndex, TaskInterpreter, TeacherTaskModel, history_text
from adapters.hybrid_retrieval import LocalBGE
from adapters.query_selector import OpenAICompletionClient, SelectorConfig
from adapters.retrieval import run_retrieval_worker
from controller.config import load_config
from controller.context_schema import EvidenceKey, RetrievalState
from controller.schema import OnlineMessage, OnlineTurn, TurnRecord
from controller.serialization import decode, to_jsonable
from training.task_data import read_jsonl


def verify_grounding(row, state, memory, client, *, record_path=None):
    refs = [ref for condition in state.conditions for ref in condition.value_spans]
    active_conditions = [state.raw_question[ref.start:ref.end] if ref.turn_id == state.current_turn_id
                         else history_text(ref, memory) for ref in refs]
    body = {"question": row["question"], "actual_history": row["actual_history"],
        "author_reference_history": row.get("reference_history", []),
        "task_query": state.query, "active_conditions": active_conditions,
        "author_grounding": [{"index": i, "quote": g["quote"], "source_id": g["passage_id"]}
                             for i, g in enumerate(row["grounding"])]}
    schema = {"type": "object", "properties": {
        "script_history_mismatch": {"type": "boolean"},
        "applicable_grounding_indices": {"type": "array", "items": {"type": "integer", "minimum": 0,
            **({"maximum": len(row["grounding"])-1} if row["grounding"] else {})},
            "maxItems": len(row["grounding"])}},
        "required": ["script_history_mismatch", "applicable_grounding_indices"], "additionalProperties": False}
    completion = client.complete([{"role": "system", "content":
        "Review original source quotes against the current task resolved from ACTUAL assistant history. "
        "Mark script_history_mismatch if the authored next question presumes an object/condition/version not established "
        "or contradicted by actual history. Select only quoted source spans that support the resolved task under its "
        "active conditions and version; topical background alone is not a positive retrieval label. "
        "Use the supplied zero-based indices."}, {"role": "user", "content": json.dumps(body, ensure_ascii=False)}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "actual_history_grounding_review", "strict": True, "schema": schema}})
    if record_path is not None:
        with Path(record_path).open("a", encoding="utf-8") as sink:
            sink.write(json.dumps({"task_id": row["task_id"], "status": completion.status,
                "finish_reason": completion.finish_reason, "input_tokens": completion.input_tokens,
                "usage": completion.usage, "elapsed_ms": completion.elapsed_ms,
                "raw_output": completion.text, "error": completion.error,
                "raw_response": completion.raw_response}, ensure_ascii=False)+"\n")
    if completion.status != "completed" or completion.finish_reason != "stop":
        raise RuntimeError(f"grounding review {row['task_id']} unfinished: {completion.status}/{completion.finish_reason}: {completion.error}")
    reviewed = json.loads(completion.text)
    return {"script_history_mismatch": reviewed["script_history_mismatch"],
        "verified_grounding": [{**row["grounding"][i], "label_origin": "qwen3_actual_history_grounding_review"}
                               for i in reviewed["applicable_grounding_indices"]],
        "verification_metrics": {"elapsed_ms": completion.elapsed_ms, "input_tokens": completion.input_tokens,
                                 "usage": completion.usage}, "verification_raw_output": completion.text}


async def label_rollouts(config, *, turns_path=None, output_path=None, concurrency=8, limit=None):
    settings = config.get("training", {})
    root = Path(settings.get("data_dir", "runs/study/data"))
    turns_path = Path(turns_path or root/"turns.jsonl")
    output = Path(output_path or root/"task_labels.jsonl")
    rows = list(read_jsonl(turns_path))
    if limit is not None:
        rows = rows[:limit]
    reader = load_config(settings.get("reader_config", "configs/qwen3_32b.yaml"))
    urls = settings.get("teacher_urls", ["http://127.0.0.1:30001"])
    clients = [OpenAICompletionClient(SelectorConfig(url, reader.model.name, reader.model.revision,
        tokenizer_path=reader.model.name, max_input_tokens=24576, max_new_tokens=4096,
        timeout_seconds=600, temperature=0)) for url in urls]
    frozen = LocalBGE(config.get("task_model", {}).get("base_path", "models/bge-m3"),
                      device="cuda:0", dtype="bfloat16", max_length=8192)
    encode_lock = Lock()

    def encode(texts):
        with encode_lock:
            return frozen.encode(texts)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["conversation_id"]].append(row)
    prior_labels = list(read_jsonl(output)) if output.exists() else []
    by_id = {row["task_id"]: row for row in prior_labels}
    output.parent.mkdir(parents=True, exist_ok=True)
    sink = output.open("a", encoding="utf-8")
    semaphore = asyncio.Semaphore(concurrency)
    started = time.monotonic()
    generated = 0

    async def label_conversation(number, conversation_id, tasks):
        nonlocal generated
        async with semaphore:
            client = clients[number % len(clients)]
            interpreter = TaskInterpreter(TeacherTaskModel(client, record_path=output.parent/"task_teacher_calls.jsonl"), HistoryIndex(frozen.tokenizer, encode,
                unit_tokens=384, candidates=16, view_tokens=8192))
            states = []
            for row in sorted(tasks, key=lambda item: item["turn"]):
                if row["task_id"] in by_id:
                    from controller.context_schema import TaskState
                    states.append(decode(TaskState, by_id[row["task_id"]]["task_state"]))
                    continue
                history = tuple(decode(TurnRecord, value) for value in row["actual_history"])
                memory = RetrievalState(len(history), history, tuple(states))
                messages = tuple(message for old in history for message in (
                    OnlineMessage("user", old.user_text), OnlineMessage("assistant", old.assistant_text)))
                turn = OnlineTurn(conversation_id, row["task_id"], row["turn"], row["snapshot_id"], row["question"], messages, (), "P3")
                state = await run_retrieval_worker(interpreter.interpret, turn, memory)
                if "source_scope" in row:
                    state = replace(state, source_scope=tuple(decode(EvidenceKey, key) for key in row["source_scope"]))
                verification = await run_retrieval_worker(verify_grounding, row, state, memory, client,
                    record_path=output.parent/"grounding_teacher_calls.jsonl")
                record = {**row, **verification, "prediction": to_jsonable(interpreter.last_prediction),
                    "task_state": to_jsonable(state), "teacher_metrics": interpreter.last_metrics,
                    "teacher_raw_output": interpreter.model.last_raw_output,
                    "label_origin": "frozen_qwen3_32b_actual_history_teacher",
                    "script_history_review_status": "reviewed"}
                sink.write(json.dumps(record, ensure_ascii=False)+"\n")
                sink.flush()
                by_id[row["task_id"]] = record
                states.append(state)
                generated += 1
                if generated % 25 == 0:
                    print(json.dumps({"generated_labels": generated, "total_labels": len(by_id),
                        "elapsed_seconds": time.monotonic()-started}), flush=True)
    try:
        async with asyncio.TaskGroup() as tasks:
            for i, (key, value) in enumerate(grouped.items()):
                tasks.create_task(label_conversation(i, key, value))
    finally:
        sink.close()
        for client in clients:
            client.close()
        frozen.close()
    manifest = {"status": "completed", "actual_history_rows": len(rows), "labels": len(by_id),
        "generated_labels": generated, "conversations": len(grouped), "elapsed_ms": (time.monotonic()-started)*1000,
        "script_history_mismatches": sum(row["script_history_mismatch"] for row in by_id.values()),
        "verified_grounding_spans": sum(len(row["verified_grounding"]) for row in by_id.values()),
        "source_path": str(turns_path.resolve()), "labels_path": str(output.resolve()),
        "frozen_history_encoder": frozen.manifest,
        "teacher_input_tokens": sum(row["teacher_metrics"]["model_metrics"]["input_tokens"]
            + row["verification_metrics"]["input_tokens"] for row in by_id.values()),
        "teacher_output_tokens": sum(row["teacher_metrics"]["model_metrics"]["usage"]["completion_tokens"]
            + row["verification_metrics"]["usage"]["completion_tokens"] for row in by_id.values()),
        "teachers": [asdict(client.config) for client in clients]}
    output.with_name(output.stem+"_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/study.yaml")
    parser.add_argument("--turns")
    parser.add_argument("--output")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    result = asyncio.run(label_rollouts(yaml.safe_load(Path(args.config).read_text()),
        turns_path=args.turns, output_path=args.output, concurrency=args.concurrency, limit=args.limit))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
