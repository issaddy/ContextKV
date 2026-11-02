"""Generate training histories through Dense5's actual controller service path."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import yaml

from adapters.corpus import CorpusIndex
from adapters.retriever_factory import create_retriever
from adapters.serving_retrieval import ServingRetriever
from controller.config import create_controller, load_config
from controller.serialization import to_jsonable
from training.task_data import read_jsonl


async def run_actual_rollouts(config, *, output_path=None, min_conversations=1000,
                              min_turns=10000, limit_turns=None, concurrency=8):
    settings = config.get("training", {})
    root = Path(settings.get("data_dir", "runs/study/data"))
    output = Path(output_path or root/"turns.jsonl")
    run_dir = Path(settings.get("rollout_state_dir", "runs/study/actual-rollout"))
    run_dir.mkdir(parents=True, exist_ok=True)
    service_config = load_config(settings.get("reader_config", "configs/qwen3_32b.yaml"))
    service_config = replace(service_config, state_dir=str(run_dir), engine=replace(service_config.engine,
        base_url=settings.get("reader_base_url", config.get("generation", {}).get("teacher_url",
            service_config.engine.base_url))))
    service = create_controller(service_config)
    retriever_config = yaml.safe_load(Path(settings.get("retriever_config",
        "configs/retriever_dialogue.example.yaml")).read_text())
    retriever_config.update(settings.get("retriever_overrides", {}))
    retriever_config.update(device="cuda:0", stage="dense", query_mode="dialogue", selector=None,
        reranker_path=None, source_window=False)
    recipe_path = run_dir/"dense5-retriever.yaml"
    recipe_path.write_text(yaml.safe_dump(retriever_config))
    retriever = create_retriever("P3", config_path=recipe_path)
    corpus = CorpusIndex.from_jsonl(settings.get("evidence_corpus", "data/multidoc2dial/dev/corpus.jsonl"),
        corpus_id="multidoc2dial", snapshot_id=retriever.corpus_snapshot_id)
    serving = ServingRetriever(retriever, corpus_id="multidoc2dial", corpus_index=corpus, events=service.events)
    conversations = {}
    partition = settings.get("rollout_split", "train")
    for source_row in read_jsonl(settings.get("rollout_questions", root/"original_train_turns.jsonl")):
        row = (source_row if "question" in source_row else {**source_row,
            "question": source_row["input"][-1]["text"], "reference_history": source_row["input"][:-1],
            "reference_answer": source_row["targets"][0]["text"]})
        conversations.setdefault(row["conversation_id"], []).append(row)
    chosen, count = [], 0
    for conversation_id in sorted(conversations):
        rows = sorted(conversations[conversation_id], key=lambda row: row["turn"])
        chosen.append((conversation_id, rows))
        count += len(rows)
        if len(chosen) >= min_conversations and count >= min_turns:
            break
    if limit_turns is not None:
        remaining, selected = limit_turns, []
        for conversation_id, rows in chosen:
            take = rows[:remaining]
            if take:
                selected.append((conversation_id, take))
            remaining -= len(take)
        chosen = selected
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = {row["task_id"] for row in read_jsonl(output)} if output.exists() else set()
    semaphore = asyncio.Semaphore(concurrency)
    started = time.monotonic()
    costs = []
    generated_conversations = set()
    output_file = output.open("a", encoding="utf-8")

    async def run_conversation(conversation_id, rows):
        async with semaphore:
            session_id = f"study-{partition}:"+conversation_id
            for row in rows:
                if row["task_id"] in completed:
                    continue
                snapshot = service.store.read_snapshot(session_id, serving.corpus_snapshot_id)
                actual_history = to_jsonable(snapshot.raw_history)
                turn_started = time.monotonic()

                async def provide(current_snapshot):
                    return await serving.retrieve(current_snapshot, row["question"], row["task_id"])

                response = await service.serve_turn(session_id, row["question"], request_id=row["task_id"],
                    corpus_snapshot_id=serving.corpus_snapshot_id, evidence_provider=provide)
                committed = service.store.get_snapshot(session_id)
                record = {"task_id": row["task_id"], "conversation_id": conversation_id,
                    "turn": row["turn"], "split": partition, "question": row["question"],
                    "actual_history": actual_history, "snapshot_id": serving.corpus_snapshot_id,
                    "rollout_policy": ("dense5_rebuild_qwen3_32b_nonthinking" if service_config.engine.kind == "sglang"
                                       else "dense5_controller_simulation"),
                    "grounding": row["grounding"], "reference_answer": row["reference_answer"],
                    "reference_history": row.get("reference_history", []),
                    "answer": response.assistant_text, "response": asdict(response),
                    "elapsed_ms": (time.monotonic()-turn_started)*1000,
                    "committed_turn": to_jsonable(committed.raw_history[-1]),
                    "script_history_mismatch": None,
                    "script_history_review_status": "pending_actual_history_object_condition_review"}
                output_file.write(json.dumps(record, ensure_ascii=False)+"\n")
                output_file.flush()
                completed.add(row["task_id"])
                generated_conversations.add(conversation_id)
                costs.append({"input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
                              "elapsed_ms": record["elapsed_ms"]})
                if len(costs) % 25 == 0:
                    print(json.dumps({"generated": len(costs), "completed": len(completed),
                        "conversations": len(generated_conversations), "elapsed_seconds": time.monotonic()-started}), flush=True)

    try:
        async with asyncio.TaskGroup() as tasks:
            for conversation_id, rows in chosen:
                tasks.create_task(run_conversation(conversation_id, rows))
    finally:
        output_file.close()
        serving.close()
        await service.aclose()
        service.store.close()
    manifest = {"status": "completed", "selected_conversations": len(chosen),
        "selected_turns": sum(len(rows) for _, rows in chosen), "generated_turns": len(costs),
        "total_completed_turns": len(completed), "elapsed_ms": (time.monotonic()-started)*1000,
        "input_tokens": sum(row["input_tokens"] for row in costs),
        "output_tokens": sum(row["output_tokens"] for row in costs),
        "sum_response_ms": sum(row["elapsed_ms"] for row in costs),
        "output_path": str(output.resolve()), "state_dir": str(run_dir.resolve()),
        "engine": service_config.engine.kind, "model": asdict(service_config.model),
        "measurement_kind": "hardware" if service_config.engine.kind == "sglang" else "simulation",
        "generation": "Dense5 controller service", "concurrency": concurrency}
    (run_dir/"rollout-manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/study.yaml")
    parser.add_argument("--output")
    parser.add_argument("--min-conversations", type=int, default=1000)
    parser.add_argument("--min-turns", type=int, default=10000)
    parser.add_argument("--limit-turns", type=int)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    result = asyncio.run(run_actual_rollouts(config, output_path=args.output,
        min_conversations=args.min_conversations, min_turns=args.min_turns,
        limit_turns=args.limit_turns, concurrency=args.concurrency))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
