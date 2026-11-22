"""Collect actual dialogue prefixes and previous engine inputs for stop supervision."""
import argparse
import asyncio
from collections import defaultdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from bench.study import run_dataset, write_jsonl
from controller.conversation_store import ConversationStore
from controller.serialization import to_jsonable
from training.stop_value import load_trajectories


async def prepare(args):
    config = yaml.safe_load(Path(args.config).read_text())
    dataset = config["retrieval"]["corpus_id"]
    targets = {r["task_id"]: r["split"] for r in load_trajectories(args.targets)}
    timeline = defaultdict(list)
    references = {}
    with Path(args.tasks).open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            key, sid, turn = row["task_id"], row["conversation_id"], int(row["turn"])
            timeline[sid].append((turn, key))
            if key in targets:
                references[key] = sid, turn, row["targets"][0]["text"]
    last = {}
    for task in targets:
        sid, turn, _ = references[task]
        last[sid] = max(last.get(sid, 0), turn)
    sessions = {sid: [task for turn, task in sorted(timeline[sid]) if turn <= end]
                for sid, end in last.items()}
    preceding = {}
    for turns in sessions.values():
        preceding.update((current, prior) for prior, current in zip(turns, turns[1:]) if current in targets)
    needed = set(preceding.values())
    seeds = {}
    def capture(task, request):
        if task in needed:
            seeds[task] = {k: getattr(request, k) for k in ("input_ids", "blocks", "demand", "query_span")}

    ids = [key for turns in sessions.values() for key in turns]
    matrix = {"datasets": {dataset: {"input": args.tasks}}, "protocol": "P3", "seed": args.seed,
              "cohorts": {"train": {dataset: ids}}, "resources": {"gpu_devices": args.gpu_devices}}
    examples = []
    with TemporaryDirectory() as directory:
        run = Path(directory)
        replies = await run_dataset(config, matrix, {"id": "contextkv"}, dataset, "train", run,
            concurrency=args.concurrency, limit_turns=len(ids), record_answer=capture)
        for reply in replies:
            if reply["status"] != "completed":
                raise RuntimeError(f"Missing training prefix at {reply['task_id']}: {reply.get('error')}")
        store = ConversationStore(run / "state" / "conversations.sqlite3")
        try:
            for sid in sessions:
                history = []
                for turn in store.get_snapshot(f"study:contextkv:{dataset}:train:{sid}").raw_history:
                    task = turn.query_context.task_state
                    key = task.task_id
                    if key in targets:
                        examples.append(dict(task_id=key, conversation_id=sid, split=targets[key], dataset=dataset,
                            turn=turn.turn_id, question=turn.user_text, actual_history=list(history), catalogue=[],
                            units=[], selected_ids=[], task_state=to_jsonable(task),
                            reference_answer=references[key][2],
                            previous_request=seeds[preceding[key]] if key in preceding else None))
                    history.append(to_jsonable(turn))
        finally:
            store.close()
    write_jsonl(args.output, examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/study.yaml")
    parser.add_argument("--targets", required=True, help="Stop records declaring task IDs and train/dev splits")
    parser.add_argument("--tasks", required=True, help="Source dialogue tasks in benchmark input format")
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--gpu-devices", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=42)
    asyncio.run(prepare(parser.parse_args()))
