"""Calibrate InfoFlow answer-quality loss on a few paired training states."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from controller.config import create_model_template, load_config
from controller.cooperative_request import compile_request
from controller.engine_client import SGLangEngineClient
from controller.schema import EvidenceUnit
from controller.serialization import decode
from training.collect_shared_values import source_blocks, history, score_answers


async def calibrate_kv_quality(args, config, run, template, engine, source):
    """Paired answer quality at the deployed recompute ratio; no text archive."""
    source = [row for row in source if row["split"] == "train"
              and row["dataset"] == "multidoc2dial"
              and not any(need["operation"] in {"count", "sum", "group", "mean", "min", "max"}
                          for need in row["task_state"]["needs"])][:args.limit]
    ratio = config["shared"]["runtime"]["recompute_ratio"]
    answers, pairs = [], []
    limit = asyncio.Semaphore(args.concurrency)

    async def pair(row):
        async with limit:
            units = tuple(decode(EvidenceUnit, unit) for unit in row["units"])
            request = compile_request(template, row["question"], history(row), source_blocks(units))
            observed = {}
            for mode in ("exact", "infoflow"):
                identity = f"kv-quality:{row['task_id']}:{mode}"
                result = await engine.compute(request, session_id=identity, request_id=identity,
                    max_new_tokens=config["execution"]["output_tokens"], mode=mode,
                    cache_namespace=identity, cache_scope=identity, recompute_ratio=ratio)
                item = {"source": row, "answer": result["text"]}
                answers.append(item)
                observed[mode] = item
            pairs.append((row["task_id"], observed))

    try:
        await asyncio.gather(*(pair(row) for row in source))
        costs = (await engine.metrics())["costs"]
    finally:
        await engine.aclose()
    await score_answers(answers, run.engine.base_url, model=run.model.name, concurrency=args.concurrency)
    records = [{"task_id": task_id, "exact": pair["exact"]["quality"],
                "infoflow": pair["infoflow"]["quality"]} for task_id, pair in pairs]
    asset = {"recompute_ratio": ratio, "quality_loss":
             sum(row["exact"]-row["infoflow"] for row in records)/len(records),
             "costs": costs, "pairs": records}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asset, ensure_ascii=False, indent=2)+"\n")
    return {"states": len(records), "quality_loss": asset["quality_loss"], "output": str(path)}


async def collect(args):
    config = yaml.safe_load(Path(args.config).read_text())
    run = load_config(config["generation"]["run_config"])
    template = create_model_template(run.model)
    engine = SGLangEngineClient(run.engine.base_url, timeout=run.engine.timeout_seconds, sampling=run.sampling)
    source = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line]
    return await calibrate_kv_quality(args, config, run, template, engine, source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/study.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="models/infoflow-quality.json")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    print(json.dumps(asyncio.run(collect(parser.parse_args()))))


if __name__ == "__main__":
    main()
