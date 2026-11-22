"""Replay future evidence demand across intervening allocations on a fixed KV pool."""
import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path

import yaml

from controller.config import apply_study_config, create_model_template, load_config
from controller.cooperative_request import CompiledRequest, compile_request
from controller.engine_client import SGLangEngineClient
from controller.schema import EvidenceBlock, EvidenceKey


def selected_rows(path, ids):
    wanted = set(ids)
    with Path(path).open() as source:
        rows = {row["task_id"]: row for row in map(json.loads, source) if row["task_id"] in wanted}
    return [rows[key] for key in ids]


def document_excerpts(template, row, budget):
    share = budget // len(row["contexts"])
    for doc in row["contexts"]:
        offsets = template.tokenizer(doc["text"], add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        count = min(share, len(offsets))
        if count:
            yield doc, doc["text"][:offsets[count-1][1]], count


async def replay(args):
    config = yaml.safe_load(Path(args.config).read_text())
    run = apply_study_config(load_config(config["generation"]["run_config"]), config)
    template = create_model_template(run.model)
    cohort = json.loads(Path(args.cohort).read_text())
    ids = cohort["revisit_ids"]
    requests = []
    for row in selected_rows(args.histories, ids):
        p = row["previous_request"]
        requests.append(CompiledRequest(tuple(p["input_ids"]), (), (), (), (), tuple(p["blocks"]),
                                        tuple(p["demand"]), tuple(p["query_span"])))
    fillers = []
    for row in selected_rows(args.documents, cohort["interference_ids"]):
        blocks = []
        for doc, text, _ in document_excerpts(template, row, args.interference_tokens):
            key = EvidenceKey("pressure", str(doc["document_id"]), "body", "1", "")
            blocks.append(EvidenceBlock(key, doc.get("source", ""), doc.get("title") or key.document_id, text))
        fillers.append(compile_request(template, "Read the documents.", (), tuple(blocks)))

    engine = SGLangEngineClient(run.engine.base_url, timeout=run.engine.timeout_seconds, sampling=run.sampling)
    async def submit(request, session, stage):
        return await engine.compute(request, session_id=session, request_id=session+"-"+stage,
                                    cache_scope=session, cache_namespace=session, max_new_tokens=1)

    records = []
    try:
        for offset in range(0, len(ids), args.concurrency):
            wave = slice(offset, offset+args.concurrency)
            sessions = ["seed"+key for key in ids[wave]]
            for demand in ((False, True) if offset // args.concurrency % 2 else (True, False)):
                await engine.profile_prepare()
                seeds = [p if demand else replace(p, demand=()) for p in requests[wave]]
                await asyncio.gather(*(submit(p, sid, "seed") for p, sid in zip(seeds, sessions)))
                await asyncio.gather(*(submit(p, "fill"+str(j), "fill") for j, p in enumerate(fillers[wave])))
                quotes = await asyncio.gather(*(engine.evidence_state([b["source_id"] for b in p.blocks],
                    cache_scope=sid, cache_namespace=sid) for p, sid in zip(seeds, sessions)))
                pool = await engine.metrics()
                replies = await asyncio.gather(*(submit(replace(p, demand=()), sid, "read")
                                                 for p, sid in zip(seeds, sessions)))
                peak = (await engine.metrics())["gpu_active_peak_tokens"]
                for key, seed, quote, reply in zip(ids[wave], seeds, quotes, replies):
                    receipt = reply["meta"]["contextkv_compute"]
                    records.append(dict(task_id=key, demand=demand, sources_retained=len(seed.demand),
                        quoted_gpu_tokens=max((p["gpu_tokens"] for p in quote["prefixes"]), default=0),
                        gpu_hit_tokens=receipt["device_hit_tokens"], host_loaded_tokens=receipt["host_loaded_tokens"],
                        prefill_tokens=receipt["actual_prefill_tokens"], elapsed_ms=reply["elapsed_ms"],
                        gpu_used_tokens=pool["pool"]["gpu_used_tokens"], active_peak_tokens=peak,
                        first_token=reply["text"]))
    finally:
        await engine.aclose()
    Path(args.output).write_text(json.dumps(records, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cohort", required=True, help="JSON with ordered revisit_ids and interference_ids")
    parser.add_argument("--histories", required=True, help="Output of training.prepare_stop_histories")
    parser.add_argument("--documents", required=True, help="Supplied document tasks for intervening allocations")
    parser.add_argument("--interference-tokens", type=int, default=5120)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", required=True)
    asyncio.run(replay(parser.parse_args()))
