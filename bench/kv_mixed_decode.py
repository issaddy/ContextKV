"""Compare Qwen decode paths on fixed waves of long revisits and Native dialogue requests."""
import argparse
import asyncio
import json
from pathlib import Path
from time import monotonic

import yaml

from bench.judge import JudgeConfig, OpenAICompletionClient
from bench.kv_residency import document_excerpts, selected_rows
from bench.loong_score import score_answer
from bench.official import reference_overlap
from controller.config import apply_study_config, create_model_template, load_config, load_kv_policy
from controller.cooperative_request import ANSWER_SCOPE, compile_request
from controller.engine_client import SGLangEngineClient
from controller.schema import EvidenceBlock, EvidenceKey, EvidenceUnit, Message, from_history_messages
from controller.serialization import decode
from training.collect_shared_values import history, source_blocks


async def replay(args):
    config = yaml.safe_load(Path(args.config).read_text())
    run = apply_study_config(load_config(config["generation"]["run_config"]), config)
    template = create_model_template(run.model)
    policy = load_kv_policy(run.cooperative)
    waves = json.loads(Path(args.cohort).read_text())["waves"]
    long_ids = [key for wave in waves for key in wave["long"]]
    short_ids = [key for wave in waves for key in wave["short"]]
    references = {r["task_id"]: r for r in selected_rows(args.documents, long_ids)}
    references.update({r["task_id"]: r for r in selected_rows(args.dialogues, short_ids)})
    requests, seeds = {}, {}
    for wave in waves:
        for key in wave["long"]:
            row = references[key]
            blocks = [EvidenceBlock(EvidenceKey("loong", str(doc["document_id"]), str(i), "v1", ""),
                        doc.get("source", ""), doc.get("title") or str(doc["document_id"]), text)
                      for i, (doc, text, _) in enumerate(document_excerpts(template, row, wave["source_tokens"]))]
            turns = from_history_messages(tuple(Message("assistant" if m["speaker"] == "agent" else m["speaker"],
                                                         m["text"]) for m in row["input"][:-1]))
            seeds[key] = compile_request(template, "Read the documents.", (), blocks[::-1], future_sources=blocks[::-1])
            requests[key] = compile_request(template, row["input"][-1]["text"], turns, blocks,
                                            instruction=ANSWER_SCOPE, future_sources=blocks)
        for key in wave["short"]:
            row = references[key]
            units = {u["unit_id"]: u for u in row["units"]}
            blocks = source_blocks(tuple(decode(EvidenceUnit, units[uid]) for uid in row["selected_ids"]))
            requests[key] = compile_request(template, row["question"], history(row), blocks, instruction=ANSWER_SCOPE)

    engine = SGLangEngineClient(run.engine.base_url, timeout=run.engine.timeout_seconds, sampling=run.sampling)
    if args.fixed_tokens:
        engine._sampling_params["ignore_eos"] = True
    async def submit(key, stage, request, mode, limit):
        return await engine.compute(request, session_id=key, request_id=key+stage, cache_namespace=key,
                                    cache_scope=key, max_new_tokens=limit, mode=mode, **policy)
    results = []
    try:
        await engine.profile_prepare()
        for wave in waves:
            start = monotonic()
            seed_mode = args.seed_route or args.route
            seeded = await asyncio.gather(*(submit(k, "-seed", seeds[k], seed_mode, 1) for k in wave["long"]))
            seed_routes = {key: reply["meta"]["contextkv_compute"]["mode"]
                           for key, reply in zip(wave["long"], seeded)}
            issued = monotonic()
            async def answer(key):
                mode = args.route if key in seeds else "exact"
                reply = await submit(key, "-read", requests[key], mode, args.fixed_tokens or run.context.max_new_tokens)
                end = monotonic()
                r = reply["meta"]["contextkv_compute"]
                return dict(task_id=key, long=key in seeds, seed_mode=seed_routes.get(key), mode=r["mode"],
                    input_tokens=reply["meta"]["prompt_tokens"],
                    output=reply["meta"]["completion_tokens"], elapsed_ms=1000*(end-issued),
                    seed_to_answer_ms=1000*(end-start), prefill=r["actual_prefill_tokens"],
                    cold=r["cold_tokens"], reused=r["reused_tokens"], score_tokens=r["score_tokens"],
                    replay_tokens=r["replay_tokens"], score_ms=r.get("score_ms", 0), replay_ms=r.get("recompute_ms", 0),
                    gpu=r["device_hit_tokens"], host=r["host_loaded_tokens"], route_cost=r["route_cost_ms"],
                    answer=reply["text"])
            results.extend(await asyncio.gather(*(answer(k) for k in (*wave["long"], *wave["short"]))))
    finally:
        await engine.aclose()
    if not args.fixed_tokens:
        judge = OpenAICompletionClient(JudgeConfig(run.engine.base_url, run.model.name, run.model.revision,
            max_new_tokens=2048, timeout_seconds=run.engine.timeout_seconds), tokenizer=template.tokenizer)
        slots = asyncio.Semaphore(4)
        async def score(result):
            key = result["task_id"]
            row = references[key]
            if result["long"]:
                async with slots:
                    outcome = await asyncio.to_thread(score_answer, row, result["answer"], judge)
                result["quality"] = outcome["score"]
            else:
                ref = {"task_id": key, "conversation_id": row["conversation_id"], "Collection": "multidoc2dial",
                       "targets": [{"text": row["reference_answer"]}]}
                pred = {"task_id": key, "predictions": [{"text": result["answer"]}]}
                result["quality"] = 100*reference_overlap([ref], [pred])["tasks"][key]["f1"]
        try:
            await asyncio.gather(*(score(r) for r in results))
        finally:
            judge.close()
    for result in results:
        del result["answer"]
    Path(args.output).write_text(json.dumps(results)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--dialogues", required=True)
    parser.add_argument("--route", choices=("auto", "exact", "infoflow"), required=True)
    parser.add_argument("--seed-route", choices=("auto", "exact", "infoflow"), help="Routing policy for the first visit")
    parser.add_argument("--fixed-tokens", type=int)
    parser.add_argument("--output", required=True)
    asyncio.run(replay(parser.parse_args()))
