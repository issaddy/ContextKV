"""Measure competing Native prefix layouts on the same long-document QA evidence."""
import argparse
import asyncio
import json
from pathlib import Path

import yaml

from bench.judge import JudgeConfig, OpenAICompletionClient
from bench.kv_residency import document_excerpts, selected_rows
from bench.loong_score import score_answer
from controller.config import apply_study_config, create_model_template, load_config, load_kv_policy
from controller.cooperative_request import ANSWER_SCOPE, CompiledRequest, EvidenceCosts, compile_request, source_address
from controller.engine_client import SGLangEngineClient
from controller.schema import EvidenceBlock, EvidenceKey, EvidenceUnit, Message, SourceSpan, from_history_messages


async def replay(args):
    config = yaml.safe_load(Path(args.config).read_text())
    run = apply_study_config(load_config(config["generation"]["run_config"]), config)
    template, policy = create_model_template(run.model), load_kv_policy(run.cooperative)
    ids = json.loads(Path(args.cohort).read_text())["interference_ids"]
    rows = selected_rows(args.documents, ids)
    cases = []
    for row in rows:
        units, blocks = [], []
        for index, (doc, text, count) in enumerate(document_excerpts(template, row, args.source_tokens)):
            key = EvidenceKey("loong", str(doc["document_id"]), str(index), "v1", "")
            block = EvidenceBlock(key, doc.get("source", ""), doc.get("title") or key.document_id, text)
            uid = source_address(block)
            units.append(EvidenceUnit(uid, (SourceSpan(uid, key, 0, len(text)),), text, body_tokens=count))
            blocks.append(block)
        history = from_history_messages(tuple(Message("assistant" if m["speaker"] == "agent" else m["speaker"],
                                                       m["text"]) for m in row["input"][:-1]))
        cases.append((row["task_id"], units, blocks, history, row["input"][-1]["text"]))

    engine = SGLangEngineClient(run.engine.base_url, timeout=run.engine.timeout_seconds, sampling=run.sampling)
    async def submit(request, task, stage, tokens):
        return await engine.compute(request, session_id=task+stage, request_id=task+stage,
            cache_scope=task, cache_namespace=task, max_new_tokens=tokens, **policy)

    def seed(blocks, wanted):
        return compile_request(template, "Read the documents.", (), blocks, future_sources=wanted)

    async def plan(case, output):
        task, units, blocks, history, question = case
        keys = tuple(unit.unit_id for unit in units)
        rotate = (*keys[1:], keys[0])
        snapshot = await engine.evidence_state(keys, cache_namespace=task, cache_scope=task)
        empty = compile_request(template, question, history, (), instruction=ANSWER_SCOPE)
        costs = EvidenceCosts(units, blocks, template.tokenizer, snapshot, prompt_tokens=len(empty.input_ids),
                              output_tokens=output, policy=policy)
        order = costs.order(rotate)
        by_id = dict(zip(keys, blocks))
        request = compile_request(template, question, history, tuple(by_id[key] for key in order), instruction=ANSWER_SCOPE)
        record = dict(id=task, has_full=any(tuple(p["source_ids"]) == keys for p in snapshot["prefixes"]),
                      full_order=order == keys)
        if args.exchange_demand:
            record["choice"] = "original" if order == keys else "rotate" if order == rotate else "other"
            for label, layout in (("original", keys), ("rotate", rotate)):
                quote = costs.quote(layout, preserve_order=True)
                record[label] = [quote["gpu_tokens"], quote["host_tokens"], quote["milliseconds"]]
                record[label+"_indexed"] = any(tuple(p["source_ids"]) == layout for p in snapshot["prefixes"])
        return request, record

    async def read(request, record, output):
        reply = await submit(request, record["id"], "-read", output)
        r = reply["meta"]["contextkv_compute"]
        record.update(gpu=r["device_hit_tokens"], host=r["host_loaded_tokens"], prefill=r["actual_prefill_tokens"],
                      output=reply["meta"]["completion_tokens"], ms=reply["elapsed_ms"], text=reply["text"])
        return record

    async def revisit(case):
        task, _, blocks, _, _ = case
        for evidence, stage in ((blocks, "-full"), (blocks[1:], "-suffix")):
            await submit(seed(evidence, evidence), task, stage, 1)
        request, record = await plan(case, run.context.max_new_tokens)
        return await read(request, record, run.context.max_new_tokens)

    async def exchange(wave, target):
        for label in ("original", "rotate"):
            jobs = []
            for task, _, blocks, _, _ in wave:
                ordered = blocks if label == "original" else [*blocks[1:], blocks[0]]
                jobs.append(submit(seed(ordered, ordered if label == target else ()), task, label, 1))
            await asyncio.gather(*jobs)
        # Allocate the same four cold prefixes, then obtain every quote before any revisit.
        jobs = []
        for task, _, blocks, _, _ in wave:
            tokens = seed(blocks, ()).input_ids[:args.interference_tokens]
            filler = CompiledRequest(tuple(tokens), (), (), (), (), (), (), (0, 0))
            jobs.append(submit(filler, task+"fill", "pressure", 1))
        await asyncio.gather(*jobs)
        plans = await asyncio.gather(*(plan(case, 1) for case in wave))
        results = await asyncio.gather(*(read(request, record, 1) for request, record in plans))
        for result in results:
            result["target"] = target
        return results

    results = []
    try:
        for start in range(0, len(cases), args.concurrency):
            wave = cases[start:start+args.concurrency]
            if args.exchange_demand:
                order = ("rotate", "original") if start // args.concurrency % 2 else ("original", "rotate")
                for target in order:
                    await engine.profile_prepare()
                    results.extend(await exchange(wave, target))
            else:
                await engine.profile_prepare()
                results.extend(await asyncio.gather(*(revisit(case) for case in wave)))
    finally:
        await engine.aclose()
    if args.exchange_demand:
        Path(args.output).write_text(json.dumps(results)+"\n")
        return
    judge = OpenAICompletionClient(JudgeConfig(run.engine.base_url, run.model.name, run.model.revision,
        max_new_tokens=2048, timeout_seconds=run.engine.timeout_seconds), tokenizer=template.tokenizer)
    try:
        for start in range(0, len(rows), args.concurrency):
            wave = results[start:start+args.concurrency]
            scores = await asyncio.gather(*(asyncio.to_thread(score_answer, row, result.pop("text"), judge)
                for row, result in zip(rows[start:start+args.concurrency], wave)))
            for result, score in zip(wave, scores):
                result["quality"] = score["score"]
    finally:
        judge.close()
    Path(args.output).write_text(json.dumps(results)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--documents", required=True, help="Loong tasks with supplied documents and official references")
    parser.add_argument("--output", required=True)
    parser.add_argument("--exchange-demand", action="store_true", help="Exchange which complete layout has future demand")
    parser.add_argument("--interference-tokens", type=int, default=2048)
    parser.add_argument("--source-tokens", type=int, default=6144)
    parser.add_argument("--concurrency", type=int, default=4)
    asyncio.run(replay(parser.parse_args()))
