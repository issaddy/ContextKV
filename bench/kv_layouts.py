"""Measure competing Native prefix layouts on the same long-document QA evidence."""
import argparse
import asyncio
import json
from pathlib import Path
from time import monotonic

import yaml

from bench.judge import JudgeConfig, OpenAICompletionClient
from bench.kv_residency import document_excerpts, selected_rows
from bench.loong_score import score_answer
from controller.config import apply_study_config, create_model_template, load_config, load_kv_policy
from controller.cooperative_request import ANSWER_SCOPE, CompiledRequest, EvidenceCosts, compile_request, source_address
from controller.engine_client import SGLangEngineClient
from controller.schema import EvidenceBlock, EvidenceKey, EvidenceUnit, Message, SourceSpan, from_history_messages


async def replay(args):
    from bench.study_metrics import _quantile

    config = yaml.safe_load(Path(args.config).read_text())
    run = apply_study_config(load_config(config["generation"]["run_config"]), config)
    engine_url = args.engine_url or run.engine.base_url
    judge_url = args.judge_url or config.get("generation", {}).get("teacher_url") or engine_url
    template, policy = create_model_template(run.model), load_kv_policy(run.cooperative)
    ids = json.loads(Path(args.cohort).read_text())["interference_ids"]
    if args.limit is not None:
        ids = ids[:args.limit]
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

    output_tokens = (args.output_tokens if args.output_tokens is not None else
                     1 if args.exchange_demand else run.context.max_new_tokens)
    engine = SGLangEngineClient(engine_url, timeout=run.engine.timeout_seconds, sampling=run.sampling)
    async def submit(request, task, stage, tokens):
        return await engine.compute(request, session_id=task+stage, request_id=task+stage,
            cache_scope=task, cache_namespace=task, max_new_tokens=tokens,
            protect_waiting=not args.no_protect_waiting, eager_demote=args.eager_demote, **policy)

    def seed(blocks, wanted):
        return compile_request(template, "Read the documents.", (), blocks, future_sources=wanted)

    async def plan(case, output):
        task, units, blocks, history, question = case
        keys = tuple(unit.unit_id for unit in units)
        rotate = (*keys[1:], keys[0])
        snapshot = await engine.evidence_state(keys, cache_namespace=task, cache_scope=task,
            session_id=task if not args.no_lease else None,
            use_lease=not args.no_lease, recent_layout_only=args.recent_layout_only)
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

    async def read(request, record, output, started):
        reply = await submit(request, record["id"], "-read", output)
        r = reply["meta"]["contextkv_compute"]
        record.update(gpu=r["device_hit_tokens"], host=r["host_loaded_tokens"], prefill=r["actual_prefill_tokens"],
                      output=reply["meta"]["completion_tokens"], ms=reply["elapsed_ms"],
                      ttft_ms=r.get("ttft_ms", reply["elapsed_ms"]), text=reply["text"],
                      seed_to_answer_ms=(monotonic()-started)*1000)
        return record

    async def revisit(case):
        started = monotonic()
        task, _, blocks, _, _ = case
        for evidence, stage in ((blocks, "-full"), (blocks[1:], "-suffix")):
            await submit(seed(evidence, evidence), task, stage, 1)
        request, record = await plan(case, output_tokens)
        return await read(request, record, output_tokens, started)

    async def exchange(wave, target):
        started = monotonic()
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
        plans = await asyncio.gather(*(plan(case, output_tokens) for case in wave))
        results = await asyncio.gather(*(read(request, record, output_tokens, started) for request, record in plans))
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
    if output_tokens > 1 and not args.exchange_demand:
        judge = OpenAICompletionClient(JudgeConfig(judge_url, run.model.name, run.model.revision,
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
    else:
        for result in results:
            result.pop("text", None)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results)+"\n")
    summary = {
        "tasks": len(rows), "records": len(results), "output_tokens": output_tokens,
        "gpu_tokens": sum(r["gpu"] for r in results),
        "host_tokens": sum(r["host"] for r in results),
        "prefill_tokens": sum(r["prefill"] for r in results),
        "p95_revisit_ms": _quantile([r["ms"] for r in results], .95),
        "p95_ttft_ms": _quantile([r["ttft_ms"] for r in results], .95),
        "p95_seed_to_answer_ms": _quantile([r["seed_to_answer_ms"] for r in results], .95),
    }
    if args.exchange_demand:
        by_id = {}
        for r in results:
            by_id.setdefault(r["id"], {})[r["target"]] = r["choice"]
        summary["matched"] = sum(r["choice"] == r["target"] for r in results)
        summary["changed_layout"] = sum(
            pair.get("original") == "original" and pair.get("rotate") == "rotate"
            for pair in by_id.values()
        )
    else:
        summary["visible_full"] = sum(r["has_full"] for r in results)
        summary["chosen_full"] = sum(r["full_order"] for r in results)
        if results and "quality" in results[0]:
            summary["mean_quality"] = sum(r["quality"] for r in results) / len(results)
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--documents", required=True, help="Loong tasks with supplied documents and official references")
    parser.add_argument("--output", required=True)
    parser.add_argument("--exchange-demand", action="store_true", help="Exchange which complete layout has future demand")
    parser.add_argument("--output-tokens", type=int, help="Override max output tokens (1 skips judge scoring)")
    parser.add_argument("--limit", type=int, help="Limit number of cohort tasks")
    parser.add_argument("--recent-layout-only", action="store_true", help="Expose only the most recent layout per source")
    parser.add_argument("--eager-demote", action="store_true", help="Demote unprioritized backed-up pages at request finish")
    parser.add_argument("--no-protect-waiting", action="store_true", help="Disable waiting-queue committed prefix protection")
    parser.add_argument("--no-lease", action="store_true", help="Disable quote soft leases")
    parser.add_argument("--engine-url", help="Override SGLang engine URL")
    parser.add_argument("--judge-url", help="Override judge URL when scoring natural answers")
    parser.add_argument("--interference-tokens", type=int, default=2048)
    parser.add_argument("--source-tokens", type=int, default=6144)
    parser.add_argument("--concurrency", type=int, default=4)
    asyncio.run(replay(parser.parse_args()))
