"""Compare cache execution on the same evidence and matched warmup state."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import time

import yaml

from controller.config import apply_study_config, create_model_template, load_config, load_kv_policy
from controller.cooperative_request import EvidenceCosts, compile_request, source_address
from controller.engine_client import SGLangEngineClient
from controller.schema import EvidenceUnit
from controller.serialization import decode
from training.collect_shared_values import history, score_answers, source_blocks


async def replay_requests(run, trace, output, *, concurrency):
    """Replay the same actual multi-turn inputs and arrivals through both cache policies."""
    from controller.sampling import SamplingRecipe
    from bench.study_metrics import _quantile
    engine = SGLangEngineClient(run.engine.base_url, timeout=run.engine.timeout_seconds, sampling=SamplingRecipe())
    first = min(item["arrived"] for item in trace)
    branches, answers = [], []
    try:
        for mode in ("exact", "auto"):
            await engine.profile_prepare("cold")
            start = time.monotonic()
            slots, sessions = asyncio.Semaphore(concurrency), {}
            records, texts = [None]*len(trace), [None]*len(trace)
            async def execute(index, item):
                arrival = item["arrived"]-first
                await asyncio.sleep(max(0., start+arrival-time.monotonic()))
                options = {**item["options"], "mode": mode, "cache_scope": "pair:"+mode,
                           "cache_namespace": "pair:"+mode}
                session = options["session_id"]
                lock = sessions.setdefault(session, asyncio.Lock())
                request = replace(item["request"], demand=()) if mode == "exact" else item["request"]
                async with lock, slots:
                    result = await engine.compute(request, **options)
                records[index] = {"complete_ms": (time.monotonic()-start-arrival)*1000,
                                  **receipt(result),
                                  "new_compute_kv_pages": result["meta"]["contextkv_compute"]["new_compute_kv_pages"]}
                texts[index] = result["text"]
            await asyncio.gather(*(execute(i, item) for i, item in enumerate(trace)))
            kv = await engine.metrics()
            branches.append({"mode": mode, "requests": len(records),
                "p95_seconds": _quantile([row["complete_ms"]/1000 for row in records], .95),
                "gpu_kv_peak_tokens": kv["gpu_peak_used_tokens"],
                "gpu_kv_active_peak_tokens": kv["gpu_active_peak_tokens"],
                "gpu_kv_peak_gib": kv["gpu_peak_used_tokens"]*kv["pool"]["bytes_per_token"]/2**30,
                **{name: sum(row[name] for row in records) for name in (
                    "input_tokens", "output_tokens", "device_hit_tokens", "host_loaded_tokens",
                    "prefill_tokens", "new_compute_kv_pages")},
                "infoflow_requests": sum(row["mode"] == "infoflow" for row in records)})
            answers.append(texts)
            for session in sessions:
                await engine.close_session(session)
    finally:
        await engine.aclose()
    value = {"conditions": "Identical actual inputs and arrival offsets, cold start per branch, greedy generation; all construction costs included.",
             "branches": branches, "identical_answers": sum(a == b for a,b in zip(*answers)),
             "questions": len(trace)}
    Path(output).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    return value


def receipt(result):
    meta = result["meta"]
    native = meta["contextkv_compute"]
    return {"engine_ms": result["elapsed_ms"], "input_tokens": meta["prompt_tokens"],
        "output_tokens": meta["completion_tokens"],
        **{key: native[key] for key in ("mode", "device_hit_tokens", "host_loaded_tokens", "prefill_tokens")}}


async def compare(args):
    config = yaml.safe_load(Path(args.config).read_text())
    run = apply_study_config(load_config(config["generation"]["run_config"]), config)
    template = create_model_template(run.model)
    engine = SGLangEngineClient(run.engine.base_url, timeout=run.engine.timeout_seconds, sampling=run.sampling)
    policy = load_kv_policy(run.cooperative)
    row = next(row for row in map(json.loads, Path(args.data).read_text().splitlines())
               if row["task_id"] == args.task_id)
    units = tuple(decode(EvidenceUnit, unit) for unit in row["units"])
    by_id = {unit.unit_id: unit for unit in units}
    original = tuple(by_id)
    blocks, turns = source_blocks(units), history(row)
    output_tokens = 1 if args.prefill_only else config["execution"]["output_tokens"]
    results, answers = [], []
    try:
        for label, mode, seed_mode in (
            ("Native ranked context", "exact", "exact"),
            ("ContextKV cache-priced order", "auto", "exact"),
            ("InfoFlow non-prefix reuse", "infoflow", "infoflow"),
        ):
            await engine.profile_prepare("cold")
            namespace = "cache-pair:"+label
            seed_sources = tuple(reversed(blocks))
            seed = compile_request(template, "Read these source documents.", turns,
                                   seed_sources, future_sources=seed_sources)
            warm = await engine.compute(seed, session_id=namespace, request_id=namespace+":seed",
                max_new_tokens=1, mode=seed_mode, cache_scope=namespace, cache_namespace=namespace, **policy)
            started = time.monotonic()
            order, estimated = original, None
            if mode == "auto":
                snapshot = await engine.evidence_state(tuple(source_address(block) for block in blocks),
                    cache_scope=namespace, cache_namespace=namespace)
                empty = compile_request(template, row["question"], turns, ())
                costs = EvidenceCosts(units, blocks, template.tokenizer, snapshot,
                    prompt_tokens=len(empty.input_ids), output_tokens=output_tokens, policy=policy, mode=mode)
                order = costs.order(original)
                estimated = costs.estimate(order)
            request = compile_request(template, row["question"], turns,
                                      source_blocks(tuple(by_id[identity] for identity in order)))
            answer = await engine.compute(request, session_id=namespace, request_id=namespace+":answer",
                max_new_tokens=output_tokens, mode=mode, cache_scope=namespace,
                cache_namespace=namespace, **policy)
            results.append({"method": label, "complete_ms": (time.monotonic()-started)*1000,
                "estimated_ms": estimated, "selected_order": [original.index(identity) for identity in order],
                "seed": receipt(warm), **receipt(answer)})
            answers.append({"source": row, "answer": answer["text"]})
    finally:
        await engine.aclose()
    if not args.prefill_only:
        await score_answers(answers, run.engine.base_url, model=run.model.name, concurrency=1)
        for result, answer in zip(results, answers):
            result.update(quality=answer["quality"], answer=answer["answer"])
    output = {"kind": "same-evidence-prefill-pair" if args.prefill_only else "same-evidence-cache-pair",
        "task_id": row["task_id"], "question": row["question"], "source_count": len(units),
        "quality_metric": None if args.prefill_only else (
            "token_f1" if row["dataset"] == "multidoc2dial" else "training_reference_correctness"),
        "conditions": "Fresh cache per branch; identical reversed-source warmup, history and evidence set. "
                      "Complete time includes quote, ordering and compilation where used.",
        "results": results}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n")
    return {"output": str(path), "methods": len(results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/study.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prefill-only", action="store_true")
    print(json.dumps(asyncio.run(compare(parser.parse_args()))))


if __name__ == "__main__":
    main()
