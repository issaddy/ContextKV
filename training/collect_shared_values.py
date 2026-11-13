"""Actual answer utility for sets; complete costs for stopping trajectories."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse

import yaml

from controller.config import create_model_template, load_config, load_kv_policy
from controller.cooperative_request import EvidenceCosts, compile_request, source_address, working_history
from controller.engine_client import SGLangEngineClient
from controller.schema import EvidenceBlock, EvidenceUnit, TurnRecord
from controller.serialization import decode
from training.shared_collection import candidate_sets
from training.shared_training import conversation


def source_blocks(units):
    return tuple(EvidenceBlock(unit.source_spans[0].source_key,
        f"{unit.source_spans[0].source_key.corpus_id}://{unit.source_spans[0].source_key.document_id}",
        unit.heading, unit.text, span=(unit.source_spans[0].start, unit.source_spans[0].end)) for unit in units)


def history(row):
    grouped = {}
    for turn, role, text in conversation(row):
        grouped.setdefault(turn, {})[role] = text
    return tuple(TurnRecord(turn, values.get("user"), values.get("assistant"))
                 for turn, values in sorted(grouped.items()))


async def score_answers(rows, endpoint, *, model, concurrency):
    """Use the public F1 for MultiDoc; structured training uses reference correctness."""
    import httpx
    from bench.official import reference_overlap
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=180) as client:
        async def score(item):
            row, answer = item["source"], item["answer"]
            if row["dataset"] == "multidoc2dial":
                reference = {"task_id": row["task_id"], "conversation_id": row["conversation_id"],
                    "Collection": "multidoc2dial", "targets": [{"text": row["reference_answer"]}]}
                report = reference_overlap([reference], [{"task_id": row["task_id"],
                                                            "predictions": [{"text": answer}]}])
                item["quality"] = report["tasks"][row["task_id"]]["f1"]
                return
            async with semaphore:
                response = await client.post(endpoint.rstrip("/") + "/v1/chat/completions", json={
                    "model": model, "temperature": 0, "max_tokens": 256,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content":
                        'Score answer correctness against the reference, including requested entities, '
                        'conditions, periods, versions, units and every requested result. '
                        'Return only {"quality": number}, from 0 for incorrect to 1 for fully correct.'},
                        {"role": "user", "content": json.dumps({"question": row["question"],
                            "reference": row["reference_answer"], "answer": answer}, ensure_ascii=False)}]})
                response.raise_for_status()
                item["quality"] = float(json.loads(response.json()["choices"][0]["message"]["content"])["quality"])
        await asyncio.gather(*(score(item) for item in rows))


async def collect(args):
    config = yaml.safe_load(Path(args.config).read_text())
    run = load_config(config["generation"]["run_config"])
    template = create_model_template(run.model)
    engine = SGLangEngineClient(run.engine.base_url, timeout=run.engine.timeout_seconds, sampling=run.sampling)
    source = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line]
    from adapters.shared_encoding import SharedQueryEncoder, SharedEvidenceEncoder
    query_encoder = SharedQueryEncoder(**config["shared"]["query"])
    kv_policy = (load_kv_policy(config["shared"]["runtime"]) if args.stage == "stop"
                 else {"recompute_ratio": config["shared"]["runtime"]["recompute_ratio"]})
    evidence_encoder = None
    if args.stage == "stop":
        from bench.study_resources import StudyResourceSampler, attribute_physical, listening_processes
        await engine.profile_prepare("cold")
        # Full-population arithmetic completes by its execution contract. The
        # learned STOP action is trained on optional evidence expansion.
        source = [row for row in source if row["task_state"]["coverage_scope"] == "selected_sources" and not any(need["operation"] in {
            "count", "sum", "group", "mean", "min", "max"} for need in row["task_state"]["needs"])]
        evidence_encoder = SharedEvidenceEncoder(**config["shared"]["evidence"])
        roots = (os.getpid(), *listening_processes((urlparse(run.engine.base_url).port,)))
        sampler = StudyResourceSampler(tuple(args.gpu_devices), roots)
        await sampler.start()
    outputs, answers, windows, queries = [], [], {}, {}
    limit = asyncio.Semaphore(args.concurrency)

    async def answer(row, units, identity, *, future=(), mode="exact", output_tokens=None, cache_scope=None):
        if args.stage == "stop":
            started = time.monotonic()
        request = compile_request(template, row["question"], working_history(queries[row["task_id"]], history(row)), source_blocks(units),
                                  future_sources=source_blocks(future))
        # Every question owns its training source namespace. Warm branches share
        # immutable independent KV; cold branches have distinct exact prefixes.
        scope = f"training:{row['task_id']}:"
        result = await engine.compute(request, session_id=scope, request_id=identity,
            max_new_tokens=config["execution"]["output_tokens"] if output_tokens is None else output_tokens, mode=mode,
            cache_namespace=identity, cache_scope=cache_scope or scope,
            **kv_policy)
        item = {"source": row, "answer": result["text"]}
        if args.stage == "stop":
            windows[identity] = started, time.monotonic()
            item.update(call_id=identity, costs={"seconds": windows[identity][1]-started},
                        native=result["meta"]["contextkv_compute"])
        answers.append(item)
        return item

    async def collect_state(row, index):
        async with limit:
            units = tuple(decode(EvidenceUnit, unit) for unit in row["units"])
            if args.stage == "stop":
                query_started = time.monotonic()
            query = await asyncio.to_thread(query_encoder.encode, conversation(row), row["question"],
                row["task_id"], row["turn"], catalogue=row.get("catalogue", ()))
            queries[row["task_id"]] = query
            if args.stage == "sets":
                if all(need["operation"] in {"count", "sum", "group", "mean", "min", "max"}
                       for need in row["task_state"]["needs"]):
                    outputs.append((row, ()))
                    return
                by_id = {unit.unit_id: unit for unit in units}
                observations = []
                for number, chosen in enumerate(candidate_sets(by_id)):
                    item = await answer(row, tuple(by_id[identity] for identity in chosen),
                                        f"sets:{row['task_id']}:{number}")
                    observations.append((chosen, item))
                outputs.append((row, observations))
                return
            from controller.value_policy import StopValueHead
            paid_seconds = time.monotonic()-query_started
            warm = index % 2 == 1
            native = {}
            if warm:
                warmup = await answer(row, units, f"warm:{row['task_id']}", future=units, mode="infoflow", output_tokens=1)
                native = warmup["native"]
                answers.remove(warmup)
            encoded, states = None, []
            batch_size = min(args.batch_size, max(1, len(units) // 2))
            for start in range(0, len(units), batch_size):
                first = time.monotonic()
                fresh = units[start:start + batch_size]
                new = await asyncio.to_thread(evidence_encoder.encode, query, fresh)
                if encoded is None:
                    encoded = new
                else:
                    encoded = encoded.merge(new)
                observed = units[:start + len(fresh)]
                blocks = source_blocks(observed)
                native_view = await engine.evidence_state(tuple(source_address(block) for block in blocks),
                    cache_scope=f"training:{row['task_id']}:")
                fixed = compile_request(template, row["question"], working_history(query, history(row)), ())
                costs = EvidenceCosts(observed, blocks, template.tokenizer, native_view,
                    prompt_tokens=len(fixed.input_ids), output_tokens=config["execution"]["output_tokens"],
                    policy=kv_policy, mode=config["shared"]["runtime"]["kv_mode"])
                chosen = encoded.select(observed, body_tokens=config["selection"]["body_tokens"], cost_model=costs,
                    seconds_price=config["shared"]["runtime"]["prices"]["seconds"])
                transition_id = f"expand:{row['task_id']}:{start}"
                windows[transition_id] = first, time.monotonic()
                paid_seconds += windows[transition_id][1]-first
                if states:
                    states[-1]["transition_id"] = transition_id
                count = sum(unit.body_tokens for unit in observed if unit.unit_id in chosen)
                features = StopValueHead.features(encoded.summary(chosen), {
                    "remaining_sources": len(units)-len(observed),
                    "open_needs": sum(need.operation in {"infer", "compare"} for need in query.task.needs),
                    "required_rows": 0, "processed_rows": 0, "input_tokens": count,
                    "remaining_tokens": config["execution"]["max_total_tokens"],
                    "elapsed_seconds": paid_seconds,
                    **{name: native.get(name, 0) for name in ("queue_depth", "gpu_ready_tokens", "host_ready_tokens",
                        "prefill_token_ms", "transfer_token_ms")}}).cpu().tolist()
                selected = tuple(unit for unit in observed if unit.unit_id in chosen)
                item = await answer(row, selected, f"stop:{row['task_id']}:{start}",
                    future=units if start+len(fresh) < len(units) else (), mode=config["shared"]["runtime"]["kv_mode"],
                    cache_scope=f"training:{row['task_id']}:" if warm else f"stop:{row['task_id']}:{start}")
                states.append({"features": features, "complete_scope": True, "answer": item})
            outputs.append((row, states))

    try:
        await asyncio.gather(*(collect_state(row, index) for index, row in enumerate(source)))
    finally:
        if args.stage == "stop":
            await sampler.stop()
        await engine.aclose()
        query_encoder.close()
        if evidence_encoder:
            evidence_encoder.close()
    if args.stage == "stop":
        attributed = attribute_physical(sampler.samples, windows)
        for item in answers:
            item["costs"].update(attributed[item["call_id"]])
    await score_answers(answers, run.engine.base_url, model=run.model.name, concurrency=args.concurrency)
    records = []
    for row, observed in outputs:
        if args.stage == "sets":
            sets = [{"unit_ids": chosen, "quality": item["quality"]}
                    for chosen, item in observed]
            records.append({**row, **({"sets": sets} if sets else {})})
        else:
            states = []
            for state in observed:
                item = state["answer"]
                value = {"features": state["features"], "complete_scope": state["complete_scope"],
                         "stop": {"quality": item["quality"], **item["costs"]}}
                if "transition_id" in state:
                    identity = state["transition_id"]
                    value["transition"] = {"seconds": windows[identity][1]-windows[identity][0], **attributed[identity]}
                states.append(value)
            records.append({"task_id": row["task_id"], "conversation_id": row["conversation_id"],
                            "split": row["split"], "states": states})
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in records))
    return {"states": len(records), "answers": len(answers), "output": str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sets", "stop"))
    parser.add_argument("--config", default="configs/study.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-devices", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(collect(args))))


if __name__ == "__main__":
    main()
