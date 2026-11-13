"""Actual answer utility for sets; complete costs for stopping trajectories."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse

import yaml

from controller.config import create_model_template, load_config, apply_study_config
from controller.cooperative import CooperativeController, execution_totals
from controller.cooperative_request import compile_request, working_history
from controller.engine_client import SGLangEngineClient
from controller.schema import EvidenceBlock, EvidencePool, EvidenceUnit, SessionSnapshot, TurnRecord
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
    run = apply_study_config(load_config(config["generation"]["run_config"]), config)
    template = create_model_template(run.model)
    engine = SGLangEngineClient(run.engine.base_url, timeout=run.engine.timeout_seconds, sampling=run.sampling)
    source = [json.loads(line) for line in Path(args.data).read_text().splitlines() if line]
    source = [row for row in source if row["task_state"]["coverage_scope"] == "selected_sources"]
    selected = []
    for split, count in (("train", args.train_states), ("dev", args.dev_states)):
        groups = {}
        for row in source:
            if row["split"] == split:
                groups.setdefault(row["conversation_id"], []).append(row)
        ordered = [rows[index] for index in range(max(map(len, groups.values()), default=0))
                   for rows in groups.values() if index < len(rows)]
        selected.extend(ordered if count is None else ordered[:count])
    source = selected
    from adapters.shared_encoding import SharedQueryEncoder, SharedEvidenceEncoder
    from adapters.shared_retrieval import SharedEvidenceState, SharedEvidenceProvider
    from adapters.document_compute import DocumentIndex
    query_encoder = SharedQueryEncoder(**config["shared"]["query"])
    # Set collection needs the existing field head, not a previously fitted set head.
    evidence_encoder = SharedEvidenceEncoder(**config["shared"]["evidence"], train=args.stage == "sets")
    evidence_encoder.model.eval()
    evidence_encoder.heads.eval()
    from controller.value_policy import StopValueHead
    controller = CooperativeController(None, engine, template, settings=run.cooperative,
        policy=run.context, stop_model=StopValueHead())
    kv_policy = controller.kv_policy
    if args.stage == "stop":
        from bench.study_resources import StudyResourceSampler, attribute_physical, listening_processes
        await engine.profile_prepare("cold")
        roots = (os.getpid(), *listening_processes((urlparse(run.engine.base_url).port,)))
        sampler = StudyResourceSampler(tuple(args.gpu_devices), roots)
        await sampler.start()
    outputs, answers, windows = [], [], {}
    limit = asyncio.Semaphore(args.concurrency)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    records = []

    def save_record(record):
        records.append(record)
        with path.open("a") as output:
            output.write(json.dumps(record, ensure_ascii=False)+"\n")

    def state_for(query, encoded, observed, all_units, identity, totals):
        blocks = source_blocks(observed)
        state = SharedEvidenceState(replace(query), encoded, EvidencePool(tuple(observed), blocks), (),
            DocumentIndex.from_blocks(source_blocks(all_units)), body_tokens=config["selection"]["body_tokens"],
            population_units=tuple(observed) if query.task.coverage_scope == "all_supplied_documents" else (),
            cache_scope=identity, cache_namespace=identity, totals=totals)
        SharedEvidenceProvider._finish_state(state)
        return state

    async def answer(row, state, snapshot, identity, previous_totals):
        started = time.monotonic()
        totals = dict(previous_totals)
        state = replace(state, totals=totals)
        complete = controller.executor(identity, identity, totals,
            cache_scope=state.cache_scope, cache_namespace=state.cache_namespace)
        if args.stage == "sets":
            await controller.read_required(state, snapshot, complete)
        request = await controller.answer_request(state, snapshot, complete)
        result = await complete(request, "answer")
        item = {"source": row, "answer": result["text"]}
        if args.stage == "stop":
            windows[identity] = started, time.monotonic()
            item.update(call_id=identity, costs={"seconds": windows[identity][1]-started})
        answers.append(item)
        return item

    async def collect_state(row, index):
        async with limit:
            units = tuple(decode(EvidenceUnit, unit) for unit in row["units"])
            started = time.monotonic()
            query = await asyncio.to_thread(query_encoder.encode, conversation(row), row["question"],
                row["task_id"], row["turn"], catalogue=row.get("catalogue", ()))
            if query.task.coverage_scope == "all_supplied_documents":
                return
            snapshot = SessionSnapshot(row["task_id"], raw_history=working_history(query, history(row)))
            totals = execution_totals()
            if args.stage == "sets":
                encoded = await asyncio.to_thread(evidence_encoder.encode, query, units)
                state = state_for(query, encoded, units, units, f"sets:{row['task_id']}:prepare", totals)
                observations = []
                for number, chosen in enumerate(candidate_sets(encoded.frontier(units))):
                    identity = f"sets:{row['task_id']}:{number}"
                    branch = replace(state, query=replace(state.query), selected_ids=chosen,
                        read_units=set(state.read_units), processed_tables=set(state.processed_tables),
                        resolved_operations=dict(state.resolved_operations),
                        cache_scope=identity, cache_namespace=identity)
                    item = await answer(row, branch, snapshot, identity, totals)
                    observations.append((chosen, item))
                await score_answers([item for _, item in observations], run.engine.base_url,
                    model=run.model.name, concurrency=args.concurrency)
                save_record({**row, "completed_needs": sorted(state.completed_needs),
                    "sets": [{"unit_ids": chosen, "quality": item["quality"]} for chosen, item in observations]})
                return
            paid_seconds = time.monotonic()-started
            warm = index % 2 == 1
            state, encoded, states = None, None, []
            batch_size = min(args.batch_size, max(1, len(units) // 2))
            for start in range(0, len(units), batch_size):
                branch_id = f"stop:{row['task_id']}:{start}"
                if warm:
                    request = compile_request(template, row["question"], snapshot.raw_history,
                        source_blocks(units), future_sources=source_blocks(units))
                    warmed = await engine.compute(request, session_id=branch_id, request_id=branch_id+":warm",
                        max_new_tokens=1, mode=config["shared"]["runtime"]["kv_mode"],
                        cache_scope=branch_id, cache_namespace=branch_id, **kv_policy)
                    controller.native_state[branch_id] = warmed["meta"]["contextkv_compute"]
                first = time.monotonic()
                fresh = units[start:start+batch_size]
                new = await asyncio.to_thread(evidence_encoder.encode, query if state is None else state.query, fresh)
                encoded = new if encoded is None else encoded.merge(new)
                observed = units[:start+len(fresh)]
                if state is None:
                    state = state_for(query, encoded, observed, units, branch_id, totals)
                else:
                    state = replace(state, evidence=encoded, pool=EvidencePool(observed, source_blocks(observed)),
                        costs=None, cache_scope=branch_id, cache_namespace=branch_id)
                    SharedEvidenceProvider._finish_state(state)
                state.pending_units = units[len(observed):]
                complete = controller.executor(branch_id, branch_id+":work", totals,
                    cache_scope=branch_id, cache_namespace=branch_id)
                await controller.prepare_state(state, snapshot, complete)
                transition_id = f"expand:{row['task_id']}:{start}"
                windows[transition_id] = first, time.monotonic()
                paid_seconds += windows[transition_id][1]-first
                if states:
                    states[-1]["transition_id"] = transition_id
                features = StopValueHead.features(encoded.summary(state.selected_ids, state.pool.units,
                    tuple(need.need_id for need in state.query.task.needs), state.completed_needs),
                    controller._value_state(state, totals, time.monotonic()-paid_seconds, branch_id)).cpu().tolist()
                item = await answer(row, state, snapshot, branch_id, totals)
                states.append({"features": features, "answer": item})
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
    for row, observed in outputs:
        states = []
        for state in observed:
            item = state["answer"]
            value = {"features": state["features"],
                     "stop": {"quality": item["quality"], **item["costs"]}}
            if "transition_id" in state:
                identity = state["transition_id"]
                value["transition"] = {"seconds": windows[identity][1]-windows[identity][0], **attributed[identity]}
            states.append(value)
        save_record({"task_id": row["task_id"], "conversation_id": row["conversation_id"],
                     "split": row["split"], "states": states})
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
    parser.add_argument("--train-states", type=int)
    parser.add_argument("--dev-states", type=int)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(collect(args))))


if __name__ == "__main__":
    main()
