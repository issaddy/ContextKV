"""Actual answer utility for sets; complete costs for stopping trajectories."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from itertools import islice
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse

import httpx
import yaml

from controller.config import create_model_template, load_config, apply_study_config
from controller.cooperative import CooperativeController, execution_totals
from controller.cooperative_request import compile_request, working_history
from controller.engine_client import SGLangEngineClient
from controller.schema import EvidenceBlock, EvidencePool, EvidenceUnit, SessionSnapshot, TurnRecord
from controller.serialization import decode, to_jsonable
from training.shared_collection import candidate_sets, reference_coverage
from training.shared_training import conversation


def source_blocks(units):
    return tuple(EvidenceBlock(unit.source_spans[0].source_key,
        f"{unit.source_spans[0].source_key.corpus_id}://{unit.source_spans[0].source_key.document_id}",
        unit.heading, unit.text, span=(unit.source_spans[0].start, unit.source_spans[0].end)) for unit in units)


def history(row):
    if "actual_history" in row:
        return tuple(decode(TurnRecord, item) for item in row["actual_history"])
    grouped = {}
    for turn, role, text in conversation(row):
        grouped.setdefault(turn, {})[role] = text
    return tuple(TurnRecord(turn, values.get("user"), values.get("assistant"))
                 for turn, values in sorted(grouped.items()))


async def score_answers(rows, endpoint, *, model, concurrency):
    """Use the public F1 for MultiDoc; structured training uses reference correctness."""
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
    batch_size = args.batch_size or config["retrieval"]["evidence_batch"]
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
        if args.stage == "sets":
            ordered.sort(key=lambda row: not bool(row.get("paired_sets")))
        elif split == "dev":
            ordered = [row for row in ordered if row["dataset"] == config["retrieval"]["corpus_id"]
                       and conversation(row)]
        else:
            ordered.sort(key=lambda row: not (row["dataset"] == config["retrieval"]["corpus_id"]
                                              and conversation(row)))
        selected.extend(ordered if args.stage == "stop" or count is None else ordered[:count])
    source = selected
    from adapters.shared_encoding import SharedQueryEncoder, SharedEvidenceEncoder
    from adapters.shared_retrieval import SharedEvidenceState, SharedEvidenceProvider
    from adapters.document_compute import DocumentIndex
    provider = (SharedEvidenceProvider(config, template)
                if any(row["dataset"] == config["retrieval"]["corpus_id"] for row in source) else None)
    if provider:
        query_encoder, evidence_encoder = provider.query_encoder, provider.evidence_encoder
    else:
        query_encoder = SharedQueryEncoder(**config["shared"]["query"])
        evidence_encoder = SharedEvidenceEncoder(**config["shared"]["evidence"])
    from controller.value_policy import StopValueHead
    controller = CooperativeController(None, engine, template, settings=run.cooperative,
        policy=run.context, stop_model=StopValueHead())
    kv_policy = controller.kv_policy
    if args.stage == "stop":
        from bench.study_resources import StudyResourceSampler, attribute_physical, listening_processes
        await engine.profile_prepare()
        roots = (os.getpid(), *listening_processes((urlparse(run.engine.base_url).port,)))
        sampler = StudyResourceSampler(tuple(args.gpu_devices), roots)
        await sampler.start()
    outputs, windows = [], {}
    limit = asyncio.Semaphore(args.concurrency)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")
    record_count = answer_count = 0

    def save_record(record):
        nonlocal record_count, answer_count
        with path.open("a") as output:
            output.write(json.dumps(record, ensure_ascii=False)+"\n")
        record_count += 1
        answer_count += len(record["sets"] if args.stage == "sets" else record["states"])

    def state_for(query, encoded, observed, all_units, identity, totals, catalogue):
        from adapters.version_structure import catalogue_versions
        blocks = source_blocks(observed)
        state = SharedEvidenceState(query, encoded, EvidencePool(tuple(observed), blocks), (),
            DocumentIndex.from_blocks(source_blocks(all_units)), body_tokens=config["selection"]["body_tokens"],
            cache_scope=identity, cache_namespace=identity, totals=totals)
        SharedEvidenceProvider._finish_state(state)
        state.version_catalogue = catalogue_versions(catalogue, query.task.version_intent.document_families)
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
        item = {"source": row, "answer": result["text"], "output_tokens": result["meta"]["completion_tokens"]}
        if args.stage == "stop":
            windows[identity] = started, time.monotonic()
            item.update(call_id=identity, costs={"seconds": windows[identity][1]-started})
        await engine.close_session(identity)
        return item

    async def collect_state(row, index):
        arrived = time.monotonic()
        async with limit:
            snapshot = SessionSnapshot(row["task_id"], raw_history=history(row))
            query = None
            if row["dataset"] == config["retrieval"]["corpus_id"]:
                query, units = await provider.retrieve_candidates(snapshot, row["question"], row["task_id"],
                                                                  initial_only=args.stage == "sets")
                row = {**row, "units": to_jsonable(units), "candidate_origin": "online_scope_hybrid"}
            else:
                units = tuple(decode(EvidenceUnit, unit) for unit in row["units"])
            if args.stage == "stop":
                by_id = {unit.unit_id: unit for unit in units}
                initial = {unit.unit_id for unit in SharedEvidenceProvider._closure(units[:batch_size], by_id)}
                if all(unit.unit_id in initial for unit in units):
                    return False
            if query is None:
                query = await asyncio.to_thread(query_encoder.encode, conversation(row), row["question"],
                    row["task_id"], row["turn"], catalogue=row.get("catalogue", ()))
            if query.task.coverage_scope == "all_supplied_documents" or not units:
                return False
            snapshot = replace(snapshot, raw_history=working_history(query, snapshot.raw_history))
            totals = execution_totals()
            if args.stage == "sets":
                encoded = await asyncio.to_thread(evidence_encoder.encode, query, units)
                state = state_for(query, encoded, units, units, f"sets:{row['task_id']}:prepare", totals, row.get("catalogue", ()))
                observations = []
                budget, _ = controller._answer_budget(state, snapshot, state.computations)
                sizes = {unit.unit_id: unit.body_tokens for unit in units}
                frontier = tuple(ids for ids in encoded.frontier(units)
                                 if sum(sizes[identity] for identity in ids) <= budget)
                await controller._select_evidence(state, snapshot, state.computations)
                choices = candidate_sets(frontier, state.selected_ids)
                for chosen in encoded.complementary_sets(units,
                        tuple(need.need_id for need in query.task.needs), state.completed_needs):
                    if chosen not in choices and sum(sizes[identity] for identity in chosen) <= budget:
                        choices += (chosen,)
                if row.get("paired_sets"):
                    by_id = {unit.unit_id: unit for unit in units}
                    for ids in row["paired_sets"]:
                        blocks = SharedEvidenceProvider._closure((by_id[key] for key in ids), by_id)
                        choice = tuple(block.unit_id for block in blocks)
                        if choice not in choices and sum(sizes[key] for key in choice) <= budget:
                            choices += (choice,)
                support, known = encoded.need_support(units, tuple(n.need_id for n in query.task.needs), state.completed_needs)
                masks = encoded.embeddings.new_tensor([[k in choice for k in encoded.unit_ids] for choice in choices])
                features = encoded.heads.features(encoded.embeddings, encoded.relevance, masks, support, known).tolist()
                for number, chosen in enumerate(choices):
                    identity = f"sets:{row['task_id']}:{number}"
                    branch = replace(state, query=replace(state.query), selected_ids=chosen,
                        read_units=set(state.read_units), processed_tables=set(state.processed_tables),
                        resolved_operations=dict(state.resolved_operations),
                        cache_scope=identity, cache_namespace=identity)
                    item = await answer(row, branch, snapshot, identity, totals)
                    observations.append((chosen, item))
                await score_answers([item for _, item in observations], run.engine.base_url,
                    model=run.model.name, concurrency=args.concurrency)
                async with httpx.AsyncClient(timeout=180) as client:
                    facts, marks = await reference_coverage(client, run.engine.base_url, run.model.name,
                        row["question"], row["reference_answer"], [item["answer"] for _, item in observations])
                save_record({**row, "completed_needs": sorted(state.completed_needs),
                    "selected_ids": state.selected_ids, "reference_facts": facts, "known": known.tolist(),
                    "sets": [{"unit_ids": chosen, "quality": item["quality"], "covered_facts": flags,
                              "output_tokens": item["output_tokens"], "features": vector}
                             for (chosen, item), flags, vector in zip(observations, marks, features)]})
                return True
            # A trajectory includes its initial queue and online work. The
            # counterfactual STOP probes and warmup fixtures are not that work.
            paid_seconds = time.monotonic()-arrived
            warm = index % 2 == 1
            state, encoded, states = None, None, []
            pending, observed = units, ()
            while pending:
                branch_id = f"stop:{row['task_id']}:{len(observed)}"
                if warm:
                    seed = []
                    request = compile_request(template, row["question"], snapshot.raw_history, ())
                    current = {unit.unit_id for unit in SharedEvidenceProvider._closure(pending[:batch_size], by_id)}
                    following = tuple(unit for unit in pending if unit.unit_id not in current)
                    upcoming = SharedEvidenceProvider._closure(following[:batch_size], by_id)
                    for unit in upcoming:
                        blocks = source_blocks((*seed, unit))
                        candidate = compile_request(template, row["question"], snapshot.raw_history,
                            blocks, future_sources=blocks)
                        if len(candidate.input_ids)+1 > run.context.max_total_model_tokens:
                            break
                        seed.append(unit)
                        request = candidate
                    if seed:
                        await engine.compute(request, session_id=branch_id, request_id=branch_id+":warm",
                            max_new_tokens=1, mode=config["shared"]["runtime"]["kv_mode"],
                            cache_scope=branch_id, cache_namespace=branch_id, **kv_policy)
                first = time.monotonic()
                seen = {unit.unit_id for unit in observed}
                fresh = tuple(unit for unit in SharedEvidenceProvider._closure(pending[:batch_size], by_id)
                              if unit.unit_id not in seen)
                new = await asyncio.to_thread(evidence_encoder.encode, query if state is None else state.query, fresh)
                encoded = new if encoded is None else encoded.merge(new)
                observed = (*observed, *fresh)
                seen.update(unit.unit_id for unit in fresh)
                pending = tuple(unit for unit in pending if unit.unit_id not in seen)
                if state is None:
                    state = state_for(query, encoded, observed, units, branch_id, totals, row.get("catalogue", ()))
                else:
                    state = replace(state, evidence=encoded, pool=EvidencePool(observed, source_blocks(observed)),
                        costs=None, cache_scope=branch_id, cache_namespace=branch_id)
                    SharedEvidenceProvider._finish_state(state)
                state.pending_units = pending
                state.evidence_batch = batch_size
                complete = controller.executor(branch_id, branch_id+":work", totals,
                    cache_scope=branch_id, cache_namespace=branch_id)
                await controller.prepare_state(state, snapshot, complete)
                if not states and (state.needs_complete or state.budget_exhausted):
                    await engine.close_session(branch_id)
                    return False
                transition_id = f"expand:{row['task_id']}:{len(observed)}"
                windows[transition_id] = first, time.monotonic()
                paid_seconds += windows[transition_id][1]-first
                if states:
                    states[-1]["transition_id"] = transition_id
                features = StopValueHead.features(encoded.summary(state.selected_ids, state.pool.units,
                    tuple(need.need_id for need in state.query.task.needs), state.completed_needs),
                    controller._value_state(state, time.monotonic()-paid_seconds)).cpu().tolist()
                item = await answer(row, state, snapshot, branch_id, totals)
                states.append({"features": features, "answer": item})
                if state.needs_complete or state.budget_exhausted:
                    break
            if len(states) > 1:
                outputs.append((row, states))
                return True
            return False

    try:
        if args.stage == "sets":
            await asyncio.gather(*(collect_state(row, index) for index, row in enumerate(source)))
        else:
            for split, count in (("train", args.train_states), ("dev", args.dev_states)):
                candidates = iter((index, row) for index, row in enumerate(source) if row["split"] == split)
                remaining = count
                while remaining is None or remaining > 0:
                    batch = tuple(islice(candidates, args.concurrency if remaining is None
                                         else min(args.concurrency, remaining)))
                    if not batch:
                        break
                    accepted = await asyncio.gather(*(collect_state(row, index) for index, row in batch))
                    if remaining is not None:
                        remaining -= sum(accepted)
    finally:
        if args.stage == "stop":
            await sampler.stop()
        await engine.aclose()
        query_encoder.close()
        evidence_encoder.close()
    if args.stage == "stop":
        attributed = attribute_physical(sampler.samples, windows)
        answers = [state["answer"] for _, states in outputs for state in states]
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
                     "split": row["split"], "candidate_origin": row.get("candidate_origin", "supplied"), "states": states})
    return {"states": record_count, "answers": answer_count, "output": str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("sets", "stop"))
    parser.add_argument("--config", default="configs/study.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-devices", type=int, nargs="+", default=[4, 5, 6, 7])
    parser.add_argument("--batch-size", type=int, help="Evidence expansion size; defaults to the online retrieval batch")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--train-states", type=int)
    parser.add_argument("--dev-states", type=int)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(collect(args))))


if __name__ == "__main__":
    main()
