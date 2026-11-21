"""Collect real answer utility for budgeted evidence sets."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path

import httpx
import yaml

from controller.config import create_model_template, load_config, apply_study_config
from controller.cooperative import CooperativeController, execution_totals
from controller.cooperative_request import working_history
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
    provider = (SharedEvidenceProvider(config, template)
                if any(row["dataset"] == config["retrieval"]["corpus_id"] for row in source) else None)
    if provider:
        query_encoder, evidence_encoder = provider.query_encoder, provider.evidence_encoder
    else:
        query_encoder = SharedQueryEncoder(**config["shared"]["query"])
        evidence_encoder = SharedEvidenceEncoder(**config["shared"]["evidence"])
    controller = CooperativeController(None, engine, template, settings=run.cooperative, policy=run.context)
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
        answer_count += len(record["sets"])

    def state_for(query, encoded, units, identity, totals, catalogue):
        from adapters.version_structure import catalogue_versions
        blocks = source_blocks(units)
        state = SharedEvidenceState(query, encoded, EvidencePool(tuple(units), blocks), (),
            DocumentIndex.from_blocks(blocks), body_tokens=config["selection"]["body_tokens"],
            cache_scope=identity, cache_namespace=identity, totals=totals)
        SharedEvidenceProvider._finish_state(state)
        state.version_catalogue = catalogue_versions(catalogue, query.task.version_intent.document_families)
        return state

    async def answer(row, state, snapshot, identity, previous_totals):
        totals = dict(previous_totals)
        state = replace(state, totals=totals)
        complete = controller.executor(identity, identity, totals,
            cache_scope=state.cache_scope, cache_namespace=state.cache_namespace)
        await controller.read_required(state, snapshot, complete)
        request = controller.answer_request(state, snapshot)
        result = await complete(request, "answer")
        item = {"source": row, "answer": result["text"], "output_tokens": result["meta"]["completion_tokens"]}
        await engine.close_session(identity)
        return item

    async def collect_state(row):
        async with limit:
            snapshot = SessionSnapshot(row["task_id"], raw_history=history(row))
            query = None
            if row["dataset"] == config["retrieval"]["corpus_id"]:
                query, units = await provider.retrieve_candidates(snapshot, row["question"], row["task_id"])
                row = {**row, "units": to_jsonable(units), "candidate_origin": "online_scope_hybrid"}
            else:
                units = tuple(decode(EvidenceUnit, unit) for unit in row["units"])
            if query is None:
                query = await asyncio.to_thread(query_encoder.encode, conversation(row), row["question"],
                    row["task_id"], row["turn"], catalogue=row.get("catalogue", ()))
            if query.task.coverage_scope == "all_supplied_documents" or not units:
                return
            snapshot = replace(snapshot, raw_history=working_history(query, snapshot.raw_history))
            totals = execution_totals()
            encoded = await asyncio.to_thread(evidence_encoder.encode, query, units)
            state = state_for(query, encoded, units, f"sets:{row['task_id']}:prepare", totals, row.get("catalogue", ()))
            observations = []
            budget, _ = controller._answer_budget(state, snapshot, state.computations)
            sizes = {unit.unit_id: unit.body_tokens for unit in units}
            frontier = tuple(ids for ids in encoded.frontier(units)
                             if sum(sizes[identity] for identity in ids) <= budget)
            await controller._select_evidence(state, snapshot, state.computations)
            choices = candidate_sets(frontier, state.selected_ids)
            support, known = encoded.need_support(units, tuple(n.need_id for n in query.task.needs), state.completed_needs)
            masks = encoded.embeddings.new_tensor([[k in choice for k in encoded.unit_ids] for choice in choices])
            features = encoded.heads.features(encoded.embeddings, encoded.relevance, masks, support, known).tolist()
            for number, chosen in enumerate(choices):
                identity = f"sets:{row['task_id']}:{number}"
                branch = replace(state, query=replace(state.query), selected_ids=chosen,
                    read_units=set(state.read_units),
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

    try:
        await asyncio.gather(*(collect_state(row) for row in source))
    finally:
        await engine.aclose()
        query_encoder.close()
        evidence_encoder.close()
    return {"states": record_count, "answers": answer_count, "output": str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/study.yaml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--train-states", type=int)
    parser.add_argument("--dev-states", type=int)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(collect(args))))


if __name__ == "__main__":
    main()
