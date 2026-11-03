"""Actual-answer collection for six evidence-set interventions on train turns."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path
import time
import uuid

from adapters.support_model import SupportModel, SupportProfile
from controller.context_schema import (ContextBudget, ContextPlan, EvidenceSet,
    RetrievalState, TaskState)
from controller.context_budget import resolve_text
from controller.engine_client import EngineError
from controller.evidence_planner import (BudgetedEvidencePlanner, close_dependencies,
    render_selection)
from controller.schema import Message, QueryContext, SessionSnapshot, TurnRecord
from controller.serialization import decode, to_jsonable
from training.support_data import answer_state_supervision, build_preference_pairs


def condition_version_mismatches(state, units, resolve):
    active = {}
    categorical = {"PLATFORM", "LOCATION", "REGION", "PRODUCT", "VERSION", "POPULATION"}
    for condition in state.conditions:
        key = condition.relation_type.upper()
        if key in categorical:
            active.setdefault(key, set()).update(resolve(ref).strip().casefold() for ref in condition.value_spans)
    source_scope = set(state.source_scope)
    expected = {}
    for unit in units:
        for span in unit.source_spans:
            if span.source_key in source_scope:
                expected.setdefault(unit.document_family, set()).add(span.source_key.version)
    result = []
    for unit in units:
        reasons = []
        for key, value in unit.applicability:
            requested = active.get(key.upper())
            if requested and str(value).strip().casefold() not in requested:
                reasons.append({"kind": "condition", "key": key, "source_value": value,
                                "requested_values": sorted(requested)})
        versions = (state.version_intent.resolved_version_ids or state.version_intent.version_ids
                    or tuple(expected.get(unit.document_family, ())))
        if versions and any(span.source_key.version not in versions for span in unit.source_spans):
            reasons.append({"kind": "version", "source_versions": sorted({span.source_key.version for span in unit.source_spans}),
                            "requested_versions": list(versions)})
        if reasons:
            result.append((unit, tuple(reasons)))
    return tuple(result)


def remove_premise(units, premise_id):
    """Remove a premise and the source units whose meaning depends on it."""
    removed = {premise_id}
    while True:
        dependents = {unit.unit_id for unit in units
                      if removed.intersection(unit.dependencies)} - removed
        if not dependents:
            return tuple(unit for unit in units if unit.unit_id not in removed)
        removed.update(dependents)


def candidate_sets(state, pool, scores, model, tokenizer, budget):
    by_id = {unit.unit_id: unit for unit in pool.units}
    all_supports = (*scores.singles, *scores.bundles)

    def make(label, units):
        ids = tuple(dict.fromkeys(unit.unit_id for unit in units))
        selected = close_dependencies(ids, by_id)
        body = len(tokenizer.encode(render_selection(selected), add_special_tokens=False))
        if body > budget.body_tokens:
            return None
        keys = {span.source_key for unit in selected for span in unit.source_spans}
        support = tuple(row for row in all_supports if set(row.unit_ids) <= {u.unit_id for u in selected})
        prediction = model.predict_set(state, selected, support)
        return EvidenceSet(f"{state.task_id}:{label}", selected,
            tuple(unit.unit_id for unit in selected if unit.unit_id not in ids), support,
            prediction.quality, body, prediction.disposition,
            source_blocks=tuple(block for block in pool.source_blocks if block.key in keys))

    def pack(label, ranked):
        selected = []
        for unit in ranked:
            proposal = make(label, (*selected, unit))
            if proposal is not None:
                selected = list(proposal.selected_units)
        return make(label, selected)

    ranked = sorted(pool.units, key=lambda unit: (-unit.retrieval_score, unit.unit_id))
    direct = {unit.unit_id: max((row.applicability*row.joint_support for row in scores.singles
              if row.unit_ids == (unit.unit_id,)), default=0) for unit in pool.units}
    direct_ranked = sorted(pool.units, key=lambda unit: (-direct[unit.unit_id],
                                                       -unit.retrieval_score, unit.unit_id))
    relevance = pack("relevance", ranked)
    supported = pack("direct_support", direct_ranked)
    complementary = BudgetedEvidencePlanner(tokenizer, model).search_sets(state, pool, scores, budget)[0]
    complementary = replace(complementary, set_id=f"{state.task_id}:complementary")
    first_words = set(ranked[0].text.split()) if ranked else set()
    redundant_ranked = sorted(ranked, key=lambda unit: (
        -len(first_words & set(unit.text.split()))/max(1, len(first_words | set(unit.text.split()))),
        -unit.retrieval_score))
    redundant = pack("redundant", redundant_ranked)
    mismatches = condition_version_mismatches(state, pool.units, model.resolve)
    contradicted = {unit.unit_id for unit, _ in mismatches}
    wrong = pack("wrong_condition_version", [unit for unit in ranked if unit.unit_id in contradicted])
    if not contradicted:
        wrong = None
    removals = []
    for removed in complementary.selected_units:
        remaining = remove_premise(complementary.selected_units, removed.unit_id)
        changed = make("deleted_premise", remaining)
        if changed is not None:
            removals.append(changed)
    deleted = min(removals, key=lambda item: item.quality_prediction) if removals else None
    return {"relevance": relevance, "direct_support": supported, "complementary": complementary,
            "redundant": redundant, "wrong_condition_version": wrong, "deleted_premise": deleted}


def build_directory(config, tokenizer):
    from adapters.corpus import CorpusIndex
    from adapters.doc2dial import structural_corpus
    from adapters.evidence_pool import PoolProfile, SourceDirectory
    from adapters.hybrid_retrieval import DenseIndex, LocalBGE
    settings = config.get("preferences", {})
    data_dir = Path(config.get("training", {}).get("data_dir", "runs/study/data"))
    sources = [json.loads(line) for line in (data_dir/"sources.jsonl").read_text().splitlines()]
    documents = json.loads(Path(settings.get("documents", "data/multidoc2dial/source/multidoc2dial_doc.json")).read_text())
    corpus_rows = []
    for source in sources:
        domain, document_id = json.loads(source["source_id"])
        corpus_rows.append({**source, "passage_id": source["source_id"],
            "document_id": source["source_id"], "provenance": [source["source_uri"]],
            "metadata": {"domain": domain, "source_document_id": document_id}})
    records = tuple(structural_corpus(documents, corpus_rows))
    if settings.get("negative_sources_path"):
        additional = [json.loads(line) for line in Path(settings["negative_sources_path"]).read_text().splitlines() if line]
        records += tuple({**source, "passage_id": source.get("passage_id", source["source_id"]),
                          "document_id": source.get("document_id", source["source_id"])} for source in additional)
    index = CorpusIndex.from_records(records, corpus_id="multidoc2dial",
        snapshot_id="multidoc2dial-train-archive-f0c034c24966")
    dense = DenseIndex.load(settings.get("dense_index", "data/multidoc2dial/bge-m3-sections"))
    encoder = LocalBGE(settings.get("encoder_path", "models/bge-m3"),
        device=settings.get("encoder_device", settings.get("device", config.get("support_model", {}).get("device", "cuda:2"))),
        dtype="bfloat16", batch_size=16,
        query_adapter_path=settings.get("query_adapter_path", "models/bge-m3-dialogue-query-v1"),
        normalize_float32=True)
    directory = SourceDirectory(index, lambda query, top_k: dense.search(query, encoder, top_k=top_k),
        tokenizer, profile=PoolProfile(**{key: value for key, value in config.get("retrieval", {}).items()
                                         if key in PoolProfile.__dataclass_fields__}))
    return directory, encoder


def _usage_delta(before, after):
    if before is None or after is None:
        return {key: None for key in ("input_tokens", "output_tokens", "compute_ms")}
    return {key: after[key]-before[key] if after[key] is not None and before[key] is not None else None
            for key in ("input_tokens", "output_tokens", "compute_ms")}


def _encoder_usage(encoder):
    if encoder is None:
        return None
    usage = encoder.usage
    return {"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
            "compute_ms": usage["elapsed_ms"]}


def _sum_known(values):
    return sum(values) if all(value is not None for value in values) else None


def failed_read_candidate(controller, row, category, settings, output_dir, *, request_window=None):
    """Retain a paid terminal READ truncation without producing a quality label."""
    session = json.dumps(["preference", row["task_id"], category], separators=(",", ":"))
    attempt = controller.store.find_request(session, "answer")
    if attempt.stage != "FAILED" or attempt.error.partition(":")[0] != "READ_STEP_OUTPUT_TRUNCATED":
        raise ValueError(f"candidate is not a terminal READ truncation: {attempt.stage}/{attempt.error}")
    records = controller.store.execution_steps_with_stage(attempt.attempt_id)
    paid = [record for _, record in records if record.receipt is not None]
    snapshot = controller.store.get_snapshot(session)
    qa_increment = len(snapshot.raw_history)-len(row["actual_history"])
    if qa_increment != 0:
        raise ValueError("failed READ candidate has committed a new QA")
    return {"task_id": row["task_id"], "category": category, "split": "train", "status": "failed",
        "failure_code": "READ_STEP_OUTPUT_TRUNCATED", "attempt_id": attempt.attempt_id,
        "reader_checkpoint": settings.get("reader_checkpoint"), "reader_profile_id": settings.get("reader_profile_id"),
        "scorer_identity": settings.get("scorer_identity"), "question": row["question"],
        "actual_history": row["actual_history"], "history_origin": "frozen_actual_rollout",
        "qa_increment": qa_increment, "version_increment": snapshot.version-attempt.expected_version,
        "quality": None, "quality_components": None, "request_window": request_window,
        "window_origin": "current_measured_request" if request_window is not None else "unrecorded_previous_invocation",
        "attempt": to_jsonable(attempt), "context_plan": to_jsonable(controller.store.context_plan(attempt.attempt_id)),
        "step_records": [{"stage": stage, "record": to_jsonable(record)} for stage, record in records],
        "phase_cost": {"complete_ms": (request_window[1]-request_window[0])*1000 if request_window is not None else None,
            "reader_input_tokens": sum(len(record.input_ids) for record in paid),
            "reader_output_tokens": sum(len(record.output_ids) for record in paid),
            "model_elapsed_ms": _sum_known([record.metrics.get("elapsed_ms") for record in paid]),
            "cpu_reduce_ms": sum(record.metrics.get("cpu_reduce_ms", 0) for _, record in records),
            "initial_auxiliary_usage": attempt.auxiliary_usage},
        "physical_cost": None,
        "original_batch_cost_paths": [str(path) for path in sorted(Path(output_dir).glob("collection_cost.*.json"))],
        "training_target": False, "retry_executed": False}


def append_record(path, row):
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(row, ensure_ascii=False)+"\n")


async def _judge(client, template, row, evidence_set, answer):
    components = ("necessary_points", "grounding", "conditions_versions", "task_score")
    schema = {"type": "object", "properties": {
        **{name: {"type": "number", "minimum": 0, "maximum": 1} for name in components},
        "disposition": {"type": "string", "enum": ["answer", "partial", "retrieval_gap", "unanswerable", "underspecified"]},
        "rationale": {"type": "string"}}, "required": [*components, "disposition", "rationale"],
        "additionalProperties": False}
    messages = (Message("system", "Evaluate the actual answer against the current question, actual history, "
        "active conditions, original sources and offline reference. Judge each required point and all "
        "condition/version constraints. Reference answers may describe a different author history; "
        "the supplied actual history and current task determine this request. Return the requested JSON."),
        Message("user", json.dumps({"question": row["question"], "actual_history": row["actual_history"],
            "task_state": row["task_state"], "reference_answer": row.get("reference_answer"),
            "offline_corpus_has_answer": row.get("corpus_has_answer"),
            "offline_scope_complete": row.get("oracle_scope_complete", False),
            "original_sources": to_jsonable(evidence_set.selected_units), "answer": answer}, ensure_ascii=False)))
    rendered = template.render(messages)
    started = time.perf_counter()
    response = await client.post("/generate", json={"input_ids": list(rendered.input_ids), "stream": False,
        "sampling_params": {"temperature": 0, "max_new_tokens": 1024,
                            "json_schema": json.dumps(schema)}})
    response.raise_for_status()
    raw = response.json()
    judged = json.loads(raw["text"])
    return judged, {"raw_output": raw["text"], "meta_info": raw["meta_info"],
        "input_tokens": len(rendered.input_ids), "output_tokens": raw["meta_info"].get("completion_tokens"),
        "elapsed_ms": (time.perf_counter()-started)*1000}


async def collect_preferences(config, *, controller=None, directory=None, limit=None):
    """Collect real train answers and rewards; return a manifest read by training."""
    import httpx
    from controller.config import create_controller, load_config
    settings = config.get("preferences", {})
    training = config.get("training", {})
    data_dir = Path(training.get("data_dir", "runs/study/data"))
    output_dir = Path(settings.get("output_dir", Path(config.get("study", {}).get("output_dir", "runs/study"))/"preferences"))
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = Path(settings.get("turns_path", training.get("task_labels", data_dir/"task_labels.jsonl")))
    rows = [json.loads(line) for line in labels_path.read_text().splitlines() if line]
    rows = rows[:limit or settings.get("turns", training.get("preference_turns", 200))]
    assert rows and all(row["split"] == "train" and "actual_history" in row for row in rows)
    own_controller = controller is None
    if own_controller:
        run = load_config(settings.get("run_config", config.get("generation", {}).get("run_config", "configs/qwen3_32b.yaml")))
        if settings.get("reader_base_url"):
            run = replace(run, engine=replace(run.engine, base_url=settings["reader_base_url"]))
        controller = create_controller(replace(run, state_dir=str(output_dir/"runtime"),
            planner=replace(run.planner, strategy="always_rebuild")))
    encoder = None
    if directory is None:
        directory, encoder = build_directory(config, controller.template.tokenizer)
    asset = settings.get("support_asset", config.get("support_model", {}).get("asset", "models/study-support"))
    model = SupportModel.from_pretrained(asset, history_resolver=lambda ref: "",
        device=settings.get("device", config.get("support_model", {}).get("device", "cuda:2")), profile=SupportProfile(**{
            key: value for key, value in config.get("support_model", {}).items()
            if key in SupportProfile.__dataclass_fields__}))
    budget = ContextBudget(body_tokens=settings.get("body_budget", 4096),
        output_tokens=settings.get("output_budget", 1024),
        read_input_tokens=settings.get("read_input_tokens", 12288),
        read_output_tokens=settings.get("read_output_tokens", 2048),
        max_steps=settings.get("max_steps", 64))
    rewards_path = output_dir/"rewards.jsonl"
    existing = [json.loads(line) for line in rewards_path.read_text().splitlines() if line] if rewards_path.exists() else []
    completed = {(row["task_id"], row["category"]) for row in existing}
    categories = ("relevance", "direct_support", "complementary", "redundant",
                  "wrong_condition_version", "deleted_premise")
    failures_path = output_dir/"failed_cases.jsonl"
    failures = [json.loads(line) for line in failures_path.read_text().splitlines() if line] if failures_path.exists() else []
    completed.update((row["task_id"], row["category"]) for row in failures)
    skipped_path = output_dir/"skipped_cases.jsonl"
    skipped = [json.loads(line) for line in skipped_path.read_text().splitlines() if line] if skipped_path.exists() else []
    completed.update((row["task_id"], row["category"]) for row in skipped)
    for row in rows:
        for category in categories:
            session = json.dumps(["preference", row["task_id"], category], separators=(",", ":"))
            previous = controller.store.find_request(session, "answer")
            if (row["task_id"], category) not in completed and previous is not None and previous.stage == "FAILED":
                failure = failed_read_candidate(controller, row, category, settings, output_dir)
                append_record(failures_path, failure)
                failures.append(failure)
                completed.add((row["task_id"], category))
                await controller.close_session(session)
    collected = list(existing)
    preparations_path = output_dir/"preparations.jsonl"
    preparations = ([json.loads(line) for line in preparations_path.read_text().splitlines() if line]
                    if preparations_path.exists() else [])
    started = time.perf_counter()
    async with httpx.AsyncClient(base_url=settings.get("judge_base_url", config.get("generation", {}).get("teacher_url", "http://127.0.0.1:30001")), timeout=180) as judge:
        for task_index, row in enumerate(rows):
            if all((row["task_id"], category) in completed for category in categories):
                continue
            preparation_started = time.perf_counter()
            support_before = dict(model.backend.usage)
            retrieval_before = _encoder_usage(encoder)
            history = tuple(decode(TurnRecord, item) for item in row["actual_history"])
            snapshot = SessionSnapshot(row["conversation_id"], raw_history=history)
            state = decode(TaskState, row["task_state"])
            model.resolve = lambda ref: resolve_text(ref, state, snapshot)
            memory = RetrievalState(len(history), history,
                tuple(turn.query_context.task_state for turn in history
                      if turn.query_context and turn.query_context.task_state), source_index=directory)
            pool = directory.retrieve_pool(state, memory)
            scores = model.score_support(state, pool)
            families = {unit.document_family for unit in pool.units} | set(state.version_intent.document_families)
            outside = [(unit, reasons) for unit, reasons in condition_version_mismatches(
                state, tuple(directory.units_by_id.values()), model.resolve)
                if unit.document_family in families and unit.unit_id not in {u.unit_id for u in pool.units}]
            outside = outside[:settings.get("negative_units", 6)]
            if outside:
                negative_units = close_dependencies(tuple(unit.unit_id for unit, _ in outside), directory.units_by_id)
                negative_keys = {span.source_key for unit in negative_units for span in unit.source_spans}
                negative_blocks = directory.source_fetch(negative_keys)
                from controller.context_schema import EvidencePool
                negative_pool = EvidencePool(negative_units, negative_blocks)
                negative_scores = model.score_support(state, negative_pool)
                scores = replace(scores, singles=tuple({s.bundle_id: s for s in (*scores.singles, *negative_scores.singles)}.values()),
                    bundles=tuple({s.bundle_id: s for s in (*scores.bundles, *negative_scores.bundles)}.values()),
                    scoring_ms=scores.scoring_ms+negative_scores.scoring_ms)
                pool = replace(pool, units=(*pool.units, *negative_units),
                    source_blocks=tuple({b.key: b for b in (*pool.source_blocks, *negative_blocks)}.values()))
            sets = candidate_sets(state, pool, scores, model, controller.template.tokenizer, budget)
            support_usage = _usage_delta(support_before, model.backend.usage)
            retrieval_usage = _usage_delta(retrieval_before, _encoder_usage(encoder))
            preparation = {"shared_cost_id": row["task_id"]+":prep:"+uuid.uuid4().hex[:12],
                "task_id": row["task_id"], "wall_ms": (time.perf_counter()-preparation_started)*1000,
                "scope": "training_candidate_family_preparation",
                "models": {"support_utility": support_usage, "query_retrieval": retrieval_usage},
                **{key: _sum_known([support_usage[key], retrieval_usage[key]])
                   for key in ("input_tokens", "output_tokens", "compute_ms")}}
            with preparations_path.open("a") as output:
                output.write(json.dumps(preparation, ensure_ascii=False)+"\n")
            preparations.append(preparation)
            charge_preparation = True
            pools_dir = output_dir/"pools"
            pools_dir.mkdir(exist_ok=True)
            pool_path = pools_dir/f"pool-{task_index:05d}.json"
            pool_path.write_text(json.dumps({"task_id": row["task_id"],
                "pool": to_jsonable(pool), "support_scores": to_jsonable(scores)}, ensure_ascii=False)+"\n")
            history_id = json.dumps(row["actual_history"], ensure_ascii=False, sort_keys=True)
            for category in categories:
                if (row["task_id"], category) in completed:
                    continue
                selected = sets[category]
                if selected is None:
                    skipped_case = {"task_id": row["task_id"], "category": category,
                                    "reason": "no_matching_source_intervention"}
                    append_record(skipped_path, skipped_case)
                    skipped.append(skipped_case)
                    completed.add((row["task_id"], category))
                    continue
                session_id = json.dumps(["preference", row["task_id"], category], separators=(",", ":"))
                plan = ContextPlan(f"{session_id}:plan", state, selected, budget=budget,
                    auxiliary_ms=(("retrieval", pool.retrieval_ms), ("support", scores.scoring_ms)))
                if settings.get("reading", False):
                    from controller.reading import build_read_plan
                    plan = build_read_plan(state, selected, "selected_sources", budget,
                        template=controller.template, snapshot=snapshot,
                        read_layout=settings.get("read_layout", "source_first"),
                        read_packing=settings.get("read_packing", "mixed"),
                        separate_needs=settings.get("separate_needs", False))
                generation_started = time.monotonic()
                try:
                    response = await controller.serve_turn(session_id, row["question"],
                        selected.source_blocks, request_id="answer", corpus_snapshot_id=row["snapshot_id"],
                        reference_history=history, context_plans=(plan,),
                        query_context=QueryContext(state.query or state.raw_question, "task", task_state=state),
                        auxiliary_usage={**preparation, "charge_scope": "once_per_candidate_family"}
                            if charge_preparation else {"input_tokens": 0, "output_tokens": 0,
                                "compute_ms": 0.0, "models": [], "shared_preparation_cost_id": preparation["shared_cost_id"]})
                except EngineError as exc:
                    if exc.code != "READ_STEP_OUTPUT_TRUNCATED":
                        raise
                    generation_ended = time.monotonic()
                    failure = failed_read_candidate(controller, row, category, settings, output_dir,
                        request_window=[generation_started, generation_ended])
                    append_record(failures_path, failure)
                    failures.append(failure)
                    completed.add((row["task_id"], category))
                    charge_preparation = False
                    await controller.close_session(session_id)
                    continue
                generation_ended = time.monotonic()
                charge_preparation = False
                actual_candidate = controller.store.get_attempt(response.attempt_id).plan.candidate
                judged, judge_record = await _judge(judge, controller.template, row, selected, response.assistant_text)
                dispositions = {support.disposition for support in selected.need_support}
                support_disposition = ("complete" if dispositions == {"complete"} else
                                       "partial" if "partial" in dispositions or "complete" in dispositions else "background")
                answer_labels, answer_origin = answer_state_supervision({
                    "answer_disposition": judged["disposition"], "disposition": support_disposition,
                    "quality": None}, corpus_has_answer=row.get("corpus_has_answer"),
                    scope_complete=row.get("oracle_scope_complete", False))
                support_rows = tuple(s for s in (*scores.singles, *scores.bundles)
                    if set(s.unit_ids) <= {u.unit_id for u in selected.selected_units})
                from adapters.support_model import query_view
                need_by_id = {need.need_id: need for need in state.needs}
                all_units = {unit.unit_id: unit for unit in pool.units}
                support_inputs = [{"need_id": support.need_id,
                    "query": query_view(state, need_by_id[support.need_id], model.resolve),
                    "units": to_jsonable(tuple(all_units[uid] for uid in support.unit_ids)),
                    "dependencies": to_jsonable(tuple(unit for unit in close_dependencies(support.unit_ids, all_units)
                                                     if unit.unit_id not in support.unit_ids))}
                    for support in support_rows]
                result = {"task_id": row["task_id"], "split": "train", "history_id": history_id,
                    "actual_history": row["actual_history"], "question": row["question"],
                    "history_origin": "frozen_actual_rollout", "measurement_kind": "hardware",
                    "reader_checkpoint": settings.get("reader_checkpoint"),
                    "reader_profile_id": settings.get("reader_profile_id"),
                    "scorer_identity": settings.get("scorer_identity"),
                    "request_window": [generation_started, generation_ended],
                    "body_budget": budget.body_tokens, "output_budget": budget.output_tokens,
                    "category": category, "action": response.action, "layout": actual_candidate.layout_id,
                    "task_state": to_jsonable(state), "units": to_jsonable(selected.selected_units),
                    "support_model_asset": str(asset),
                    "supports": to_jsonable(support_rows), "support_inputs": support_inputs,
                    "answer": response.assistant_text, "reader_model": "qwen3_32b_nonthinking",
                    "quality_components": {name: judged[name] for name in (
                        "necessary_points", "grounding", "conditions_versions", "task_score")},
                    "disposition": answer_labels["answer_disposition"],
                    "raw_judge_disposition": judged["disposition"],
                    "disposition_supervision_origin": answer_origin,
                    "corpus_has_answer": row.get("corpus_has_answer"),
                    "oracle_scope_complete": row.get("oracle_scope_complete", False),
                    "judge_record": judge_record,
                    "condition_version_interventions": [
                        {"unit_id": unit.unit_id, "reasons": reasons}
                        for unit, reasons in condition_version_mismatches(state, selected.selected_units, model.resolve)],
                    "response": to_jsonable(response), "input_tokens": response.input_tokens,
                    "input_hash": actual_candidate.input_hash,
                    "reader_input_tokens": response.reader_input_tokens,
                    "reader_output_tokens": response.reader_output_tokens,
                    "input_total_tokens": response.total_input_tokens, "output_total_tokens": response.total_output_tokens,
                    "shared_preparation_cost": preparation,
                    "comparable_cost": {"complete_ms": response.complete_ms+preparation["wall_ms"]
                            if response.complete_ms is not None else None,
                        "input_tokens": _sum_known([response.reader_input_tokens, preparation["input_tokens"]]),
                        "output_tokens": _sum_known([response.reader_output_tokens, preparation["output_tokens"]]),
                        "scope": "reader_plus_shared_preparation_attributed_equally"},
                    "support_usage": dict(model.backend.usage), "candidate_pool_path": str(pool_path),
                    "evidence_set": to_jsonable(selected)}
                with rewards_path.open("a") as output:
                    output.write(json.dumps(result, ensure_ascii=False)+"\n")
                collected.append(result)
                completed.add((row["task_id"], category))
                await controller.close_session(session_id)
            model.release_task(state.task_id)
    await controller.drain()
    if own_controller:
        await controller.aclose()
    if encoder is not None:
        encoder.close()
    weights = settings.get("reward_weights", training.get("reward_weights", {
        "necessary_points": .3, "grounding": .2, "conditions_versions": .2, "task_score": .3}))
    preferences_path = Path(settings.get("preferences_path", training.get("preferences", output_dir/"preferences.jsonl")))
    pair_manifest = build_preference_pairs(rewards_path, preferences_path, weights=weights)
    manifest = {"turns_path": str(labels_path), "planned_turns": len(rows), "answers": len(collected),
        "tasks_with_answers": len({row["task_id"] for row in collected}),
        "categories": {name: sum(row["category"] == name for row in collected) for name in categories},
        "skipped": skipped, "failed_candidates": len(failures), "failure_codes": {
            "READ_STEP_OUTPUT_TRUNCATED": len(failures)}, "legal_actions": ["REBUILD"],
        "elapsed_ms": (time.perf_counter()-started)*1000,
        "reader_input_tokens": sum(row["reader_input_tokens"] for row in collected),
        "reader_output_tokens": sum(row["reader_output_tokens"] for row in collected),
        "preparation_input_tokens": _sum_known([item["input_tokens"] for item in preparations]),
        "preparation_output_tokens": _sum_known([item["output_tokens"] for item in preparations]),
        "preparations_path": str(preparations_path),
        "judge_input_tokens": sum(row["judge_record"]["input_tokens"] for row in collected),
        "judge_output_tokens": _sum_known([row["judge_record"].get("output_tokens") for row in collected]),
        "support_usage": model.backend.usage, "pairs": pair_manifest,
        "outputs": {"rewards": str(rewards_path), "preferences": str(preferences_path),
                    "failed_cases": str(failures_path), "skipped_cases": str(skipped_path)}}
    (output_dir/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    model.close()
    return manifest


def main(argv=None):
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--turns", type=int)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    print(json.dumps(asyncio.run(collect_preferences(config, limit=args.turns)), ensure_ascii=False))


if __name__ == "__main__":
    main()
