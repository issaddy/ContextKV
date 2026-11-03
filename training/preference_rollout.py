"""Actual-answer collection for six evidence-set interventions on train turns."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import sqlite3
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
from training.reader_data import digest, file_digest, history_digest


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


def candidate_sets(state, pool, scores, model, tokenizer, budget, *, current_set=None, expand_interventions=False):
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
    result = {"relevance": relevance, "direct_support": supported, "complementary": complementary,
              "redundant": redundant, "wrong_condition_version": wrong, "deleted_premise": deleted}
    if current_set is not None:
        result = {"current": make("current", current_set.selected_units), **result}
    if expand_interventions:
        # Each extra candidate removes observed source units and their dependent
        # branches. No invented source or task/quality label enters the family.
        result = {name: result[name] for name in ("current", "complementary", "wrong_condition_version",
            "deleted_premise", "relevance", "direct_support", "redundant") if name in result}
        seen = {tuple(sorted(unit.unit_id for unit in evidence.selected_units))
                for evidence in result.values() if evidence is not None}
        for unit in sorted(complementary.selected_units, key=lambda item: item.unit_id):
            if len(seen) >= 6:
                break
            remaining = remove_premise(complementary.selected_units, unit.unit_id)
            ids = tuple(sorted(item.unit_id for item in remaining))
            if ids in seen:
                continue
            label = "source_deletion:"+digest(unit.unit_id)[:12]
            evidence = make(label, remaining)
            if evidence is not None:
                result[label] = evidence
                seen.add(ids)
    return result




def bind_saved_turns(turns, state_db):
    """Read task state and the actually selected evidence from native attempts."""
    from controller.session_store import Attempt, load_record
    database = Path(state_db).resolve()
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    bound = []
    for turn in turns:
        identity = turn["response"]["attempt_id"]
        payload = connection.execute("SELECT payload FROM attempts WHERE attempt_id=?", (identity,)).fetchone()[0]
        attempt = load_record(connection, Attempt, payload)
        plan = attempt.reading_outcome.plan if attempt.reading_outcome is not None else attempt.context_plan
        if "task_state" in turn:
            task_state = turn["task_state"]
        else:
            if attempt.query_context is None or attempt.query_context.task_state is None:
                raise ValueError(f"saved turn {turn['task_id']} needs its reviewed task_labels row")
            task_state = to_jsonable(attempt.query_context.task_state)
        bound.append({**turn, "task_state": task_state,
            "current_evidence_set": to_jsonable(plan.selected_set) if plan is not None else None,
            "current_evidence_status": "stored_context_plan" if plan is not None else "stored_original_blocks",
            "current_source_blocks": to_jsonable(attempt.evidence)})
    connection.close()
    return bound


def current_evidence_from_blocks(row, tokenizer):
    """Express the actually supplied originals as source units, without editing text."""
    from adapters.corpus import CorpusIndex
    from adapters.evidence_pool import SourceDirectory
    from controller.schema import EvidenceBlock
    blocks = tuple(decode(EvidenceBlock, block) for block in row["current_source_blocks"])
    if not blocks:
        return EvidenceSet(f"{row['task_id']}:current", ())
    index = CorpusIndex(blocks, corpus_id=blocks[0].key.corpus_id, snapshot_id=row["snapshot_id"])
    directory = SourceDirectory(index, None, tokenizer)
    pool = directory.full_scope(tuple(block.key for block in blocks))
    return EvidenceSet(f"{row['task_id']}:current", pool.units, source_blocks=pool.source_blocks)

def real_candidate_family(state, pool, sets, *, minimum=4, maximum=6):
    """Keep distinct real-source interventions, with their dependency closure."""
    by_id = {unit.unit_id: unit for unit in pool.units}
    selected, seen, excluded = {}, set(), []
    for category, evidence in sets.items():
        if evidence is None:
            excluded.append({"category": category, "reason": "no_matching_source_intervention"})
            continue
        ids = tuple(sorted(unit.unit_id for unit in evidence.selected_units))
        for unit in evidence.selected_units:
            if unit.unit_id not in by_id or by_id[unit.unit_id] != unit:
                raise ValueError(f"candidate source differs from actual pool: {unit.unit_id}")
            if not set(unit.dependencies) <= set(ids):
                raise ValueError(f"candidate has an unresolved source dependency: {unit.unit_id}")
        if ids in seen:
            excluded.append({"category": category, "reason": "duplicate_evidence_set"})
            continue
        seen.add(ids)
        selected[category] = evidence
        if len(selected) == maximum:
            break
    return selected, {"task_id": state.task_id, "distinct_candidates": len(selected),
        "minimum": minimum, "maximum": maximum, "ready": len(selected) >= minimum,
        "exclusions": excluded, "candidates": [{"category": category,
            "unit_ids": [unit.unit_id for unit in evidence.selected_units],
            "source_keys": to_jsonable(tuple(block.key for block in evidence.source_blocks)),
            "evidence_sha256": digest(to_jsonable(evidence))}
            for category, evidence in selected.items()]}


def load_candidate_family(saved, row, state, budget, cohort_path):
    """Replay the exact saved real-source family across preparation and collection."""
    from controller.context_schema import EvidencePool, SupportScores
    expected = {"actual_history_sha256": history_digest(row["actual_history"]),
        "cohort_manifest_sha256": file_digest(cohort_path), "task_state_sha256": digest(to_jsonable(state)),
        "budget": to_jsonable(budget)}
    for key, value in expected.items():
        if saved[key] != value:
            raise ValueError(f"saved candidate family differs from {key}: {row['task_id']}")
    path = Path(saved["candidate_pool_path"])
    if file_digest(path) != saved["candidate_pool_sha256"]:
        raise ValueError(f"saved real candidate pool changed: {row['task_id']}")
    payload = json.loads(path.read_text())
    pool, scores = decode(EvidencePool, payload["pool"]), decode(SupportScores, payload["support_scores"])
    sets = {category: decode(EvidenceSet, evidence) for category, evidence in saved["evidence_sets"].items()}
    return pool, scores, sets, path

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
        **{key: row[key] for key in ("conversation_id", "turn", "snapshot_id") if key in row},
        "cohort_split": settings.get("cohort_split"),
        "response": {"attempt_id": attempt.attempt_id},
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
    cohort = None
    if settings.get("cohort_manifest"):
        cohort_path = Path(settings["cohort_manifest"])
        cohort = json.loads(cohort_path.read_text())
        cohort_split = settings.get("cohort_split", "train")
        members = {item["task_id"]: item for item in cohort["members"] if item["split"] == cohort_split}
        if "eligible_task_ids" in settings:
            eligible = set(settings["eligible_task_ids"])
            if not eligible <= set(members):
                raise ValueError("eligible task labels include states outside the registered cohort split")
            members = {identity: member for identity, member in members.items() if identity in eligible}
        rows = [row for row in rows if row["task_id"] in members]
        for row in rows:
            member = members[row["task_id"]]
            if (row["conversation_id"] != member["conversation_id"] or
                    history_digest(row["actual_history"]) != member["actual_history_sha256"]):
                raise ValueError(f"candidate state differs from declared cohort: {row['task_id']}")
        if set(members) != {row["task_id"] for row in rows}:
            raise ValueError("candidate turns do not contain every declared cohort state")
    rows = rows[:limit or (len(rows) if cohort else settings.get("turns", training.get("preference_turns", 200)))]
    assert rows and all(row["split"] == "train" and "actual_history" in row for row in rows)
    if settings.get("state_db"):
        rows = bind_saved_turns(rows, settings["state_db"])
    own_controller = controller is None
    if own_controller:
        run = load_config(settings.get("run_config", config.get("generation", {}).get("run_config", "configs/qwen3_32b.yaml")))
        if settings.get("reader_base_url"):
            run = replace(run, engine=replace(run.engine, base_url=settings["reader_base_url"]))
        controller = create_controller(replace(run, state_dir=str(output_dir/"runtime"),
            planner=replace(run.planner, strategy="always_rebuild")))
    if settings.get("reader_checkpoint"):
        info = await controller.engine.info()
        if info.fingerprint.model_revision != settings["reader_checkpoint"]:
            raise ValueError("active reader differs from the declared candidate collection checkpoint")
        binding = {"reader_checkpoint": settings["reader_checkpoint"],
            "reader_profile_id": settings["reader_profile_id"],
            "cohort_manifest_sha256": file_digest(settings["cohort_manifest"]) if cohort else None,
            "reader_data_only": settings.get("reader_data_only", False)}
        binding_path = output_dir/"reader_collection_binding.json"
        if binding_path.exists() and json.loads(binding_path.read_text()) != binding:
            raise ValueError("existing candidate collection belongs to a different reader or cohort")
        binding_path.write_text(json.dumps(binding, indent=2)+"\n")
    encoder = None
    if directory is None:
        directory, encoder = build_directory(config, controller.template.tokenizer)
    asset = settings.get("support_asset", config.get("support_model", {}).get("asset", "models/study-support"))
    profile = SupportProfile(**{key: value for key, value in config.get("support_model", {}).items()
                                if key in SupportProfile.__dataclass_fields__})
    if settings.get("support_mode", "trained") == "teacher":
        from adapters.teacher_support import TeacherSupportBackend
        backend = TeacherSupportBackend(settings["support_teacher_url"], controller.template,
            record_path=output_dir/"support_teacher_calls.jsonl")
        model = SupportModel(backend, history_resolver=lambda ref: "", profile=profile)
        asset = "teacher:"+settings["support_teacher_url"]
    else:
        model = SupportModel.from_pretrained(asset, history_resolver=lambda ref: "",
            device=settings.get("device", config.get("support_model", {}).get("device", "cuda:2")), profile=profile)
    budget = ContextBudget(window_tokens=settings.get("window_tokens", 32768),
        body_tokens=settings.get("body_budget", 4096), history_tokens=settings.get("history_tokens", 8192),
        output_tokens=settings.get("output_budget", 1024),
        read_input_tokens=settings.get("read_input_tokens", 12288),
        read_output_tokens=settings.get("read_output_tokens", 2048),
        max_steps=settings.get("max_steps", 64), max_total_tokens=settings.get("max_total_tokens", 4194304))
    rewards_path = output_dir/"rewards.jsonl"
    existing = [json.loads(line) for line in rewards_path.read_text().splitlines() if line] if rewards_path.exists() else []
    actual_turns_path = output_dir/"turns.jsonl"
    actual_turns = [json.loads(line) for line in actual_turns_path.read_text().splitlines() if line] if actual_turns_path.exists() else []
    if settings.get("reader_data_only", False):
        actual_turns_path.touch(exist_ok=True)
    completed = {(row["task_id"], row["category"]) for row in existing}
    if settings.get("reader_data_only", False):
        completed.update((row["task_id"], row["category"]) for row in actual_turns)
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
    family_path = Path(settings.get("candidate_families_path", output_dir/"candidate_families.jsonl"))
    saved_families = ({item["task_id"]: item for item in (
        json.loads(line) for line in family_path.read_text().splitlines() if line)}
        if cohort is not None and family_path.exists() else {})
    candidate_families = {identity: family for identity, family in saved_families.items()
                          if identity in {row["task_id"] for row in rows}}
    started = time.perf_counter()
    async with httpx.AsyncClient(base_url=settings.get("judge_base_url", config.get("generation", {}).get("teacher_url", "http://127.0.0.1:30001")), timeout=180) as judge:
        for task_index, row in enumerate(rows):
            expected_categories = tuple(saved_families[row["task_id"]]["evidence_sets"]) if row["task_id"] in saved_families else categories
            if all((row["task_id"], category) in completed for category in expected_categories):
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
            reused_family = cohort is not None and row["task_id"] in saved_families
            if reused_family:
                pool, scores, sets, pool_path = load_candidate_family(saved_families[row["task_id"]], row,
                    state, budget, settings["cohort_manifest"])
            else:
                pool = directory.retrieve_pool(state, memory)
                current_set = decode(EvidenceSet, row["current_evidence_set"]) if row.get("current_evidence_set") else None
                if cohort is not None and current_set is None and "current_source_blocks" in row:
                    current_set = current_evidence_from_blocks(row, controller.template.tokenizer)
                if cohort is not None and current_set is not None:
                    pool = replace(pool, units=tuple({unit.unit_id: unit for unit in (*pool.units, *current_set.selected_units)}.values()),
                        source_blocks=tuple({block.key: block for block in (*pool.source_blocks, *current_set.source_blocks)}.values()))
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
                sets = candidate_sets(state, pool, scores, model, controller.template.tokenizer, budget,
                    current_set=current_set if cohort is not None else None,
                    expand_interventions=cohort is not None)
            active_categories = categories
            if cohort is not None:
                sets, family = real_candidate_family(state, pool, sets,
                    minimum=cohort["minimum_candidates_per_state"], maximum=cohort["maximum_candidates_per_state"])
                active_categories = tuple(sets)
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
            if not reused_family:
                pools_dir = output_dir/"pools"
                pools_dir.mkdir(exist_ok=True)
                pool_path = pools_dir/(f"pool-{digest(row['task_id'])[:24]}.json" if cohort else f"pool-{task_index:05d}.json")
                pool_path.write_text(json.dumps({"task_id": row["task_id"],
                    "pool": to_jsonable(pool), "support_scores": to_jsonable(scores)}, ensure_ascii=False)+"\n")
            if cohort is not None:
                family.update(task_id=row["task_id"], task_state_id=state.task_id,
                    candidate_pool_path=str(pool_path.resolve()),
                    candidate_pool_sha256=file_digest(pool_path),
                    actual_history_sha256=history_digest(row["actual_history"]),
                    cohort_manifest_sha256=file_digest(settings["cohort_manifest"]),
                    task_state_sha256=digest(to_jsonable(state)), budget=to_jsonable(budget),
                    evidence_sets={category: to_jsonable(evidence) for category, evidence in sets.items()})
                if not reused_family:
                    append_record(output_dir/"candidate_families.jsonl", family)
                candidate_families[row["task_id"]] = family
                if settings.get("candidate_preparation_only", False) or not family["ready"]:
                    model.release_task(state.task_id)
                    continue
            history_id = json.dumps(row["actual_history"], ensure_ascii=False, sort_keys=True)
            for category in active_categories:
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
                        separate_needs=settings.get("separate_needs", False),
                        reading_mode=settings.get("reading_mode", "typed_reduce"),
                        numeric_identity=settings.get("numeric_identity", "names"),
                        output_tokens_per_fact=settings.get("output_tokens_per_fact", 64),
                        answer_system_prompt=settings.get("answer_system_prompt"))
                previous = controller.store.find_request(session_id, "answer")
                if previous is not None and previous.stage == "FAILED":
                    failure = failed_read_candidate(controller, row, category, settings, output_dir)
                    append_record(failures_path, failure)
                    failures.append(failure)
                    completed.add((row["task_id"], category))
                    await controller.close_session(session_id)
                    continue
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
                actual_turn = {"task_id": row["task_id"], "conversation_id": row["conversation_id"],
                    "turn": row["turn"], "snapshot_id": row["snapshot_id"], "split": "train",
                    "cohort_split": settings.get("cohort_split", "train") if cohort else None,
                    "cohort_manifest_sha256": file_digest(settings["cohort_manifest"]) if cohort else None,
                    "actual_history": row["actual_history"], "question": row["question"],
                    "category": category, "response": to_jsonable(response), "answer": response.assistant_text,
                    "reader_checkpoint": settings.get("reader_checkpoint"),
                    "reader_profile_id": settings.get("reader_profile_id"),
                    "request_window": [generation_started, generation_ended],
                    "evidence_set": to_jsonable(selected), "candidate_pool_path": str(pool_path),
                    "reader_input_tokens": response.reader_input_tokens,
                    "reader_output_tokens": response.reader_output_tokens}
                if (row["task_id"], category) not in {(item["task_id"], item["category"]) for item in actual_turns}:
                    append_record(actual_turns_path, actual_turn)
                    actual_turns.append(actual_turn)
                if settings.get("reader_data_only", False):
                    completed.add((row["task_id"], category))
                    await controller.close_session(session_id)
                    continue
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
                result = {"task_id": row["task_id"], "split": settings.get("cohort_split", "train") if cohort else "train", "history_id": history_id,
                    "conversation_id": row["conversation_id"], "turn": row["turn"], "snapshot_id": row["snapshot_id"],
                    "cohort_split": settings.get("cohort_split", "train") if cohort else None,
                    "cohort_manifest_sha256": file_digest(settings["cohort_manifest"]) if cohort else None,
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
    if settings.get("reader_data_only", False):
        pair_manifest = {"pairs": 0, "reason": "reader_training_precedes_policy_reward_recollection"}
    elif settings.get("candidate_preparation_only", False):
        pair_manifest = {"pairs": 0, "reason": "candidate_preparation_only"}
    elif not collected:
        pair_manifest = {"pairs": 0, "reason": "no_actual_reader_answers"}
    elif cohort is not None and settings.get("cohort_split", "train") == "dev":
        pair_manifest = {"pairs": 0, "reason": "heldout_dev_rewards_do_not_enter_policy_training"}
    else:
        pair_manifest = build_preference_pairs(rewards_path, preferences_path, weights=weights)
    manifest = {"cohort_manifest": settings.get("cohort_manifest"),
        "cohort_split": settings.get("cohort_split") if cohort else None,
        "candidate_preparation_only": settings.get("candidate_preparation_only", False),
        "candidate_family_states": len(candidate_families),
        "candidate_family_ready_states": sum(row["ready"] for row in candidate_families.values()),
        "candidate_family_shortfall_states": sum(not row["ready"] for row in candidate_families.values()),
        "candidate_families_path": str((output_dir/"candidate_families.jsonl").resolve()) if cohort else None,
        "turns_path": str(labels_path), "planned_turns": len(rows), "answers": len(collected),
        "reader_data_only": settings.get("reader_data_only", False),
        "actual_candidate_answers": len(actual_turns),
        "actual_turns_path": str(actual_turns_path.resolve()),
        "tasks_with_answers": len({row["task_id"] for row in collected}),
        "categories": {name: sum(row["category"] == name for row in collected) for name in dict.fromkeys((*categories, *(row["category"] for row in collected)))},
        "skipped": skipped, "failed_candidates": len(failures), "failure_codes": {
            "READ_STEP_OUTPUT_TRUNCATED": len(failures)}, "legal_actions": ["REBUILD"],
        "elapsed_ms": (time.perf_counter()-started)*1000,
        "reader_input_tokens": sum(row["reader_input_tokens"] for row in actual_turns) if actual_turns else sum(row["reader_input_tokens"] for row in collected),
        "reader_output_tokens": sum(row["reader_output_tokens"] for row in actual_turns) if actual_turns else sum(row["reader_output_tokens"] for row in collected),
        "preparation_input_tokens": _sum_known([item["input_tokens"] for item in preparations]),
        "preparation_output_tokens": _sum_known([item["output_tokens"] for item in preparations]),
        "preparations_path": str(preparations_path),
        "judge_input_tokens": sum(row["judge_record"]["input_tokens"] for row in collected),
        "judge_output_tokens": _sum_known([row["judge_record"].get("output_tokens") for row in collected]),
        "support_usage": model.backend.usage, "pairs": pair_manifest,
        "outputs": {"turns": str(actual_turns_path), "rewards": str(rewards_path), "preferences": str(preferences_path),
                    "failed_cases": str(failures_path), "skipped_cases": str(skipped_path)}}
    (output_dir/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    model.close()
    return manifest



async def measured_preferences(config, *, limit=None):
    """Keep the collector's full process/GPU window, including partial failures."""
    settings = config.get("preferences", {})
    if "gpu_devices" not in settings:
        return await collect_preferences(config, limit=limit)
    import os
    from urllib.parse import urlparse
    from bench.study_resources import StudyResourceSampler, integrate_physical, listening_processes
    from controller.config import load_config
    run = load_config(settings["run_config"])
    urls = [settings.get("reader_base_url", run.engine.base_url)]
    if not settings.get("reader_data_only", False):
        urls.append(settings["judge_base_url"])
    if settings.get("support_mode", "trained") == "teacher":
        urls.append(settings["support_teacher_url"])
    processes = [os.getpid(), *listening_processes([urlparse(url).port for url in urls])]
    sampler = StudyResourceSampler(settings["gpu_devices"], processes,
        interval=settings.get("resource_sample_seconds", .5))
    await sampler.start()
    started = time.monotonic()
    completed = False
    try:
        result = await collect_preferences(config, limit=limit)
        completed = True
        return result
    finally:
        ended = time.monotonic()
        await sampler.stop()
        output = Path(settings["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        invocation = str(time.time_ns())
        (output/f"collection_physical.{invocation}.json").write_text(json.dumps(sampler.observation(), indent=2)+"\n")
        cost = {"status": "completed" if completed else "incomplete", "start": started, "end": ended,
            "complete_ms": (ended-started)*1000, "physical": integrate_physical(sampler.samples, started, ended),
            "reader_checkpoint": settings.get("reader_checkpoint"), "reader_profile_id": settings.get("reader_profile_id"),
            "cohort_manifest_sha256": file_digest(settings["cohort_manifest"]) if settings.get("cohort_manifest") else None,
            "reader_data_only": settings.get("reader_data_only", False),
            "gpu_devices": settings["gpu_devices"], "process_roots": processes}
        (output/f"collection_cost.{invocation}.json").write_text(json.dumps(cost, indent=2)+"\n")

def main(argv=None):
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--turns", type=int)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    print(json.dumps(asyncio.run(measured_preferences(config, limit=args.turns)), ensure_ascii=False))


if __name__ == "__main__":
    main()
