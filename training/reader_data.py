"""Reader SFT data from actual training-partition controller attempts."""
from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from controller.serialization import to_jsonable
from controller.session_store import Attempt, load_record
from controller.templates import token_hash


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1048576), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, ensure_ascii=False)+"\n" for value in values))


def history_digest(history):
    return digest([{name: turn[name] for name in ("turn_id", "user_text", "assistant_text")}
                   for turn in history])


def export_reader_traces(config):
    settings = config["reader_training"]
    output = Path(settings["data"]["output_dir"])
    traces, excluded, sources = [], [], []
    for spec in settings["data"]["rollouts"]:
        if spec["split"] != "train":
            raise ValueError("reader training traces must retain the native training partition")
        database = Path(spec["state_db"]).resolve()
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        turns = rows(spec["turns_path"])
        if settings["data"].get("export_turns") is not None:
            turns = turns[:settings["data"]["export_turns"]]
        sources.append({**spec, "turns_sha256": file_digest(spec["turns_path"]),
                        "exported_turns": len(turns), "has_execution_steps": "execution_steps" in tables})
        for turn in turns:
            if turn.get("split", spec["split"]) != "train":
                raise ValueError(f"non-training actual turn: {turn['task_id']}")
            attempt_id = turn["response"]["attempt_id"]
            payload = connection.execute("SELECT payload FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()[0]
            attempt = load_record(connection, Attempt, payload)
            history = turn["actual_history"]
            plan = attempt.reading_outcome.plan if attempt.reading_outcome is not None else attempt.context_plan
            blocks = {block.key: block for block in attempt.evidence}
            if plan is not None:
                blocks.update((block.key, block) for block in plan.selected_set.source_blocks)
            coordinates = [{"source_key": to_jsonable(block.key), "start": block.span[0] if block.span else 0,
                "end": block.span[1] if block.span else len(block.text), "text": block.text,
                "content_sha256": block.key.content_sha256, "source_uri": block.source_uri}
                for block in blocks.values()]
            common = {"task_id": turn["task_id"], "conversation_id": turn["conversation_id"],
                "turn": turn["turn"], "attempt_id": attempt_id, "split": "train",
                "source_partition": spec["source_partition"], "actual_history": history,
                "actual_history_sha256": history_digest(history), "question": turn["question"],
                "reader_identity": spec["reader_identity"], "snapshot_id": turn["snapshot_id"],
                "task_state": to_jsonable(attempt.query_context.task_state) if attempt.query_context is not None else None,
                "context_plan": to_jsonable(plan), "original_sources": coordinates,
                "trace_origin": {"turns_path": spec["turns_path"], "state_db": str(database)}}
            step_records = (connection.execute("SELECT payload FROM execution_steps WHERE attempt_id=? ORDER BY rowid",
                                                (attempt_id,)).fetchall() if "execution_steps" in tables else ())
            from controller.context_schema import StepRecord
            for record_payload, in step_records:
                record = load_record(connection, StepRecord, record_payload)
                if not record.input_messages:
                    excluded.append({"task_id": turn["task_id"], "step_id": record.step_id, "reason": "cpu_step_has_no_reader_call"})
                    continue
                if record.result is None:
                    excluded.append({"task_id": turn["task_id"], "step_id": record.step_id, "reason": "reader_call_has_no_observed_output"})
                    continue
                traces.append({**common, "example_id": f"{attempt_id}:{record.step_id}", "step_id": record.step_id,
                    "phase": record.task.kind, "messages": to_jsonable(record.input_messages),
                    "prompt_token_hash": token_hash(record.input_ids), "prompt_tokens": len(record.input_ids),
                    "actual_output": record.result.raw_output, "actual_output_ids": list(record.output_ids),
                    "read_task": to_jsonable(record.task), "read_result": to_jsonable(record.result),
                    "output_schema": record.task.output_schema,
                    "read_protocol_version": record.metrics.get("read_protocol_version", spec.get("read_protocol_version", 1)),
                    "generation_metrics": record.metrics})
            if attempt.stage == "COMMITTED":
                candidate = attempt.plan.candidate
                traces.append({**common, "example_id": f"{attempt_id}:ANSWER", "step_id": "ANSWER", "phase": "ANSWER",
                    "messages": to_jsonable(candidate.messages), "prompt_token_hash": candidate.input_hash,
                    "prompt_tokens": len(candidate.input_ids), "actual_output": attempt.result.text,
                    "actual_output_ids": list(attempt.result.output_ids), "read_protocol_version": None,
                    "reference_map": to_jsonable(candidate.reference_map),
                    "generation_metrics": {"finish_reason": attempt.result.finish_reason,
                        "input_tokens": len(candidate.input_ids), "output_tokens": len(attempt.result.output_ids),
                        "elapsed_ms": attempt.result.elapsed_ms}})
        connection.close()
    write_rows(output/"traces.jsonl", traces)
    write_rows(output/"export_exclusions.jsonl", excluded)
    manifest = {"schema_version": 1, "evidence_kind": "actual_training_partition_runtime",
        "turns": len({row["task_id"] for row in traces}), "examples": len(traces),
        "phases": dict(Counter(row["phase"] for row in traces)), "sources": sources,
        "traces_path": str((output/"traces.jsonl").resolve()), "excluded": len(excluded),
        "actual_history": "committed generated QA; never replaced by authored reference history",
        "target_status": "pending_independent_actual_history_and_source_support_review"}
    (output/"export_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest


def augment_retrieved_reads(config, *, template=None):
    """Compile protocol-2 inputs from actually retrieved training originals.

    These are new teacher-generation inputs, not fabricated historical READs.
    Explicit source edits always require a fresh prompt-bound teacher target.
    """
    from dataclasses import replace
    from adapters.corpus import CorpusIndex
    from adapters.evidence_pool import SourceDirectory
    from controller.context_schema import ContextBudget, EvidenceSet, Need, TaskState, TextRef
    from controller.reading import build_read_plan, step_messages
    from controller.schema import EvidenceBlock, EvidenceKey, SessionSnapshot, TurnRecord
    from controller.serialization import decode
    from controller.templates import HuggingFaceTemplate
    settings = config["reader_training"]
    data = settings["data"]
    root = Path(data["output_dir"])
    if template is None:
        from engine_patches.contextkv_model_scope import load_model_scope
        scope = load_model_scope(settings["base_scope"])
        template = HuggingFaceTemplate(scope.model_path, None, local_files_only=True,
                                      model_identity=scope.identity)
    original = rows(root/"traces.jsonl")
    answered = [row for row in original if row["phase"] == "ANSWER"][:data.get("retrieved_read_turns", 256)]
    edits = rows(data["paired_mutations_path"]) if data.get("paired_mutations_path") else []
    generated = []
    for trace in answered:
        variants = [("original", None), *((edit["mutation_id"], edit) for edit in edits if edit["task_id"] == trace["task_id"])]
        for variant, edit in variants:
            documents = []
            changes = []
            for source in trace["original_sources"]:
                key = decode(EvidenceKey, source["source_key"])
                body = source["text"]
                if edit is not None and source["source_key"] == edit["source_key"]:
                    start, end = edit["start"]-source["start"], edit["end"]-source["start"]
                    if body[start:end] != edit["original_text"]:
                        raise ValueError(f"paired edit differs from training original: {edit['mutation_id']}")
                    body = body[:start]+edit["replacement"]+body[end:]
                    key = replace(key, content_sha256=hashlib.sha256(body.encode()).hexdigest(),
                                  version=edit.get("new_version", key.version))
                    changes.append({**edit, "changed_source_key": to_jsonable(key),
                        "changed_end": edit["start"]+len(edit["replacement"])})
                documents.append(EvidenceBlock(key, source["source_uri"], "", body,
                    span=(source["start"], source["start"]+len(body)), provenance=(source["source_uri"],)))
            if edit is not None and not changes:
                raise ValueError(f"paired edit has no matching actually retrieved training source: {edit['mutation_id']}")
            history = tuple(decode(TurnRecord, turn) for turn in trace["actual_history"])
            current = trace["turn"]
            if trace["task_state"] is None:
                refs = tuple(TextRef(turn.turn_id, role, 0, len(text)) for turn in history
                    for role, text in (("user", turn.user_text), ("assistant", turn.assistant_text)))
                state = TaskState(trace["task_id"], trace["question"], working_history=refs,
                    needs=(Need("question", anchor_spans=(TextRef(current, "user", 0, len(trace["question"])),)),),
                    current_turn_id=current)
            else:
                state = decode(TaskState, trace["task_state"])
            corpus = CorpusIndex(documents, corpus_id=documents[0].key.corpus_id, snapshot_id=trace["snapshot_id"])
            directory = SourceDirectory(corpus, None, template.tokenizer)
            pool = directory.full_scope(tuple(block.key for block in documents))
            evidence = EvidenceSet(f"actual-retrieval:{variant}", pool.units, source_blocks=pool.source_blocks)
            budget = ContextBudget(window_tokens=32768, body_tokens=8192, read_input_tokens=12288,
                                   read_output_tokens=2048, max_steps=512)
            snapshot = SessionSnapshot(trace["conversation_id"], len(history), history, archive=tuple(documents))
            plan = build_read_plan(state, evidence, "all_supplied_documents", budget,
                template=template, snapshot=snapshot, read_layout="source_first", reading_mode="typed_reduce")
            by_id = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
            for step in plan.steps:
                if step.kind != "READ":
                    continue
                messages = step_messages(plan, step, tuple(by_id[identity] for identity in step.input_unit_ids), snapshot=snapshot)
                prompt = template.render(messages)
                generated.append({**trace, "example_id": f"{trace['attempt_id']}:replanned:{variant}:{step.step_id}",
                    "step_id": f"replanned:{variant}:{step.step_id}", "phase": "READ", "messages": to_jsonable(messages),
                    "prompt_token_hash": prompt.input_hash, "prompt_tokens": len(prompt.input_ids),
                    "actual_output": "", "actual_output_ids": [], "output_schema": step.output_schema,
                    "read_task": to_jsonable(step), "read_result": None, "context_plan": to_jsonable(plan),
                    "read_protocol_version": 2, "read_layout": "source_first", "pair_id": trace["example_id"],
                    "source_mutations": changes,
                    "original_sources": [{"source_key": to_jsonable(block.key), "start": block.span[0],
                        "end": block.span[1], "text": block.text, "content_sha256": block.key.content_sha256,
                        "source_uri": block.source_uri} for block in documents],
                    "trace_origin": {**trace["trace_origin"], "input_origin": "actual_retrieval_replanned_protocol_2",
                        "target_origin": "pending_real_teacher_generation"},
                    "generation_metrics": {"status": "not_executed_by_original_R0", "input_tokens": 0, "output_tokens": 0}})
    prior = {row["example_id"]: row for row in original}
    prior.update((row["example_id"], row) for row in generated)
    write_rows(root/"traces.jsonl", prior.values())
    manifest = {"turns": len(answered), "read_examples": len(generated), "read_protocol_version": 2,
        "read_layout": "source_first", "source_origin": "actual native-training retrieval",
        "paired_source_edits": len(edits), "targets": "require fresh actual-history/source/prompt-bound teacher review",
        "public_dev_test_sources_read": []}
    (root/"retrieved_read_inputs_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def reader_review_body(trace, *, answer_output_tokens=1024):
    """Deduplicate literal views while preserving the complete online messages."""
    messages = trace["messages"]
    contents = [message["content"] for message in messages]
    structured = trace["phase"] != "ANSWER"
    extra_history = []
    for turn in trace["actual_history"]:
        user_present = any(message["role"] == "user" and turn["user_text"] in message["content"] for message in messages)
        assistant_present = any(message["role"] == "assistant" and turn["assistant_text"] == message["content"] for message in messages)
        if not (user_present and assistant_present):
            extra_history.append(turn)
    sources = []
    for source in trace["original_sources"]:
        metadata = {name: value for name, value in source.items() if name != "text"}
        if structured:
            metadata["text_view"] = "original window units and claims are preserved in exact messages"
        elif any(source["text"] in content for content in contents):
            metadata["text_view"] = "complete original already present in exact messages"
        else:
            metadata["text"] = source["text"]
        sources.append(metadata)
    return {"phase": trace["phase"], "question": trace["question"], "messages": messages,
        "task_state": trace["task_state"], "additional_actual_history": extra_history,
        "full_actual_history_sha256": trace["actual_history_sha256"],
        "history_view": "actual QA already in exact messages plus unchanged additional_actual_history",
        "original_sources": sources, "actual_output": trace["actual_output"],
        "target_output_budget_tokens": trace["read_task"]["output_budget"] if structured else answer_output_tokens}


async def review_reader_traces(config, *, limit_turns=None, concurrency=8):
    from adapters.query_selector import OpenAICompletionClient, SelectorConfig
    from engine_patches.contextkv_model_scope import load_model_scope
    settings = config["reader_training"]
    data, review = settings["data"], settings["review"]
    root = Path(data["output_dir"])
    trace_rows = rows(root/"traces.jsonl")
    allowed_protocols = data.get("read_protocol_versions", [2])
    trace_rows = [row for row in trace_rows if row["phase"] == "ANSWER" or row["read_protocol_version"] in allowed_protocols]
    if limit_turns is not None:
        selected = set(dict.fromkeys(row["task_id"] for row in trace_rows))
        selected = set(sorted(selected)[:limit_turns])
        trace_rows = [row for row in trace_rows if row["task_id"] in selected]
    scope = load_model_scope(settings["base_scope"])
    clients = [OpenAICompletionClient(SelectorConfig(url, scope.model_path, scope.identity,
        tokenizer_path=scope.model_path, max_input_tokens=review["max_input_tokens"],
        max_new_tokens=review["max_output_tokens"], context_window=review["context_window"],
        timeout_seconds=review["timeout_seconds"], temperature=0)) for url in review["base_urls"]]
    path = root/"reviews.jsonl"
    done = {row["example_id"] for row in rows(path)} if path.exists() else set()
    sink = path.open("a", encoding="utf-8")
    ledger = (root/"review_calls.jsonl").open("a", encoding="utf-8")
    semaphore = asyncio.Semaphore(concurrency)
    started = time.monotonic()

    async def review_one(index, trace):
        if trace["example_id"] in done:
            return
        async with semaphore:
            structured = trace["phase"] != "ANSWER"
            target = trace["output_schema"] if structured else {"type": "string"}
            schema = {"type": "object", "properties": {"target": target,
                "quality": {"type": "number", "minimum": 0, "maximum": 1},
                "source_support_verified": {"type": "boolean"}, "correct_task_binding": {"type": "boolean"}},
                "required": ["target", "quality", "source_support_verified", "correct_task_binding"], "additionalProperties": False}
            system = ("Produce a high-quality supervised target for the exact recorded online prompt, using its ACTUAL "
                "conversation history and supplied original sources only. Correct an erroneous old output. "
                "Resolve current object, active conditions and versions; never inherit an unsupported earlier answer as fact. "
                "Preserve the original READ JSON contract or ANSWER citation rules. For READ, a window with no new grounded "
                "fact has claims=[]; background notes and statements saying there are no additional records are not records. "
                "Each fact belongs to its exact need_id. Use followups=[] unless the current need truly requires a missing "
                "source relation; do not repeatedly request already supplied quantities or invent a gap for irrelevant notes. "
                "For ANSWER give the supported answer completely and concisely with only allowed citations. "
                "Return target and assess its correctness and source support against the actual task. "
                "The question's authored/reference history is not provided and must not be assumed.")
            body = reader_review_body(trace, answer_output_tokens=data.get("answer_output_tokens", 1024))
            output_budget = body["target_output_budget_tokens"]
            result = await asyncio.to_thread(clients[index % len(clients)].complete,
                [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(body, ensure_ascii=False)}],
                response_format={"type": "json_schema", "json_schema": {"name": "reader_target_review", "strict": True, "schema": schema}})
            budget_excluded = result.status == "input_over_budget"
            ledger.write(json.dumps({"example_id": trace["example_id"], **asdict(result),
                "model_request_executed": not budget_excluded,
                "model_input_tokens": 0 if budget_excluded else result.input_tokens,
                "model_output_tokens": 0 if budget_excluded else
                    result.usage.get("completion_tokens") if result.usage is not None else None}, ensure_ascii=False)+"\n")
            ledger.flush()
            if result.status == "input_over_budget":
                sink.write(json.dumps({"example_id": trace["example_id"], "task_id": trace["task_id"],
                    "step_id": trace["step_id"], "status": "input_over_budget", "exclusion_reason": "review_input_over_budget",
                    "actual_history_sha256": trace["actual_history_sha256"], "prompt_token_hash": trace["prompt_token_hash"],
                    "reader_identity": trace["reader_identity"], "review_input_tokens": result.input_tokens,
                    "review_input_budget": review["max_input_tokens"], "model_request_executed": False,
                    "target_origin": "not_generated", "deduplication": "complete exact messages retained; no history truncation"},
                    ensure_ascii=False)+"\n")
                sink.flush()
                return
            if result.status == "completed" and result.finish_reason == "length":
                sink.write(json.dumps({"example_id": trace["example_id"], "task_id": trace["task_id"],
                    "step_id": trace["step_id"], "status": "teacher_target_truncated",
                    "exclusion_reason": "teacher_target_truncated", "target_origin": "truncated_output_not_a_training_target",
                    "actual_history_sha256": trace["actual_history_sha256"], "prompt_token_hash": trace["prompt_token_hash"],
                    "reader_identity": trace["reader_identity"], "raw_output": result.text,
                    "finish_reason": result.finish_reason, "input_tokens": result.input_tokens,
                    "usage": result.usage, "elapsed_ms": result.elapsed_ms,
                    "model_request_executed": True}, ensure_ascii=False)+"\n")
                sink.flush()
                return
            if result.status != "completed" or result.finish_reason != "stop":
                raise RuntimeError(f"reader review unfinished for {trace['example_id']}: {result.status}/{result.finish_reason}")
            value = json.loads(result.text)
            target_text = json.dumps(value["target"], ensure_ascii=False, separators=(",", ":")) if structured else value["target"]
            target_tokens = len(clients[index % len(clients)].tokenizer.encode(target_text, add_special_tokens=False))+1
            accepted = (value["quality"] >= data["minimum_quality"] and value["source_support_verified"]
                        and value["correct_task_binding"] and target_tokens <= output_budget)
            record = {"example_id": trace["example_id"], "task_id": trace["task_id"], "step_id": trace["step_id"],
                "status": "accepted" if accepted else "rejected", "target_origin": "teacher_corrected_actual_runtime",
                "target_text": target_text, "target_tokens_including_eos": target_tokens, "target_output_budget": output_budget,
                "quality": value["quality"], "source_support_verified": value["source_support_verified"],
                "correct_task_binding": value["correct_task_binding"], "actual_history_sha256": trace["actual_history_sha256"],
                "prompt_token_hash": trace["prompt_token_hash"], "reader_identity": trace["reader_identity"],
                "reviewer_identity": scope.manifest(), "read_protocol_version": trace["read_protocol_version"]}
            record["quality_measurement"] = "same-call teacher target assessment; separate from benchmark evaluation"
            sink.write(json.dumps(record, ensure_ascii=False)+"\n")
            sink.flush()
    try:
        async with asyncio.TaskGroup() as tasks:
            for index, trace in enumerate(trace_rows):
                tasks.create_task(review_one(index, trace))
    finally:
        sink.close()
        ledger.close()
        for client in clients:
            client.close()
    saved = rows(path)
    manifest = {"processed_examples": len(saved),
        "reviewed_examples": sum(row["status"] in {"accepted", "rejected"} for row in saved),
        "requested_examples": len(trace_rows),
        "status_counts": dict(Counter(row["status"] for row in saved)),
        "elapsed_ms": (time.monotonic()-started)*1000, "reviews_path": str(path.resolve()),
        "cost_ledger": str((root/"review_calls.jsonl").resolve()), "reviewer": scope.manifest()}
    (root/"review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest


def prepare_reader_data(config):
    settings = config["reader_training"]["data"]
    root = Path(settings["output_dir"])
    reviews = {row["example_id"]: row for row in rows(settings.get("reviews_path", root/"reviews.jsonl"))}
    selected, excluded = [], []
    for trace in rows(root/"traces.jsonl"):
        review = reviews.get(trace["example_id"])
        reason = ("not_native_train" if trace["split"] != "train" else
            "old_read_protocol" if trace["phase"] != "ANSWER" and trace["read_protocol_version"] not in settings["read_protocol_versions"] else
            "missing_review" if review is None else
            "review_input_over_budget" if review["status"] == "input_over_budget" else
            "teacher_target_truncated" if review["status"] == "teacher_target_truncated" else
            "review_rejected" if review["status"] != "accepted" else
            "insufficient_quality" if review["quality"] < settings["minimum_quality"] else
            "unsupported_target" if not review["source_support_verified"] or not review["correct_task_binding"] else
            "actual_history_binding_mismatch" if review["actual_history_sha256"] != trace["actual_history_sha256"] else
            "prompt_binding_mismatch" if review["prompt_token_hash"] != trace["prompt_token_hash"] else
            "reader_binding_mismatch" if review["reader_identity"] != trace["reader_identity"] else None)
        if reason is not None:
            excluded.append({"example_id": trace["example_id"], "task_id": trace["task_id"], "reason": reason})
            continue
        selected.append({**trace, "target_text": review["target_text"], "target_origin": review["target_origin"],
            "quality": review["quality"], "supervision": review,
            "data_split": "validation" if int(digest([settings["split_seed"], trace["conversation_id"]])[:8], 16)
                /16**8 < settings["validation_fraction"] else "train"})
    for split in ("train", "validation"):
        write_rows(root/f"{split}.jsonl", [row for row in selected if row["data_split"] == split])
    write_rows(root/"selection_exclusions.jsonl", excluded)
    manifest = {"schema_version": 1, "evidence_kind": "reviewed_actual_training_partition_reader_data",
        "turns": len({row["task_id"] for row in selected}), "examples": len(selected),
        "training_turns": len({row["task_id"] for row in selected if row["data_split"] == "train"}),
        "data_splits": dict(Counter(row["data_split"] for row in selected)),
        "phases": dict(Counter(row["phase"] for row in selected)), "target_origins": dict(Counter(row["target_origin"] for row in selected)),
        "read_protocol_versions": settings["read_protocol_versions"], "review_filter": settings,
        "source_traces_sha256": file_digest(root/"traces.jsonl"),
        "reviews_sha256": file_digest(settings.get("reviews_path", root/"reviews.jsonl")),
        "selection_exclusions": dict(Counter(row["reason"] for row in excluded)),
        "validation_origin": "conversation-disjoint heldout from training partition; no official dev/test",
        "evaluation_results": "separate artifacts; no answer quality improvement inferred from dataset preparation"}
    (root/"data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest
