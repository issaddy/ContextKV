"""Audit exact native reader traces and independently verify protocol outputs."""
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



REVIEW_PROTOCOL = "visible_target_independent_review_v6"

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
    settings = config["reader_audit"]
    output = Path(settings["data"]["output_dir"])
    traces, excluded, sources = [], [], []
    cohort_path = settings["data"].get("cohort_manifest")
    members = ({member["task_id"]: member for member in json.loads(Path(cohort_path).read_text())["members"]}
               if cohort_path else None)
    for spec in settings["data"]["rollouts"]:
        if spec["split"] != "train":
            raise ValueError("reader audit traces must retain the native training partition")
        database = Path(spec["state_db"]).resolve()
        connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        turns = rows(spec["turns_path"])
        if spec.get("failed_turns_path") and Path(spec["failed_turns_path"]).exists():
            turns.extend(rows(spec["failed_turns_path"]))
        if settings["data"].get("export_turns") is not None:
            turns = turns[:settings["data"]["export_turns"]]
        sources.append({**spec, "turns_sha256": file_digest(spec["turns_path"]),
                        "exported_turns": len(turns), "has_execution_steps": "execution_steps" in tables})
        for turn in turns:
            if turn.get("split", spec["split"]) != "train":
                raise ValueError(f"non-training actual turn: {turn['task_id']}")
            if members is not None:
                member = members[turn["task_id"]]
                if (turn["conversation_id"] != member["conversation_id"] or turn["turn"] != member["turn"] or
                        history_digest(turn["actual_history"]) != member["actual_history_sha256"] or
                        turn["cohort_split"] != member["split"]):
                    raise ValueError(f"reader trace differs from registered actual state: {turn['task_id']}")
            attempt_id = turn["response"]["attempt_id"]
            payload = connection.execute("SELECT payload FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()[0]
            attempt = load_record(connection, Attempt, payload)
            history = [{key: item[key] for key in ("turn_id", "user_text", "assistant_text")}
                       for item in turn["actual_history"]]
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
                "cohort_split": turn.get("cohort_split"), "candidate_category": turn.get("category"),
                "task_state": to_jsonable(attempt.query_context.task_state) if attempt.query_context is not None else None,
                "context_plan": to_jsonable(plan), "original_sources": coordinates,
                "trace_origin": {"turns_path": spec["turns_path"], "state_db": str(database)}}
            step_records = (connection.execute("SELECT payload FROM execution_steps WHERE attempt_id=? ORDER BY rowid",
                                                (attempt_id,)).fetchall() if "execution_steps" in tables else ())
            from controller.context_schema import StepRecord
            reading_steps = []
            for record_payload, in step_records:
                record = load_record(connection, StepRecord, record_payload)
                if record.result is not None:
                    reading_steps.append({"step_id": record.step_id, "phase": record.task.kind,
                        "operation": record.task.operation, "claims": to_jsonable(record.result.claims),
                        "covered_units": list(record.result.covered_units)})
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
                    "read_protocol_version": record.metrics["read_protocol_version"] if "read_protocol_version" in record.metrics
                        else record.task.output_schema["properties"]["format_version"]["const"],
                    "generation_metrics": record.metrics})
            if attempt.stage == "COMMITTED":
                candidate = attempt.plan.candidate
                traces.append({**common, "example_id": f"{attempt_id}:ANSWER", "step_id": "ANSWER", "phase": "ANSWER",
                    "messages": to_jsonable(candidate.messages), "prompt_token_hash": candidate.input_hash,
                    "prompt_tokens": len(candidate.input_ids), "actual_output": attempt.result.text,
                    "actual_output_ids": list(attempt.result.output_ids), "read_protocol_version": None,
                    "reference_map": to_jsonable(candidate.reference_map),
                    "reading_steps": reading_steps,
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


def reader_target_body(trace, *, answer_output_tokens=1024):
    """Give the target writer exactly the factual input available to the reader."""
    return {"phase": trace["phase"], "messages": trace["messages"],
        "target_output_budget_tokens": trace["read_task"]["output_budget"]
            if trace["phase"] != "ANSWER" else answer_output_tokens}


def reader_review_body(trace, *, answer_output_tokens=1024, target_text=None, reading_steps=()):
    """Put the fixed target first; every other field is verification evidence."""
    messages = trace["messages"]
    contents = [message["content"] for message in messages]
    structured = trace["phase"] != "ANSWER"
    extra_history = []
    for turn in trace["actual_history"]:
        user_present = any(message["role"] == "user" and turn["user_text"] in message["content"] for message in messages)
        assistant_present = any(message["role"] == "assistant" and turn["assistant_text"] == message["content"] for message in messages)
        if not (user_present and assistant_present):
            extra_history.append({key: turn[key] for key in ("turn_id", "user_text", "assistant_text")})
    sources = []
    for source in trace["original_sources"]:
        metadata = {name: value for name, value in source.items() if name != "text"}
        if any(source["text"] in content for content in contents):
            metadata["text_view"] = "complete original already present in exact messages"
        else:
            metadata["text"] = source["text"]
        sources.append(metadata)
    body = ({"target_output": json.loads(target_text) if structured else target_text}
            if target_text is not None else {})
    body.update(phase=trace["phase"], question=trace["question"], messages=messages)
    body.update({
        "task_state": trace["task_state"], "additional_actual_history": extra_history,
        "full_actual_history_sha256": trace["actual_history_sha256"],
        "history_view": "actual QA already in exact messages plus unchanged additional_actual_history",
        "original_sources": sources,
        "target_output_budget_tokens": trace["read_task"]["output_budget"] if structured else answer_output_tokens})
    if target_text is not None:
        body.update(reading_steps=trace.get("reading_steps", reading_steps),
                    source_role="messages, input claims and reading_steps are evidence/upstream attribution only; target_output is the only evaluated continuation")
    return body


async def review_reader_trace(trace, client, *, minimum_quality, reviewer_identity,
                              review_input_budget, record_call, answer_output_tokens=1024,
                              reading_steps=(), fixed_target_text=None):
    """Write from the visible prompt, then verify the fixed target independently."""
    structured = trace["phase"] != "ANSWER"
    common = {"example_id": trace["example_id"], "task_id": trace["task_id"],
        "phase": trace["phase"], "step_id": trace["step_id"] if structured else None,
        "actual_history_sha256": trace["actual_history_sha256"],
        "prompt_token_hash": trace["prompt_token_hash"], "reader_identity": trace["reader_identity"],
        "review_protocol": REVIEW_PROTOCOL}
    for key in ("original_example_id", "original_attempt_id", "source_asset_id", "source_traces", "source_reviews"):
        if key in trace:
            common[key] = trace[key]
    assessment = {"quality": {"type": "number", "minimum": 0, "maximum": 1},
        "source_support_verified": {"type": "boolean"}, "correct_task_binding": {"type": "boolean"}}
    async def complete(stage, messages, schema=None, *, output_budget=None):
        limit = client.config.max_new_tokens if output_budget is None else output_budget
        response_format = ({"type": "json_schema", "json_schema": {
            "name": "reader_"+stage, "strict": True, "schema": schema}} if schema is not None else None)
        result = await asyncio.to_thread(client.complete, messages,
            response_format=response_format, max_new_tokens=limit,
            chat_template_kwargs={"enable_thinking": stage == "review"})
        executed = result.status != "input_over_budget"
        record_call({"example_id": trace["example_id"], "stage": stage, "review_protocol": REVIEW_PROTOCOL,
            "teacher_messages_sha256": digest(messages), "max_new_tokens": limit,
            "enable_thinking": stage == "review", **asdict(result),
            "model_request_executed": executed, "model_input_tokens": result.input_tokens if executed else 0,
            "model_output_tokens": (result.usage.get("completion_tokens")
                if result.usage is not None else None) if executed else 0})
        return result

    def terminal_record(result, stage):
        if result.status == "input_over_budget":
            return {**common, "status": "input_over_budget", "exclusion_reason": "review_input_over_budget",
                "review_stage": stage, "review_input_tokens": result.input_tokens,
                "review_input_budget": review_input_budget, "model_request_executed": False,
                "target_origin": "not_generated" if stage == "target" else "visible_prompt_target_awaiting_review"}
        if result.status == "completed" and result.finish_reason == "length":
            return {**common, "status": "teacher_"+stage+"_truncated",
                "exclusion_reason": "teacher_"+stage+"_truncated", "review_stage": stage,
                "target_origin": "truncated_output_not_a_training_target" if stage == "target"
                    else "visible_prompt_target_awaiting_review",
                "raw_output": result.text, "finish_reason": result.finish_reason,
                "input_tokens": result.input_tokens, "usage": result.usage,
                "elapsed_ms": result.elapsed_ms, "model_request_executed": True}
        if result.status == "completed" and not result.text.strip():
            return {**common, "status": "teacher_"+stage+"_empty",
                "exclusion_reason": "teacher_returned_no_final_output", "review_stage": stage,
                "target_origin": "not_generated" if stage == "target" else "visible_prompt_target_awaiting_review",
                "input_tokens": result.input_tokens, "usage": result.usage,
                "elapsed_ms": result.elapsed_ms, "model_request_executed": True}
        if result.status != "completed" or result.finish_reason != "stop":
            raise RuntimeError(f"reader {stage} unfinished for {trace['example_id']}: {result.status}/{result.finish_reason}")
        return None

    body = reader_target_body(trace, answer_output_tokens=answer_output_tokens)
    if fixed_target_text is None:
        instruction = ("Complete the task in the preceding messages with the actual assistant continuation. "
            "Use only the facts, sources, Reading conclusions and conversation visible in those messages. "
            "Recheck extraction, reasoning, quantities, units, conditions and versions before responding. "
            "Perform the requested task fully; do not output a phase label, placeholder, schema description "
            "or an object wrapping the continuation in a target field. "
            f"Your entire continuation must fit {body['target_output_budget_tokens']} output tokens. ")
        schema = trace["output_schema"] if structured else None
        if structured:
            instruction += ("Return the complete structured step result as one JSON object matching the schema below, "
                "using the exact Need/source/parent IDs from the preceding messages. Empty claims/facts are valid "
                "only when this visible window contains no relevant fact for the active Needs. Output schema:\n"+
                json.dumps(schema, ensure_ascii=False))
        else:
            instruction += ("Return the supported final answer directly, including the required conclusions and "
                "citations under the original response rules. Be concise and omit unrequested examples.")
        messages = [*trace["messages"], {"role": "user", "content": instruction}]
        result = await complete("target", messages, schema, output_budget=body["target_output_budget_tokens"])
        terminal = terminal_record(result, "target")
        if terminal is not None:
            return terminal
        target_text = (json.dumps(json.loads(result.text), ensure_ascii=False, separators=(",", ":"))
                       if structured else result.text)
    else:
        target_text = fixed_target_text
    target_tokens = len(client.tokenizer.encode(target_text, add_special_tokens=False))+1
    review_properties = {**assessment, "visible_prompt_support": {"type": "boolean"},
        "reason": {"type": "string", "maxLength": 512},
        "upstream_errors": {"type": "array", "items": {"type": "object", "properties": {
            "phase": {"type": "string", "enum": ["READ", "REDUCE"]},
            "step_id": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["phase", "step_id", "reason"], "additionalProperties": False}}}
    review_schema = {"type": "object", "properties": review_properties,
        "required": list(review_properties), "additionalProperties": False}
    phase_review = {
        "READ": ("Assess only atomic facts this CURRENT visible source window can contribute to its active Needs. "
            "Do not require an answer to the whole question or facts located elsewhere in full originals. An irrelevant "
            "document window, an isolated heading, or text such as a delivery-method heading without a relevant relation "
            "can correctly yield empty claims/facts/support/conflicts/open_needs. If no relevant visible fact is omitted "
            "and no unsupported assertion is made, this empty extraction is correct: source_support_verified, "
            "visible_prompt_support and correct_task_binding can all be true and quality high. Absence of a claim is "
            "not hallucination. Respect step_output_schema: never penalize missing followups/open_needs when the schema "
            "forbids them. Judge completeness against this window, not the entire source archive."),
        "REDUCE": ("Assess only reasoning over the claims supplied in the CURRENT visible reduction input. Check "
            "arithmetic, units, deduplication, conditions and parent/source chains. Do not demand facts missing from "
            "those input claims merely because they occur elsewhere in complete originals. Full originals verify "
            "asserted facts independently; observed upstream extraction mistakes belong in upstream_errors."),
        "ANSWER": ("Assess whether the fixed final answer uses the visible evidence/conclusions to address the whole "
            "current question, with the requested conditions, coverage, values and citations. A phase label, generic "
            "placeholder or empty shell is not a complete final answer.")}[trace["phase"]]
    review_system = ("The first body field target_output is the ONLY continuation being evaluated. Inspect that "
        "exact object/string, not any output embedded in messages or reading_steps. Those other fields are evidence "
        "or upstream attribution and must never replace the evaluated target. Return assessments only and do not "
        "rewrite target_output. "+phase_review+" For READ/REDUCE, begin reason with the actual target_output counts "
        "claims=N, facts=M (zero when absent), then identify the concrete erroneous claim/field or omitted relevant "
        "fact if rejecting. A nonempty target can never be described as an empty extraction. A heading fragment "
        "tagged as support for an unrelated Need is not a correct empty extraction. For ANSWER, reason must name "
        "a specific target answer point and its error or explain which requested point it correctly addresses. "
        "visible_prompt_support means every asserted "
        "fact follows from the exact messages; source_support_verified checks those assertions against complete "
        "original_sources independently. Extra originals/history are verification evidence, never additional required "
        "content for a READ window or REDUCE input. When an observed READ/REDUCE caused an error, identify it using "
        "reading_steps in upstream_errors; do not invent attribution. Give a short reason for the lowest score or any "
        "false assessment, or explain why a valid empty extraction is appropriate. Return the object matching "
        "review_output_schema.")
    verification = reader_review_body(trace, answer_output_tokens=answer_output_tokens,
        target_text=target_text, reading_steps=reading_steps)
    verification["review_output_schema"] = review_schema
    if structured:
        verification["step_output_schema"] = trace["output_schema"]
    result = await complete("review", [{"role": "system", "content": review_system},
        {"role": "user", "content": json.dumps(verification, ensure_ascii=False)}], review_schema)
    terminal = terminal_record(result, "review")
    if terminal is not None:
        return {**terminal, "target_text": target_text}
    value = json.loads(result.text)
    accepted = (value["quality"] >= minimum_quality and value["source_support_verified"]
                and value["correct_task_binding"] and target_tokens <= body["target_output_budget_tokens"]
                and value["visible_prompt_support"])
    record = {**common, "status": "accepted" if accepted else "rejected",
        "target_origin": "teacher_visible_runtime_prompt" if fixed_target_text is None else "fixed_existing_target_audit",
        "target_text": target_text,
        "target_tokens_including_eos": target_tokens, "target_output_budget": body["target_output_budget_tokens"],
        "quality": value["quality"], "reason": value["reason"],
        "source_support_verified": value["source_support_verified"],
        "correct_task_binding": value["correct_task_binding"], "reviewer_identity": reviewer_identity,
        "read_protocol_version": trace["read_protocol_version"],
        "quality_measurement": "independent assessment of fixed target against complete original sources",
        "visible_prompt_support": value["visible_prompt_support"], "upstream_errors": value["upstream_errors"],
        "target_sha256": digest(target_text)}
    return record


async def review_reader_traces(config, *, limit_turns=None, concurrency=8):
    from adapters.query_selector import OpenAICompletionClient, SelectorConfig
    from engine_patches.contextkv_model_scope import load_model_scope
    settings = config["reader_audit"]
    data, review = settings["data"], settings["review"]
    root = Path(data["output_dir"])
    trace_rows = rows(root/"traces.jsonl")
    reading_steps = {}
    for trace in trace_rows:
        if trace["phase"] in {"READ", "REDUCE"} and trace.get("read_result") is not None:
            reading_steps.setdefault(trace["attempt_id"], []).append({
                "step_id": trace["read_task"]["step_id"], "phase": trace["phase"],
                "operation": trace["read_task"]["operation"], "claims": trace["read_result"]["claims"],
                "covered_units": trace["read_result"]["covered_units"]})
    allowed_protocols = data.get("read_protocol_versions", [2, 3, 4])
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
    fixed_targets = ({row["example_id"]: row for row in rows(data["fixed_targets_path"])}
                     if data.get("fixed_targets_path") else None)
    if fixed_targets is not None:
        trace_rows = [row for row in trace_rows if row["example_id"] in fixed_targets]
        for trace in trace_rows:
            target = fixed_targets[trace["example_id"]]
            for key in ("prompt_token_hash", "actual_history_sha256", "reader_identity"):
                if target[key] != trace[key]:
                    raise ValueError(f"fixed target {key} differs from trace: {trace['example_id']}")
    path = root/"reviews.jsonl"
    prior_targets = {row["example_id"]: row for row in rows(path)
        if row.get("review_protocol") in {"visible_target_independent_review_v5", REVIEW_PROTOCOL}
        and "target_text" in row} if path.exists() else {}
    calls_path = root/"review_calls.jsonl"
    generated_targets = {row["example_id"]: row["text"] for row in rows(calls_path)
        if row.get("review_protocol") == REVIEW_PROTOCOL and row["stage"] == "target"
        and row["status"] == "completed" and row["finish_reason"] == "stop" and row["text"]} if calls_path.exists() else {}
    for trace in trace_rows:
        identity = trace["example_id"]
        if identity in generated_targets and identity not in prior_targets:
            value = generated_targets[identity]
            prior_targets[identity] = {"target_text": json.dumps(json.loads(value), ensure_ascii=False,
                separators=(",", ":")) if trace["phase"] != "ANSWER" else value}
    done = {row["example_id"] for row in rows(path) if row.get("review_protocol") == REVIEW_PROTOCOL} if path.exists() else set()
    sink = path.open("a", encoding="utf-8")
    ledger = (root/"review_calls.jsonl").open("a", encoding="utf-8")
    semaphore = asyncio.Semaphore(concurrency)
    started = time.monotonic()

    def record_call(record):
        ledger.write(json.dumps(record, ensure_ascii=False)+"\n")
        ledger.flush()

    async def review_one(index, trace):
        if trace["example_id"] in done:
            return
        async with semaphore:
            record = await review_reader_trace(trace, clients[index % len(clients)],
                minimum_quality=data["minimum_quality"], reviewer_identity=scope.manifest(),
                review_input_budget=review["max_input_tokens"], record_call=record_call,
                answer_output_tokens=data.get("answer_output_tokens", 1024),
                reading_steps=reading_steps.get(trace.get("attempt_id"), ()),
                fixed_target_text=(fixed_targets[trace["example_id"]]["target_text"] if fixed_targets is not None
                    else prior_targets[trace["example_id"]]["target_text"] if trace["example_id"] in prior_targets else None))
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
    saved = {row["example_id"]: row for row in rows(path) if row.get("review_protocol") == REVIEW_PROTOCOL}
    manifest = {"review_protocol": REVIEW_PROTOCOL, "processed_examples": len(saved),
        "reviewed_examples": sum(row["status"] in {"accepted", "rejected"} for row in saved.values()),
        "requested_examples": len(trace_rows),
        "status_counts": dict(Counter(row["status"] for row in saved.values())),
        "elapsed_ms": (time.monotonic()-started)*1000, "reviews_path": str(path.resolve()),
        "cost_ledger": str((root/"review_calls.jsonl").resolve()), "reviewer": scope.manifest()}
    (root/"review_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest



def audit_reader_targets(config):
    """Inventory saved targets without generating, rewriting or judging any text.

    The output is a separate traces/reviews workspace. Its fixed_targets_path can
    be passed to review_reader_traces to independently verify the old targets.
    """
    settings = config["reader_audit"]["data"]
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    inventory, frozen, traces, upstream, sources = [], {}, {}, [], []
    original_reviews = {}
    phase_counts = {phase: Counter() for phase in ("READ", "REDUCE", "ANSWER")}
    for spec in settings["audit_sources"]:
        trace_path, review_path = Path(spec["traces_path"]), Path(spec["reviews_path"])
        if output.resolve() in {trace_path.parent.resolve(), review_path.parent.resolve()}:
            raise ValueError("audit output must be separate from the saved reader assets")
        saved_reviews = {row["example_id"]: row for row in rows(review_path)}
        asset_id = spec.get("asset_id", "asset:"+digest([
            str(trace_path.resolve()), str(review_path.resolve())])[:20])
        provenance = {"source_asset_id": asset_id, "source_traces": str(trace_path.resolve()),
                      "source_reviews": str(review_path.resolve())}
        sources.append({**spec, **provenance, "traces_sha256": file_digest(trace_path),
                        "reviews_sha256": file_digest(review_path)})
        for saved_trace in rows(trace_path):
            original_id, phase = saved_trace["example_id"], saved_trace["phase"]
            identity = f"{asset_id}:{original_id}"
            trace = {**saved_trace, **provenance, "original_example_id": original_id,
                "example_id": identity}
            if "attempt_id" in saved_trace:
                trace.update(original_attempt_id=saved_trace["attempt_id"],
                             attempt_id=f"{asset_id}:{saved_trace['attempt_id']}")
            if identity in traces and traces[identity] != trace:
                raise ValueError(f"conflicting saved reader trace: {identity}")
            traces[identity] = trace
            review = saved_reviews.get(original_id)
            if review is not None:
                original_reviews[identity] = {**review, **provenance,
                    "example_id": identity, "original_example_id": original_id}
            counts = phase_counts[phase]
            counts["examples"] += 1
            findings = []
            if review is None or "target_text" not in review:
                findings.append("no_saved_target")
            else:
                frozen[identity] = {"example_id": identity, "original_example_id": original_id,
                    **provenance, "task_id": trace["task_id"],
                    "target_text": review["target_text"], "target_sha256": digest(review["target_text"]),
                    "prompt_token_hash": trace["prompt_token_hash"],
                    "actual_history_sha256": trace["actual_history_sha256"],
                    "reader_identity": trace["reader_identity"], "source_review": str(review_path.resolve())}
                counts["saved_targets"] += 1
                for name in ("prompt_token_hash", "actual_history_sha256", "reader_identity"):
                    if review[name] != trace[name]:
                        findings.append(name+"_mismatch")
                independent = review.get("review_protocol") in {
                    "visible_target_independent_review_v1", "visible_target_independent_review_v2", "visible_target_independent_review_v3", "visible_target_independent_review_v4", REVIEW_PROTOCOL}
                counts["independently_reviewed" if independent else "needs_independent_review"] += 1
                if "visible_prompt_support" not in review:
                    findings.append("visibility_unverified")
                elif not review["visible_prompt_support"]:
                    findings.append("hidden_information_target")
                if review.get("source_support_verified") is False:
                    findings.append("source_unsupported_target")
                if review.get("correct_task_binding") is False:
                    findings.append("wrong_task_binding")
                for error in review.get("upstream_errors", ()):
                    steps = trace.get("reading_steps", ())
                    matching = [step for step in steps if step["step_id"] == error["step_id"]
                                and step["phase"] == error["phase"]]
                    observed = matching or [step for step in traces.values()
                        if step.get("attempt_id") == trace.get("attempt_id")
                        and step.get("step_id") == error["step_id"] and step["phase"] == error["phase"]]
                    upstream.append({"example_id": identity, "original_example_id": original_id,
                        **provenance, "task_id": trace["task_id"],
                        "attempt_id": trace.get("attempt_id"), **error,
                        "observed_step": bool(observed), "source_review": str(review_path.resolve())})
            counts.update(findings)
            inventory.append({"example_id": identity, "original_example_id": original_id,
                **provenance, "task_id": trace["task_id"],
                "phase": phase, "source_traces": str(trace_path.resolve()),
                "source_reviews": str(review_path.resolve()), "findings": findings,
                "status": review["status"] if review is not None else "missing_review"})
    write_rows(output/"traces.jsonl", traces.values())
    write_rows(output/"fixed_targets.jsonl", frozen.values())
    write_rows(output/"original_reviews.jsonl", original_reviews.values())
    write_rows(output/"target_audit.jsonl", inventory)
    for phase in ("READ", "REDUCE"):
        write_rows(output/f"{phase.lower()}_upstream_errors.jsonl", [row for row in upstream if row["phase"] == phase])
    manifest = {"schema_version": 2, "sources": sources, "phases": {
        phase: dict(counts) for phase, counts in phase_counts.items()},
        "upstream_errors": dict(Counter(row["phase"] for row in upstream)),
        "unobserved_upstream_attributions": sum(not row["observed_step"] for row in upstream),
        "traces_path": str((output/"traces.jsonl").resolve()),
        "fixed_targets_path": str((output/"fixed_targets.jsonl").resolve()),
        "original_reviews_path": str((output/"original_reviews.jsonl").resolve()),
        "model_requests": 0, "target_generation": "none; saved targets remain fixed"}
    (output/"target_audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest
