"""Parallel frozen-reader task labels and actual-history grounding review."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path
from threading import Lock
import time

import yaml

from adapters.history_state import (HistoryIndex, TaskInterpreter, TeacherTaskModel, history_text,
    TASK_NEED_SEMANTICS, TASK_NEED_SEMANTICS_VERSION, task_source_catalogue)
from adapters.hybrid_retrieval import LocalBGE
from adapters.query_selector import OpenAICompletionClient, SelectorConfig
from adapters.retrieval import run_retrieval_worker
from controller.config import load_config
from controller.context_schema import EvidenceKey, RetrievalState
from controller.schema import OnlineMessage, OnlineTurn, TurnRecord
from controller.serialization import decode, to_jsonable
from training.task_data import read_jsonl, write_jsonl
from training.reader_data import file_digest, history_digest


def verify_grounding(row, state, memory, client, *, record_path=None):
    refs = [ref for condition in state.conditions for ref in condition.value_spans]
    active_conditions = [state.raw_question[ref.start:ref.end] if ref.turn_id == state.current_turn_id
                         else history_text(ref, memory) for ref in refs]
    body = {"question": row["question"], "actual_history": [
        {name: turn[name] for name in ("turn_id", "user_text", "assistant_text")}
        for turn in row["actual_history"]],
        "task_state": to_jsonable(state), "need_semantics_version": TASK_NEED_SEMANTICS_VERSION,
        "author_reference_history": row.get("reference_history", []),
        "task_query": state.query, "active_conditions": active_conditions,
        "author_grounding": [{"index": i, "quote": g["quote"], "source_id": g["passage_id"]}
                             for i, g in enumerate(row["grounding"])]}
    schema = {"type": "object", "properties": {
        "script_history_mismatch": {"type": "boolean"},
        "applicable_grounding_indices": {"type": "array", "items": {"type": "integer", "minimum": 0,
            **({"maximum": len(row["grounding"])-1} if row["grounding"] else {})},
            "maxItems": len(row["grounding"])}},
        "required": ["script_history_mismatch", "applicable_grounding_indices"], "additionalProperties": False}
    need_check = {"type": "object", "properties": {
        "operation_correct": {"type": "boolean"}, "scope_correct": {"type": "boolean"},
        "reason": {"type": "string"}},
        "required": ["operation_correct", "scope_correct", "reason"], "additionalProperties": False}
    schema["properties"].update(correct_task_binding={"type": "boolean"},
        need_checks={"type": "object", "properties": {need.need_id: need_check for need in state.needs},
            "required": [need.need_id for need in state.needs], "additionalProperties": False},
        history_references_correct={"type": "boolean"}, history_reference_reason={"type": "string"})
    schema["required"].extend(("correct_task_binding", "need_checks", "history_references_correct", "history_reference_reason"))
    cohort_bound = "cohort_manifest_sha256" in row
    if cohort_bound:
        body.update(original_sources=row["original_sources"],
            grounding_original_sources=row["grounding_original_sources"],
            verification_role="independent review of fixed TaskState; actual history establishes task intent, original text establishes source facts")
        for name in ("source_support_verified",):
            schema["properties"][name] = {"type": "boolean"}
            schema["required"].append(name)
    body["output_schema"] = schema
    completion = client.complete([{"role": "system", "content":
        "Review original source quotes against the current task resolved from ACTUAL assistant history. "
        "Mark script_history_mismatch if the authored next question presumes an object/condition/version not established "
        "or contradicted by actual history. Select only quoted source spans that support the resolved task under its "
        "active conditions and version; topical background alone is not a positive retrieval label. "
        "Use the supplied zero-based indices. When a fixed task_state is provided, assess correct_task_binding "
        "against the unchanged ACTUAL question and history, including Need references, active conditions and versions. "
        "Assess source_support_verified for the selected grounding quotes against complete grounding_original_sources; "
        "original_sources identifies the actual runtime evidence. Independently assess operation_correct and "
        "scope_correct for EVERY fixed Need using the semantics below; do not merely check that enum values "
        "are allowed. correct_task_binding must also reflect those semantic checks. Check history_references_correct "
        "for working_history, Need anchors/subjects, condition values/predicates and version time/source references: "
        "the turn, speaker and selected substring must exist in the unchanged actual messages AND identify the "
        "requested object, active conditions and versions. Fixed TaskState TextRefs are half-open [start,end), "
        "unlike the generation wire's included boundary_chars. The current turn owns only the current user "
        "question; earlier assistant answers can identify conversation intent but are not source facts. "
        "Author reference history never establishes actual intent. Explain each Need's operation/scope judgment "
        "and the history-reference judgment. Return the assessment JSON matching output_schema; "
        "do not rewrite the fixed TaskState.\n\n"+
        TASK_NEED_SEMANTICS}, {"role": "user", "content": json.dumps(body, ensure_ascii=False)}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "actual_history_grounding_review", "strict": True, "schema": schema}},
        chat_template_kwargs={"enable_thinking": True})
    if record_path is not None:
        with Path(record_path).open("a", encoding="utf-8") as sink:
            sink.write(json.dumps({"task_id": row["task_id"], "status": completion.status,
                "need_semantics_version": TASK_NEED_SEMANTICS_VERSION,
                "task_label_review_protocol": "fixed_task_state_semantics_v2",
                "finish_reason": completion.finish_reason, "input_tokens": completion.input_tokens,
                "usage": completion.usage, "elapsed_ms": completion.elapsed_ms,
                "raw_output": completion.text, "error": completion.error,
                "raw_response": completion.raw_response}, ensure_ascii=False)+"\n")
    if (completion.status == "input_over_budget" or
            completion.status == "completed" and (completion.finish_reason == "length" or not completion.text.strip())):
        return None
    if completion.status != "completed" or completion.finish_reason != "stop":
        raise RuntimeError(f"grounding review {row['task_id']} unfinished: {completion.status}/{completion.finish_reason}: {completion.error}")
    reviewed = json.loads(completion.text)
    semantics_verified = (reviewed["history_references_correct"] and all(
        check["operation_correct"] and check["scope_correct"] for check in reviewed["need_checks"].values()))
    independent = {"correct_task_binding": reviewed["correct_task_binding"],
        "need_checks": reviewed["need_checks"], "history_references_correct": reviewed["history_references_correct"],
        "history_reference_reason": reviewed["history_reference_reason"],
        "need_semantics_verified": semantics_verified, "need_semantics_version": TASK_NEED_SEMANTICS_VERSION,
        "task_label_review_protocol": "fixed_task_state_semantics_v2"}
    if cohort_bound:
        independent["source_support_verified"] = reviewed["source_support_verified"]
    return {**independent, "script_history_mismatch": reviewed["script_history_mismatch"],
        "verified_grounding": [{**row["grounding"][i], "label_origin": "qwen3_actual_history_grounding_review"}
                               for i in reviewed["applicable_grounding_indices"]],
        "verification_metrics": {"elapsed_ms": completion.elapsed_ms, "input_tokens": completion.input_tokens,
                                 "usage": completion.usage}, "verification_raw_output": completion.text}


async def label_rollouts(config, *, turns_path=None, output_path=None, concurrency=8, limit=None):
    settings = config.get("training", {})
    root = Path(settings.get("data_dir", "runs/study/data"))
    turns_path = Path(turns_path or settings.get("task_label_turns", root/"turns.jsonl"))
    output = Path(output_path or root/"task_labels.jsonl")
    rows = list(read_jsonl(turns_path))
    if limit is not None:
        rows = rows[:limit]
    cohort_path = settings.get("cohort_manifest")
    cohort = json.loads(Path(cohort_path).read_text()) if cohort_path else None
    if cohort is not None:
        members = {member["task_id"]: member for member in cohort["members"]}
        if set(members) != {row["task_id"] for row in rows}:
            raise ValueError("labeling input must contain every registered cohort state")
        for row in rows:
            member = members[row["task_id"]]
            if (row["conversation_id"] != member["conversation_id"] or row["turn"] != member["turn"] or
                    history_digest(row["actual_history"]) != member["actual_history_sha256"] or
                    row["cohort_split"] != member["split"] or
                    row["cohort_manifest_sha256"] != file_digest(cohort_path)):
                raise ValueError(f"labeling state differs from registered cohort: {row['task_id']}")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["conversation_id"]].append(row)
    prior_labels = list(read_jsonl(output)) if output.exists() else []
    by_id = {row["task_id"]: row for row in prior_labels}
    if cohort is not None:
        expected = {row["task_id"]: row for row in rows}
        for old in prior_labels:
            if old["task_id"] not in expected:
                raise ValueError("saved task label is outside the registered cohort")
            actual = expected[old["task_id"]]
            if any(old[key] != actual[key] for key in ("conversation_id", "turn", "question", "snapshot_id")) or (
                    history_digest(old["actual_history"]) != history_digest(actual["actual_history"]) or
                    old["cohort_split"] != actual["cohort_split"] or
                    old.get("correct_task_binding") is not True or old.get("source_support_verified") is not True or
                    old.get("task_label_review_protocol") != "fixed_task_state_semantics_v2" or
                    "task_catalogue" not in old or
                    old.get("need_semantics_version") != TASK_NEED_SEMANTICS_VERSION or
                    old.get("need_semantics_verified") is not True):
                raise ValueError(f"saved task label differs from actual cohort state: {old['task_id']}")
    reader = load_config(settings.get("reader_config", "configs/qwen3_32b.yaml"))
    urls = settings.get("teacher_urls", ["http://127.0.0.1:30001"])
    clients = [OpenAICompletionClient(SelectorConfig(url, reader.model.name, reader.model.revision,
        tokenizer_path=reader.model.name, max_input_tokens=settings.get("task_teacher_input_tokens", 24576),
        max_new_tokens=settings.get("task_teacher_output_tokens", 4096),
        timeout_seconds=settings.get("task_teacher_timeout_seconds", 600), temperature=0)) for url in urls]
    frozen = LocalBGE(config.get("task_model", {}).get("base_path", "models/bge-m3"),
                      device=settings.get("task_history_encoder_device", "cuda:0"),
                      dtype=settings.get("task_history_encoder_dtype", "bfloat16"), max_length=8192)
    encode_lock = Lock()

    def encode(texts):
        with encode_lock:
            return frozen.encode(texts)

    output.parent.mkdir(parents=True, exist_ok=True)
    sink = output.open("a", encoding="utf-8")
    semaphore = asyncio.Semaphore(concurrency)
    started = time.monotonic()
    generated = 0
    invocation = str(time.time_ns())
    call_paths = {stage: output.parent/f"{stage}_teacher_calls.{invocation}.jsonl"
                  for stage in ("task", "grounding")}

    async def label_conversation(number, conversation_id, tasks):
        nonlocal generated
        async with semaphore:
            client = clients[number % len(clients)]
            interpreter = TaskInterpreter(TeacherTaskModel(client, record_path=call_paths["task"]), HistoryIndex(frozen.tokenizer, encode,
                unit_tokens=384, candidates=16, view_tokens=8192))
            states = []
            for row in sorted(tasks, key=lambda item: item["turn"]):
                if row["task_id"] in by_id:
                    from controller.context_schema import TaskState
                    states.append(decode(TaskState, by_id[row["task_id"]]["task_state"]))
                    continue
                history = tuple(decode(TurnRecord, value) for value in row["actual_history"])
                memory = RetrievalState(len(history), history, tuple(states))
                messages = tuple(message for old in history for message in (
                    OnlineMessage("user", old.user_text), OnlineMessage("assistant", old.assistant_text)))
                turn = OnlineTurn(conversation_id, row["task_id"], row["turn"], row["snapshot_id"], row["question"], messages, (), "P3")
                state = await run_retrieval_worker(interpreter.interpret, turn, memory)
                if "source_scope" in row:
                    state = replace(state, source_scope=tuple(decode(EvidenceKey, key) for key in row["source_scope"]))
                verification = await run_retrieval_worker(verify_grounding, row, state, memory, client,
                    record_path=call_paths["grounding"])
                if verification is None:
                    with (output.parent/"task_label_rejections.jsonl").open("a", encoding="utf-8") as rejected:
                        rejected.write(json.dumps({"task_id": row["task_id"], "invocation_id": invocation,
                            "task_state": to_jsonable(state), "review_status": "no_complete_judgment",
                            "call_journal": str(call_paths["grounding"])}, ensure_ascii=False)+"\n")
                    continue
                accepted = verification["correct_task_binding"] and verification["need_semantics_verified"]
                if cohort is not None:
                    accepted = accepted and verification["source_support_verified"]
                if not accepted:
                    with (output.parent/"task_label_rejections.jsonl").open("a", encoding="utf-8") as rejected:
                        rejected.write(json.dumps({"task_id": row["task_id"], "invocation_id": invocation,
                            "task_state": to_jsonable(state), **verification}, ensure_ascii=False)+"\n")
                    continue
                record = {**row, **verification, "prediction": to_jsonable(interpreter.last_prediction),
                    "task_catalogue": task_source_catalogue(memory),
                    "task_state": to_jsonable(state), "teacher_metrics": interpreter.last_metrics,
                    "teacher_raw_output": interpreter.model.last_raw_output,
                    "label_origin": "frozen_qwen3_32b_actual_history_teacher",
                    "script_history_review_status": "reviewed"}
                sink.write(json.dumps(record, ensure_ascii=False)+"\n")
                sink.flush()
                by_id[row["task_id"]] = record
                states.append(state)
                generated += 1
                if generated % 25 == 0:
                    print(json.dumps({"generated_labels": generated, "total_labels": len(by_id),
                        "elapsed_seconds": time.monotonic()-started}), flush=True)
    completed = False
    try:
        async with asyncio.TaskGroup() as tasks:
            for i, (key, value) in enumerate(grouped.items()):
                tasks.create_task(label_conversation(i, key, value))
        completed = True
    finally:
        sink.close()
        for client in clients:
            client.close()
        frozen.close()
        calls = []
        for stage, path in call_paths.items():
            if path.exists():
                for call in read_jsonl(path):
                    executed = call["status"] != "input_over_budget"
                    usage = call["usage"]
                    calls.append({**call, "stage": stage, "invocation_id": invocation,
                        "cohort_manifest_sha256": file_digest(cohort_path) if cohort_path else None,
                        "model_request_executed": executed,
                        "model_input_tokens": call["input_tokens"] if executed else 0,
                        "model_output_tokens": usage.get("completion_tokens") if usage is not None and executed else
                            0 if not executed else None})
        ledger_path = output.parent/"task_labeling_calls.jsonl"
        with ledger_path.open("a", encoding="utf-8") as ledger:
            for call in calls:
                ledger.write(json.dumps(call, ensure_ascii=False)+"\n")
        known_inputs = [call["model_input_tokens"] for call in calls]
        known_outputs = [call["model_output_tokens"] for call in calls]
        manifest = {"status": "completed" if completed and len(by_id) == len(rows) else "incomplete", "invocation_id": invocation,
            "need_semantics_version": TASK_NEED_SEMANTICS_VERSION,
            "task_label_review_protocol": "fixed_task_state_semantics_v2",
            "actual_history_rows": len(rows), "labels": len(by_id), "generated_labels": generated,
            "pending_labels": len(rows)-len(by_id), "conversations": len(grouped),
            "need_operations": dict(Counter(need["operation"] for row in by_id.values() for need in row["task_state"]["needs"])),
            "need_scopes": dict(Counter(need["scope"] for row in by_id.values() for need in row["task_state"]["needs"])),
            "elapsed_ms": (time.monotonic()-started)*1000,
            "script_history_mismatches": sum(row["script_history_mismatch"] for row in by_id.values()),
            "verified_grounding_spans": sum(len(row["verified_grounding"]) for row in by_id.values()),
            "source_path": str(turns_path.resolve()), "source_sha256": file_digest(turns_path),
            "labels_path": str(output.resolve()), "frozen_history_encoder": frozen.manifest,
            "history_encoder_usage": dict(frozen.usage),
            "teacher_input_tokens": sum(known_inputs) if all(value is not None for value in known_inputs) else None,
            "teacher_unknown_input_token_calls": sum(value is None for value in known_inputs),
            "teacher_unknown_output_token_calls": sum(value is None for value in known_outputs),
            "teacher_output_tokens": sum(known_outputs) if all(value is not None for value in known_outputs) else None,
            "teacher_calls": len(calls), "cost_ledger": str(ledger_path.resolve()),
            "call_journals": {stage: str(path.resolve()) for stage, path in call_paths.items()},
            "teachers": [asdict(client.config) for client in clients],
            "cohort_manifest": str(Path(cohort_path).resolve()) if cohort_path else None,
            "cohort_manifest_sha256": file_digest(cohort_path) if cohort_path else None,
            "cohort_splits": {split: sum(row.get("cohort_split") == split for row in by_id.values())
                              for split in ("train", "dev")},
            "token_cost_scope": "current invocation including unsuccessful paid calls; reused labels retain original metrics"}
        if cohort is not None:
            manifest["split_labels"] = {}
            for split in ("train", "dev"):
                path = output.parent/f"{split}_task_labels.jsonl"
                write_jsonl(path, [row for row in by_id.values() if row["cohort_split"] == split])
                manifest["split_labels"][split] = str(path.resolve())
        output.with_name(output.stem+"_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        output.with_name(output.stem+f"_manifest.{invocation}.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/study.yaml")
    parser.add_argument("--turns")
    parser.add_argument("--output")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    result = asyncio.run(label_rollouts(yaml.safe_load(Path(args.config).read_text()),
        turns_path=args.turns, output_path=args.output, concurrency=args.concurrency, limit=args.limit))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
