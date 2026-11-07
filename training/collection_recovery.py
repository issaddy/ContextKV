"""Explicit collector retry routing; interrupted native ledgers stay unchanged."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys
import time

import yaml

from controller.context_schema import StepRecord
from controller.serialization import to_jsonable
from controller.session_store import Attempt, load_record
from training.reader_data import digest, file_digest, rows, write_rows


def candidate_session(task_id, category):
    return json.dumps(["preference", task_id, category], separators=(",", ":"))


def candidate_key(task_id, category):
    return json.dumps([task_id, category], separators=(",", ":"))


def _epoch_restart(epoch):
    if epoch is None:
        return None
    return epoch["engine_restart_id"] if isinstance(epoch, dict) else epoch.engine_restart_id


def _interrupted_record(attempt, steps):
    native = [record for _, record in steps if record.receipt is not None]
    unresolved = [{"step_id": record.step_id, "phase": record.task.kind, "stage": stage,
        "submitted_input_tokens": len(record.input_ids), "executed_input_tokens": None,
        "output_tokens": None, "elapsed_ms": None, "physical_cost": None}
        for stage, record in steps if stage == "SUBMITTED" and record.receipt is None]
    if attempt.stage == "SUBMITTED":
        unresolved.append({"step_id": "ANSWER", "phase": "ANSWER", "stage": "SUBMITTED",
            "submitted_input_tokens": len(attempt.plan.candidate.input_ids), "executed_input_tokens": None,
            "output_tokens": None, "elapsed_ms": None, "physical_cost": None})
    restarts = {_epoch_restart(attempt.epoch), *(_epoch_restart(record.engine_epoch) for _, record in steps)} - {None, ""}
    if not restarts:
        raise ValueError(f"interrupted candidate has no recorded old engine restart identity: {attempt.attempt_id}")
    return {"old_attempt_id": attempt.attempt_id, "old_session_id": attempt.session_id,
        "old_request_id": attempt.request_id, "old_engine_restart_ids": sorted(restarts),
        "old_attempt": to_jsonable(attempt), "old_attempt_sha256": digest(to_jsonable(attempt)),
        "old_steps": [{"stage": stage, "record": to_jsonable(record)} for stage, record in steps],
        "old_steps_sha256": digest([(stage, to_jsonable(record)) for stage, record in steps]),
        "cost": {"receipted_input_tokens": sum(len(record.input_ids) for record in native),
            "receipted_output_tokens": sum(len(record.output_ids) for record in native),
            "unresolved_submissions": unresolved, "reader_input_tokens_total": None,
            "reader_output_tokens_total": None, "complete_ms": None, "physical_cost": None,
            "scope": "interrupted old attempt; submitted-without-receipt costs remain unknown"},
        "training_target": False}


def prepare_collection_resume(config, *, pause_receipt, confirmed_stopped, output_dir=None):
    """CPU-only snapshot of stopped work plus an explicit immutable retry route."""
    if not confirmed_stopped:
        raise ValueError("resume preparation requires confirmation that the old collector and engine stopped")
    settings = config["preferences"]
    collection = Path(settings["output_dir"]).resolve()
    receipt = Path(pause_receipt).resolve()
    evidence = json.loads(receipt.read_text())
    resume_id = str(time.time_ns())
    output = Path(output_dir).resolve() if output_dir else collection/"recovery"/resume_id
    if output.exists():
        raise ValueError("collection recovery output must be a new directory")
    database = collection/"runtime"/"sessions.sqlite3"
    families_path = Path(settings.get("candidate_families_path", collection/"candidate_families.jsonl"))
    families = {family["task_id"]: family for family in rows(families_path)}
    previous_routes, inherited = {}, {}
    if settings.get("resume_manifest"):
        previous_path = Path(settings["resume_manifest"]).resolve()
        previous = json.loads(previous_path.read_text())
        previous_routes = {candidate_key(entry["task_id"], entry["category"]): entry
                           for entry in previous["interrupted_candidates"]}
        inherited.update((entry["old_attempt_id"], entry) for entry in previous.get("inherited_interrupted_costs", ()))
        for entry in previous["interrupted_candidates"]:
            inherited[entry["old_attempt_id"]] = {"old_attempt_id": entry["old_attempt_id"], "cost": entry["cost"],
                "resume_manifest_path": str(previous_path), "resume_manifest_sha256": file_digest(previous_path)}
    completed = {}
    turns_path = collection/"turns.jsonl"
    for row in rows(turns_path):
        key = candidate_key(row["task_id"], row["category"])
        if key in completed:
            raise ValueError(f"candidate already has multiple completed output rows: {key}")
        completed[key] = row
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    interrupted = []
    for task_id, family in families.items():
        for category, selected in family["evidence_sets"].items():
            key = candidate_key(task_id, category)
            prior_route = previous_routes.get(key)
            session = prior_route["retry_session_id"] if prior_route else candidate_session(task_id, category)
            request = prior_route["retry_request_id"] if prior_route else "answer"
            payload = connection.execute("SELECT payload FROM attempts WHERE session_id=? AND request_id=?", (session, request)).fetchone()
            if key in completed:
                committed = connection.execute("SELECT payload FROM attempts WHERE attempt_id=?",
                    (completed[key]["response"]["attempt_id"],)).fetchone()
                if committed is None or load_record(connection, Attempt, committed[0]).stage != "COMMITTED":
                    raise ValueError(f"completed collector row has no matching committed native attempt: {key}")
                continue
            if payload is None:
                if prior_route is not None:
                    interrupted.append(prior_route)
                continue
            attempt = load_record(connection, Attempt, payload[0])
            if attempt.stage == "RESULT_READY":
                raise ValueError(f"candidate has a stored completed result; recover and reconcile it before retry: {key}")
            if attempt.stage == "COMMITTED":
                raise ValueError(f"native candidate committed without a turns row; reconcile its stored answer before retry: {key}")
            if attempt.stage == "FAILED":
                continue
            steps = [(stage, load_record(connection, StepRecord, value)) for stage, value in connection.execute(
                "SELECT stage,payload FROM execution_steps WHERE attempt_id=? ORDER BY rowid", (attempt.attempt_id,))]
            entry = _interrupted_record(attempt, steps)
            entry.update(task_id=task_id, category=category,
                original_batch_cost_paths=[str(path.resolve()) for path in sorted(collection.glob("collection_cost.*.json"))],
                original_physical_paths=[str(path.resolve()) for path in sorted(collection.glob("collection_physical.*.json"))],
                candidate_family_sha256=digest(family),
                evidence_set_sha256=digest(selected), actual_history_sha256=family["actual_history_sha256"],
                task_state_sha256=family["task_state_sha256"], budget=family["budget"],
                candidate_pool_path=family["candidate_pool_path"], candidate_pool_sha256=family["candidate_pool_sha256"],
                retry_session_id=json.dumps(["preference-retry", task_id, category, resume_id], separators=(",", ":")),
                retry_request_id="answer:resume:"+resume_id)
            interrupted.append(entry)
    connection.close()
    output.mkdir(parents=True)
    records_path = output/"interrupted_candidates.jsonl"
    write_rows(records_path, interrupted)
    manifest_path = output/"resume_manifest.json"
    manifest = {"schema_version": 1, "resume_id": resume_id, "collection_dir": str(collection),
        "state_db": str(database), "confirmed_old_collector_and_engine_stopped": True,
        "pause_receipt": {"path": str(receipt), "sha256": file_digest(receipt), "evidence": evidence},
        "candidate_families_path": str(families_path.resolve()), "completed_candidates": len(completed),
        "completed_output_rows": [{"task_id": row["task_id"], "category": row["category"],
            "attempt_id": row["response"]["attempt_id"], "row_sha256": digest(row)} for row in completed.values()],
        "interrupted_candidates": interrupted, "interrupted_records_path": str(records_path),
        "inherited_interrupted_costs": list(inherited.values()),
        "old_engine_restart_ids": sorted({restart for entry in interrupted for restart in entry["old_engine_restart_ids"]}),
        "run_config_sha256": file_digest(settings["run_config"]) if settings.get("run_config") else None,
        "model_requests": 0, "retry_executed": False,
        "cost_status": "old unresolved submissions remain unknown and linked separately from future retry costs"}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    resumed = deepcopy(config)
    resumed["preferences"]["resume_manifest"] = str(manifest_path)
    config_path = output/"resume_collection.yaml"
    config_path.write_text(yaml.safe_dump(resumed, sort_keys=False))
    return {"resume_manifest": str(manifest_path), "resume_config": str(config_path),
        "completed_candidates": len(completed), "interrupted_candidates": len(interrupted),
        "model_requests": 0, "resume_command": [sys.executable, "-m", "training.preference_rollout", "--config", str(config_path)]}


async def validate_collection_resume(settings, controller):
    """Check stopped-worker evidence and the new engine before creating model work."""
    if not settings.get("resume_manifest"):
        return None, {}
    path = Path(settings["resume_manifest"])
    manifest = json.loads(path.read_text())
    if not manifest["confirmed_old_collector_and_engine_stopped"]:
        raise ValueError("resume manifest does not confirm old workers stopped")
    if file_digest(manifest["pause_receipt"]["path"]) != manifest["pause_receipt"]["sha256"]:
        raise ValueError("collector pause evidence changed after recovery preparation")
    if Path(settings["output_dir"]).resolve() != Path(manifest["collection_dir"]):
        raise ValueError("resume manifest belongs to a different collection directory")
    if manifest["run_config_sha256"] is not None and file_digest(settings["run_config"]) != manifest["run_config_sha256"]:
        raise ValueError("reader run configuration changed after recovery preparation")
    for inherited in manifest.get("inherited_interrupted_costs", ()):
        if file_digest(inherited["resume_manifest_path"]) != inherited["resume_manifest_sha256"]:
            raise ValueError("an inherited interrupted-cost ledger changed")
    info = await controller.engine.info()
    if info.engine_restart_id in manifest["old_engine_restart_ids"]:
        raise ValueError("collector retry requires a different engine restart than the stopped original")
    existing_rows = {candidate_key(row["task_id"], row["category"]): row
                     for row in rows(Path(manifest["collection_dir"])/"turns.jsonl")}
    for completed in manifest["completed_output_rows"]:
        key = candidate_key(completed["task_id"], completed["category"])
        if key not in existing_rows or digest(existing_rows[key]) != completed["row_sha256"]:
            raise ValueError("completed candidate rows changed after recovery preparation")
    routes = {}
    for entry in manifest["interrupted_candidates"]:
        attempt = controller.store.get_attempt(entry["old_attempt_id"])
        steps = controller.store.execution_steps_with_stage(attempt.attempt_id)
        if digest(to_jsonable(attempt)) != entry["old_attempt_sha256"] or digest([
                (stage, to_jsonable(record)) for stage, record in steps]) != entry["old_steps_sha256"]:
            raise ValueError("interrupted native attempt changed after the recovery snapshot")
        if file_digest(entry["candidate_pool_path"]) != entry["candidate_pool_sha256"]:
            raise ValueError("interrupted candidate source pool changed after the recovery snapshot")
        routes[candidate_key(entry["task_id"], entry["category"])] = entry
    proof = {"resume_manifest": str(path.resolve()), "resume_manifest_sha256": file_digest(path),
        "new_engine_restart_id": info.engine_restart_id, "checked_monotonic": time.monotonic(),
        "inherited_interrupted_costs": manifest.get("inherited_interrupted_costs", [])}
    return proof, routes
