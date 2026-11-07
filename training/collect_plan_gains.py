"""Import already measured/scored direct/counterfactual runs; never generate QA."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from controller.context_schema import ReadingOutcome
from controller.plan_policy import PlanCandidate
from controller.plan_gain_protocol import validate_input_protocol, validate_pre_action_input
from controller.resident_models import evidence_selection_contract, resident_model_digest
from controller.schema import SessionSnapshot
from controller.serialization import decode
from .train_plan_gain import build_gain_record


def served_branch(row, *, semantic_kind, plan_id, metric, normalization, scorer_identity, comparison_budget):
    """Adapt a stored run whose frozen state key was recorded before execution.

    `normalization` declares scale/offset and its metric protocol source. Raw
    scores, complete run and native receipts remain in the imported branch.
    Missing metrics or state/checkpoint bindings are not synthesized.
    """
    if row["status"] != "completed":
        raise ValueError("plan gain collection needs completed, scored runs")
    if not normalization["protocol_source"]:
        raise ValueError("gain score normalization requires its metric protocol source")
    if row["scoring"]["status"] != "scored" or row["scoring"]["identity"] != scorer_identity:
        raise ValueError("gain score rubric differs from the actual recorded scorer")
    if row["comparison_budget"] != comparison_budget:
        raise ValueError("gain comparison budget differs from the recorded execution protocol")
    raw = row["metrics"][metric]
    quality = raw*normalization["scale"]+normalization["offset"]
    attempt = row["attempt"]
    if row["plan_id"] != plan_id or attempt["plan"]["candidate"]["context_plan_id"] != plan_id:
        raise ValueError("gain branch plan identity differs from its actually executed context plan")
    candidate = attempt["plan"]["candidate"]
    validate_input_protocol(row.get("input_protocol"), reader_fingerprint=candidate["fingerprint"])
    validate_pre_action_input(row)
    if candidate["fingerprint"]["model_revision"] != row["reader_checkpoint"]:
        raise ValueError("gain reader label differs from the actual executed input fingerprint")
    budgets = {step["step_id"]: step["task"]["output_budget"] for step in row["steps"]}
    budgets["ANSWER"] = candidate["output_budget"]
    if row["executed_output_budgets"] != budgets:
        raise ValueError("gain output budgets differ from the actual executed steps")
    observed = row.get("observed_state_binding")
    reads = row["new_action_counts"]["READ"]
    if reads != sum(step["task"]["kind"] == "READ" for step in row["steps"]):
        raise ValueError("gain READ count differs from the actual executed steps")
    if observed is not None and set(row["new_step_ids"]) != {step["step_id"] for step in row["steps"]}:
        raise ValueError("observed action IDs differ from the actual newly executed steps")
    receipts = [step["receipt"] for step in row["steps"] if step["receipt"] is not None]
    receipts.append(attempt["result"]["receipt"])
    return {"question": row["question"], "actual_history": row["actual_history"],
        "state_key": row["frozen_state_key"], "reader_checkpoint": row["reader_checkpoint"],
        "reader_profile_id": row["reader_profile_id"], "split": row["split"],
        "cohort_split": row["cohort_split"], "cohort_manifest_sha256": row["cohort_manifest_sha256"],
        "comparison_budget": comparison_budget, "answer": row["answer"], "quality": quality,
        "scorer_identity": row["scoring"]["identity"], "registered_scorer_identity": row["scorer_identity"],
        "native_receipts": receipts, "cost": row["cost"],
        "measurement_kind": row["measurement_kind"], "history_origin": row["history_origin"],
        "semantic_kind": semantic_kind, "plan_id": plan_id, "read_count": reads,
        "observed_state_binding": observed,
        "input_protocol": row.get("input_protocol"),
        "continuation_gap_limits": row["continuation_gap_limits"],
        "new_action_counts": row.get("new_action_counts"),
        "new_step_ids": row.get("new_step_ids"),
        "remaining_cost": row.get("actual_branch_execution_cost"),
        "raw_metric": {"name": metric, "value": raw, "normalization": normalization},
        "output_budgets": row["executed_output_budgets"], "source_run": row}


def collect_pair(packet, direct_row, candidate_row):
    plan = decode(PlanCandidate, packet["plan_candidate"])
    state = decode(ReadingOutcome, packet["working_state"])
    snapshot = decode(SessionSnapshot, packet["snapshot"])
    shared = {name: packet[name] for name in ("metric", "normalization", "scorer_identity", "comparison_budget")}
    direct = served_branch(direct_row, semantic_kind="direct_answer", plan_id=direct_row["plan_id"], **shared)
    candidate = served_branch(candidate_row, semantic_kind=plan.semantic_kind, plan_id=plan.plan_id, **shared)
    record = build_gain_record(plan, state, snapshot, direct, candidate)
    record["collection_protocol"] = ("observed_bounded_continuation_actual_final_answers_v1"
        if direct["observed_state_binding"] is not None else "recorded_frozen_state_actual_served_answers_v1")
    return record


def export_observed_packets(observations, actual_turns, output):
    """Resolve runtime attempts against completed or failed actual capture rows."""
    tasks, capture_statuses, source_turns = {}, {}, {}
    for row in actual_turns:
        response = row.get("response")
        actual = response if response is not None else row.get("attempt")
        if actual is None:
            continue
        attempt = actual["attempt_id"]
        binding = {name: row[name] for name in ("task_id", "cohort_split", "cohort_manifest_sha256")}
        if attempt in tasks and tasks[attempt] != binding:
            raise ValueError("one actual attempt maps to different task/cohort identities")
        tasks[attempt] = binding
        source_turns[attempt] = row
        capture_statuses[attempt] = row.get("status", "completed" if response is not None else "failed")
    packets, excluded, seen = [], [], set()
    for observation in observations:
        identity = observation["observation_id"]
        if identity in seen:
            continue
        seen.add(identity)
        attempt = observation["attempt_id"]
        if attempt not in tasks:
            excluded.append({"observation_id": identity, "attempt_id": attempt, "reason": "no_actual_turn_attempt_mapping"})
            continue
        if observation["decision_mode"] == "mandatory_coverage":
            excluded.append({"observation_id": identity, "attempt_id": attempt,
                             "reason": "mandatory_source_coverage_incomplete"})
            continue
        binding = tasks[attempt]
        if any(name in observation and observation[name] != value for name, value in binding.items()):
            raise ValueError("observed packet differs from its actual turn's task/cohort identity")
        source = source_turns[attempt]
        if not isinstance(source.get("reference"), dict):
            raise ValueError("observed export requires the actual capture raw offline reference")
        source_hash = hashlib.sha256(json.dumps(source, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest()
        generation_dir = (Path(source["generation_dir"]) if source.get("response") is not None
                          else Path(source["state_db"]).parent.parent)
        manifest_path = (generation_dir/"measurements"/
                         source["measurement_segment_id"]/"generation_manifest.json")
        manifest_bytes = manifest_path.read_bytes()
        resident_manifest = json.loads(manifest_bytes)["retrieval"]
        resident_digest = resident_model_digest(resident_manifest)
        packet = {**observation, **binding, "source_capture_status": capture_statuses[attempt],
            "source_capture_sha256": source_hash, "resident_model_manifest_digest": resident_digest,
            "evidence_selection_contract": evidence_selection_contract(resident_manifest),
            "capture_resident_binding": {"recorded_digest": observation["resident_model_manifest_digest"],
                "manifest_path": str(manifest_path.resolve()),
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}}
        if observation.get("gap_request") is not None:
            packet["gap_request"] = {**observation["gap_request"], "retriever_identity": {
                **observation["gap_request"]["retriever_identity"],
                "resident_model_manifest_digest": resident_digest}}
        packets.append(packet)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for packet in packets:
            stream.write(json.dumps(packet, ensure_ascii=False, allow_nan=False)+"\n")
    manifest = {"schema_version": 1, "scope": "observed decision packets mapped through actual response or failed attempt IDs",
        "packets": len(packets), "excluded": excluded, "output": str(path.resolve()),
        "cohort_bindings": [{"cohort_split": split, "cohort_manifest_sha256": identity}
            for split, identity in sorted({(packet["cohort_split"], packet["cohort_manifest_sha256"])
                                          for packet in packets})]}
    Path(str(path)+".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--pairs", help="JSONL packets naming direct_run and candidate_run JSON files")
    source_group.add_argument("--observations", help="Runtime observed-state JSONL with attempt/request IDs")
    source_group.add_argument("--terminal-branches", help="Completed observed-actions collection with train/dev READ branches")
    parser.add_argument("--actual-turns", help="JSONL of completed response.attempt_id and failed attempt.attempt_id capture rows")
    parser.add_argument("--reader-config", help="Actual reader config for terminal-state action eligibility")
    parser.add_argument("--output", required=True, help="New JSONL gain data file")
    args = parser.parse_args(argv)
    if args.terminal_branches:
        if not args.reader_config:
            parser.error("--terminal-branches requires --reader-config")
        from controller.config import create_model_template, load_config
        from training.terminal_observations import export_terminal_observations
        export_terminal_observations(args.terminal_branches,
            create_model_template(load_config(args.reader_config).model), args.output)
        return
    if args.observations:
        if not args.actual_turns:
            parser.error("--observations requires --actual-turns")
        def read(path):
            return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        export_observed_packets(read(args.observations), read(args.actual_turns), args.output)
        return
    source = Path(args.pairs)
    packets = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    rows = []
    manifest = []
    for packet in packets:
        paths = {name: source.parent/packet[name] for name in ("direct_run", "candidate_run")}
        raw = {name: path.read_bytes() for name, path in paths.items()}
        record = collect_pair(packet, json.loads(raw["direct_run"]), json.loads(raw["candidate_run"]))
        record["source_files"] = {name: {"path": str(paths[name].resolve()),
            "sha256": hashlib.sha256(value).hexdigest()} for name, value in raw.items()}
        rows.append(record)
        manifest.append({"record_id": record["record_id"], "source_files": record["source_files"]})
    with Path(args.output).open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
    Path(args.output+".manifest.json").write_text(json.dumps({"schema_version": 1,
        "source_pairs": str(source.resolve()), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "records": len(rows), "readers": sorted({row["reader_checkpoint"] for row in rows}),
        "splits": sorted({row["split"] for row in rows}), "pairs": manifest}, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
