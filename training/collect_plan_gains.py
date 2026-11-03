"""Import already measured/scored direct/counterfactual runs; never generate QA."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from controller.context_schema import ReadingOutcome
from controller.plan_policy import PlanCandidate
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
    if candidate["fingerprint"]["model_revision"] != row["reader_checkpoint"]:
        raise ValueError("gain reader label differs from the actual executed input fingerprint")
    budgets = {step["step_id"]: step["task"]["output_budget"] for step in row["steps"]}
    budgets["ANSWER"] = candidate["output_budget"]
    if row["executed_output_budgets"] != budgets:
        raise ValueError("gain output budgets differ from the actual executed steps")
    observed = row.get("observed_state_binding")
    reads = row["new_action_counts"]["READ"] if observed is not None else row["response"]["read_count"]
    if reads != sum(step["task"]["kind"] == "READ" for step in row["steps"]):
        raise ValueError("gain READ count differs from the actual executed steps")
    if observed is not None and set(row["new_step_ids"]) != {step["step_id"] for step in row["steps"]}:
        raise ValueError("observed action IDs differ from the actual newly executed steps")
    receipts = [step["receipt"] for step in row["steps"] if step["receipt"] is not None]
    receipts.append(attempt["result"]["receipt"])
    return {"question": row["question"], "actual_history": row["actual_history"],
        "state_key": row["frozen_state_key"], "reader_checkpoint": row["reader_checkpoint"],
        "reader_profile_id": row["reader_profile_id"], "split": row["split"],
        "comparison_budget": comparison_budget, "answer": row["answer"], "quality": quality,
        "scorer_identity": scorer_identity, "native_receipts": receipts, "cost": row["cost"],
        "measurement_kind": row["measurement_kind"], "history_origin": row["history_origin"],
        "semantic_kind": semantic_kind, "plan_id": plan_id, "read_count": reads,
        "observed_state_binding": observed,
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
    record["collection_protocol"] = ("observed_single_action_actual_final_answers_v1"
        if direct["observed_state_binding"] is not None else "recorded_frozen_state_actual_served_answers_v1")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True, help="JSONL packets naming direct_run and candidate_run JSON files")
    parser.add_argument("--output", required=True, help="New JSONL gain data file")
    args = parser.parse_args(argv)
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
