"""Actual completion/coverage labels and completion-weighted plan utility.

Raw judge Q remains missing for absent answers. The declared operational rule
assigns utility zero to a terminal attempt with no committed QA; complete QA
uses its actually judged Q. Unresolved attempts and unscored QA stay unlabelled.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from controller.context_schema import ReadingOutcome
from controller.plan_policy import FEATURE_NAMES, PlanCandidate, plan_features
from controller.schema import SessionSnapshot
from controller.serialization import decode
from training.train_plan_gain import PlanGainModel, working_state_key


RULE = "terminal_no_committed_QA_has_utility_0; committed_QA_uses_actual_fixed_judge_Q; raw_missing_Q_remains_null"
LABEL = "actual_completion_weighted_final_utility_minus_direct_utility"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, values):
    with Path(path).open("x") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")


def _size(intervals):
    result = 0
    for spans in intervals.values():
        end = -1
        for left, right in sorted(spans):
            result += max(0, right-max(left, end))
            end = max(end, right)
    return result


def completion_record(before, run):
    planned = decode(PlanCandidate, before["plan_candidate"])
    state = decode(ReadingOutcome, before["working_state"])
    snapshot = decode(SessionSnapshot, before["snapshot"])
    key = working_state_key(state, snapshot)
    if (before["frozen_state_key"] != key or run["frozen_state_key"] != key or run["plan_id"] != planned.plan_id
            or before["registered_at"] >= run["actual_branch_execution_cost"]["window"]["start"]):
        raise ValueError("completion labels require the actual pre-execution plan/state registration")
    if run["measurement_kind"] != "hardware" or run["split"] not in {"train", "dev"}:
        raise ValueError("completion supervision uses actual hardware train/development branches")
    if any(run[name] != before[name] for name in ("reader_checkpoint", "reader_profile_id", "split", "scorer_identity")):
        raise ValueError("completion branch reader/partition/scorer differs from its pre-registration")
    if run["qa_increment"] not in {0, 1}:
        raise ValueError("completion branch did not retain the one-current-QA protocol")
    calls_complete = all(call.get("receipt", {}).get("generation_complete") for call in run["actual_native_calls"])
    terminal = run["attempt"]["stage"] in {"COMMITTED", "FAILED"} and calls_complete
    complete = int(run["qa_increment"] == 1) if terminal else None
    quality = run["metrics"].get("plan_gain_quality") if run["scoring"]["status"] == "scored" else None
    necessary = run.get("quality_components", {}).get("necessary_points")
    utility = 0.0 if complete == 0 else quality if complete == 1 else None
    effective_coverage = 0.0 if complete == 0 else necessary if complete == 1 else None
    units = {unit.unit_id: unit for unit in planned.context_plan.selected_set.selected_units}
    required, covered = {}, {}
    for unit in units.values():
        for span in unit.source_spans:
            required.setdefault(str(span.source_key), []).append((span.start, span.end))
    for step in run["steps"]:
        result = step["result"]
        if step["task"]["kind"] == "READ" and result is not None:
            for identity in result["covered_units"]:
                for span in units[identity].source_spans:
                    covered.setdefault(str(span.source_key), []).append((span.start, span.end))
    required_chars, covered_chars = _size(required), _size(covered)
    return {"schema_version": 1, "source": "actual_plan_completion", "record_id": digest([
            key, run["plan_id"], run["actual_native_calls"], run["qa_increment"]]),
        "state_key": key, "task_id": run["task_id"], "conversation_id": run["conversation_id"],
        "plan_id": planned.plan_id, "semantic_kind": planned.semantic_kind,
        "reader_checkpoint": run["reader_checkpoint"], "reader_profile_id": run["reader_profile_id"],
        "split": run["split"], "scorer_identity": run["scorer_identity"],
        "completion": complete, "raw_judge_Q": quality, "final_required_point_coverage": necessary,
        "effective_utility": utility, "effective_final_coverage": effective_coverage,
        "utility_rule": RULE, "read_source_char_fraction": covered_chars/required_chars if required_chars else None,
        "read_original_characters": covered_chars, "required_original_characters": required_chars,
        "features": plan_features(planned, state),
        "pre_registration": before, "actual_run": run}


def finalize(collection, output):
    source = Path(collection)
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    completed, effective = [], []
    for task in sorted((source/"tasks").iterdir()):
        labelled = []
        for registration_path in sorted(task.glob("branch-*/before.json")):
            run_path = registration_path.with_name("run.json")
            if not run_path.exists():
                continue
            registration, run = json.loads(registration_path.read_text()), json.loads(run_path.read_text())
            record = completion_record(registration, run)
            record["source_files"] = {"registration": str(registration_path.resolve()), "run": str(run_path.resolve()),
                "registration_sha256": hashlib.sha256(registration_path.read_bytes()).hexdigest(),
                "run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest()}
            completed.append(record)
            labelled.append(record)
        direct = [record for record in labelled if record["semantic_kind"] == "direct_answer"]
        if not direct:
            continue
        if len(direct) != 1 or direct[0]["actual_run"]["steps"]:
            raise ValueError("effective gain baseline must be the unique zero-READ direct branch")
        base = direct[0]
        for candidate in labelled:
            if candidate is base or base["effective_utility"] is None or candidate["effective_utility"] is None:
                continue
            if base["actual_run"]["comparison_budget"] != candidate["actual_run"]["comparison_budget"]:
                raise ValueError("effective counterfactuals use the same declared comparison budgets")
            effective.append({"schema_version": 1, "source": "actual_completion_weighted_counterfactual",
                "record_id": digest([base["record_id"], candidate["record_id"]]),
                **{key: candidate[key] for key in ("state_key", "task_id", "conversation_id", "plan_id",
                    "semantic_kind", "reader_checkpoint", "reader_profile_id", "split", "scorer_identity", "features")},
                "utility_rule": RULE, "no_answer_utility": 0.0,
                "raw_Q_direct": base["raw_judge_Q"], "raw_Q_counterfactual": candidate["raw_judge_Q"],
                "utility_direct": base["effective_utility"], "utility_counterfactual": candidate["effective_utility"],
                "effective_delta_q": candidate["effective_utility"]-base["effective_utility"],
                "completion_direct": base["completion"], "completion_counterfactual": candidate["completion"],
                "direct": base, "counterfactual": candidate})
    write_rows(target/"completion.jsonl", completed)
    write_rows(target/"effective-gains.jsonl", effective)
    manifest = {"source_collection": str(source.resolve()), "utility_rule": RULE,
        "completion_records": len(completed), "effective_gain_records": len(effective),
        "completion_counts": dict(Counter(str(record["completion"]) for record in completed)),
        "raw_quality_missing": sum(record["raw_judge_Q"] is None for record in completed),
        "raw_quality_policy": "missing raw judge Q preserved; zero is only the predeclared no-QA operational utility",
        "reader_bindings": sorted({(record["reader_checkpoint"], record["reader_profile_id"]) for record in completed})}
    (target/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest


def fit_effective(training, *, ridge=1.0, path=None):
    import numpy as np
    if not training or any(row["split"] != "train" or row["source"] != "actual_completion_weighted_counterfactual"
                           or row["utility_rule"] != RULE for row in training):
        raise ValueError("effective gain fit uses actual train-only completion-weighted counterfactuals")
    binding = {(row["reader_checkpoint"], row["reader_profile_id"], digest(row["scorer_identity"])) for row in training}
    if len(binding) != 1:
        raise ValueError("effective gain asset binds one actual reader/profile/scorer")
    for row in training:
        if row["effective_delta_q"] != row["utility_counterfactual"]-row["utility_direct"]:
            raise ValueError("effective target differs from the observed branch utilities")
        for name in ("direct", "counterfactual"):
            branch = row[name]
            observed = completion_record(branch["pre_registration"], branch["actual_run"])
            if observed["effective_utility"] != branch["effective_utility"] or observed["raw_judge_Q"] != branch["raw_judge_Q"]:
                raise ValueError("effective utility differs from actual completion/raw judge evidence")
    started = time.perf_counter()
    x = np.asarray([[row["features"][name] for name in FEATURE_NAMES] for row in training], dtype=np.float64)
    y = np.asarray([row["effective_delta_q"] for row in training], dtype=np.float64)
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    z, intercept = (x-center)/scale, float(y.mean())
    coefficients = np.linalg.solve(z.T@z+ridge*np.eye(z.shape[1]), z.T@(y-intercept))
    return PlanGainModel(training[0]["reader_checkpoint"], training[0]["reader_profile_id"],
        "effective-plan-gain:"+digest(training), FEATURE_NAMES, tuple(center.tolist()), tuple(scale.tolist()),
        tuple(coefficients.tolist()), intercept,
        {"label": LABEL, "utility_rule": RULE, "no_answer_utility": 0.0, "raw_missing_quality_preserved": True,
         "training_split": "train", "training_records": len(training), "counterfactual_runs_retained": True,
         "input_sha256": digest(training), "input_path": str(Path(path).resolve()) if path else None,
         "training_state_keys": sorted({row["state_key"] for row in training}),
         "training_conversations": sorted({row["conversation_id"] for row in training}),
         "scorer_identity": training[0]["scorer_identity"], "ridge": ridge,
         "completion_counts": dict(Counter(str(row["completion_counterfactual"]) for row in training)),
         "fit_elapsed_ms": (time.perf_counter()-started)*1000})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    final = commands.add_parser("finalize")
    final.add_argument("--collection", required=True)
    final.add_argument("--output", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--train", required=True)
    fit.add_argument("--dev")
    fit.add_argument("--output", required=True)
    fit.add_argument("--ridge", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.command == "finalize":
        print(json.dumps(finalize(args.collection, args.output), ensure_ascii=False))
        return
    model = fit_effective(rows(args.train), ridge=args.ridge, path=args.train)
    model.save(args.output)
    if args.dev:
        development = rows(args.dev)
        if any(row["split"] != "dev" or row["reader_checkpoint"] != model.reader_checkpoint or
               row["reader_profile_id"] != model.reader_profile_id or row["scorer_identity"] != model.provenance["scorer_identity"]
               for row in development):
            raise ValueError("effective gain development must match the actual reader/profile/scorer")
        if {row["conversation_id"] for row in development} & set(model.provenance["training_conversations"]):
            raise ValueError("effective development conversations overlap fitted training conversations")
        report = {"records": len(development), "mae": sum(abs(model.predict(row["features"])-row["effective_delta_q"])
            for row in development)/len(development), "reader_checkpoint": model.reader_checkpoint,
            "reader_profile_id": model.reader_profile_id, "asset_id": model.asset_id, "utility_rule": RULE}
        Path(args.output+".development.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"asset": args.output, "asset_id": model.asset_id, "training_records": model.provenance["training_records"]}))


if __name__ == "__main__":
    main()
