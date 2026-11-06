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
import math
from pathlib import Path
import time

from controller.context_schema import ReadingOutcome
from controller.action_proposals import BOUNDED_CONTINUATION, validate_action_counts
from controller.plan_policy import OBSERVED_FEATURE_SCHEMA, CompleteCost, NATIVE_COST_SCHEMA, PlanCandidate, plan_cost_features, plan_features
from controller.schema import CacheQuoteBatch, SessionSnapshot
from controller.serialization import decode
from controller.plan_gain_protocol import validate_input_protocol, validate_pre_action_input
from training.train_plan_gain import (COST_FIELDS, PlanGainModel, _fit_features, _ridge,
    actual_history, fit_cost_heads, working_state_key)


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


def _native_finished(call):
    receipt = call.get("receipt")
    if receipt is not None and receipt.get("generation_complete") is True:
        return True
    terminal = call.get("terminal_status")
    return (terminal is not None and terminal.get("terminal") is True and terminal.get("inflight") is False
            and terminal.get("state") in {"FAILED", "CANCELLED", "RETIRED", "REJECTED"})


def _measured(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _completion(run, *, observed):
    termination = run.get("termination", {})
    if termination.get("user_paused") is True or run["status"] in {"paused", "unresolved"}:
        return None, "paused_or_unresolved"
    calls = run["actual_native_calls"]
    if not all(_native_finished(call) for call in calls):
        return None, "native_outcome_unresolved"
    if run["attempt"] is None:
        return None, "no_native_attempt"
    if run["qa_increment"] == 1 and run["attempt"]["stage"] == "COMMITTED" and calls:
        return 1, "committed_QA"
    if run["qa_increment"] == 0 and run["attempt"]["stage"] == "FAILED":
        if observed and not (termination.get("terminal") is True and termination.get("user_paused") is False
                and termination.get("native_calls_finished") is True and termination.get("reason")
):
            return None, "terminal_failure_not_confirmed"
        return 0, "confirmed_terminal_no_QA"
    return None, "attempt_not_terminal"


def completion_record(before, run):
    planned = decode(PlanCandidate, before["plan_candidate"])
    state = decode(ReadingOutcome, before["working_state"])
    snapshot = decode(SessionSnapshot, before["snapshot"])
    key = working_state_key(state, snapshot)
    protocol = validate_input_protocol(before.get("input_protocol"))
    if run.get("input_protocol") != protocol:
        raise ValueError("completion branch changed its actual input protocol")
    if (before["frozen_state_key"] != key or run["frozen_state_key"] != key or run["plan_id"] != planned.plan_id
            or before["registered_at"] >= run["actual_branch_execution_cost"]["window"]["start"]):
        raise ValueError("completion labels require the actual pre-execution plan/state registration")
    if run["measurement_kind"] != "hardware" or run["split"] not in {"train", "dev"}:
        raise ValueError("completion supervision uses actual hardware train/development branches")
    if (any(run[name] != before[name] for name in ("reader_checkpoint", "reader_profile_id", "split", "scorer_identity"))
            or planned.reader_checkpoint != run["reader_checkpoint"]):
        raise ValueError("completion branch reader/partition/scorer differs from its pre-registration")
    if (run["question"] != state.plan.task_state.raw_question or run["actual_history"] != actual_history(snapshot)
            or run["history_origin"] not in {"frozen_actual_rollout", "model_actual"}):
        raise ValueError("completion branch differs from its actual frozen question/history")
    if run["qa_increment"] not in {0, 1}:
        raise ValueError("completion branch did not retain the one-current-QA protocol")
    observed = before.get("observed_state_binding")
    if observed != run.get("observed_state_binding"):
        raise ValueError("completion branch changed its observed paid/native state")
    if observed is not None and (observed["state_key"] != key or observed["actual_history"] != actual_history(snapshot)
            or observed["reader_checkpoint"] != run["reader_checkpoint"]
            or observed["reader_profile_id"] != run["reader_profile_id"]):
        raise ValueError("observed completion binding differs from its actual frozen state")
    validate_pre_action_input(run)
    complete, reason = _completion(run, observed=observed is not None)
    if observed is not None:
        counts = run["new_action_counts"]
        if (observed.get("continuation_protocol") != BOUNDED_CONTINUATION
                or before.get("continuation_protocol") != BOUNDED_CONTINUATION
                or run.get("continuation_protocol") != BOUNDED_CONTINUATION):
            raise ValueError("completion labels require the registered bounded continuation protocol")
        if before["continuation_gap_limits"] != run["continuation_gap_limits"]:
            raise ValueError("bounded continuation GAP limits changed after registration")
        try:
            validate_action_counts(planned.semantic_kind, counts, complete=complete == 1)
        except ValueError:
            complete, reason = None, "registered_action_not_executed"
        new_ids = set(run["new_step_ids"])
        if not new_ids <= {step["step_id"] for step in run["steps"]}:
            raise ValueError("observed completion names unavailable new physical steps")
        for kind in ("READ", "REDUCE"):
            if counts[kind] != sum(step["task"]["kind"] == kind and step["step_id"] in new_ids for step in run["steps"]):
                raise ValueError("observed new action count differs from its actual new step IDs")
    quality = (run["metrics"].get("plan_gain_quality") if run["qa_increment"] == 1
               and run["scoring"]["status"] == "scored" else None)
    if quality is not None and (not _measured(quality) or not 0 <= quality <= 1
            or run["scoring"]["identity"] != run["scorer_identity"]):
        raise ValueError("completion raw quality requires its actual normalized fixed scorer")
    necessary = (run.get("quality_components", {}).get("necessary_points") if quality is not None else None)
    utility = 0.0 if complete == 0 else quality if complete == 1 else None
    effective_coverage = 0.0 if complete == 0 else necessary if complete == 1 else None
    units = {unit.unit_id: unit for unit in planned.context_plan.selected_set.selected_units}
    required, covered = {}, {}
    for unit in units.values():
        for span in unit.source_spans:
            required.setdefault(str(span.source_key), []).append((span.start, span.end))
    # This diagnostic measures the registered source scope. A bounded GAP can
    # add sources after registration; those remain in the actual run's outcome
    # and GAP result, rather than changing this diagnostic's denominator.
    covered_ids = {identity for result in state.results for identity in result.covered_units}
    covered_ids.update(identity for step in run["steps"]
        if step["task"]["kind"] == "READ" and step["result"] is not None
        for identity in step["result"]["covered_units"])
    for identity in covered_ids & units.keys():
        for span in units[identity].source_spans:
            covered.setdefault(str(span.source_key), []).append((span.start, span.end))
    required_chars, covered_chars = _size(required), _size(covered)
    paid = None if observed is None else decode(CompleteCost, observed["paid_cost"])
    cost_features = None
    if observed is not None and (native := run.get("pre_action_quote")) is not None:
        batch = decode(CacheQuoteBatch, native["batch"])
        first = (None if native["candidate_id"] is None else
                 next(quote for quote in batch.quotes if quote.candidate_id == native["candidate_id"]))
        cost_features = plan_cost_features(plan_features(planned, state,
            output_budgets=run["executed_output_budgets"], paid_cost=paid), batch, first)
    return {"schema_version": 2, "source": "actual_plan_completion", "record_id": digest([
            key, run["plan_id"], run["actual_native_calls"], run["qa_increment"]]),
        "input_protocol": protocol,
        "state_key": key, "task_id": run["task_id"], "conversation_id": run["conversation_id"],
        "plan_id": planned.plan_id, "semantic_kind": planned.semantic_kind,
        "collection_mode": "observed_actions" if observed is not None else "whole_plan",
        "reader_checkpoint": run["reader_checkpoint"], "reader_profile_id": run["reader_profile_id"],
        "split": run["split"], "cohort_split": run["cohort_split"],
        "cohort_manifest_sha256": run["cohort_manifest_sha256"],
        "scorer_identity": run["scorer_identity"], "observed_state_binding": observed,
        "completion": complete, "completion_reason": reason,
        "raw_judge_Q": quality, "final_required_point_coverage": necessary,
        "effective_utility": utility, "effective_final_coverage": effective_coverage,
        "utility_rule": RULE, "read_source_char_fraction": covered_chars/required_chars if required_chars else None,
        "read_original_characters": covered_chars, "required_original_characters": required_chars,
        "features": plan_features(planned, state, paid_cost=paid), "cost_features": cost_features,
        "pre_registration": before, "actual_run": run}


def _cost_targets(base, candidate):
    """Retain null per dimension; subtract only two actually known costs."""
    direct = base["actual_run"]["actual_branch_execution_cost"]
    action = candidate["actual_run"]["actual_branch_execution_cost"]
    total_direct, total_action = base["actual_run"]["cost"], candidate["actual_run"]["cost"]
    branch, baseline, delta = {}, {}, {}
    for name in COST_FIELDS:
        left, right = direct[name], action[name]
        for value in (left, right, total_direct[name], total_action[name]):
            if value is not None and (not _measured(value) or value < 0):
                raise ValueError("completion costs must be observed nonnegative values or unknown")
        baseline[name], branch[name] = left, right
        delta[name] = right-left if left is not None and right is not None else None
        if (delta[name] is not None and total_direct[name] is not None and total_action[name] is not None
                and not math.isclose(delta[name], total_action[name]-total_direct[name], rel_tol=1e-9, abs_tol=1e-9)):
            raise ValueError("completion paired costs do not share paid state and preparation")
    return {"branch_complete_cost": branch, "direct_complete_cost": baseline, "incremental_complete_cost": delta}


def effective_pair(base, candidate):
    for name in ("state_key", "task_id", "conversation_id", "reader_checkpoint", "reader_profile_id", "split",
                 "scorer_identity", "observed_state_binding", "input_protocol"):
        if base[name] != candidate[name]:
            raise ValueError(f"effective branches differ in {name}")
    for name in ("comparison_budget", "paid_cost", "continuation_gap_limits"):
        if base["actual_run"].get(name) != candidate["actual_run"].get(name):
            raise ValueError(f"effective counterfactuals differ in {name}")
    if (base["actual_run"]["cost"]["shared_preparation_cost_id"]
            != candidate["actual_run"]["cost"]["shared_preparation_cost_id"]):
        raise ValueError("effective counterfactuals differ in shared preparation")
    observed = candidate["observed_state_binding"] is not None
    if base["semantic_kind"] != "direct_answer":
        raise ValueError("effective baseline must be direct ANSWER")
    if observed:
        if base["actual_run"]["new_action_counts"] != {"READ": 0, "REDUCE": 0, "GAP_RETRIEVAL": 0, "ANSWER": 1}:
            raise ValueError("effective observed direct must have zero newly executed reading actions")
    elif base["actual_run"]["steps"]:
        raise ValueError("effective whole-plan direct must have zero READ/REDUCE steps")
    if base["effective_utility"] is None or candidate["effective_utility"] is None:
        return None
    result = {"schema_version": 2, "source": "actual_completion_weighted_counterfactual",
        "record_id": digest([base["record_id"], candidate["record_id"]]),
        **{key: candidate[key] for key in ("state_key", "task_id", "conversation_id", "plan_id",
            "semantic_kind", "reader_checkpoint", "reader_profile_id", "split", "scorer_identity", "features",
            "collection_mode", "observed_state_binding", "cohort_split", "cohort_manifest_sha256", "input_protocol")},
        "utility_rule": RULE, "no_answer_utility": 0.0,
        "raw_Q_direct": base["raw_judge_Q"], "raw_Q_counterfactual": candidate["raw_judge_Q"],
        "utility_direct": base["effective_utility"], "utility_counterfactual": candidate["effective_utility"],
        "effective_delta_q": candidate["effective_utility"]-base["effective_utility"],
        "completion_direct": base["completion"], "completion_counterfactual": candidate["completion"],
        "direct_features": base["features"], "direct": base, "counterfactual": candidate}
    if observed:
        result["cost_targets"] = _cost_targets(base, candidate)
        result["incremental_complete_cost"] = result["cost_targets"]["incremental_complete_cost"]
    return result


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
        groups = {}
        for record in labelled:
            group = digest([record[name] for name in ("state_key", "conversation_id", "reader_checkpoint",
                "reader_profile_id", "split", "scorer_identity", "observed_state_binding")])
            groups.setdefault(group, []).append(record)
        for members in groups.values():
            direct = {record["record_id"]: record for record in members if record["semantic_kind"] == "direct_answer"}
            if not direct:
                continue
            if len(direct) != 1:
                raise ValueError("effective gain baseline must be the unique direct branch at this frozen decision state")
            base = next(iter(direct.values()))
            for candidate in members:
                if candidate["semantic_kind"] == "direct_answer":
                    continue
                paired = effective_pair(base, candidate)
                if paired is not None:
                    effective.append(paired)
    write_rows(target/"completion.jsonl", completed)
    write_rows(target/"effective-gains.jsonl", effective)
    manifest = {"source_collection": str(source.resolve()), "utility_rule": RULE,
        "completion_records": len(completed), "effective_gain_records": len(effective),
        "completion_counts": dict(Counter(str(record["completion"]) for record in completed)),
        "raw_quality_missing": sum(record["raw_judge_Q"] is None for record in completed),
        "collection_modes": dict(Counter(record["collection_mode"] for record in completed)),
        "completion_reasons": dict(Counter(record["completion_reason"] for record in completed)),
        "cost_labelled_actions": sum("cost_targets" in row and all(
            value is not None for values in row["cost_targets"].values() for value in values.values()) for row in effective),
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
    if len({digest(validate_input_protocol(row.get("input_protocol"))) for row in training}) != 1:
        raise ValueError("effective gain asset binds one actual source/compiler/reader input protocol")
    limits = {digest(row["counterfactual"]["actual_run"]["continuation_gap_limits"]) for row in training}
    if len(limits) != 1:
        raise ValueError("one bounded gain asset requires the same declared GAP limits")
    samples, direct_seen, cost_kinds, cache_scopes = [], set(), set(), set()
    cost_record_ids = []
    resident_digests = set()
    for row in training:
        branches = {name: completion_record(row[name]["pre_registration"], row[name]["actual_run"])
                    for name in ("direct", "counterfactual")}
        expected = effective_pair(branches["direct"], branches["counterfactual"])
        fields = ("state_key", "conversation_id", "semantic_kind", "reader_checkpoint", "reader_profile_id", "split",
                  "scorer_identity", "features", "raw_Q_direct", "raw_Q_counterfactual", "utility_direct",
                  "utility_counterfactual", "effective_delta_q", "direct_features", "observed_state_binding", "input_protocol")
        if expected is None or any(row[name] != expected[name] for name in fields):
            raise ValueError("effective target/features differ from the actual fixed-state branch evidence")
        if expected.get("cost_targets") != row.get("cost_targets"):
            raise ValueError("effective cost labels differ from the paired actual paid branch costs")
        if row["observed_state_binding"] is None:
            continue
        resident_digests.add(row["observed_state_binding"]["resident_model_manifest_digest"])
        costs = row["cost_targets"]
        for side, target in (("direct", "direct_complete_cost"), ("counterfactual", "branch_complete_cost")):
            branch = branches[side]
            if branch["cost_features"] is None or any(value is None for value in costs[target].values()):
                continue
            if side == "direct":
                if branch["record_id"] in direct_seen:
                    continue
                direct_seen.add(branch["record_id"])
            samples.append((branch["cost_features"], costs[target]))
            cost_kinds.add(branch["semantic_kind"])
            cost_record_ids.append(branch["record_id"])
            cache_scopes.add(branch["actual_run"]["counterfactual_cache_start"])
    started = time.perf_counter()
    names = _fit_features(training)
    x = np.asarray([[row["features"][name] for name in names] for row in training], dtype=np.float64)
    y = np.asarray([row["effective_delta_q"] for row in training], dtype=np.float64)
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    coefficients, intercept = _ridge((x-center)/scale, y, ridge)
    cost_models, cost_provenance = {}, {}
    if samples:
        cost_names = tuple(samples[0][0])
        cost_x = np.asarray([[features[name] for name in cost_names] for features, _ in samples], dtype=np.float64)
        cost_center, cost_scale = cost_x.mean(axis=0), cost_x.std(axis=0)
        cost_scale[cost_scale == 0] = 1.0
        cost_models, cost_mse = fit_cost_heads(samples, cost_names, cost_center, cost_scale,
            ridge=ridge, targets=("branch_complete_cost",))
        cost_provenance = {"observed_feature_schema": OBSERVED_FEATURE_SCHEMA, "continuation_protocol": BOUNDED_CONTINUATION,
            "source": "actual_observed_terminal_branch_and_paired_direct_costs",
            "prefix_control_scope": "native_excludes_prefix_control",
            "feature_schema": NATIVE_COST_SCHEMA, "feature_names": cost_names,
            "center": cost_center.tolist(), "scale": cost_scale.tolist(),
            "training_cache_starts": sorted(cache_scopes), "training_samples": len(samples),
            "action_samples": len(samples)-len(direct_seen), "unique_direct_samples": len(direct_seen),
            "record_ids": cost_record_ids, "trained_semantic_kinds": sorted(cost_kinds), "cost_fields": COST_FIELDS,
            "branch_scope": "actual new action through committed ANSWER or confirmed terminal no-QA failure; excludes common paid state/preparation/cleanup",
            "incremental_scope": "same_observation_predicted_action_absolute_minus_direct_absolute",
            "training_mse": cost_mse}
    return PlanGainModel(training[0]["reader_checkpoint"], training[0]["reader_profile_id"],
        "effective-plan-gain:"+digest(training), names, tuple(center.tolist()), tuple(scale.tolist()),
        tuple(coefficients.tolist()), float(intercept),
        {"input_protocol": training[0]["input_protocol"],
         "observed_feature_schema": OBSERVED_FEATURE_SCHEMA if any(
             row["observed_state_binding"] is not None for row in training) else None,
         "continuation_protocol": BOUNDED_CONTINUATION,
         "continuation_gap_limits": training[0]["counterfactual"]["actual_run"]["continuation_gap_limits"],
         "resident_model_manifest_digests": sorted(resident_digests),
         "label": LABEL, "utility_rule": RULE, "no_answer_utility": 0.0, "raw_missing_quality_preserved": True,
         "training_split": "train", "training_records": len(training), "counterfactual_runs_retained": True,
         "input_sha256": digest(training), "input_path": str(Path(path).resolve()) if path else None,
         "training_state_keys": sorted({row["state_key"] for row in training}),
         "training_conversations": sorted({row["conversation_id"] for row in training}),
         "scorer_identity": training[0]["scorer_identity"], "ridge": ridge,
         "trained_semantic_kinds": sorted({row["semantic_kind"] for row in training}),
         "direct_baseline_identity": any(row["observed_state_binding"] is not None for row in training),
         "completion_counts": dict(Counter(str(row["completion_counterfactual"]) for row in training)),
         "fit_elapsed_ms": (time.perf_counter()-started)*1000}, cost_models, cost_provenance)


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
               or row.get("input_protocol") != model.provenance["input_protocol"]
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
