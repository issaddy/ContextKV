"""Reader-bound plan gains from actual direct/counterfactual ANSWER scores.

This module imports saved runs, fits a small ridge regressor, and loads its
portable JSON asset.  It does not generate answers or derive labels from support
scores.  A new reader checkpoint/profile gets a separate gain-data cohort.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import time

from controller.plan_policy import (FEATURE_NAMES, LEGACY_FEATURE_NAMES, LEGACY_PLAN_KINDS,
                                     PLAN_KINDS, PlanCandidate, complete_plan, plan_features)
from controller.context_schema import ReadingOutcome
from controller.schema import SessionSnapshot
from controller.serialization import to_jsonable


COST_FIELDS = ("complete_ms", "gpu_gib_seconds", "host_gib_seconds",
               "total_input_tokens", "total_output_tokens")
COST_TARGETS = ("branch_complete_cost", "incremental_complete_cost")


def observed_cost_targets(direct, counterfactual):
    """Actual future response costs exclude common paid state, preparation and cleanup."""
    measured = []
    for branch in (direct, counterfactual):
        cost = branch["remaining_cost"]
        if cost is None or any(not isinstance(cost[name], (int, float))
                or not math.isfinite(cost[name]) or cost[name] < 0 for name in COST_FIELDS):
            raise ValueError("observed actions require measured nonnegative remaining complete response costs")
        measured.append({name: cost[name] for name in COST_FIELDS})
    baseline, future = measured
    delta = {name: counterfactual["cost"][name]-direct["cost"][name] for name in COST_FIELDS}
    if any(not math.isclose(delta[name], future[name]-baseline[name], rel_tol=1e-9, abs_tol=1e-9)
           for name in COST_FIELDS):
        raise ValueError("observed paired costs must share the same paid state and preparation")
    return {"branch_complete_cost": future, "direct_complete_cost": baseline,
            "incremental_complete_cost": delta}


def _digest(value):
    return hashlib.sha256(json.dumps(to_jsonable(value), ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def actual_history(snapshot: SessionSnapshot):
    return [{"turn_id": row.turn_id, "user_text": row.user_text,
             "assistant_text": row.assistant_text} for row in snapshot.raw_history]


def working_state_key(state: ReadingOutcome, snapshot: SessionSnapshot):
    return _digest({"task_state": state.plan.task_state, "results": state.results,
        "claims": state.claims, "coverage": state.coverage, "open_needs": state.open_needs,
        "available_sources": tuple((unit.unit_id, unit.text, unit.source_spans, unit.dependencies)
            for unit in state.plan.selected_set.selected_units),
        "coverage_scope": state.plan.coverage_scope, "required_scope": state.plan.coverage_units,
        "actual_history": actual_history(snapshot)})


def build_gain_record(plan: PlanCandidate, state: ReadingOutcome, snapshot: SessionSnapshot,
                      direct, counterfactual):
    """Bind two already served/scored branches at one frozen actual-history state.

    Branches contain question, actual_history, state_key, reader_checkpoint,
    reader_profile_id, split, comparison_budget, answer, quality (normalized
    scorer output), scorer_identity, native_receipts, cost, measurement_kind,
    and history_origin. The baseline additionally declares semantic_kind and
    read_count. Full native receipts, raw answers/scores and all costs survive.
    """
    expected_state = working_state_key(state, snapshot)
    expected_history = actual_history(snapshot)
    for branch in (direct, counterfactual):
        if branch["measurement_kind"] != "hardware" or branch["history_origin"] not in {
                "model_actual", "frozen_actual_rollout"}:
            raise ValueError("plan gain supervision requires hardware answers on actual model history")
        history = [{name: row[name] for name in ("turn_id", "user_text", "assistant_text")}
                   for row in branch["actual_history"]]
        if (branch["question"] != state.plan.task_state.raw_question or history != expected_history
                or branch["state_key"] != expected_state):
            raise ValueError("counterfactual branches must share the frozen logical state and actual history")
        if branch["reader_checkpoint"] != plan.reader_checkpoint:
            raise ValueError("counterfactual answer belongs to a different reader checkpoint")
        if branch["split"] not in {"train", "dev"}:
            raise ValueError("test answers cannot enter plan gain supervision")
        if not branch["native_receipts"] or any(not receipt["generation_complete"] or
                not receipt["input_hash"] for receipt in branch["native_receipts"]):
            raise ValueError("gain labels require completed native generation receipts")
        quality = branch["quality"]
        if not isinstance(quality, (int, float)) or not math.isfinite(quality) or not 0 <= quality <= 1:
            raise ValueError("gain quality must be the declared scorer's observed normalized score")
        for name in ("complete_ms", "gpu_gib_seconds", "host_gib_seconds",
                     "total_input_tokens", "total_output_tokens"):
            if branch["cost"][name] is None:
                raise ValueError("counterfactual records retain complete latency, physical integrals and token costs")
    for name in ("split", "reader_checkpoint", "reader_profile_id", "comparison_budget", "scorer_identity"):
        if direct[name] != counterfactual[name]:
            raise ValueError(f"direct and counterfactual answers differ in {name}")
    observed = direct.get("observed_state_binding")
    if observed != counterfactual.get("observed_state_binding"):
        raise ValueError("direct and counterfactual answers differ in their observed paid/native state")
    if observed is not None:
        if (observed["state_key"] != expected_state or observed["actual_history"] != expected_history
                or observed["reader_checkpoint"] != plan.reader_checkpoint
                or observed["reader_profile_id"] != direct["reader_profile_id"]):
            raise ValueError("observed decision binding differs from the actual frozen reader/history state")
        expected = {"READ": 0, "REDUCE": 0, "GAP_RETRIEVAL": 0, "ANSWER": 1}
        if direct["new_action_counts"] != expected:
            raise ValueError("observed direct baseline must perform only one new ANSWER")
        action = {"read_once": "READ", "reduce_once": "REDUCE", "gap_retrieval": "GAP_RETRIEVAL"}[plan.semantic_kind]
        expected[action] = 1
        if counterfactual["new_action_counts"] != expected:
            raise ValueError("observed counterfactual must perform exactly its one registered action and ANSWER")
    if direct["semantic_kind"] != "direct_answer" or direct["read_count"] != 0:
        raise ValueError("DeltaQ baseline must be direct ANSWER without newly paid reading")
    if counterfactual["plan_id"] != plan.plan_id:
        raise ValueError("counterfactual answer must execute the labelled complete plan")
    record = {"schema_version": 1, "record_id": _digest([expected_state, plan.plan_id,
            direct["native_receipts"], counterfactual["native_receipts"]]),
        "source": "actual_answer_counterfactual", "split": direct["split"],
        "state_key": expected_state, "actual_history_sha256": _digest(expected_history),
        "reader_checkpoint": direct["reader_checkpoint"], "reader_profile_id": direct["reader_profile_id"],
        "scorer_identity": direct["scorer_identity"], "comparison_budget": direct["comparison_budget"],
        "plan_id": plan.plan_id, "semantic_kind": plan.semantic_kind,
        "features": plan_features(plan, state, output_budgets=counterfactual.get("output_budgets")),
        "quality_direct": direct["quality"], "quality_counterfactual": counterfactual["quality"],
        "delta_q": counterfactual["quality"]-direct["quality"],
        "incremental_complete_cost": {name: counterfactual["cost"][name]-direct["cost"][name]
            for name in ("complete_ms", "gpu_gib_seconds", "host_gib_seconds",
                         "total_input_tokens", "total_output_tokens")},
        "observed_state_binding": observed,
        "direct_run": direct, "counterfactual_run": counterfactual,
        "plan": to_jsonable(plan), "working_state": to_jsonable(state)}
    if observed is not None:
        direct_plan = complete_plan(replace(state.plan, plan_id=direct["plan_id"],
            steps=tuple(step.task for step in state.steps)), (), semantic_kind="direct_answer",
            reader_checkpoint=plan.reader_checkpoint)
        record["direct_features"] = plan_features(direct_plan, state, output_budgets=direct.get("output_budgets"))
        record["cost_targets"] = observed_cost_targets(direct, counterfactual)
    return record


@dataclass(frozen=True)
class PlanGainModel:
    reader_checkpoint: str
    reader_profile_id: str
    asset_id: str
    feature_names: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    provenance: dict
    cost_models: dict = field(default_factory=dict)
    cost_provenance: dict = field(default_factory=dict)

    def supports_semantic_kind(self, kind, *, costs=False):
        if costs:
            return bool(self.cost_models) and kind in self.cost_provenance["trained_semantic_kinds"]
        declared = self.provenance.get("trained_semantic_kinds", LEGACY_PLAN_KINDS)
        return kind == "direct_answer" or (kind in declared and (kind in LEGACY_PLAN_KINDS or kind in self.feature_names))

    def _check_action(self, features, *, costs=False):
        for kind in PLAN_KINDS:
            if features.get(kind, 0) > .5 and (not self.supports_semantic_kind(kind, costs=costs)
                    or (kind not in LEGACY_PLAN_KINDS and kind not in self.feature_names)):
                raise ValueError(f"asset has not learned semantic action {kind}")

    def _linear(self, features, coefficients, intercept):
        return intercept+sum(weight*(features[name]-center)/scale for name, center, scale, weight
            in zip(self.feature_names, self.center, self.scale, coefficients, strict=True))

    def predict(self, features):
        self._check_action(features)
        if self.provenance.get("direct_baseline_identity", False) and features.get("direct_answer", 0) > .5:
            return 0.0
        value = self._linear(features, self.coefficients, self.intercept)
        return min(1.0, max(-1.0, value))

    def predict_costs(self, features):
        """Return future branch costs and signed differences in the recorded COST_FIELDS units."""
        if not self.cost_models:
            raise ValueError("asset has no measured observed-action complete-cost heads")
        self._check_action(features, costs=True)
        result = {}
        for target in COST_TARGETS:
            result[target] = {}
            for name in COST_FIELDS:
                head = self.cost_models[target][name]
                value = self._linear(features, head["coefficients"], head["intercept"])
                if target == "incremental_complete_cost" and features.get("direct_answer", 0) > .5:
                    value = 0.0
                result[target][name] = max(0.0, value) if target == "branch_complete_cost" else value
        return result

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"schema_version": 2, "model_type": "reader_bound_plan_gain_cost_ridge",
            **asdict(self)}, ensure_ascii=False, indent=2)+"\n")

    @classmethod
    def load(cls, path, *, reader_checkpoint, reader_profile_id):
        data = json.loads(Path(path).read_text())
        schema, kind = data.pop("schema_version"), data.pop("model_type")
        if (schema, kind) not in {(1, "reader_bound_plan_gain_ridge"),
                                  (2, "reader_bound_plan_gain_cost_ridge")}:
            raise ValueError("unsupported plan gain asset schema")
        if data["reader_checkpoint"] != reader_checkpoint or data["reader_profile_id"] != reader_profile_id:
            raise ValueError("gain asset requires its measured reader checkpoint and runtime profile")
        if (len(set(data["feature_names"])) != len(data["feature_names"])
                or not set(data["feature_names"]) <= set(FEATURE_NAMES)):
            raise ValueError("gain asset declares unknown or duplicate online plan features")
        for name in ("feature_names", "center", "scale", "coefficients"):
            data[name] = tuple(data[name])
        return cls(**data)


def _supervision(rows, *, split):
    if not rows or any(row["source"] != "actual_answer_counterfactual" or row["split"] != split for row in rows):
        raise ValueError(f"plan gain {split} data must contain actual answer counterfactuals from that split")
    identities = {(row["reader_checkpoint"], row["reader_profile_id"],
                   json.dumps(row["scorer_identity"], sort_keys=True)) for row in rows}
    if len(identities) != 1:
        raise ValueError("fit one reader checkpoint/profile and scorer identity per gain asset")
    for row in rows:
        if row["delta_q"] != row["quality_counterfactual"]-row["quality_direct"]:
            raise ValueError("gain labels must equal observed counterfactual quality minus direct quality")
        for name, quality in (("direct_run", row["quality_direct"]),
                              ("counterfactual_run", row["quality_counterfactual"])):
            branch = row[name]
            if (branch["measurement_kind"] != "hardware" or not branch["native_receipts"]
                    or branch["quality"] != quality or branch["state_key"] != row["state_key"]
                    or branch["reader_checkpoint"] != row["reader_checkpoint"]
                    or branch["reader_profile_id"] != row["reader_profile_id"]
                    or branch["split"] != row["split"]
                    or branch["scorer_identity"] != row["scorer_identity"]
                    or branch["comparison_budget"] != row["comparison_budget"]):
                raise ValueError("gain labels must retain their bound native answer runs and raw scores")
        direct, counterfactual = row["direct_run"], row["counterfactual_run"]
        histories = [[{key: turn[key] for key in ("turn_id", "user_text", "assistant_text")}
                      for turn in branch["actual_history"]] for branch in (direct, counterfactual)]
        if (direct["semantic_kind"] != "direct_answer" or direct["read_count"] != 0
                or direct["question"] != counterfactual["question"]
                or histories[0] != histories[1]):
            raise ValueError("gain baseline must use the same actual question/history and direct ANSWER")
        if row.get("observed_state_binding") is not None:
            if (row["observed_state_binding"] != direct.get("observed_state_binding")
                    or row["observed_state_binding"] != counterfactual.get("observed_state_binding")):
                raise ValueError("observed cost labels must retain their bound paid/native decision state")
            if row["cost_targets"] != observed_cost_targets(direct, counterfactual):
                raise ValueError("cost labels must equal the actual observed complete response costs")
            if row["incremental_complete_cost"] != row["cost_targets"]["incremental_complete_cost"]:
                raise ValueError("signed cost difference differs from the paired actual runs")
    return next(iter(identities))


def _fit_features(rows):
    names = tuple(rows[0]["features"])
    selected = FEATURE_NAMES if set(names) == set(FEATURE_NAMES) else LEGACY_FEATURE_NAMES
    if any(set(row["features"]) != set(selected) for row in rows):
        raise ValueError("fit one declared current or legacy feature schema without zero-filling new actions")
    if any(row["semantic_kind"] not in LEGACY_PLAN_KINDS and row["semantic_kind"] not in selected for row in rows):
        raise ValueError("single-action labels require their declared action one-hot features")
    if any(row["features"][kind] != float(kind == row["semantic_kind"])
           for row in rows for kind in PLAN_KINDS if kind in selected):
        raise ValueError("declared action one-hot values differ from the actual paired semantic action")
    return selected


def _cost_samples(rows):
    """One direct observation per actual state/run, independently of pair count."""
    samples, seen = [], set()
    for row in rows:
        if row.get("observed_state_binding") is None:
            continue
        samples.append((row["features"], row["cost_targets"]["branch_complete_cost"],
                        row["cost_targets"]["incremental_complete_cost"]))
        key = _digest([row["state_key"], row["direct_run"]["native_receipts"]])
        if key not in seen:
            seen.add(key)
            samples.append((row["direct_features"], row["cost_targets"]["direct_complete_cost"],
                            {name: 0.0 for name in COST_FIELDS}))
    return samples, len(seen)


def _ridge(z, y, ridge):
    import numpy as np
    x_mean, y_mean = z.mean(axis=0), y.mean(axis=0)
    x = z-x_mean
    coefficients = np.linalg.solve(x.T@x+ridge*np.eye(x.shape[1]), x.T@(y-y_mean))
    return coefficients, y_mean-x_mean@coefficients



def fit_cost_heads(samples, feature_names, center, scale, *, ridge=1.0):
    """Fit the existing two complete-cost targets from fully observed samples."""
    import numpy as np
    if not samples:
        return {}, {}
    cost_x = np.asarray([[sample[0][name] for name in feature_names] for sample in samples], dtype=np.float64)
    cost_y = np.asarray([[sample[index][name] for index in (1, 2) for name in COST_FIELDS]
                         for sample in samples], dtype=np.float64)
    cost_z = (cost_x-center)/scale
    weights, bias = _ridge(cost_z, cost_y, ridge)
    errors = np.mean((cost_z@weights+bias-cost_y)**2, axis=0)
    models = {target: {name: {"coefficients": weights[:, index*len(COST_FIELDS)+column].tolist(),
        "intercept": float(bias[index*len(COST_FIELDS)+column])} for column, name in enumerate(COST_FIELDS)}
        for index, target in enumerate(COST_TARGETS)}
    mse = {target: {name: float(errors[index*len(COST_FIELDS)+column]) for column, name in enumerate(COST_FIELDS)}
           for index, target in enumerate(COST_TARGETS)}
    return models, mse

def fit_gain(rows, *, ridge=1.0, input_path=None):
    """Closed-form regression; called only by the explicit training command."""
    import numpy as np
    reader, profile, _ = _supervision(rows, split="train")
    started = time.perf_counter()
    names = _fit_features(rows)
    cost_samples, direct_count = _cost_samples(rows)
    features = [row["features"] for row in rows]
    normalization = (*features, *(sample[0] for sample in cost_samples if sample[0]["direct_answer"] > .5))
    all_x = np.asarray([[row[name] for name in names] for row in normalization], dtype=np.float64)
    x = np.asarray([[row[name] for name in names] for row in features], dtype=np.float64)
    y = np.asarray([row["delta_q"] for row in rows], dtype=np.float64)
    center, scale = all_x.mean(axis=0), all_x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x-center)/scale
    coefficients, intercept = _ridge(z, y, ridge)
    predicted = np.clip(intercept+z@coefficients, -1, 1)
    cost_models, cost_provenance = {}, {}
    if cost_samples:
        cost_models, cost_mse = fit_cost_heads(cost_samples, names, center, scale, ridge=ridge)
        cost_provenance = {"source": "actual_observed_action_and_paired_direct_complete_response_costs",
            "training_samples": len(cost_samples), "action_samples": len(cost_samples)-direct_count,
            "unique_direct_samples": direct_count,
            "record_ids": [row["record_id"] for row in rows if row.get("observed_state_binding") is not None],
            "trained_semantic_kinds": sorted({"direct_answer", *(row["semantic_kind"] for row in rows
                if row.get("observed_state_binding") is not None)}),
            "cost_fields": COST_FIELDS, "branch_scope": "new_action_and_complete_ANSWER_excludes_paid_preparation_cleanup",
            "incremental_scope": "signed_action_plus_ANSWER_minus_direct_ANSWER",
            "training_mse": cost_mse}
    digest = _digest(rows)
    return PlanGainModel(reader, profile, "plan-gain:"+digest, names,
        tuple(center.tolist()), tuple(scale.tolist()), tuple(coefficients.tolist()), float(intercept),
        {"label": "actual_final_quality_minus_actual_direct_answer_quality",
         "training_split": "train", "training_records": len(rows),
         "training_state_count": len({row["state_key"] for row in rows}),
         "record_ids": [row["record_id"] for row in rows], "input_sha256": digest,
         "input_path": str(Path(input_path).resolve()) if input_path is not None else None,
         "scorer_identity": rows[0]["scorer_identity"], "ridge": ridge,
         "training_mse": float(np.mean((predicted-y)**2)),
         "fit_elapsed_ms": (time.perf_counter()-started)*1000,
         "implementation": "float64 standardized linear ridge; shared pre-action features for quality and complete costs",
         "counterfactual_runs_retained": True,
         "trained_semantic_kinds": sorted({row["semantic_kind"] for row in rows}),
         "direct_baseline_identity": bool(cost_samples)}, cost_models, cost_provenance)


def evaluate_gain(model, rows, *, training_state_keys=()):
    reader, profile, _ = _supervision(rows, split="dev")
    if reader != model.reader_checkpoint or profile != model.reader_profile_id:
        raise ValueError("development gains belong to a different reader checkpoint/profile")
    if rows[0]["scorer_identity"] != model.provenance["scorer_identity"]:
        raise ValueError("development and training gain labels use different scoring rubrics")
    if set(training_state_keys) & {row["state_key"] for row in rows}:
        raise ValueError("gain development states overlap fitted training states")
    errors = [model.predict(row["features"])-row["delta_q"] for row in rows]
    cost_report = {}
    samples, direct_count = _cost_samples(rows)
    if model.cost_models and samples:
        for index, target in enumerate(COST_TARGETS, 1):
            cost_report[target] = {}
            predicted = [model.predict_costs(features)[target] for features, _, _ in samples]
            for name in COST_FIELDS:
                values = [estimate[name]-sample[index][name] for estimate, sample in zip(predicted, samples, strict=True)]
                cost_report[target][name] = {"mae": sum(abs(value) for value in values)/len(values),
                                            "mse": sum(value*value for value in values)/len(values)}
    return {"split": "dev", "records": len(rows), "mae": sum(abs(e) for e in errors)/len(errors),
            "mse": sum(e*e for e in errors)/len(errors), "reader_checkpoint": reader,
            "reader_profile_id": profile, "asset_id": model.asset_id,
            "cost_errors": cost_report, "cost_samples": len(samples), "unique_direct_samples": direct_count}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dev")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ridge", type=float, default=1.0)
    args = parser.parse_args(argv)
    rows = [json.loads(line) for line in Path(args.train).read_text().splitlines() if line.strip()]
    model = fit_gain(rows, ridge=args.ridge, input_path=args.train)
    model.save(args.output)
    if args.dev:
        development = [json.loads(line) for line in Path(args.dev).read_text().splitlines() if line.strip()]
        report = evaluate_gain(model, development, training_state_keys=[row["state_key"] for row in rows])
        Path(args.output+".development.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
