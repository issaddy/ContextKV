"""Reader-bound plan gains from actual direct/counterfactual ANSWER scores.

This module imports saved runs, fits a small ridge regressor, and loads its
portable JSON asset.  It does not generate answers or derive labels from support
scores.  A new reader checkpoint/profile gets a separate gain-data cohort.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

from controller.plan_policy import FEATURE_NAMES, PlanCandidate, plan_features
from controller.context_schema import ReadingOutcome
from controller.schema import SessionSnapshot
from controller.serialization import to_jsonable


def _digest(value):
    return hashlib.sha256(json.dumps(to_jsonable(value), ensure_ascii=False,
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def actual_history(snapshot: SessionSnapshot):
    return [{"turn_id": row.turn_id, "user_text": row.user_text,
             "assistant_text": row.assistant_text} for row in snapshot.raw_history]


def working_state_key(state: ReadingOutcome, snapshot: SessionSnapshot):
    return _digest({"task_state": state.plan.task_state, "results": state.results,
        "claims": state.claims, "coverage": state.coverage, "open_needs": state.open_needs,
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
    if direct["semantic_kind"] != "direct_answer" or direct["read_count"] != 0:
        raise ValueError("DeltaQ baseline must be direct ANSWER without newly paid reading")
    if counterfactual["plan_id"] != plan.plan_id:
        raise ValueError("counterfactual answer must execute the labelled complete plan")
    return {"schema_version": 1, "record_id": _digest([expected_state, plan.plan_id,
            direct["native_receipts"], counterfactual["native_receipts"]]),
        "source": "actual_answer_counterfactual", "split": direct["split"],
        "state_key": expected_state, "actual_history_sha256": _digest(expected_history),
        "reader_checkpoint": direct["reader_checkpoint"], "reader_profile_id": direct["reader_profile_id"],
        "scorer_identity": direct["scorer_identity"], "comparison_budget": direct["comparison_budget"],
        "plan_id": plan.plan_id, "semantic_kind": plan.semantic_kind,
        "features": plan_features(plan, state, output_budgets=counterfactual.get("output_budgets")),
        "quality_direct": direct["quality"], "quality_counterfactual": counterfactual["quality"],
        "delta_q": counterfactual["quality"]-direct["quality"],
        "direct_run": direct, "counterfactual_run": counterfactual,
        "plan": to_jsonable(plan), "working_state": to_jsonable(state)}


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

    def predict(self, features):
        value = self.intercept+sum(weight*(features[name]-center)/scale for name, center, scale, weight
            in zip(self.feature_names, self.center, self.scale, self.coefficients, strict=True))
        return min(1.0, max(-1.0, value))

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"schema_version": 1, "model_type": "reader_bound_plan_gain_ridge",
            **asdict(self)}, ensure_ascii=False, indent=2)+"\n")

    @classmethod
    def load(cls, path, *, reader_checkpoint, reader_profile_id):
        data = json.loads(Path(path).read_text())
        if data.pop("schema_version") != 1 or data.pop("model_type") != "reader_bound_plan_gain_ridge":
            raise ValueError("unsupported plan gain asset schema")
        if data["reader_checkpoint"] != reader_checkpoint or data["reader_profile_id"] != reader_profile_id:
            raise ValueError("gain asset requires its measured reader checkpoint and runtime profile")
        if tuple(data["feature_names"]) != FEATURE_NAMES:
            raise ValueError("gain asset and online plan feature schemas differ")
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
    return next(iter(identities))


def fit_gain(rows, *, ridge=1.0, input_path=None):
    """Closed-form regression; called only by the explicit training command."""
    import numpy as np
    reader, profile, _ = _supervision(rows, split="train")
    started = time.perf_counter()
    x = np.asarray([[row["features"][name] for name in FEATURE_NAMES] for row in rows], dtype=np.float64)
    y = np.asarray([row["delta_q"] for row in rows], dtype=np.float64)
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1.0
    z, intercept = (x-center)/scale, float(y.mean())
    coefficients = np.linalg.solve(z.T@z+ridge*np.eye(z.shape[1]), z.T@(y-intercept))
    predicted = np.clip(intercept+z@coefficients, -1, 1)
    digest = _digest(rows)
    return PlanGainModel(reader, profile, "plan-gain:"+digest, FEATURE_NAMES,
        tuple(center.tolist()), tuple(scale.tolist()), tuple(coefficients.tolist()), intercept,
        {"label": "actual_final_quality_minus_actual_direct_answer_quality",
         "training_split": "train", "training_records": len(rows),
         "training_state_count": len({row["state_key"] for row in rows}),
         "record_ids": [row["record_id"] for row in rows], "input_sha256": digest,
         "input_path": str(Path(input_path).resolve()) if input_path is not None else None,
         "scorer_identity": rows[0]["scorer_identity"], "ridge": ridge,
         "training_mse": float(np.mean((predicted-y)**2)),
         "fit_elapsed_ms": (time.perf_counter()-started)*1000,
         "implementation": "float64 standardized linear ridge; joint complete-plan features",
         "counterfactual_runs_retained": True})


def evaluate_gain(model, rows, *, training_state_keys=()):
    reader, profile, _ = _supervision(rows, split="dev")
    if reader != model.reader_checkpoint or profile != model.reader_profile_id:
        raise ValueError("development gains belong to a different reader checkpoint/profile")
    if rows[0]["scorer_identity"] != model.provenance["scorer_identity"]:
        raise ValueError("development and training gain labels use different scoring rubrics")
    if set(training_state_keys) & {row["state_key"] for row in rows}:
        raise ValueError("gain development states overlap fitted training states")
    errors = [model.predict(row["features"])-row["delta_q"] for row in rows]
    return {"split": "dev", "records": len(rows), "mae": sum(abs(e) for e in errors)/len(errors),
            "mse": sum(e*e for e in errors)/len(errors), "reader_checkpoint": reader,
            "reader_profile_id": profile, "asset_id": model.asset_id}


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
