"""Exploratory, session-grouped fitting of a pre-submit output-length proxy.

NumPy ridge fitting stays offline. Repeated cache states/replicates share one
case; current generated answers supply labels only, never runtime features.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import statistics

from bench.metrics import percentile
from controller.cost_model import output_length_prior
from controller.output_length import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, OutputLengthModel, candidate_features
from controller.schema import PromptCandidate, SessionSnapshot
from controller.serialization import decode


@dataclass(frozen=True)
class LengthCase:
    case_id: str
    session_id: str
    snapshot: SessionSnapshot
    candidates: tuple[PromptCandidate, ...]
    targets: dict[str, float]
    observations: dict[str, tuple[int, ...]]
    max_new_tokens: int
    truncated_counts: dict[str, int]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def build_cases(frozen_rows, pair_rows):
    """Validate ingestion once; outputs are used only for aggregated labels."""
    frozen = {row["case_id"]: row for row in frozen_rows}
    if len(frozen) != len(frozen_rows):
        raise ValueError("duplicate frozen case IDs")
    observations, seen, failures = defaultdict(list), set(), []
    for row in pair_rows:
        identity = (row["case_id"], row["cache_state"], row["repetition"])
        if identity in seen:
            raise ValueError("duplicate case/cache-state/repetition observation")
        seen.add(identity)
        case = frozen.get(row["case_id"])
        if case is None or row["conversation_id"] != case["conversation_id"]:
            raise ValueError("pair does not match a frozen case/session")
        if row["status"] != "completed":
            failures.append({"case_id": row["case_id"], "status": row["status"]})
            continue
        if row.get("measurement_source") != "hardware":
            raise ValueError("length fitting requires actual completed generation labels")
        candidates = {c["action"]: c for c in case["candidates"]}
        if set(candidates) != {"APPEND", "REBUILD"} or set(row["observations"]) != set(candidates):
            raise ValueError("length training needs paired APPEND and REBUILD inputs")
        for action, observation in row["observations"].items():
            candidate = candidates[action]
            if (observation["input_hash"] != candidate["input_hash"]
                    or observation["input_tokens"] != len(candidate["input_ids"])):
                raise ValueError("length label describes a different candidate input")
            if not 0 <= observation["output_tokens"] <= case["max_new_tokens"]:
                raise ValueError("served output length exceeds frozen generation budget")
        observations[row["case_id"]].append(row)
    result = []
    for case_id, rows in sorted(observations.items()):
        frozen_case = frozen[case_id]
        targets, values, truncated = {}, {}, {}
        for action in ("APPEND", "REBUILD"):
            states = defaultdict(list)
            for row in rows:
                states[row["cache_state"]].append(row["observations"][action]["output_tokens"])
            targets[action] = statistics.mean(statistics.mean(v) for v in states.values())
            values[action] = tuple(row["observations"][action]["output_tokens"] for row in rows)
            truncated[action] = sum(row["observations"][action]["result"]["finish_reason"]
                                    in {"length", "max_tokens"} for row in rows)
        result.append(LengthCase(case_id, frozen_case["conversation_id"],
            decode(SessionSnapshot, frozen_case["snapshot"]),
            tuple(decode(PromptCandidate, c) for c in frozen_case["candidates"]),
            targets, values, frozen_case["max_new_tokens"], truncated))
    fingerprints = {json.dumps(asdict(c.fingerprint), sort_keys=True)
                    for case in result for c in case.candidates if c.fingerprint is not None}
    if (not result or len(fingerprints) != 1 or len({c.max_new_tokens for c in result}) != 1
            or any(c.fingerprint is None for case in result for c in case.candidates)):
        raise ValueError("one complete target fingerprint and output budget are required")
    return result, failures


def shared(case, priors):
    return output_length_prior(case.snapshot, global_mean=priors[0], global_p90=priors[1],
                               max_new_tokens=case.max_new_tokens)


def fit_regression(cases, model_kind, alpha, target_scale, priors):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("offline length fitting requires the optional calibration extra") from exc
    names = FEATURE_NAMES if model_kind == "candidate" else ("is_rebuild",)
    matrix, target = [], []
    for case in cases:
        baseline = shared(case, priors).mean_tokens
        for candidate in case.candidates:
            features = candidate_features(candidate, case.snapshot)
            matrix.append([features[name] for name in names])
            length = case.targets[candidate.action]
            target.append(length - baseline if target_scale == "linear" else math.log1p(length) - math.log1p(baseline))
    x, y = np.asarray(matrix), np.asarray(target)
    centers, scales = x.mean(axis=0), x.std(axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    z = (x - centers) / scales
    intercept = float(y.mean())
    coefficients = np.linalg.solve(z.T @ z / len(y) + alpha * np.eye(len(names)),
                                   z.T @ (y - intercept) / len(y))
    return {"schema_version": 1, "kind": "contextkv_output_length_ridge_v1", "model_kind": model_kind,
            "target": {"fingerprint": asdict(cases[0].candidates[0].fingerprint),
                       "max_new_tokens": cases[0].max_new_tokens,
                       "length_semantics": "served_output_tokens_capped_not_natural_eos"},
            "feature_schema": {"version": FEATURE_SCHEMA_VERSION, "names": list(names),
                               "available_at": "before_submit", "current_outputs_used": False},
            "regression": {"target_scale": target_scale, "reference": "shared_history_prior",
                           "alpha": alpha, "centers": centers.tolist(), "scales": scales.tolist(),
                           "coefficients": coefficients.tolist(), "intercept": intercept},
            "training_range": {"input_tokens": [min(len(c.input_ids) for case in cases for c in case.candidates),
                                                  max(len(c.input_ids) for case in cases for c in case.candidates)]},
            "uncertainty": {"absolute_error_p90_tokens": 0, "positive_residual_p90_tokens": 0},
            "metadata": {"training_scope": "development_exploration", "global_priors": list(priors)}}


def predict_cases(cases, model, priors):
    predictions = []
    for case in cases:
        prior = shared(case, priors)
        means, upper, errors_by_arm, ranges = {}, {}, {}, {}
        for candidate in case.candidates:
            prediction = (model.predict_prior(candidate, case.snapshot, case.max_new_tokens, prior)
                          if model is not None else None)
            means[candidate.action] = prediction.mean_tokens if prediction else prior.mean_tokens
            upper[candidate.action] = prediction.p90_tokens if prediction else prior.p90_tokens
            errors_by_arm[candidate.action] = prediction.error_tokens if prediction else None
            ranges[candidate.action] = prediction.within_training_range if prediction else True
        predictions.append(case_prediction(case, means, upper, errors_by_arm, ranges))
    return predictions


def case_prediction(case, means, upper, errors_by_arm, ranges):
    errors = [abs(means[action] - label) for action in means for label in case.observations[action]]
    positives = [max(0, label - means[action]) for action in means for label in case.observations[action]]
    return {"case_id": case.case_id, "session_id": case.session_id,
            "predicted_tokens": means, "target_mean_tokens": case.targets,
            "p90_tokens": upper, "error_tokens": errors_by_arm,
            "observed_tokens": case.observations, "within_training_range": ranges,
            "predicted_delta_tokens": means["APPEND"] - means["REBUILD"],
            "target_delta_tokens": case.targets["APPEND"] - case.targets["REBUILD"],
            "mean_arm_error_tokens": statistics.mean(abs(means[a] - case.targets[a]) for a in means),
            "observation_mean_absolute_error_tokens": statistics.mean(errors),
            "max_absolute_error_tokens": max(errors), "max_positive_residual_tokens": max(positives)}


def pair_features(case):
    """Only pre-submit candidate differences; common text is never replaced by labels."""
    arms = {c.action: candidate_features(c, case.snapshot) for c in case.candidates}
    return [arms["APPEND"][name] - arms["REBUILD"][name] for name in FEATURE_NAMES]


def fit_delta(cases, alpha):
    import numpy as np
    x = np.asarray([pair_features(c) for c in cases])
    y = np.asarray([c.targets["APPEND"] - c.targets["REBUILD"] for c in cases])
    centers, scales = x.mean(axis=0), x.std(axis=0)
    scales = np.where(scales > 0, scales, 1.0)
    z, intercept = (x - centers) / scales, float(y.mean())
    coefficients = np.linalg.solve(z.T @ z / len(y) + alpha * np.eye(len(FEATURE_NAMES)),
                                   z.T @ (y - intercept) / len(y))
    return {"schema_version": 1, "kind": "contextkv_paired_length_delta_ridge_v1",
            "feature_names": list(FEATURE_NAMES), "feature_operation": "APPEND minus REBUILD",
            "centers": centers.tolist(), "scales": scales.tolist(),
            "coefficients": coefficients.tolist(), "intercept": intercept, "alpha": alpha,
            "input_range": [min(len(c.input_ids) for case in cases for c in case.candidates),
                            max(len(c.input_ids) for case in cases for c in case.candidates)],
            "uncertainty": {"absolute_error_p90_tokens": 0, "positive_residual_p90_tokens": 0},
            "target": {"fingerprint": asdict(cases[0].candidates[0].fingerprint),
                       "max_new_tokens": cases[0].max_new_tokens,
                       "length_semantics": "served_output_tokens_capped_not_natural_eos"},
            "deployment": "offline_exploration_requires_both_frozen_candidates; not accepted by OutputLengthModel.load"}


def predict_delta_cases(cases, spec, priors):
    predictions = []
    for case in cases:
        prior = shared(case, priors)
        covered = all(spec["input_range"][0] <= len(c.input_ids) <= spec["input_range"][1] for c in case.candidates)
        delta = spec["intercept"] + sum(w * ((f - c) / s) for f, w, c, s in
                    zip(pair_features(case), spec["coefficients"], spec["centers"], spec["scales"]))
        means = {a: min(case.max_new_tokens, max(1, round(prior.mean_tokens + sign * delta / 2)))
                 if covered else prior.mean_tokens for a, sign in (("APPEND", 1), ("REBUILD", -1))}
        uncertainty = spec["uncertainty"]
        upper = {a: min(case.max_new_tokens, math.ceil(n + uncertainty["positive_residual_p90_tokens"]))
                 if covered else prior.p90_tokens for a, n in means.items()}
        predictions.append(case_prediction(case, means, upper,
            {a: uncertainty["absolute_error_p90_tokens"] for a in means}, {a: covered for a in means}))
    return predictions


def delta_oof(cases, alpha, priors):
    return [p for held in sorted({c.session_id for c in cases})
            for p in predict_delta_cases([c for c in cases if c.session_id == held],
                       fit_delta([c for c in cases if c.session_id != held], alpha), priors)]


def select_delta(cases, priors, alphas=(.01, .1, 1.0, 10.0)):
    scores = [{"alpha": alpha, **summarize(delta_oof(cases, alpha, priors))} for alpha in alphas]
    chosen = min(scores, key=lambda s: (s["paired_delta_mae_tokens"], s["mean_arm_mae_tokens"], -s["alpha"]))
    return chosen, scores


def residual_uncertainty(predictions):
    return {
        "absolute_error_p90_tokens": percentile([p["max_absolute_error_tokens"] for p in predictions], .9),
        "positive_residual_p90_tokens": percentile([p["max_positive_residual_tokens"] for p in predictions], .9),
        "definition": "empirical P90 of per-case worst observed arm residual from session LOSO",
        "use": "error_tokens diagnostic only; positive residual sets p90 risk length, not switch error",
        "confidence_guarantee": False, "sessions": len({p["session_id"] for p in predictions}),
        "cases": len(predictions),
    }


def fixed_choice_oof(cases, model_kind, chosen, priors):
    predictions = []
    for held in sorted({c.session_id for c in cases}):
        training = [c for c in cases if c.session_id != held]
        testing = [c for c in cases if c.session_id == held]
        model = OutputLengthModel(fit_regression(training, model_kind, chosen["alpha"], chosen["target_scale"], priors))
        predictions.extend(predict_cases(testing, model, priors))
    return predictions


def summarize(predictions):
    differences = [abs(p["predicted_delta_tokens"] - p["target_delta_tokens"]) for p in predictions]
    non_ties = [p for p in predictions if abs(p["target_delta_tokens"]) > 1]
    return {"cases": len(predictions), "sessions": len({p["session_id"] for p in predictions}),
            "mean_arm_mae_tokens": statistics.mean(p["mean_arm_error_tokens"] for p in predictions),
            "case_equal_observation_mae_tokens": statistics.mean(p["observation_mean_absolute_error_tokens"] for p in predictions),
            "paired_delta_mae_tokens": statistics.mean(differences),
            "paired_delta_p90_absolute_error_tokens": percentile(differences, .9),
            "non_tie_cases": len(non_ties), "tie_tolerance_tokens": 1,
            "non_tie_sign_correct": sum(p["predicted_delta_tokens"] * p["target_delta_tokens"] > 0 for p in non_ties),
            "out_of_range_arms": sum(not covered for p in predictions for covered in p["within_training_range"].values())}


def select_hyperparameters(cases, model_kind, priors, alphas=(.01, .1, 1.0, 10.0)):
    sessions = sorted({case.session_id for case in cases})
    scores = []
    for target_scale in ("linear", "log"):
        for alpha in alphas:
            predictions = []
            for held in sessions:
                training = [c for c in cases if c.session_id != held]
                testing = [c for c in cases if c.session_id == held]
                if not training:
                    continue
                model = OutputLengthModel(fit_regression(training, model_kind, alpha, target_scale, priors))
                predictions.extend(predict_cases(testing, model, priors))
            if predictions:
                score = summarize(predictions)
                scores.append({"alpha": alpha, "target_scale": target_scale,
                               "paired_delta_mae_tokens": score["paired_delta_mae_tokens"],
                               "mean_arm_mae_tokens": score["mean_arm_mae_tokens"]})
    chosen = min(scores, key=lambda s: (s["paired_delta_mae_tokens"], s["mean_arm_mae_tokens"], -s["alpha"]))
    return chosen, scores


def fit_exploration(cases, *, priors=(433, 512)):
    sessions = sorted({case.session_id for case in cases})
    if len(sessions) < 3:
        raise ValueError("session LOSO exploration needs at least three independent sessions")
    report = {"scope": "exploratory_development_session_LOSO_not_final_test",
              "case_weighting": "one vote per case; equal cache-state means, then two arms equally weighted",
              "hyperparameter_selection": "inner training-session LOSO; delta MAE first, arm MAE second",
              "sessions": sessions, "cases": len(cases), "global_priors": list(priors),
              "prior_scope": "existing frozen shared prior reused by every family; LOSO isolates the new residual fit, not historical prior construction",
              "fit_dependencies": {"numpy": version("numpy")},
              "quality_evaluated": False, "test_data_used": False, "models": {}}
    artifacts = {}
    for kind in ("shared_history", "action_only", "candidate"):
        predictions, folds = [], []
        for held in sessions:
            training = [case for case in cases if case.session_id != held]
            testing = [case for case in cases if case.session_id == held]
            chosen = None
            if kind == "shared_history":
                model = None
            else:
                chosen, _ = select_hyperparameters(training, kind, priors)
                spec = fit_regression(training, kind, chosen["alpha"], chosen["target_scale"], priors)
                spec["uncertainty"] = residual_uncertainty(fixed_choice_oof(training, kind, chosen, priors))
                model = OutputLengthModel(spec)
            fold_predictions = predict_cases(testing, model, priors)
            predictions.extend(fold_predictions)
            folds.append({"heldout_session": held, "training_sessions": sorted({c.session_id for c in training}),
                          "chosen": chosen, "summary": summarize(fold_predictions),
                          "uncertainty_scope": "training-session inner OOF only" if model else "frozen shared prior"})
        per_session = {sid: summarize([p for p in predictions if p["session_id"] == sid]) for sid in sessions}
        report["models"][kind] = {"summary": summarize(predictions), "by_session": per_session,
                                  "folds": folds, "oof_predictions": predictions}
        if kind != "shared_history":
            chosen, search = select_hyperparameters(cases, kind, priors)
            spec = fit_regression(cases, kind, chosen["alpha"], chosen["target_scale"], priors)
            spec["uncertainty"] = residual_uncertainty(predictions)
            spec["metadata"].update(validation=report["models"][kind], final_training_choice=chosen,
                                    final_training_search=search, fit_dependencies=report["fit_dependencies"],
                                    adaptation="SSJF pre-submit external proxy; LTR/PARS relative comparison emphasis; independent ridge, no upstream predictor code copied",
                                    final_test_untouched=True, default_enabled=False,
                                    prior_scope=report["prior_scope"],
                                    source_label_scope="current outputs used only as offline served-length labels")
            artifacts[kind] = spec
    predictions, folds = [], []
    for held in sessions:
        training = [case for case in cases if case.session_id != held]
        testing = [case for case in cases if case.session_id == held]
        chosen, _ = select_delta(training, priors)
        spec = fit_delta(training, chosen["alpha"])
        spec["uncertainty"] = residual_uncertainty(delta_oof(training, chosen["alpha"], priors))
        fold_predictions = predict_delta_cases(testing, spec, priors)
        predictions.extend(fold_predictions)
        folds.append({"heldout_session": held, "training_sessions": sorted({c.session_id for c in training}),
                      "chosen": chosen, "summary": summarize(fold_predictions),
                      "uncertainty_scope": "training-session inner OOF only"})
    report["models"]["paired_delta"] = {
        "summary": summarize(predictions),
        "by_session": {sid: summarize([p for p in predictions if p["session_id"] == sid]) for sid in sessions},
        "folds": folds, "oof_predictions": predictions,
        "mean_decomposition": "frozen shared history mean plus/minus predicted delta/2, then served-budget clipping"}
    chosen, search = select_delta(cases, priors)
    spec = fit_delta(cases, chosen["alpha"])
    spec["uncertainty"] = residual_uncertainty(predictions)
    spec["metadata"] = {"validation": report["models"]["paired_delta"], "final_training_choice": chosen,
                        "final_training_search": search, "global_priors": list(priors),
                        "prior_scope": report["prior_scope"], "fit_dependencies": report["fit_dependencies"],
                        "training_scope": "development_exploration", "final_test_untouched": True,
                        "default_enabled": False, "current_outputs_used_as_features": False}
    artifacts["paired_delta"] = spec
    return artifacts, report


def training_priors(baseline, source_configuration):
    """Use the serving prior center for residuals, including recipe fallback.

    Only prior settings/metadata are needed; timing points and stage simulation
    are not loaded or changed by length fitting.
    """
    from controller.cost_model import CostModel
    from controller.sampling import SamplingRecipe
    model = CostModel(default_output_tokens=baseline["settings"]["default_output_tokens"],
                      p90_output_tokens=baseline["settings"]["p90_output_tokens"])
    model.calibration_metadata = baseline.get("metadata", {})
    if source_configuration is None:
        return model.effective_output_prior, {"status": "unknown_source_configuration_legacy_prior"}
    configured = source_configuration.get("sampling")
    sampling = SamplingRecipe(**configured) if configured is not None else None
    settings = source_configuration.get("cost_model") or {}
    report = model.configure_sampling(sampling, fallback_mean=settings.get("default_output_tokens", 128),
                                      fallback_p90=settings.get("p90_output_tokens", 256))
    return model.effective_output_prior, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--baseline-cost-artifact", required=True)
    parser.add_argument("--output", required=True, help="candidate-feature artifact")
    parser.add_argument("--action-output", help="optional action-only artifact")
    parser.add_argument("--delta-output", help="optional offline paired-delta specification (requires both candidates)")
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    paths = {"cases": Path(args.cases), "pairs": Path(args.pairs),
             "cases_manifest": Path(args.cases).with_suffix(".manifest.json"),
             "pairs_manifest": Path(args.pairs).with_suffix(".manifest.json"),
             "baseline_cost_artifact": Path(args.baseline_cost_artifact)}
    for name in ("cases_manifest", "pairs_manifest"):
        if json.loads(paths[name].read_text()).get("split") != "dev":
            raise ValueError("length exploration accepts development data only")
    for path in (args.output, args.action_output, args.delta_output, args.report):
        if path and Path(path).exists():
            raise FileExistsError(path)
    baseline = json.loads(paths["baseline_cost_artifact"].read_text())
    source_configuration = json.loads(paths["pairs_manifest"].read_text()).get("config")
    priors, prior_binding = training_priors(baseline, source_configuration)
    cases, failures = build_cases(read_jsonl(paths["cases"]), read_jsonl(paths["pairs"]))
    artifacts, report = fit_exploration(cases, priors=priors)
    provenance = {name: {"path": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest()}
                  for name, path in paths.items()}
    report.update(provenance=provenance, excluded_failures=failures, training_prior_sampling=prior_binding,
                  truncated_arms=sum(sum(c.truncated_counts.values()) for c in cases),
                  label_semantics="served tokens at frozen budget; truncation is not natural EOS length")
    from controller.sampling import SamplingRecipe, request_sampling
    for kind, path in (("candidate", args.output), ("action_only", args.action_output)):
        if source_configuration is not None:
            artifacts[kind]["target"]["sampling_recipe"] = request_sampling(SamplingRecipe(**source_configuration["sampling"])
                if source_configuration.get("sampling") is not None else None)
        if path:
            artifacts[kind]["metadata"].update(provenance=provenance, excluded_failures=failures,
                                               truncated_arms=report["truncated_arms"],
                                               source_configuration=source_configuration, training_prior_sampling=prior_binding)
            OutputLengthModel(artifacts[kind]).save(path)
    if args.delta_output:
        artifacts["paired_delta"]["metadata"].update(provenance=provenance,
            truncated_arms=report["truncated_arms"], source_configuration=source_configuration,
            training_prior_sampling=prior_binding)
        path = Path(args.delta_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(artifacts["paired_delta"], stream, indent=2, allow_nan=False)
    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": args.output, "report": args.report,
                      "models": {k: v["summary"] for k, v in report["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
