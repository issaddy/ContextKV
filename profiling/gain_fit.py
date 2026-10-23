"""Offline nested-session calibration of APPEND minus REBUILD gain.

Small ridge residual models use only pre-submit cost/candidate features. Current
times are labels; length-label stacking is cross-fitted again inside every gain
fold. Nothing in this module changes the online planner or serving defaults.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from bench.metrics import percentile
from controller.config import load_config
from controller.output_length import OutputLengthModel, OutputLengthPrediction
from controller.planner import choose_plan
from controller.schema import CacheQuoteBatch
from controller.serialization import decode
from profiling.decision_pairs import _model_and_knobs, load_frozen, rescore_pairs
from profiling.length_decisions import FAMILIES, OOFLengthPrior
from profiling import length_fit


FEATURE_NAMES = ("prefill_gain_ms", "decode_gain_ms", "input_delta_tokens")
ALPHAS = (.1, 1., 10., 100.)
QUANTILES = (0., .5, .8, .9)
GATES = ("configured", "no_margin", "no_stability", "no_margin_or_stability")


class SavedLengthPrior:
    def __init__(self, predictions, cases, family):
        by_case = {p["case_id"]: p for p in predictions}
        self.predictions = {}
        for case in cases:
            row = by_case[case.case_id]
            for candidate in case.candidates:
                action = candidate.action
                self.predictions[candidate.candidate_id] = OutputLengthPrediction(
                    row["predicted_tokens"][action], row["p90_tokens"][action],
                    row["error_tokens"][action], family + ":nested_oof",
                    row["within_training_range"][action])

    def predict_prior(self, candidate, session, max_new_tokens, shared_prior):
        return self.predictions[candidate.candidate_id]


def predict_lengths(training, testing, family, priors):
    """All fitting, hyperparameters and uncertainty use this training subset."""
    if family == "shared_history":
        return length_fit.predict_cases(testing, None, priors)
    if family == "paired_delta":
        chosen, _ = length_fit.select_delta(training, priors)
        spec = length_fit.fit_delta(training, chosen["alpha"])
        spec["uncertainty"] = length_fit.residual_uncertainty(
            length_fit.delta_oof(training, chosen["alpha"], priors))
        return length_fit.predict_delta_cases(testing, spec, priors)
    chosen, _ = length_fit.select_hyperparameters(training, family, priors)
    spec = length_fit.fit_regression(training, family, chosen["alpha"], chosen["target_scale"], priors)
    spec["uncertainty"] = length_fit.residual_uncertainty(
        length_fit.fixed_choice_oof(training, family, chosen, priors))
    return length_fit.predict_cases(testing, OutputLengthModel(spec), priors)


def prepared_pairs(cases, pair_rows, model, knobs, engine):
    # Reuse the existing source/receipt/quote validation once at ingestion. The
    # compact result below separates the saved pre-submit quote from outcomes.
    rescore_pairs(pair_rows, cases, model, knobs, engine)
    records, seen = [], set()
    for row in pair_rows:
        identity = (row["case_id"], row["cache_state"], row["repetition"])
        if identity in seen:
            raise ValueError("duplicate case/state/repetition")
        seen.add(identity)
        if row["status"] != "completed" or row.get("measurement_source") != "hardware":
            raise ValueError("gain fitting requires completed hardware pairs")
        times = {a: row["observations"][a]["server_elapsed_ms"] for a in ("APPEND", "REBUILD")}
        if any(t is None or not 0 <= t < float("inf") for t in times.values()):
            raise ValueError("gain labels require finite server_elapsed_ms for both arms")
        records.append({"case_id": row["case_id"], "session_id": row["conversation_id"],
            "cache_state": row["cache_state"], "repetition": row["repetition"],
            "quote": decode(CacheQuoteBatch, row["observations"][row["execution_order"][0]]["quote"]),
            "actual_gain_ms": times["APPEND"]-times["REBUILD"], "actual_costs_ms": times})
    if {r["case_id"] for r in records} != {c.case_id for c in cases}:
        raise ValueError("hardware pairs must cover every frozen case")
    return records


def prediction_features(case, quote, model, knobs):
    """No current result, timing label, case ID or answer text is accepted here."""
    plan = choose_plan(case.candidates, quote, model, case.snapshot, knobs,
                       concurrency=1, control_ms=0)
    candidates = {c.action: c for c in case.candidates}
    by_id = {e.candidate_id: e for e in plan.predicted}
    quote_by_id = {q.candidate_id: q for q in quote.quotes}
    estimates = {action: by_id[c.candidate_id] if c.candidate_id in by_id else
                 model.predict_native(c, quote_by_id[c.candidate_id], case.snapshot,
                                      batch=quote, concurrency=1, control_ms=0)
                 for action, c in candidates.items()}
    a, r = estimates["APPEND"], estimates["REBUILD"]
    features = {"prefill_gain_ms": a.restore_prefill_ms-r.restore_prefill_ms,
                "decode_gain_ms": a.decode_ms-r.decode_ms,
                "input_delta_tokens": len(candidates["APPEND"].input_ids)-len(candidates["REBUILD"].input_ids)}
    return {"features": features, "baseline_gain_ms": a.cost-r.cost,
        "baseline_action": plan.candidate.action, "baseline_reason": plan.reason,
        "margin_ms": plan.switch_margin_ms,
        "stable": case.snapshot.turns_since_rebuild >= knobs.min_stable_turns,
        "eligible": plan.reason in {"cost_saving", "hysteresis"},
        "calibrated": a.calibrated and r.calibrated,
        "length_means": {action: estimate.predicted_output_tokens for action, estimate in estimates.items()}}


class NestedFeatures:
    """Offline subset memoization; never reuses full-cohort OOF training rows."""
    def __init__(self, cases, lengths, pairs, model, knobs, validation=None):
        self.cases, self.lengths, self.pairs = cases, lengths, pairs
        self.model, self.knobs = model, knobs
        self.priors = (model.default_output_tokens, model.p90_output_tokens)
        self.sessions = frozenset(c.conversation_id for c in cases)
        self.saved = ({family: OOFLengthPrior(validation, family, cases) for family in FAMILIES}
                      if validation is not None else None)
        if validation is not None and tuple(validation["global_priors"]) != self.priors:
            raise ValueError("OOF length validation and frozen cost priors differ")
        self.cache = {}
        self.fit_audit = []

    def rows(self, family, training_sessions, testing_sessions):
        training_sessions, testing_sessions = frozenset(training_sessions), frozenset(testing_sessions)
        if training_sessions & testing_sessions:
            raise ValueError("length training and testing sessions overlap")
        key = (family, tuple(sorted(training_sessions)), tuple(sorted(testing_sessions)))
        if key in self.cache:
            return self.cache[key]
        testing = [c for c in self.cases if c.conversation_id in testing_sessions]
        # Only the final outer held-out rows may reuse the prior validation. A
        # training row from that report could depend on the current held-out H.
        outer = training_sessions | testing_sessions == self.sessions and len(testing_sessions) == 1
        if self.saved is not None and outer:
            adapter = self.saved[family]
            source = "saved outer-session OOF; never reused for gain training"
        else:
            predictions = predict_lengths([c for c in self.lengths if c.session_id in training_sessions],
                [c for c in self.lengths if c.session_id in testing_sessions], family, self.priors)
            adapter = SavedLengthPrior(predictions, testing, family)
            source = "refit using only listed training sessions"
        self.fit_audit.append({"family": family, "training_sessions": sorted(training_sessions),
                              "testing_sessions": sorted(testing_sessions), "source": source})
        self.model.output_length_model = adapter
        by_case = {c.case_id: c for c in testing}
        result = []
        for pair in self.pairs:
            if pair["session_id"] not in testing_sessions:
                continue
            result.append({k: v for k, v in pair.items() if k != "quote"} | prediction_features(
                by_case[pair["case_id"]], pair["quote"], self.model, self.knobs))
        self.cache[key] = result
        return result

    def crossfit(self, family, sessions):
        sessions = frozenset(sessions)
        return [row for held in sorted(sessions)
                for row in self.rows(family, sessions-{held}, {held})]


def cells(rows):
    """Collapse repetitions before fitting; cases and sessions retain equal weight."""
    groups = defaultdict(list)
    for row in rows:
        if row["eligible"]:
            groups[row["session_id"], row["case_id"], row["cache_state"]].append(row)
    result = []
    for (session, case, state), values in sorted(groups.items()):
        result.append({"session_id": session, "case_id": case, "cache_state": state,
            "features": {name: mean(r["features"][name] for r in values) for name in FEATURE_NAMES},
            "baseline_gain_ms": mean(r["baseline_gain_ms"] for r in values),
            "actual_gain_ms": mean(r["actual_gain_ms"] for r in values), "repetitions": len(values)})
    cases_by_session, states_by_case = defaultdict(set), Counter()
    for row in result:
        cases_by_session[row["session_id"]].add(row["case_id"])
        states_by_case[row["case_id"]] += 1
    for row in result:
        row["weight"] = 1 / (len(cases_by_session) * len(cases_by_session[row["session_id"]]) * states_by_case[row["case_id"]])
    return result


def fit_gain(rows, alpha):
    import numpy as np
    training = cells(rows)
    if len({r["session_id"] for r in training}) < 2:
        raise ValueError("gain ridge needs at least two supported training sessions")
    x = np.asarray([[r["features"][name] for name in FEATURE_NAMES] for r in training])
    y = np.asarray([r["actual_gain_ms"]-r["baseline_gain_ms"] for r in training])
    w = np.asarray([r["weight"] for r in training])
    centers = (x*w[:, None]).sum(axis=0)
    scales = np.sqrt((((x-centers)**2)*w[:, None]).sum(axis=0))
    scales = np.where(scales > 0, scales, 1.)
    z, intercept = (x-centers)/scales, float((w*y).sum())
    coefficients = np.linalg.solve(z.T @ (w[:, None]*z) + alpha*np.eye(len(FEATURE_NAMES)),
                                   z.T @ (w*(y-intercept)))
    return {"kind": "offline_gain_residual_ridge_v1", "feature_names": list(FEATURE_NAMES),
        "alpha": alpha, "centers": centers.tolist(), "scales": scales.tolist(),
        "coefficients": coefficients.tolist(), "intercept": intercept,
        "training_sessions": sorted({r["session_id"] for r in training}),
        "training_cases": len({r["case_id"] for r in training}), "training_cells": len(training),
        "target": "actual APPEND minus REBUILD server_elapsed_ms minus frozen predicted gain",
        "weighting": "equal sessions, then cases, then supported cache states; repetitions averaged"}


def predict_gain(row, spec):
    return row["baseline_gain_ms"] + spec["intercept"] + sum(
        coefficient*(row["features"][name]-center)/scale
        for name, coefficient, center, scale in zip(spec["feature_names"], spec["coefficients"], spec["centers"], spec["scales"]))


def one_sided_threshold(rows, quantile):
    """Training inner OOF optimism, case maximum over mean cache-state residuals."""
    groups = defaultdict(list)
    for row in rows:
        if row["eligible"]:
            groups[row["case_id"], row["cache_state"]].append(row["gain_ms"]-row["actual_gain_ms"])
    worst = defaultdict(float)
    for (case, _), values in groups.items():
        worst[case] = max(worst[case], mean(values), 0.)
    return {"quantile": quantile, "threshold_ms": percentile(list(worst.values()), quantile) if quantile else 0.,
            "cases": len(worst), "cells": len(groups), "definition": "empirical positive gain optimism; case maximum across mean state residuals",
            "confidence_guarantee": False}


def decide(row, gain_ms, threshold_ms=0., gate="configured"):
    if gate not in GATES:
        raise ValueError("unknown matched gate ablation")
    margin = 0. if gate in {"no_margin", "no_margin_or_stability"} else (row["margin_ms"] or 0.)
    stable = row["stable"] or gate in {"no_stability", "no_margin_or_stability"}
    if row["eligible"]:
        action = "REBUILD" if stable and gain_ms > margin+threshold_ms else "APPEND"
        reason = "gain_saving" if action == "REBUILD" else "gain_gate"
    else:
        action, reason = row["baseline_action"], row["baseline_reason"]
    actual = row["actual_gain_ms"]
    return {**row, "predicted_gain_ms": gain_ms, "one_sided_threshold_ms": threshold_ms,
        "effective_margin_ms": margin+threshold_ms, "gate": gate, "selected_action": action, "reason": reason,
        "regret_ms": max(0., actual if action == "APPEND" else -actual),
        "false_rebuild_harm_ms": max(0., -actual) if action == "REBUILD" else 0.,
        "missed_rebuild_gain_ms": max(0., actual) if action == "APPEND" else 0.}


def summarize(rows):
    by_session = defaultdict(list)
    for row in rows:
        by_session[row["session_id"]].append(row)
    def metrics(values):
        return {name: mean(row[name] for row in values) for name in
                ("regret_ms", "false_rebuild_harm_ms", "missed_rebuild_gain_ms")}
    sessions = {sid: metrics(values) | {"pairs": len(values),
        "selected_actions": dict(Counter(r["selected_action"] for r in values))} for sid, values in sorted(by_session.items())}
    supported = [r for r in rows if r["eligible"]]
    return {"pairs": len(rows), "cases": len({r["case_id"] for r in rows}), "sessions": sessions,
        "equal_session_mean_regret_ms": mean(s["regret_ms"] for s in sessions.values()),
        "equal_session_false_rebuild_harm_ms": mean(s["false_rebuild_harm_ms"] for s in sessions.values()),
        "equal_session_missed_rebuild_gain_ms": mean(s["missed_rebuild_gain_ms"] for s in sessions.values()),
        "selected_actions": dict(Counter(r["selected_action"] for r in rows)),
        "reasons": dict(Counter(r["reason"] for r in rows)), "supported_pairs": len(supported),
        "supported_gain_mae_ms": mean(abs(r["predicted_gain_ms"]-r["actual_gain_ms"]) for r in supported) if supported else None,
        "supported_sign_accuracy": mean((r["predicted_gain_ms"] > 0) == (r["actual_gain_ms"] > 0) for r in supported) if supported else None}


def fit_outer(features, family, heldout, *, alphas=ALPHAS, quantiles=QUANTILES):
    training_sessions = features.sessions-{heldout}
    inner = {alpha: [] for alpha in alphas}
    audits = []
    for held in sorted(training_sessions):
        fitting = training_sessions-{held}
        train_rows = features.crossfit(family, fitting)
        valid_rows = features.rows(family, fitting, {held})
        audits.append({"heldout_session": held, "gain_training_sessions": sorted(fitting),
                       "length_training": "crossfit only inside gain_training_sessions"})
        for alpha in alphas:
            spec = fit_gain(train_rows, alpha)
            inner[alpha].extend({**row, "gain_ms": predict_gain(row, spec)} for row in valid_rows)
    search = []
    for alpha in alphas:
        for quantile in quantiles:
            uncertainty = one_sided_threshold(inner[alpha], quantile)
            score = summarize([decide(row, row["gain_ms"], uncertainty["threshold_ms"]) for row in inner[alpha]])
            search.append({"alpha": alpha, **uncertainty,
                "regret_ms": score["equal_session_mean_regret_ms"],
                "false_rebuild_harm_ms": score["equal_session_false_rebuild_harm_ms"]})
    def best(values):
        return min(values, key=lambda p: (p["regret_ms"], p["false_rebuild_harm_ms"], -p["alpha"], -p["quantile"]))
    choices = {"ridge": best([p for p in search if p["quantile"] == 0.]), "one_sided": best(search)}
    final_train = features.crossfit(family, training_sessions)
    testing = features.rows(family, training_sessions, {heldout})
    models = {kind: fit_gain(final_train, chosen["alpha"]) for kind, chosen in choices.items()}
    records = {}
    for kind in ("baseline", "ridge", "one_sided"):
        for gate in GATES:
            records[f"{family}-{kind}-{gate}"] = [decide(row,
                row["baseline_gain_ms"] if kind == "baseline" else predict_gain(row, models[kind]),
                0. if kind == "baseline" else choices[kind]["threshold_ms"], gate) for row in testing]
    return records, {"family": family, "heldout_session": heldout,
        "training_sessions": sorted(training_sessions), "inner_folds": audits,
        "selection_objective": "equal-session observed regret with original margin/stability; training inner validation only",
        "search": search, "chosen": choices, "models": models}


def fit_exploration(features, *, families=FAMILIES, alphas=ALPHAS, quantiles=QUANTILES):
    if len(features.sessions) < 5:
        raise ValueError("nested gain/length exploration needs at least five sessions")
    if not alphas or min(alphas) <= 0 or 0. not in quantiles or any(not 0 <= q <= 1 for q in quantiles):
        raise ValueError("use positive ridge alphas and quantiles including the zero-threshold baseline")
    records, folds = defaultdict(list), []
    for family in families:
        for held in sorted(features.sessions):
            values, fold = fit_outer(features, family, held, alphas=alphas, quantiles=quantiles)
            folds.append(fold)
            for name, rows in values.items():
                records[name].extend(rows)
    summaries = {name: summarize(rows) for name, rows in records.items()}
    exemplar = next(iter(records.values()))
    fixed = {}
    for action in ("APPEND", "REBUILD"):
        values = [decide({**r, "eligible": False, "baseline_action": action, "baseline_reason": "fixed"}, 0.) for r in exemplar]
        fixed[action] = summarize(values)
    stable_control = []
    for row in exemplar:
        action = ("REBUILD" if row["stable"] else "APPEND") if row["eligible"] else row["baseline_action"]
        stable_control.append(decide({**row, "eligible": False, "baseline_action": action,
                                      "baseline_reason": "stable_supported_rebuild_control"}, 0.))
    return records, {"schema_version": 1,
        "scope": "exploratory development nested session LOSO, not independent final test",
        "new_engine_requests": 0, "quality_evaluated": False, "online_policy_changed": False,
        "feature_names": list(FEATURE_NAMES), "current_outcomes_are_features": False,
        "target": "server_elapsed_ms APPEND minus REBUILD; actual measurements used only as labels/evaluation",
        "prior_scope": "shared 433/512-style frozen artifact prior reused; OOF isolates new residual fits, not original prior construction",
        "one_sided_scope": "training inner OOF empirical optimism threshold; no conformal or population guarantee",
        "gate_scope": "original base-cost margin/stability preserved; matched ablations reuse the same selected fits/thresholds",
        "alpha_grid": list(alphas), "quantile_grid": list(quantiles), "folds": folds,
        "length_fit_audit": features.fit_audit, "variants": summaries, "fixed_actions": fixed,
        "stable_supported_rebuild_control": summarize(stable_control)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cases", "pairs", "validation", "config", "cost-artifact", "output"):
        parser.add_argument("--"+name, required=True)
    args = parser.parse_args(argv)
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("choose a new gain exploration directory")
    paths = {name: Path(getattr(args, name)) for name in ("cases", "pairs", "validation", "config", "cost_artifact")}
    paths["pairs_manifest"] = paths["pairs"].with_suffix(".manifest.json")
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    identities = {name: sha256(value).hexdigest() for name, value in payloads.items()}
    cases, frozen = load_frozen(args.cases)
    manifest, validation = json.loads(payloads["pairs_manifest"]), json.loads(payloads["validation"])
    if (manifest.get("split") != "dev" or manifest.get("status") != "complete"
            or manifest["frozen"]["cases_sha256"] != frozen["cases_sha256"]
            or validation.get("provenance", {}).get("cases", {}).get("sha256") != identities["cases"]
            or validation.get("provenance", {}).get("pairs", {}).get("sha256") != identities["pairs"]):
        raise ValueError("gain fit requires matching frozen dev cases, completed pairs and length OOF source")
    config = load_config(args.config)
    if config.cost_model.output_length_artifact or config.planner.strategy != "cost":
        raise ValueError("gain replay requires the cost planner and no full-fit length artifact")
    config = replace(config, cost_model=replace(config.cost_model, uncertainty_mode="conditional"))
    model, knobs = _model_and_knobs(config, cases, args.cost_artifact)
    pair_rows = [json.loads(line) for line in payloads["pairs"].decode().splitlines() if line]
    lengths, failures = length_fit.build_cases([json.loads(line) for line in payloads["cases"].decode().splitlines() if line], pair_rows)
    if failures:
        raise ValueError("complete paired measurements before gain fitting")
    pairs = prepared_pairs(cases, pair_rows, model, knobs, manifest["engine"])
    features = NestedFeatures(cases, lengths, pairs, model, knobs, validation)
    records, report = fit_exploration(features)
    report.update(provenance={name: {"path": str(path.resolve()), "sha256": identities[name]} for name, path in paths.items()},
                  config=asdict(config), frozen=frozen, prediction_pressure_concurrency=1, prediction_control_ms=0)
    target.mkdir(parents=True)
    for name, rows in records.items():
        (target/f"decision-{name}.jsonl").write_text("".join(json.dumps(row, allow_nan=False)+"\n" for row in rows))
    (target/"validation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(target), "configured_regret_ms": {name: value["equal_session_mean_regret_ms"]
        for name, value in report["variants"].items() if name.endswith("-configured")}}, indent=2))


if __name__ == "__main__":
    main()
