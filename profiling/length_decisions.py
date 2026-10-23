"""Replay the real planner with held-out per-case length predictions, offline.

This accepts session-LOSO validation records, never a full-fit length artifact.
Only saved means, train-side P90/errors and range flags enter prediction. Actual
pair outputs and times remain evaluation outcomes passed to analyze_pairs.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path

from controller.config import load_config
from controller.output_length import OutputLengthPrediction
from controller.serialization import dumps
from profiling.decision_pairs import _model_and_knobs, analyze_pairs, load_frozen, rescore_pairs


FAMILIES = ("shared_history", "action_only", "candidate", "paired_delta")


class OOFLengthPrior:
    """One startup binding from frozen candidate IDs to their held-out predictions."""

    def __init__(self, validation, family, cases):
        if validation.get("scope") != "exploratory_development_session_LOSO_not_final_test":
            raise ValueError("length decision replay requires a session-LOSO validation report, not a full-fit artifact")
        if family not in FAMILIES or family not in validation.get("models", {}):
            raise ValueError("missing OOF model family")
        report = validation["models"][family]
        rows = report.get("oof_predictions", ())
        by_case = {row["case_id"]: row for row in rows}
        if len(by_case) != len(rows) or set(by_case) != {case.case_id for case in cases}:
            raise ValueError("OOF cases must exactly cover the frozen cases once")
        sessions = {case.conversation_id for case in cases}
        folds = {fold["heldout_session"]: fold for fold in report.get("folds", ())}
        if len(folds) != len(report.get("folds", ())) or set(folds) != sessions:
            raise ValueError("OOF folds must cover every frozen session once")
        for heldout, fold in folds.items():
            expected = "frozen shared prior" if family == "shared_history" else "training-session inner OOF only"
            if (set(fold["training_sessions"]) != sessions - {heldout}
                    or fold.get("uncertainty_scope") != expected):
                raise ValueError("OOF means/P90/errors must come from the held-out session's training folds only")
        self.predictions = {}
        for case in cases:
            row = by_case[case.case_id]
            fields = ("predicted_tokens", "p90_tokens", "error_tokens", "within_training_range")
            if (row.get("session_id") != case.conversation_id
                    or any(not isinstance(row.get(key), dict) or set(row[key]) != {"APPEND", "REBUILD"} for key in fields)):
                raise ValueError("OOF prediction schema requires per-arm mean/P90/error/range and the matching session")
            for candidate in case.candidates:
                action = candidate.action
                mean, p90 = row["predicted_tokens"][action], row["p90_tokens"][action]
                error, in_range = row["error_tokens"][action], row["within_training_range"][action]
                if (type(mean) is not int or type(p90) is not int
                        or not 1 <= mean <= case.max_new_tokens or not 1 <= p90 <= case.max_new_tokens
                        or type(in_range) is not bool
                        or (error is None and family != "shared_history")
                        or (error is not None and (not math.isfinite(error) or error < 0))):
                    raise ValueError("invalid OOF length prediction values or frozen output budget")
                if candidate.candidate_id in self.predictions:
                    raise ValueError("frozen candidate IDs must be unique across OOF cases")
                kind = family + ":oof" + ("" if in_range else ":shared_range_fallback")
                self.predictions[candidate.candidate_id] = OutputLengthPrediction(mean, p90, error, kind, in_range)
        self.model_kind = family + ":oof"
        self.metadata = {"training_scope": "heldout_session_predictions_only",
                         "uncertainty_scope": "training-session inner OOF only; shared prior remains frozen"}

    def predict_prior(self, candidate, session, max_new_tokens, shared_prior):
        # Inputs and budgets were bound once above; no answers, hashes or feature
        # fitting are consulted at this per-candidate lookup.
        return self.predictions[candidate.candidate_id]


def evaluate_decisions(cases, pair_rows, validation, config, cost_artifact, engine_info,
                       *, uncertainty_modes=("global", "conditional")):
    if config.cost_model.output_length_artifact:
        raise ValueError("OOF replay cannot mix in a configured full-fit output_length_artifact")
    adapters = {family: OOFLengthPrior(validation, family, cases) for family in FAMILIES}
    records, summaries = {}, {}
    for mode in uncertainty_modes:
        configured = replace(config, cost_model=replace(config.cost_model, uncertainty_mode=mode))
        for family, adapter in adapters.items():
            model, knobs = _model_and_knobs(configured, cases, cost_artifact)
            if (model.default_output_tokens, model.p90_output_tokens) != tuple(validation["global_priors"]):
                raise ValueError("cost artifact shared priors differ from those used in OOF length validation")
            model.output_length_model = adapter
            variant = f"{mode}-{family}"
            rows = rescore_pairs(pair_rows, cases, model, knobs, engine_info)
            for row in rows:
                row["length_oof"] = {"family": family, "heldout_session": row["conversation_id"],
                                     "scope": "outer held-out mean, training-only inner OOF P90/error"}
            records[variant], summaries[variant] = rows, analyze_pairs(rows)
    return records, summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cost-artifact", required=True)
    parser.add_argument("--output", required=True, help="new directory for decision variants and combined summary")
    parser.add_argument("--uncertainty-modes", nargs="+", choices=("global", "conditional"), default=("global", "conditional"))
    args = parser.parse_args(argv)
    target = Path(args.output)
    if target.exists():
        raise FileExistsError("choose a new decision replay directory")
    cases, frozen = load_frozen(args.cases)
    paths = {"pairs": Path(args.pairs), "validation": Path(args.validation), "config": Path(args.config),
             "cost_artifact": Path(args.cost_artifact), "pairs_manifest": Path(args.pairs).with_suffix(".manifest.json")}
    payloads = {key: path.read_bytes() for key, path in paths.items()}
    identities = {key: sha256(value).hexdigest() for key, value in payloads.items()}
    validation, manifest = json.loads(payloads["validation"]), json.loads(payloads["pairs_manifest"])
    provenance = validation.get("provenance", {})
    if (provenance.get("cases", {}).get("sha256") != frozen["cases_sha256"]
            or provenance.get("pairs", {}).get("sha256") != identities["pairs"]
            or manifest.get("split") != "dev" or manifest.get("status") not in {"complete", "complete_with_failures"}
            or manifest["frozen"]["cases_sha256"] != frozen["cases_sha256"]):
        raise ValueError("OOF validation, measured pairs and frozen dev cases must name the same source observations")
    rows = [json.loads(line) for line in payloads["pairs"].decode().splitlines() if line]
    records, summaries = evaluate_decisions(cases, rows, validation, load_config(args.config), args.cost_artifact,
                                            manifest["engine"], uncertainty_modes=tuple(dict.fromkeys(args.uncertainty_modes)))
    target.mkdir(parents=True)
    for variant, values in records.items():
        (target/f"decision-{variant}.jsonl").write_text("".join(dumps(row)+"\n" for row in values))
    report = {"schema_version": 1, "scope": "exploratory development session-LOSO decision replay, not independent final test",
        "new_generations": 0, "new_engine_requests": 0, "full_fit_length_artifacts_used": False,
        "current_outputs_read_for_prediction": False, "quality_evaluated": False,
        "prior_scope": validation.get("prior_scope"),
        "uncertainty_scope": "per-case outer held-out means; P90/errors from inner training-session OOF; frozen shared baseline",
        "pressure_concurrency": 1, "prediction_control_ms": 0,
        "provenance": {key: {"path": str(path.resolve()), "sha256": identities[key]} for key, path in paths.items()},
        "frozen": frozen, "length_oof_metrics": {key: value["summary"] for key, value in validation["models"].items()},
        "variants": summaries}
    (target/"summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"output": str(target), "variants": {name: {key: value[key] for key in
        ("completed_pairs", "calibration_coverage", "plan_reasons", "sign_accuracy_on_calibrated", "equal_session_mean_regret_ms")}
        for name, value in summaries.items()}}, indent=2))


if __name__ == "__main__":
    main()
