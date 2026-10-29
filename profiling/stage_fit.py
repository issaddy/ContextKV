"""Offline stage lookup fit from bound server timings, never scheduler rank labels.

forward_first is the server forward-to-first-token interval. pre_forward is the
remainder of server TTFT, including admission/restore/waiting; it is not pure queue
or a CUDA duration. Empirical pre_forward cells cover homogeneous bursts only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from statistics import mean

from controller.cost_model import CostModel
from controller.sampling import bound_recipe
from profiling.common import load_measurements
from profiling.fit import measurement_provenance, output_priors
from profiling.prefill import PrefillCase

FORWARD_FEATURES = ("prefix_tokens", "suffix_tokens", "host_load_tokens", "concurrency")
WAITING_FEATURES = (*FORWARD_FEATURES, "requested_output_budget", "gpu_capacity_tokens")
TARGETS = ("forward_ms", "pre_forward_ms", "total_ttft_ms")


def quantile(values, probability):
    """Empirical nearest-rank quantile; no distributional coverage guarantee."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered))-1)] if ordered else None


def features(row, *, budget=False):
    keys = WAITING_FEATURES if budget else FORWARD_FEATURES
    return tuple(row[key] for key in keys)


def feature_cell(row):
    # Output budgets, all repetitions and batch members of the same forward
    # feature cell stay together. Budget variants cannot leak into heldout cells.
    return (*features(row), row["gpu_capacity_tokens"])


def load_bound_telemetry(telemetry_paths, measurements_path, manifests):
    """Join native request telemetry to its measured batches and source manifests."""
    if set(telemetry_paths) != set(manifests):
        raise ValueError("telemetry groups and supplied profile manifests must match")
    records, files = [], []
    for group, source in telemetry_paths.items():
        path = Path(source)
        content = path.read_bytes()
        files.append({"group": group, "path": str(path.resolve())})
        for line in content.decode("utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["event"] != "request":
                continue
            if (type(record.get("repetition")) is not int or record["repetition"] < 1
                    or type(record.get("warmup")) is not bool):
                raise ValueError("native request telemetry requires explicit repetition and warmup identity")
            result = record["result"]
            receipt = result["receipt"]
            if not result["generation_complete"] or not receipt["generation_complete"]:
                raise ValueError("native request telemetry contains incomplete generation")
            if result.get("timing_source") != "sglang_server_monotonic":
                raise ValueError("stage labels require native server-monotonic telemetry")
            records.append({"group": group, "run_id": record["run_id"],
                "repetition": record["repetition"], "case": record["case"],
                "index": record["index"], "warmup": record["warmup"],
                "server_ttft_ms": result.get("server_ttft_ms"),
                "forward_to_first_token_ms": result.get("forward_to_first_token_ms"),
                "output_tokens": len(result["output_ids"]), "input_hash": receipt["input_hash"],
                "actual_device_hit_tokens": receipt["actual_device_hit_tokens"],
                "actual_host_loaded_tokens": receipt["actual_host_loaded_tokens"],
                "actual_prefill_tokens": receipt["actual_prefill_tokens"]})
    return _bind_stage_rows(records, measurements_path, manifests, source_key="stage_telemetry",
                            source_info={"files": files})


def _bind_stage_rows(records, measurements_path, manifests, *, source_key, source_info):
    measurements = [asdict(row) for row in load_measurements(measurements_path)]
    provenance = measurement_provenance(measurements, measurements_path,
                                       profile_manifest_path=list(manifests.values()))
    manifest_rows = {group: json.loads(Path(path).read_text()) for group, path in manifests.items()}
    if any(row.get("source") != "hardware" or row.get("status") != "complete" for row in manifest_rows.values()):
        raise ValueError("stage fit requires completed hardware profile manifests")
    grouped = defaultdict(list)
    for row in measurements:
        if row["phase"] == "first_token":
            if row["source"] != "hardware" or row.get("timing_scope") != "server_request":
                raise ValueError("stage labels require hardware server-request TTFT measurements")
            grouped[row["run_id"]].append(row)
    if {row["group"] for row in records} != set(manifest_rows):
        raise ValueError("stage request groups and supplied profile manifests must match")
    positions = defaultdict(int)
    rows, batches = [], defaultdict(list)
    for record in records:
        manifest = manifest_rows[record["group"]]
        run_id = manifest["run_id"]
        if record["run_id"] != run_id:
            raise ValueError("native telemetry run_id differs from its profile manifest")
        if record["warmup"]:
            continue
        position = positions[run_id]
        if position >= len(grouped[run_id]):
            raise ValueError("telemetry contains more formal requests than original measurements")
        observed = grouped[run_id][position]
        positions[run_id] += 1
        if record["repetition"] != observed["repetition"]:
            raise ValueError("native telemetry repetition differs from original measurements")
        case = record["case"]
        if case not in manifest.get("cases",[]):
            raise ValueError("telemetry fixture is absent from its original profile manifest")
        fixture = PrefillCase(**case)
        prepared_device, prepared_host = fixture.prefix_split()
        if (fixture.cache_state == "mixed" and
                (record["actual_device_hit_tokens"], record["actual_host_loaded_tokens"]) != (prepared_device, prepared_host)):
            raise ValueError("mixed device/host split changed from its prepared fixture")
        prefix = record["actual_device_hit_tokens"] + record["actual_host_loaded_tokens"]
        expected = {"prompt_tokens": case["prompt_tokens"], "prefix_tokens": prefix,
                    "host_load_tokens": record["actual_host_loaded_tokens"], "concurrency": case["concurrency"],
                    "output_tokens": record["output_tokens"], "cache_state": case["cache_state"],
                    "elapsed_ms": record["server_ttft_ms"]}
        if any(observed.get(key) != value for key, value in expected.items()):
            raise ValueError("telemetry request order/labels differ from original first_token measurements")
        if not isinstance(observed.get("repetition"), int):
            raise ValueError("original measurements must preserve batch repetition")
        if prefix != case["prefix_tokens"] or record["actual_prefill_tokens"] != case["prompt_tokens"]-prefix:
            raise ValueError("execution prefix changed from the prepared fixture; cannot fit this controlled stage contract")
        values = (record["forward_to_first_token_ms"], record["server_ttft_ms"])
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values) or not 0 <= values[0] <= values[1]:
            raise ValueError("stage intervals must satisfy 0 <= forward <= server TTFT")
        row = {"run_id": run_id, "batch_id": f"{run_id}:{observed['repetition']}",
               "group": record["group"], "request_index": record["index"], "input_hash": record["input_hash"],
               "prefix_tokens": case["prefix_tokens"], "suffix_tokens": case["prompt_tokens"]-case["prefix_tokens"],
               "host_load_tokens": prepared_host,
               "actual_host_loaded_tokens": observed["host_load_tokens"],
               "prefix_tier_binding": "prepared_and_execution_exact" if fixture.cache_state == "mixed" else "prepared",
               "prepared_device_prefix_tokens": prepared_device,
               "concurrency": case["concurrency"], "requested_output_budget": case["output_tokens"],
               "gpu_capacity_tokens": manifest["pool"]["gpu_capacity_tokens"],
               "forward_ms": values[0], "pre_forward_ms": values[1]-values[0], "total_ttft_ms": values[1]}
        rows.append(row)
        batches[row["batch_id"]].append(row)
    if any(positions[run_id] != len(values) for run_id, values in grouped.items()):
        raise ValueError("telemetry does not cover every formal first_token measurement")
    for batch in batches.values():
        concurrency = batch[0]["concurrency"]
        if (len(batch) != concurrency or {r["request_index"] for r in batch} != set(range(concurrency))
                or len({features(r, budget=True) for r in batch}) != 1):
            raise ValueError("original batch must retain every member of one homogeneous fixture")
    capacities = {row["gpu_capacity_tokens"] for row in rows}
    if len(capacities) != 1:
        raise ValueError("fit one physical pool configuration at a time")
    provenance[source_key] = {**source_info,
        "warmup_requests_excluded": sum(bool(row["warmup"]) for row in records),
        "formal_requests": len(rows), "batches": len(batches),
        "prepared_vs_actual_host_load_changed_requests": sum(r["host_load_tokens"] != r["actual_host_loaded_tokens"] for r in rows),
        "groups": {group: {
            "prefix_tier_binding": sorted({r["prefix_tier_binding"] for r in rows if r["group"] == group}),
            "formal_requests": sum(r["group"] == group for r in rows),
            "prepared_vs_actual_host_load_changed_requests": sum(r["host_load_tokens"] != r["actual_host_loaded_tokens"]
                                                                  for r in rows if r["group"] == group)}
            for group in manifest_rows},
        "batch_binding": "explicit_telemetry_run_id_repetition_and_original_first_token_order_and_exact_fields"}
    return rows, provenance


def fit_cells(rows, target, *, budget=False):
    groups = defaultdict(list)
    for row in rows:
        groups[features(row, budget=budget)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        batches = defaultdict(list)
        for row in group:
            batches[row["batch_id"]].append(row[target])
        values = [row[target] for row in group]
        batch_means = [mean(batch) for batch in batches.values()]
        result.append({"features": list(key), "mean_ms": mean(batch_means),
                       "p95_ms": quantile(values,.95), "p99_ms": quantile(values,.99), "max_ms": max(values),
                       "batches": len(batches), "requests": len(values)})
    return result


def lookup(cells, row, *, budget=False):
    """Reuse the production interpolation and total-input support semantics."""
    predictor = CostModel()
    target = features(row, budget=budget)
    value, _, supported = predictor._interpolate(target, [(tuple(p["features"]), p["mean_ms"], 0.0) for p in cells])
    totals = [sum(p["features"][:2]) for p in cells]
    supported = supported and min(totals) <= sum(target[:2]) <= max(totals)
    return value, supported


def grouped_folds(rows, folds=5, seed=42):
    keys = sorted({feature_cell(row) for row in rows})
    if folds < 2 or len(keys) < 2:
        return []
    random.Random(seed).shuffle(keys)
    bins = [set(keys[i::min(folds,len(keys))]) for i in range(min(folds,len(keys)))]
    result = []
    for heldout in bins:
        train = [row for row in rows if feature_cell(row) not in heldout]
        test = [row for row in rows if feature_cell(row) in heldout]
        if {r["batch_id"] for r in train} & {r["batch_id"] for r in test}:
            raise ValueError("a batch crosses feature-cell folds")
        result.append((train,test))
    return result


def error_metrics(rows):
    errors = [abs(row["predicted_ms"]-row["actual_ms"]) for row in rows]
    relative = [abs(row["predicted_ms"]-row["actual_ms"])/row["actual_ms"] for row in rows if row["actual_ms"]>0]
    return {"count":len(rows), "mae_ms":mean(errors) if errors else None,
            "mre":mean(relative) if relative else None, "absolute_error_p95_ms":quantile(errors,.95),
            "absolute_error_p99_ms":quantile(errors,.99), "max_absolute_error_ms":max(errors) if errors else None}


def summarize_errors(rows):
    supported = [row for row in rows if row["supported"]]
    grouped = {}
    for unit in ("batch_id", "feature_cell"):
        groups = defaultdict(list)
        for row in rows:
            key = tuple(row[unit]) if isinstance(row[unit],list) else row[unit]
            groups[key].append(row)
        means = [{"predicted_ms":mean(r["predicted_ms"] for r in group),
                  "actual_ms":mean(r["actual_ms"] for r in group), "supported":all(r["supported"] for r in group)}
                 for group in groups.values()]
        grouped[unit] = {**error_metrics(means), "supported_only":error_metrics([r for r in means if r["supported"]])}
    return {**error_metrics(rows), "supported_requests":len(supported),
            "coverage":len(supported)/len(rows) if rows else None,
            "supported_only":error_metrics(supported), "group_equal_weight":grouped}


def validate(rows, *, folds=5, seed=42):
    predictions = defaultdict(list)
    fold_records = []
    for index,(train,test) in enumerate(grouped_folds(rows,folds,seed)):
        forward = fit_cells(train,"forward_ms")
        waiting = fit_cells(train,"pre_forward_ms",budget=True)
        total = fit_cells(train,"total_ttft_ms")
        total_budget = fit_cells(train,"total_ttft_ms",budget=True)
        fold_records.append({"fold":index, "train_cells":len({feature_cell(r) for r in train}),
                             "heldout_cells":[list(k) for k in sorted({feature_cell(r) for r in test})],
                             "train_batches":len({r['batch_id'] for r in train}),"heldout_batches":len({r['batch_id'] for r in test})})
        for row in test:
            f,fs = lookup(forward,row)
            w,ws = lookup(waiting,row,budget=True)
            t,ts = lookup(total,row)
            b,bs = lookup(total_budget,row,budget=True)
            for name,target,point,supported in (
                ("forward_first","forward_ms",f,fs), ("pre_forward","pre_forward_ms",w,ws),
                ("stage_sum","total_ttft_ms",f+w,fs and ws),
                ("total_ttft_original_features","total_ttft_ms",t,ts),
                ("total_ttft_budget_features","total_ttft_ms",b,bs)):
                predictions[name].append({"fold":index,"run_id":row["run_id"],"batch_id":row["batch_id"],
                    "input_hash":row["input_hash"],"feature_cell":list(feature_cell(row)),
                    "features":list(features(row,budget=name in {"pre_forward","stage_sum","total_ttft_budget_features"})),
                    "predicted_ms":point,"actual_ms":row[target],"supported":supported})
    return {"protocol":"whole_forward_feature_cell_kfold_all_budgets_repetitions_batch_members_together",
            "seed":seed,"folds":fold_records,"sorting_rank_used_as_feature":False,
            "training_scope":"homogeneous_burst_only", "cross_configuration_validation":False,
            "metrics":{name:summarize_errors(values) for name,values in predictions.items()},
            "predictions":dict(predictions)}


def fit_artifact(rows, provenance, base_artifact, *, folds=5, seed=42, development_report=None):
    base = json.loads(Path(base_artifact).read_text())
    original_manifest = (base.get("metadata") or {}).get("profile_manifest") or {}
    manifest = provenance["profile_manifest"]
    for key in ("fingerprint","cache_config_hash"):
        old = original_manifest.get(key) if key=="fingerprint" else None
        old = old or original_manifest.get("engine",{}).get(key)
        new = manifest.get(key) if key=="fingerprint" else None
        new = new or manifest.get("engine",{}).get(key)
        if not old or old != new:
            raise ValueError("base decode/prior artifact must bind the same engine configuration")
    if bound_recipe(original_manifest.get("sampling_recipe")) != bound_recipe(manifest.get("sampling_recipe")):
        raise ValueError("base decode and new forward observations must bind the same sampling recipe")
    validation = validate(rows,folds=folds,seed=seed)
    forward = fit_cells(rows,"forward_ms")
    waiting = fit_cells(rows,"pre_forward_ms",budget=True)
    heldout = [r for r in validation["predictions"].get("forward_first",[]) if r["supported"]]
    errors = [abs(r["predicted_ms"]-r["actual_ms"]) for r in heldout]
    error = quantile(errors,.99)
    model = deepcopy(base)
    model["prefill_points"] = [{"prefix_tokens":cell["features"][0],"suffix_tokens":cell["features"][1],
        "host_load_tokens":cell["features"][2],"concurrency":cell["features"][3],"elapsed_ms":cell["mean_ms"],
        "error_ms":error if error is not None else cell["mean_ms"],"source":"hardware","scope":"forward_first"} for cell in forward]
    model["stage_model"] = {
        "version": 1, "waiting_calibration_scope": "homogeneous_burst_only",
        "pre_forward": {"features": list(WAITING_FEATURES),
            "cells": [{k: cell[k] for k in ("features", "mean_ms", "requests")}
                      for cell in waiting if cell["features"][3] == 1],
            "online_heterogeneous_queue_calibrated": False},
        "gpu_capacity_tokens": rows[0]["gpu_capacity_tokens"]}
    frozen_lengths = (base.get("stage_model") or {}).get("development_output_lengths")
    prior_metadata = {}
    if development_report:
        report = json.loads(Path(development_report).read_text())
        fingerprint = manifest.get("fingerprint") or manifest.get("engine", {}).get("fingerprint")
        if report.get("fingerprint") is not None and report["fingerprint"] != fingerprint:
            raise ValueError("development outputs and timing profile use different model fingerprints")
        turns = report.get("report", report).get("turns", [])
        completed = [row for row in turns if row.get("status") == "success"]
        lengths = [row["result"]["output_tokens"] for row in completed]
        mean, p90 = output_priors(report)
        model["settings"].update(default_output_tokens=mean, p90_output_tokens=p90)
        prior_metadata = {"output_prior_sampling": report.get("sampling_recipe"),
                          "output_prior_system_prompt": report.get("system_prompt"),
                          "output_prior_fingerprint": report.get("fingerprint")}
        model["stage_model"]["development_output_lengths"] = {
            "sampling_recipe": report.get("sampling_recipe"), "system_prompt": report.get("system_prompt"),
            "lengths": lengths, "source": {"path": str(Path(development_report).resolve())},
            "tasks": len(lengths), "sessions": len({row['conversation_id'] for row in completed}),
            "truncated_outputs": sum(bool(row['result'].get('truncated')) for row in completed)}
    elif frozen_lengths:
        model["stage_model"]["development_output_lengths"] = deepcopy(frozen_lengths)
    # The base artifact remains separately bound below. Its old stage source
    # must not appear to be an additional input to this fit.
    base_metadata = {key: value for key, value in base.get("metadata",{}).items()
                     if key not in {"stage_telemetry"}}
    model["metadata"] = {**base_metadata, **provenance, **prior_metadata,
        "prefill_estimator":"lookup", "validation":{"metrics":validation["metrics"]},
        "stage_fit":{"source":"hardware","formal_requests":len(rows),"batches":len({r['batch_id'] for r in rows}),
                     "forward_error_p99_ms":error,"oof_residuals":len(errors),
                     "base_artifact":{"path":str(Path(base_artifact).resolve())},
                     "decode_source":"base artifact",
                     "online_heterogeneous_waiting_validation":False}}
    # Runtime needs identity, sampling and page size. Full manifests, validation
    # predictions and unused comparison tables stay with the original run files.
    for key in ("profile_manifest", "profile_manifests"):
        profiles = model["metadata"].get(key)
        if not profiles:
            continue
        compact = [{name: value for name, value in profile.items() if name in
                    {"run_id", "source", "status", "fingerprint", "cache_config", "sampling_recipe"}}
                   | {"engine": {name: value for name, value in profile.get("engine", {}).items()
                                 if name in {"fingerprint", "cache_config_hash"}}}
                   for profile in (profiles if isinstance(profiles, list) else [profiles])]
        model["metadata"][key] = compact if isinstance(profiles, list) else compact[0]
    if len(model["metadata"].get("profile_manifests", [])) == 1:
        model["metadata"].pop("profile_manifests")
    return model, validation


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry",required=True,action="append",metavar="GROUP=PATH",
                             help="native SGLangProbe telemetry with explicit batch repetition")
    parser.add_argument("--measurements",required=True)
    parser.add_argument("--profile-manifest",required=True,action="append",metavar="GROUP=PATH")
    parser.add_argument("--base-artifact",required=True)
    parser.add_argument("--development-report",help="completed development outputs for the output-length prior")
    parser.add_argument("--output",required=True)
    parser.add_argument("--validation",required=True)
    parser.add_argument("--folds",type=int,default=5)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args(argv)
    manifests={}
    for item in args.profile_manifest:
        group,path=item.split("=",1)
        if group in manifests:
            parser.error("profile manifest group must be unique")
        manifests[group]=path
    paths=[Path(args.output),Path(args.validation)]
    if paths[0].resolve()==paths[1].resolve() or any(path.exists() for path in paths):
        parser.error("use distinct new artifact/validation paths; existing results are preserved")
    telemetry={}
    for item in args.telemetry:
        group,path=item.split("=",1)
        if group in telemetry:
            parser.error("telemetry group must be unique")
        telemetry[group]=path
    rows,provenance=load_bound_telemetry(telemetry,args.measurements,manifests)
    model,validation=fit_artifact(rows,provenance,args.base_artifact,folds=args.folds,seed=args.seed,
                                  development_report=args.development_report)
    for path,value in zip(paths,(model,validation)):
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x') as stream:
            json.dump(value,stream,indent=2,allow_nan=False)
    print(json.dumps({"artifact":str(paths[0]),"validation":str(paths[1]),"requests":len(rows),
                      "forward_cells":len(model['prefill_points']),"waiting_calibration_scope":"homogeneous_burst_only"}))


if __name__=="__main__":
    main()
