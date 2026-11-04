"""Fixed-window Native/TTL15/TTL60/TTL180 pairs with split isolation."""
import argparse
import asyncio
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

from controller.prefix_retention import EmpiricalRetentionModel
from controller.reading import READ_SYSTEM
from profiling.retention_coverage import audit_coverage
from profiling.retention_trace import choose_windows, trace_generations, verify_trace, window_trace, write_json


async def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    capture_stages = []
    trace_paths = {}
    for split, count in (("train", args.train_capture_turns), ("dev", args.dev_capture_turns)):
        state_db, records = getattr(args, split+"_state_db"), getattr(args, split+"_records")
        reader_traces = getattr(args, split+"_reader_traces")
        if state_db or reader_traces:
            directory = output/(split+"-current-capture")
            source = (["capture-reader-traces", "--reader-traces", reader_traces] if reader_traces else
                      ["capture", "--state-db", state_db, "--records", records])
            command = [args.python or sys.executable, "-m", "profiling.retention_trace", *source,
                "--config", args.config, "--base-url", args.base_url, "--gpu-device", str(args.gpu_device),
                "--read-packing", "source_windows", "--max-capture-turns", str(count),
                "--output-dir", str(directory)]
            for option in ("study_config", "resource_gpu_devices", "resource_process_roots"):
                if getattr(args, option):
                    command += ["--"+option.replace("_", "-"), getattr(args, option)]
            with (output/(split+"-capture.log")).open("wb") as log:
                process = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=asyncio.subprocess.STDOUT)
                code = await process.wait()
            capture_stages.append(dict(split=split, command=command, returncode=code,
                input_selection="first preregistered source attempts, not replaced after completed failure",
                capture_entry=str(directory/"capture-entry.json")))
            write_json(output/"capture-stages.json", capture_stages)
            if code:
                raise RuntimeError("capture stopped with unresolved infrastructure; inspect "+str(directory))
            trace_paths[split] = str(directory/"capture-trace.json")
        else:
            trace_paths[split] = getattr(args, split+"_trace")
    traces = {split: json.loads(Path(path).read_text()) for split, path in
              trace_paths.items()}
    reports = {}
    protocol_scopes = {}
    for split, trace in traces.items():
        verify_trace(trace)
        if not trace["training_eligible"] or any(turn["split"] != split for turn in trace["turns"]):
            raise ValueError("paired value fitting needs separate actual train/development traces")
        reports[split] = audit_coverage(trace, horizons=(args.horizon_seconds,), page_size=args.page_size,
            max_candidates=2, max_prefixes_per_decision=args.max_prefixes_per_decision,
            history_steps=args.history_steps, max_replay_steps=args.max_replay_steps, seed=args.seed)
        write_json(output/(split+"-coverage.json"), reports[split])
        prompts = {event.get("system_prompt") for turn in trace["turns"]
                   for event in turn["events"] if event["kind"] == "READ"}
        protocol_scopes[split] = dict(
            read_system_prompt_sha256=sorted(hashlib.sha256(prompt.encode()).hexdigest()
                for prompt in prompts if prompt is not None),
            matches_current_read_prompt=prompts == {READ_SYSTEM})
    horizon = args.horizon_seconds
    if horizon < max(180, *args.ttl_seconds):
        raise ValueError("common observation window must cover 180 seconds and all TTL arms")
    selected = {split: choose_windows(trace, horizon_seconds=horizon, max_candidates=2,
        max_prefixes_per_decision=args.max_prefixes_per_decision, history_steps=args.history_steps,
        max_replay_steps=args.max_replay_steps, seed=args.seed)[0] for split, trace in traces.items()}
    calls = {split: sum((1+len(args.ttl_seconds))*len(trace_generations(window_trace(traces[split], item)))
        for item in values) for split, values in selected.items()}
    plan = dict(scope="actual source-first fixed-input same-window multi-TTL C costs", maximum_windows=4,
        choice_basis="same preregistered observation window and request DAG for Native and every TTL; no shorter-window fallback or outcome selection",
        observation_seconds=horizon, ttl_seconds=list(args.ttl_seconds), seed=args.seed,
        train_trace=str(Path(trace_paths["train"]).resolve()), dev_trace=str(Path(trace_paths["dev"]).resolve()),
        capture_stages=capture_stages, protocol_scopes=protocol_scopes,
        current_prompt_joint_qualification=all(scope["matches_current_read_prompt"]
                                               for scope in protocol_scopes.values()), stages=[])
    if any(len(values) != 2 for values in selected.values()) or sum(calls.values()) > args.max_total_generations:
        plan.update(status="no_common_observable_budgeted_window", planned_calls=calls, value_asset=None)
        write_json(output/"paired-workflow.json", plan)
        print(json.dumps({"status": plan["status"], "pairs": 0}))
        return plan
    plan.update(status="running", horizon_seconds=horizon, planned_calls=calls,
        source_windows={split: [{key: value for key, value in item.items() if key != "turns"}
                               for item in values] for split, values in selected.items()})
    write_json(output/"paired-workflow.json", plan)
    for split, path in trace_paths.items():
        if args.deadline and datetime.now().astimezone().timestamp() >= datetime.fromisoformat(args.deadline).timestamp():
            plan["stages"].append({"split": split, "status": "not_started_deadline"})
            break
        command = [args.python or sys.executable, "-m", "profiling.retention_trace", "replay",
            "--trace", str(Path(path).resolve()), "--config", str(Path(args.config).resolve()),
            "--base-url", args.base_url, "--gpu-device", str(args.gpu_device),
            "--output-dir", str(output/split), "--horizon-seconds", str(horizon),
            "--ttl-seconds", ",".join(str(ttl) for ttl in args.ttl_seconds),
            "--max-candidates", "2", "--max-prefixes-per-decision", str(args.max_prefixes_per_decision),
            "--history-steps", str(args.history_steps), "--max-replay-steps", str(args.max_replay_steps),
            "--max-total-generations", str(calls[split]), "--residency-interval", ".5", "--seed", str(args.seed)]
        with (output/(split+"-replay.log")).open("wb") as log:
            process = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=asyncio.subprocess.STDOUT)
            code = await process.wait()
        plan["stages"].append({"split": split, "returncode": code,
                               "status": "completed" if code == 0 else "stopped_before_more_cold"})
        write_json(output/"paired-workflow.json", plan)
        if code:
            plan["status"] = "replay_error; inspect exact native handles before continuing"
            break
    train_rows = output/"train"/"paired-rows.jsonl"
    if train_rows.exists():
        command = [args.python or sys.executable, "-m", "profiling.retention_trace", "fit",
            "--records", str(train_rows), "--split", "train", "--output", str(output/"value-model.json")]
        with (output/"fit.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=asyncio.subprocess.STDOUT)
            code = await process.wait()
        plan["value_asset"] = str(output/"value-model.json") if code == 0 else None
    asset = output/"value-model.json"
    if asset.exists():
        model_data = json.loads(asset.read_text())
        model_data["measurement_protocol_scope"] = protocol_scopes
        model_data["current_prompt_joint_qualification"] = plan["current_prompt_joint_qualification"]
        write_json(asset, model_data)
    dev_rows = output/"dev"/"paired-rows.jsonl"
    if asset.exists() and dev_rows.exists():
        model = EmpiricalRetentionModel.from_dict(json.loads(asset.read_text()))
        validation = []
        for line in dev_rows.read_text().splitlines():
            row = json.loads(line)
            try:
                predicted = model.predict(row["features"], row["cache_scope_hash"], row["horizon_seconds"])
            except KeyError:
                predicted = None
            validation.append(dict(event_id=row["event_id"], calibrated=predicted is not None,
                predicted_target_saved_ms=predicted.target_saved_ms if predicted else None,
                predicted_retention_control_ms=predicted.retention_control_ms if predicted else None,
                predicted_collateral_saved_ms=predicted.expected_collateral_saved_ms if predicted else None,
                predicted_policy_value_ms=predicted.saved_ms if predicted else None,
                policy_value_scope="next READ target minus control; before residency prices and protected-event deduction",
                predicted_full_window_saved_ms=predicted.full_window_saved_ms if predicted else None,
                actual_full_window_saved_ms=row["full_window_cost_difference_ms"],
                semantic_quality_label=None, split="dev"))
        write_json(output/"dev-validation.json", {"fit_split": "train", "evaluation_split": "dev",
            "comparison_scope": "single-prefix complete response windows; no protected-set externality estimate",
            "rows": validation})
    if plan["status"] == "running":
        plan["status"] = "completed_with_actual_signed_costs"
    write_json(output/"paired-workflow.json", plan)
    print(json.dumps({"status": plan["status"], "observation_seconds": horizon, "planned_pairs": 4*len(args.ttl_seconds),
                      "value_asset": plan.get("value_asset")}))
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for split, count in (("train", 7), ("dev", 14)):
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--"+split+"-trace")
        source.add_argument("--"+split+"-state-db")
        source.add_argument("--"+split+"-reader-traces")
        parser.add_argument("--"+split+"-records")
        parser.add_argument("--"+split+"-capture-turns", type=int, default=count)
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--gpu-device", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python")
    parser.add_argument("--study-config")
    parser.add_argument("--resource-gpu-devices")
    parser.add_argument("--resource-process-roots")
    parser.add_argument("--horizon-seconds", type=float, default=180)
    parser.add_argument("--ttl-seconds", type=lambda value: tuple(map(float, value.split(","))), default=(15, 60, 180))
    parser.add_argument("--page-size", type=int)
    parser.add_argument("--max-prefixes-per-decision", type=int, default=2)
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--max-replay-steps", type=int, default=48)
    parser.add_argument("--max-total-generations", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deadline")
    args = parser.parse_args(argv)
    for split in ("train", "dev"):
        if getattr(args, split+"_state_db") and not getattr(args, split+"_records"):
            parser.error("--"+split+"-state-db requires the actual --"+split+"-records partition ledger")
    return asyncio.run(run(args))


if __name__ == "__main__":
    main()
