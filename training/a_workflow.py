"""Reuse an existing P1 policy to collect, review and prepare the next reader."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

from bench.study import merge_config
from training.reader_data import audit_reader_targets, file_digest
from training.reader_curriculum import prepare_stage_cohorts
from training.task_data import prepare_cohort_task_labels
from training.reader_stage_workflow import prepare_reader_stage, prepare_reader_phases
from training.train_evidence_policy import cohort_directory, require_policy_reader_binding, reader_binding


def stage(output, name, arguments, *, completed_artifact=None):
    logs = output/"logs"
    logs.mkdir(parents=True, exist_ok=True)
    invocation = time.time_ns()
    log = logs/f"{name}.{invocation}.log"
    started = time.monotonic()
    reused = completed_artifact is not None and completed_artifact.exists()
    returncode = 0
    if not reused:
        with log.open("w") as stream:
            result = subprocess.run(arguments, stdout=stream, stderr=subprocess.STDOUT, check=False)
        returncode = result.returncode
    receipt = {"stage": name, "arguments": arguments, "returncode": returncode,
        "status": "reused_existing_artifact" if reused else "completed" if returncode == 0 else "failed",
        "complete_ms": (time.monotonic()-started)*1000, "log": str(log) if not reused else None,
        "completed_artifact": str(completed_artifact) if completed_artifact is not None else None,
        "cost_scope": "orchestration wall time; phase GPU/PSS/token costs remain in each stage's original artifacts"}
    (logs/f"{name}.{invocation}.json").write_text(json.dumps(receipt, indent=2)+"\n")
    if returncode:
        raise subprocess.CalledProcessError(returncode, arguments)
    return receipt


def load_policy_config(path, asset):
    config = yaml.safe_load(Path(path).read_text())
    config = merge_config(yaml.safe_load(Path(config["base_study_config"]).read_text()), config)
    config["evidence_policy_training"]["output_asset"] = asset
    binding = reader_binding(config)
    require_policy_reader_binding(asset, reader_checkpoint=binding["reader_checkpoint"],
                                  reader_profile_id=binding["reader_profile_id"])
    return config


def data_workflow(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_policy_config(args.policy_config, args.policy_asset)
    effective = output/"existing_p1_config.yaml"
    effective.write_text(yaml.safe_dump(config, sort_keys=False))
    policy = json.loads((Path(args.policy_asset)/"support-model.json").read_text())
    rollout = cohort_directory(config)/"next_reader_rollout"/policy["policy_checkpoint"].split(":")[-1][:20]
    alternation = rollout/"alternation_manifest.json"
    receipts = [stage(output, "p1-own-p3", [sys.executable, "-m", "training.train_evidence_policy", "rollout",
        "--config", str(effective), "--limit", str(args.conversations)], completed_artifact=alternation)]
    paths = json.loads(alternation.read_text())
    mixed = yaml.safe_load(Path(paths["reader_mix_config"]).read_text())
    training = Path(mixed["reader_mix"]["output_dir"])/"reader_r2_training.yaml"
    receipts.append(stage(output, "new-p1-export-review-mix", [sys.executable, "-m", "training.reader_mix", "all",
        "--config", paths["reader_mix_config"], "--p1-config", paths["new_P1_review_config"],
        "--concurrency", str(args.review_concurrency)], completed_artifact=training))
    result = {"existing_policy_asset": args.policy_asset, "policy_checkpoint": policy["policy_checkpoint"],
        "policy_was_retrained": False, "actual_new_turns": paths["actual_turns"],
        "actual_new_conversations": paths["actual_conversations"], "stages": receipts,
        "reader_training_config": str(training.resolve()), "reader_training_config_sha256": file_digest(training),
        "single_gpu_command": f"CUDA_VISIBLE_DEVICES=6 {sys.executable} -m training.train_reader train --config {training}",
        "four_gpu_command": f"CUDA_VISIBLE_DEVICES=0,3,6,7 torchrun --standalone --nproc_per_node=4 -m training.train_reader train --config {training}",
        "execution": "data prepared only; training and service lifecycle are explicitly left to root"}
    (output/"workflow_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({name: result[name] for name in ("actual_new_turns", "actual_new_conversations",
        "reader_training_config", "four_gpu_command")}, ensure_ascii=False))


def comparison_workflow(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    comparison = yaml.safe_load(Path(args.comparison_config).read_text())
    comparison["policy_comparison"]["policies"]["P1"] = args.policy_asset
    effective = output/"comparison_config.yaml"
    effective.write_text(yaml.safe_dump(comparison, sort_keys=False))
    command = [sys.executable, "-m", "bench.evidence_policy_comparison", args.phase,
        "--config", str(effective), "--matrix", args.matrix, "--dataset", args.dataset, "--split", args.split,
        "--output", str(output/"comparison")]
    for flag, value in (("--records", args.records), ("--goal-policy", args.goal_policy),
                        ("--limit-states", args.limit_states), ("--limit-conversations", args.conversations)):
        if value is not None:
            command.extend((flag, str(value)))
    stage(output, "four-cells-and-own-p3", command)
    print(json.dumps({"comparison_manifest": str(output/"comparison"/"comparison_manifest.json")}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("data", "compare", "audit", "cohort", "prepare-labels", "reader-stage", "reader-prepare"))
    parser.add_argument("--policy-asset", default="models/study-support-r1-prototype22")
    parser.add_argument("--policy-config", default="configs/evidence_policy_r1.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", help="reader audit, cohort or task_cohort_labeling configuration")
    parser.add_argument("--conversations", type=int, default=2)
    parser.add_argument("--review-concurrency", type=int, default=8)
    parser.add_argument("--comparison-config", default="configs/evidence_policy_comparison.yaml")
    parser.add_argument("--phase", choices=("frozen", "p3", "both"), default="both")
    parser.add_argument("--matrix")
    parser.add_argument("--dataset", default="dynamic_goal_dev")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--records")
    parser.add_argument("--goal-policy")
    parser.add_argument("--limit-states", type=int)
    args = parser.parse_args(argv)
    if args.command == "compare" and args.matrix is None:
        parser.error("compare requires --matrix")
    if args.command in {"audit", "cohort", "prepare-labels", "reader-stage", "reader-prepare"}:
        if args.config is None:
            parser.error("audit/cohort/prepare-labels requires --config")
        config = yaml.safe_load(Path(args.config).read_text())
        if args.command == "audit":
            config["reader_training"]["data"]["output_dir"] = args.output
            result = audit_reader_targets(config)
        elif args.command in {"reader-stage", "reader-prepare"}:
            config["reader_stage"]["output_dir"] = args.output
            result = prepare_reader_stage(config) if args.command == "reader-stage" else prepare_reader_phases(config)
        elif args.command == "prepare-labels":
            config["task_cohort_labeling"]["output_dir"] = args.output
            result = prepare_cohort_task_labels(config)
        else:
            config["reader_stage_cohort"]["output_dir"] = args.output
            result = prepare_stage_cohorts(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "data":
        data_workflow(args)
    else:
        comparison_workflow(args)


if __name__ == "__main__":
    main()
