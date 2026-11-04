"""Prepare fixed-reader evidence cohorts and compare evidence policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import yaml

from training.reader_data import audit_reader_targets
from training.reader_curriculum import prepare_evidence_cohorts
from training.task_data import build_task_data, prepare_cohort_task_labels


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
    subprocess.run(command, check=True)
    print(json.dumps({"comparison_manifest": str(output/"comparison"/"comparison_manifest.json")}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("compare", "audit", "cohort", "prepare-data", "prepare-labels"))
    parser.add_argument("--policy-asset", default="models/study-support-fixed-qwen")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", help="reader audit, cohort or task_cohort_labeling configuration")
    parser.add_argument("--conversations", type=int)
    parser.add_argument("--comparison-config")
    parser.add_argument("--phase", choices=("frozen", "p3", "both"), default="both")
    parser.add_argument("--matrix")
    parser.add_argument("--dataset", default="dynamic_goal_dev")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--records")
    parser.add_argument("--goal-policy")
    parser.add_argument("--limit-states", type=int)
    args = parser.parse_args(argv)
    if args.command == "compare" and (args.matrix is None or args.comparison_config is None):
        parser.error("compare requires --comparison-config and --matrix")
    if args.command in {"audit", "cohort", "prepare-data", "prepare-labels"}:
        if args.config is None:
            parser.error(f"{args.command} requires --config")
        config = yaml.safe_load(Path(args.config).read_text())
        if "base_study_config" in config:
            from bench.study import merge_config
            config = merge_config(yaml.safe_load(Path(config["base_study_config"]).read_text()), config)
        if args.command == "prepare-data":
            config["training"]["data_dir"] = args.output
            result = build_task_data(config)
        elif args.command == "audit":
            config["reader_audit"]["data"]["output_dir"] = args.output
            result = audit_reader_targets(config)
        elif args.command == "prepare-labels":
            config["task_cohort_labeling"]["output_dir"] = args.output
            result = prepare_cohort_task_labels(config)
        else:
            config["evidence_cohort"]["output_dir"] = args.output
            result = prepare_evidence_cohorts(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        comparison_workflow(args)


if __name__ == "__main__":
    main()
