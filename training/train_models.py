"""Train or evaluate auxiliary assets on the registered train/dev cohorts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def train(config, stage):
    settings = config["training"]
    if stage == "task":
        from training.task_model import train_task
        return train_task(config)
    if stage == "task-eval":
        from training.task_model import evaluate_task
        return evaluate_task(config)
    if stage == "support-eval":
        from training.support_model import evaluate_support
        return evaluate_support(config)
    if stage == "support":
        from training.support_model import train_support
        return train_support(settings["support_labels"], settings["support_parent_asset"],
            settings["support_output"], base_path=config["support_model"]["base_path"],
            device=config["support_model"]["device"], epochs=settings["epochs"],
            learning_rate=settings["head_lr"], encoder_learning_rate=settings.get("support_encoder_lr", 1e-5),
            batch_size=settings.get("support_batch_size", 4), seed=settings.get("seed", 42),
            cohort_manifest=settings["cohort_manifest"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("task", "task-eval", "support", "support-eval"))
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    if "base_study_config" in config:
        from bench.study import merge_config
        config = merge_config(yaml.safe_load(Path(config["base_study_config"]).read_text()), config)
    print(json.dumps(train(config, args.stage), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
