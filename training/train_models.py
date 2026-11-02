"""Train the task asset and the support/utility asset with explicit stage inputs."""
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
    if stage == "support":
        from training.support_model import train_support
        return train_support(settings["support_labels"], config["support_model"]["base_path"],
            settings["support_output"], device=config["support_model"]["device"], epochs=settings["epochs"],
            learning_rate=settings["head_lr"])
    from training.support_model import train_set_utility
    return train_set_utility(settings["preferences"], settings["support_output"],
        settings["preference_output"], device=config["support_model"]["device"], epochs=settings["epochs"],
        learning_rate=settings["head_lr"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("task", "support", "set_utility"))
    args = parser.parse_args(argv)
    print(json.dumps(train(yaml.safe_load(Path(args.config).read_text()), args.stage),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
