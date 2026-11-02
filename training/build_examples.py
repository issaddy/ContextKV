"""Build support supervision candidates from recorded training histories."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from training.support_data import build_support_examples
from training.task_data import read_jsonl


def build_examples(config):
    from controller.config import create_model_template, load_config
    from training.task_labeling import label_rollouts
    from training.support_data import annotate_support_parallel

    settings = config["training"]
    root = Path(settings["data_dir"])
    reader = load_config(settings["reader_config"])
    template = create_model_template(reader.model)
    task_labels = asyncio.run(label_rollouts(config, concurrency=settings.get("teacher_concurrency", 8)))
    build_support_examples(settings["task_labels"], root/"sources.jsonl", root/"candidate_build",
        max_turns=settings["min_rollout_turns"], tokenizer=template.tokenizer)
    teacher = annotate_support_parallel(root/"candidate_build"/"support_candidates.jsonl",
        settings["support_teacher_labels"], template=template,
        base_urls=settings["teacher_urls"], workers=settings.get("teacher_concurrency", 8),
        max_examples=settings.get("support_teacher_examples"))
    result = build_support_examples(settings["task_labels"], root/"sources.jsonl", root,
        teacher_labels_path=settings["support_teacher_labels"], max_turns=settings["min_rollout_turns"],
        tokenizer=template.tokenizer)
    manifest = {"task_data": json.loads((root/"task_data_manifest.json").read_text()),
        "support_data": result, "task_labeling": task_labels, "support_teacher": teacher,
        "actual_turns": sum(1 for _ in read_jsonl(root/"turns.jsonl")),
        "task_labels": sum(1 for _ in read_jsonl(root/"task_labels.jsonl")),
        "split": "train", "history_origin": "model_actual",
        "paths": {name: str((root/name).resolve()) for name in
                  ("turns.jsonl", "sources.jsonl", "task_labels.jsonl", "support_candidates.jsonl")}}
    (root/"training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build_examples(yaml.safe_load(Path(args.config).read_text())),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
