"""Build support supervision candidates from recorded training histories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from training.support_data import build_support_examples
from training.task_data import read_jsonl
from training.reader_data import file_digest


def build_examples(config):
    from controller.config import create_model_template, load_config
    from training.support_data import annotate_support_parallel

    settings = config["training"]
    root = Path(settings["support_data_dir"])
    sources = Path(settings["data_dir"])/"sources.jsonl"
    labels = Path(settings["task_labels"])
    reader = load_config(settings["reader_config"])
    template = create_model_template(reader.model)
    build_support_examples(labels, sources, root/"candidate_build",
        cohort_manifest=settings["cohort_manifest"],
        max_turns=settings["min_rollout_turns"], tokenizer=template.tokenizer)
    teacher = annotate_support_parallel(root/"candidate_build"/"support_candidates.jsonl",
        settings["support_teacher_labels"], template=template,
        base_urls=settings["teacher_urls"], workers=settings.get("teacher_concurrency", 8),
        max_examples=settings.get("support_teacher_examples"))
    result = build_support_examples(labels, sources, root,
        cohort_manifest=settings["cohort_manifest"],
        teacher_labels_path=settings["support_teacher_labels"], max_turns=settings["min_rollout_turns"],
        tokenizer=template.tokenizer)
    manifest = {"support_data": result, "support_teacher": teacher,
        "task_labels": {"path": str(labels.resolve()), "sha256": file_digest(labels),
                        "states": sum(1 for _ in read_jsonl(labels))},
        "cohort_manifest": settings["cohort_manifest"], "history_origin": "model_actual",
        "paths": {"sources": str(sources.resolve()),
                  "candidates": str((root/"support_candidates.jsonl").resolve()),
                  "support_labels": str((root/"support_labels.jsonl").resolve())}}
    (root/"training_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    if "base_study_config" in config:
        from bench.study import merge_config
        config = merge_config(yaml.safe_load(Path(config["base_study_config"]).read_text()), config)
    print(json.dumps(build_examples(config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
