"""Pinned Loong reference rubric and score extraction over saved answers.

The configured judge endpoint preserves the author's question/reference
construction and score parser.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from importlib.resources import files
import json
from pathlib import Path
import re
from statistics import mean

from adapters.loong import COMMIT
from bench.judge import OpenAICompletionClient, JudgeConfig
from bench.official import load_jsonl


RUBRIC = files("bench").joinpath("assets/loong/evaluation.txt").read_text(encoding="utf-8")


def evaluation_prompt(reference, prediction):
    """Preserve the upstream evaluator's question construction and answer types."""
    original = reference["source_task"]
    template = original["prompt_template"]
    if original["type"] != "paper":
        template = template.replace("{docs}", "")
    question = template.replace("{question}", original["question"]).replace("{instruction}", original["instruction"])
    return RUBRIC.format(question, reference["reference_value"], prediction)


def parse_score(text):
    # Loong src/utils/metric.py::extract_number at COMMIT.
    match = re.search(r"\[\[([0-9]*\.?[0-9]+)\]\]", text)
    if match:
        return float(match.group(1))
    match = re.search(r"\[([0-9]*\.?[0-9]+)\]", text)
    if match:
        return float(match.group(1))
    return None


def score_answer(reference, prediction, client):
    base = {"task_id": reference["task_id"], "domain": reference["source_task"]["type"],
            "level": reference["source_task"]["level"], "length_set": reference["source_task"]["set"]}
    if prediction is None:
        return {**base, "status": "missing_prediction", "score": None}
    prompt = evaluation_prompt(reference, prediction)
    completion = client.complete([{"role": "user", "content": prompt}])
    value = (parse_score(completion.text) if completion.status == "completed" and
             completion.finish_reason == "stop" and isinstance(completion.text, str) else None)
    return {**base, "status": "scored" if value is not None else "unscored", "score": value,
            "completion": asdict(completion)}


def summarize(rows):
    values = [r["score"] for r in rows if r["score"] is not None]
    missing = [r["task_id"] for r in rows if r["score"] is None]
    return {"requested": len(rows), "scored": len(values), "missing_task_ids": missing,
            "observed_mean_0_100": mean(values) if values else None,
            "observed_perfect_rate": sum(v == 100 for v in values)/len(values) if values else None,
            "full_cohort_mean_bounds": [sum(values)/len(rows), (sum(values)+100*len(missing))/len(rows)] if rows else None}


def grouped_summary(rows):
    return {"summary": summarize(rows),
            "by_domain": {name: summarize([row for row in rows if row["domain"] == name])
                          for name in sorted({row["domain"] for row in rows})},
            "by_level": {str(level): summarize([row for row in rows if row["level"] == level])
                         for level in sorted({row["level"] for row in rows})}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True, help="JSON JudgeConfig for the existing local completion transport")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    references, predictions = load_jsonl(args.references), load_jsonl(args.predictions)
    by_id = {r["task_id"]: r["predictions"][0]["text"] for r in predictions}
    ids = {r["task_id"] for r in references}
    if len(by_id) != len(predictions) or len(ids) != len(references) or by_id.keys()-ids:
        raise ValueError("duplicate or unknown evaluation task IDs")
    config = JudgeConfig(**json.loads(args.judge_config.read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output/"scores.jsonl"
    results = load_jsonl(target) if target.exists() else []
    finished = {row["task_id"] for row in results}
    client = OpenAICompletionClient(config)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor, target.open("a", encoding="utf-8") as output:
            def evaluate(reference):
                return score_answer(reference, by_id.get(reference["task_id"]), client)
            pending = [reference for reference in references if reference["task_id"] not in finished]
            for row in executor.map(evaluate, pending):
                results.append(row)
                output.write(json.dumps(row, ensure_ascii=False)+"\n")
                output.flush()
    finally:
        client.close()
    report = {"status": "completed", **grouped_summary(results),
              "protocol": "loong-reference-rubric-local-judge-v1", "upstream_commit": COMMIT,
              "references": str(args.references.resolve()),
              "predictions": str(args.predictions.resolve()),
              "judge": client.manifest, "workers": args.workers, "answer_regeneration": False,
              "adaptations": "Configured judge endpoint and budgets; original rubric, question/reference construction and score extraction retained",
              "scope": "Reference accuracy/completeness under the configured judge"}
    (args.output/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
