"""Pinned Loong reference rubric over saved answers through the existing transport.

The local judge is an explicit substitution for the paper's GPT-4 judge. This is
not a leaderboard reproduction or a source-faithfulness metric. Failed, absent,
over-budget or truncated grades stay missing; no regeneration or repair calls.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
from importlib.resources import files
import json
from pathlib import Path
import re
from statistics import mean

from adapters.loong import COMMIT
from adapters.query_selector import OpenAICompletionClient, SelectorConfig
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
    # Upstream accepts double brackets or a single-bracket fallback. Require
    # one unambiguous, in-range grade instead of silently using the first.
    found = re.findall(r"\[\[([0-9]*\.?[0-9]+)\]\]", text)
    if not found:
        found = re.findall(r"\[([0-9]*\.?[0-9]+)\]", text)
    if len(found) != 1:
        return None
    value = float(found[0])
    return value if 0 <= value <= 100 else None


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
            "prompt_sha256": sha256(prompt.encode()).hexdigest(),
            "answer_sha256": sha256(prediction.encode()).hexdigest(), "completion": asdict(completion)}


def summarize(rows):
    values = [r["score"] for r in rows if r["score"] is not None]
    missing = [r["task_id"] for r in rows if r["score"] is None]
    return {"requested": len(rows), "scored": len(values), "missing_task_ids": missing,
            "observed_mean_0_100": mean(values) if values else None,
            "observed_perfect_rate": sum(v == 100 for v in values)/len(values) if values else None,
            "full_cohort_mean_bounds": [sum(values)/len(rows), (sum(values)+100*len(missing))/len(rows)] if rows else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True, help="JSON SelectorConfig for the existing local completion transport")
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
    config = SelectorConfig(**json.loads(args.judge_config.read_text()))
    args.output.mkdir(parents=True, exist_ok=False)
    client = OpenAICompletionClient(config)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor, (args.output/"scores.jsonl").open("w", encoding="utf-8") as output:
            def evaluate(reference):
                return score_answer(reference, by_id.get(reference["task_id"]), client)
            results = []
            for row in executor.map(evaluate, references):
                results.append(row)
                output.write(json.dumps(row, ensure_ascii=False)+"\n")
                output.flush()
    finally:
        client.close()
    report = {"status": "completed", "summary": summarize(results),
              "by_domain": {name: summarize([r for r in results if r["domain"] == name])
                            for name in sorted({r["domain"] for r in results})},
              "by_level": {str(n): summarize([r for r in results if r["level"] == n])
                           for n in sorted({r["level"] for r in results})},
              "protocol": "loong-reference-rubric-local-judge-v1", "upstream_commit": COMMIT,
              "rubric_sha256": sha256(RUBRIC.encode()).hexdigest(),
              "references_sha256": sha256(args.references.read_bytes()).hexdigest(),
              "predictions_sha256": sha256(args.predictions.read_bytes()).hexdigest(),
              "judge": client.manifest, "workers": args.workers, "answer_regeneration": False,
              "adaptations": "Configured local judge/budgets, one call per answer, explicit missing/truncated grades, unique in-range bracket score; original rubric and question/reference construction retained",
              "scope": "Reference accuracy/completeness diagnostic; no source-groundedness guarantee or comparability to the original GPT-4 leaderboard"}
    (args.output/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
