"""Offline adapter for the pinned IBM evaluator; never imported by serving code.

Upstream dependencies are installed separately, since its vLLM/torch/ragas stack
is not a controller dependency. The adapter preserves upstream metric algorithms.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

from adapters.mtrag import DATASET_COMMIT


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def merge_offline_labels(predictions: Sequence[Mapping], references: Sequence[Mapping]) -> list[dict]:
    """Join labels after generation, retaining each method's own input and contexts."""
    reference = {r["task_id"]: r for r in references}
    result, seen = [], set()
    for prediction in predictions:
        task_id = prediction["task_id"]
        if task_id in seen:
            raise ValueError(f"duplicate prediction task: {task_id}")
        seen.add(task_id)
        source = reference[task_id]
        row = dict(prediction)
        for key in ("targets", "enrichments"):
            if key in source:
                row[key] = source[key]
        row["evaluation_history_matches_reference"] = row["input"] == source["input"]
        result.append(row)
    return result


def evaluator_command(checkout: str | Path, mode: str, input_path: str | Path,
                      output_path: str | Path, *, python: str = sys.executable,
                      provider: str | None = None, model: str | None = None) -> list[str]:
    """Arguments verified against the pinned scripts, not the current README."""
    root = Path(checkout).resolve()
    inp, out = str(Path(input_path).resolve()), str(Path(output_path).resolve())
    scripts = root / "scripts/evaluation"
    if mode == "retrieval":
        return [python, str(scripts / "run_retrieval_eval.py"), "--input_file", inp, "--output_file", out]
    if mode == "algorithmic":
        return [python, str(scripts / "run_algorithmic.py"), "-i", inp, "-o", out,
                "-e", str(scripts / "config.yaml")]
    if mode == "generation":
        if provider not in {"hf", "vllm"} or not model:
            raise ValueError("official generation adapter requires hf/vllm and an explicitly frozen judge model")
        return [python, str(scripts / "run_generation_eval.py"), "-i", inp, "-o", out,
                "-e", str(scripts / "config.yaml"), "--provider", provider, "--judge_model", model]
    raise ValueError("mode must be retrieval, algorithmic or generation")


def check_format(checkout, input_path, prediction_path, *, mode="rag_taskc", python=sys.executable):
    # Upstream CLI prints validation errors but does not exit nonzero. Call its
    # public validator and set an accurate exit status without changing its rules.
    if mode not in {"rag_taskc", "generation_taskb", "retrieval_taska"}:
        raise ValueError("invalid official checker mode")
    code = ("import sys,json; sys.path.insert(0,sys.argv[1]); import format_checker as f; "
            "errors,warnings=f.validate_prediction_file(sys.argv[2],sys.argv[3],sys.argv[4]); "
            "print(json.dumps(dict(errors=errors,warnings=warnings))); sys.exit(bool(errors))")
    return subprocess.run([python, "-c", code, str(Path(checkout).resolve() / "scripts/evaluation"),
                           str(Path(input_path).resolve()), str(Path(prediction_path).resolve()), mode],
                          capture_output=True, text=True)


def run_evaluator(checkout, mode, input_path, output_path, *, judge_revision=None, **kwargs):
    root = Path(checkout).resolve()
    manifest = json.loads((root / "contextkv-source.json").read_text())
    if manifest["commit"] != DATASET_COMMIT:
        raise ValueError("official source checkout is not the frozen MTRAG revision")
    if mode == "generation" and not judge_revision:
        raise ValueError("official judge revision must be recorded explicitly")
    if mode == "retrieval":
        # Pinned upstream still uses the old directory name. A layout alias fixes
        # that path only; labels and metric implementation remain unchanged.
        alias = root / "human/retrieval_tasks_convid"
        alias.parent.mkdir(parents=True, exist_ok=True)
        if not alias.exists():
            alias.symlink_to(root / "mtrag-human/retrieval_tasks", target_is_directory=True)
    Path(output_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    command = evaluator_command(root, mode, input_path, output_path, **kwargs)
    env_command = [kwargs.get("python", sys.executable), "-c",
                   "import importlib.metadata as m,json,sys; print(json.dumps(dict(python=sys.version,packages=sorted([(d.metadata['Name'],d.version) for d in m.distributions()]))))"]
    environment = json.loads(subprocess.check_output(env_command, text=True))
    Path(str(output_path) + ".environment.json").write_text(json.dumps(environment, indent=2))
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
    receipt = {"mode": mode, "source_commit": DATASET_COMMIT, "command": command,
               "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr,
               "status": "completed" if completed.returncode == 0 else "failed",
               "judge_revision": judge_revision, "judge_revision_provenance": "operator_declared" if judge_revision else None}
    Path(str(output_path) + ".execution.json").write_text(json.dumps(receipt, indent=2))
    completed.check_returncode()
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "check", "retrieval", "algorithmic", "generation"])
    parser.add_argument("--checkout")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--references")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--provider", choices=["hf", "vllm"])
    parser.add_argument("--model")
    parser.add_argument("--judge-revision")
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        if not args.references:
            parser.error("prepare requires --references (offline only)")
        result = merge_offline_labels(load_jsonl(args.input), load_jsonl(args.references))
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result))
    elif args.mode == "check":
        if not args.checkout:
            parser.error("check requires --checkout")
        completed = check_format(args.checkout, args.input, args.output, python=args.python)
        print(completed.stdout)
        return completed.returncode
    else:
        if not args.checkout:
            parser.error("evaluation requires --checkout")
        run_evaluator(args.checkout, args.mode, args.input, args.output,
                      python=args.python, provider=args.provider, model=args.model, judge_revision=args.judge_revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
