"""Offline reference scoring and pinned IBM evaluator adapter; no serving imports.

Upstream dependencies are installed separately, since its vLLM/torch/ragas stack
is not a controller dependency. The adapter preserves upstream metric algorithms.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import importlib
import json
import os
import re
from statistics import mean
import string
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

from adapters.mtrag import DATASET_COMMIT


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def reference_overlap(references, predictions):
    """SQuAD-style answer F1/EM used by MultiDoc2Dial, aligned by task ID.

    This lexical measure supplements task/source evaluation. Missing predictions
    score zero and remain visible in coverage, as in the upstream evaluator.
    See IBM/multidoc2dial scripts/sharedtask_eval.py at 2456a5609b447e1a12b716f4d658728f346f3041.
    """
    def normalize(text):
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        return re.sub(r"\b(a|an|the)\b", " ", text).split()

    predictions = list(predictions)
    answers = {row["task_id"]: row["predictions"][0]["text"] for row in predictions}
    if len(answers) != len(predictions):
        raise ValueError("duplicate prediction task ID")
    scores, domains, sessions, missing = {}, defaultdict(list), defaultdict(list), []
    for row in references:
        tid = row["task_id"]
        if tid in scores:
            raise ValueError("duplicate reference task ID")
        if tid not in answers:
            missing.append(tid)
        prediction = normalize(answers.get(tid, ""))
        f1 = em = 0.0
        for target in row["targets"]:
            reference = normalize(target["text"])
            overlap = sum((Counter(prediction) & Counter(reference)).values())
            f1 = max(f1, 2 * overlap / (len(prediction) + len(reference)) if overlap else 0.0)
            em = max(em, float(prediction == reference))
        scores[tid] = {"f1": f1, "exact_match": em}
        domains[row["Collection"]].append(f1)
        sessions[row["conversation_id"]].append(f1)
    if answers.keys() - scores.keys():
        raise ValueError("predictions contain task IDs absent from references")
    return {"metric": "squad_style_reference_overlap", "tasks": scores,
        "requested_tasks": len(scores), "predicted_tasks": len(answers), "missing_task_ids": missing,
        "mean_f1": mean(v["f1"] for v in scores.values()) if scores else None,
        "mean_exact_match": mean(v["exact_match"] for v in scores.values()) if scores else None,
        "domain_mean_f1": {domain: mean(values) for domain, values in domains.items()},
        "conversation_mean_f1": {sid: mean(values) for sid, values in sessions.items()},
        "scope": "lexical answer overlap on the declared exchange units; not semantic correctness or source faithfulness"}


def merge_offline_labels(predictions: Sequence[Mapping], references: Sequence[Mapping]) -> list[dict]:
    """Join labels after generation, retaining each method's own input and contexts."""
    reference = {r["task_id"]: r for r in references}
    result, seen = [], set()
    def dialogue(messages):
        return [(m["speaker"], m["text"]) for m in messages]
    for prediction in predictions:
        task_id = prediction["task_id"]
        if task_id in seen:
            raise ValueError(f"duplicate prediction task: {task_id}")
        seen.add(task_id)
        source = reference[task_id]
        row = dict(prediction)
        for key in ("targets", "enrichments", "answerability", "Answerability", "Question Type", "Multi-Turn"):
            if key in source:
                row[key] = source[key]
        # Author/timestamp metadata is intentionally absent from model inputs.
        # Compare the actual dialogue, not those offline metadata fields.
        row["evaluation_history_matches_reference"] = dialogue(row["input"]) == dialogue(source["input"])
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
    if mode == "retrieval":
        rows = load_jsonl(input_path)
        if any("retrieval_contexts" in row for row in rows):
            # Context expansion is reading material, not another ranked hit.
            projected = Path(str(output_path) + ".retrieval-input.jsonl")
            projected.write_text("".join(json.dumps({**row,
                "contexts": row.get("retrieval_contexts", row["contexts"])}, ensure_ascii=False) + "\n"
                for row in rows))
            input_path = projected
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


def create_http_judge(model, endpoint, *, max_tokens=2048, enable_thinking=None):
    """Adapt the upstream string call and RAGAS interface to one cached HTTP client."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    class OfficialJudge(LangchainLLMWrapper):
        def __call__(self, prompt, **kwargs):
            return self.langchain_llm.invoke(prompt, **kwargs)

    extra = {} if enable_thinking is None else {
        "chat_template_kwargs": {"enable_thinking": enable_thinking}}
    client = ChatOpenAI(model=model, base_url=endpoint,
        api_key=os.environ.get("OPENAI_API_KEY", "local"), temperature=0, top_p=1,
        max_tokens=max_tokens, max_retries=0, timeout=180, cache=True, extra_body=extra)
    judge = OfficialJudge(client)
    # RAGAS otherwise changes the declared greedy sampler to temperature 0.01.
    judge.bypass_temperature = True
    return judge


def score_http_stages(official, model, input_path, output_directory, *, answerability_adjustment=True):
    """Reuse upstream metric functions; atomically retain each completed stage."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(input_path)
    task_ids = [row["task_id"] for row in rows]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("official scoring input contains duplicate task IDs")
    has_underspecified = answerability_adjustment and any(
        "UNDERSPECIFIED" in row.get("answerability", row.get("Answerability", [])) for row in rows)
    stages = [("idk", official.run_idk_judge)] if answerability_adjustment else []
    stages += [("rb_llm", official.run_radbench_judge), ("rl_f", official.run_ragas_judges_local)]
    if has_underspecified:
        stages.append(("underspecified", lambda provider, name, inp, out:
                       official.underspecified_eval(name, inp, out)))
    previous = Path(input_path)
    for name, operation in stages:
        target = output / f"{name}.jsonl"
        if not target.exists():
            pending = target.with_suffix(".pending.jsonl")
            operation("vllm", model, str(previous), str(pending))
            scored_ids = [r["task_id"] for r in load_jsonl(pending)]
            if sorted(scored_ids) != sorted(task_ids):
                raise ValueError(f"official {name} output changed the task cohort")
            pending.replace(target)
        previous = target
        (output / "progress.json").write_text(json.dumps({"completed_stage": name, "tasks": len(rows)})+"\n")
    scored = load_jsonl(previous)
    suffix = None
    if answerability_adjustment:
        condition = official.get_idk_underspec_score if has_underspecified else official.get_idk_score
        suffix = "_idk_underspecified" if has_underspecified else "_idk"
        for row in scored:
            for metric in ("RB_llm", "RL_F"):
                row["metrics"][metric+suffix] = [condition(row, metric)]
    target = output / "scores.jsonl"
    pending = target.with_suffix(".pending.jsonl")
    pending.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in scored))
    pending.replace(target)
    return {"tasks": len(scored), "stages": [name for name, _ in stages],
            "conditioned_suffix": suffix, "output": str(target)}


def run_http_evaluator(checkout, input_path, output_directory, *, model, judge_revision,
                       endpoint, max_tokens=2048, enable_thinking=None, answerability_adjustment=True):
    """Offline official RB_llm/RL_F/IDK over frozen answers, with resumable HTTP calls.

    Run in the isolated evaluator environment. This function does not start a
    server, load weights, regenerate answers or alter official metric prompts.
    """
    root = Path(checkout).resolve()
    source = json.loads((root / "contextkv-source.json").read_text())
    if source["commit"] != DATASET_COMMIT:
        raise ValueError("official source checkout is not the frozen MTRAG revision")
    if not model or not judge_revision or not endpoint:
        raise ValueError("HTTP scoring requires model, judge revision and endpoint")
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = {"input_sha256": sha256(Path(input_path).read_bytes()).hexdigest(),
                "source_commit": DATASET_COMMIT, "judge_model": model,
                "judge_revision": judge_revision, "temperature": 0, "top_p": 1,
                "max_tokens": max_tokens, "enable_thinking": enable_thinking}
    if not answerability_adjustment:
        identity["answerability_adjustment"] = False
    identity_path = output / "identity.json"
    if not identity_path.exists() and any(output.glob("*.jsonl")):
        raise ValueError("existing scoring files have no bound identity; use a new directory")
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("scoring directory belongs to different answers or judge settings; use a new directory")
    identity_path.write_text(json.dumps(identity, indent=2)+"\n")
    from importlib.metadata import distributions
    environment = {"python": sys.version,
                   "packages": sorted((d.metadata["Name"], d.version) for d in distributions())}
    (output / "environment.json").write_text(json.dumps(environment, indent=2)+"\n")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    os.environ.setdefault("LANGSMITH_TRACING", "false")
    from langchain_community.cache import SQLiteCache
    from langchain_core.globals import set_llm_cache

    set_llm_cache(SQLiteCache(database_path=str(output / "judge-cache.sqlite3")))
    sys.path.insert(0, str(root / "scripts/evaluation"))
    official = importlib.import_module("judge_wrapper")
    loader = official.load_chat_openai_llm
    official.load_chat_openai_llm = lambda name, max_new_tokens=None: create_http_judge(
        name, endpoint, max_tokens=max_tokens, enable_thinking=enable_thinking)
    try:
        result = score_http_stages(official, model, input_path, output,
                                   answerability_adjustment=answerability_adjustment)
    finally:
        official.load_chat_openai_llm = loader
    receipt = {**result, **identity, "status": "completed", "endpoint": endpoint,
               "adapter_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
               "backend": "OpenAI-compatible HTTP via the official vllm client branch",
               "generation_answers_modified": False,
               "scope": "Official metric functions on the declared task protocol; scorer identity and source/label differences accompany interpretation."}
    (output / "completion.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "check", "retrieval", "algorithmic", "generation", "generation-http", "overlap"])
    parser.add_argument("--checkout")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--references")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--provider", choices=["hf", "vllm"])
    parser.add_argument("--model")
    parser.add_argument("--judge-revision")
    parser.add_argument("--endpoint", help="OpenAI-compatible /v1 URL for generation-http")
    parser.add_argument("--max-tokens", type=int, default=2048, help="HTTP judge output budget")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--answerability-adjustment", action=argparse.BooleanOptionalAction, default=True,
                        help="Disable for datasets without the MT-RAG answerability labels; score raw RB_llm and RL_F only")
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        if not args.references:
            parser.error("prepare requires --references (offline only)")
        result = merge_offline_labels(load_jsonl(args.input), load_jsonl(args.references))
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result))
    elif args.mode == "overlap":
        if not args.references:
            parser.error("overlap requires --references")
        result = reference_overlap(load_jsonl(args.references), load_jsonl(args.input))
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    elif args.mode == "check":
        if not args.checkout:
            parser.error("check requires --checkout")
        completed = check_format(args.checkout, args.input, args.output, python=args.python)
        print(completed.stdout)
        return completed.returncode
    elif args.mode == "generation-http":
        if not args.checkout or not args.model or not args.judge_revision or not args.endpoint:
            parser.error("generation-http requires --checkout, --model, --judge-revision and --endpoint")
        run_http_evaluator(args.checkout, args.input, args.output, model=args.model,
            judge_revision=args.judge_revision, endpoint=args.endpoint,
            max_tokens=args.max_tokens, enable_thinking=args.enable_thinking,
            answerability_adjustment=args.answerability_adjustment)
    else:
        if not args.checkout:
            parser.error("evaluation requires --checkout")
        run_evaluator(args.checkout, args.mode, args.input, args.output,
                      python=args.python, provider=args.provider, model=args.model, judge_revision=args.judge_revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
