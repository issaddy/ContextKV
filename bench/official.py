"""Offline reference scoring and pinned IBM evaluator adapter; no serving imports.

Upstream dependencies are installed separately, since its vLLM/torch/ragas stack
is not a controller dependency. The adapter preserves upstream metric algorithms.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import importlib
import json
import os
import re
from statistics import mean
import string
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence
from types import SimpleNamespace

from adapters.mtrag import DATASET_COMMIT


def remap(value, mapping):
    if isinstance(value, dict):
        return {key: remap(item, mapping) for key, item in value.items() if key != "path_map"}
    if isinstance(value, list):
        return [remap(item, mapping) for item in value]
    if isinstance(value, str):
        for old, new in sorted(mapping.items(), key=lambda pair: -len(pair[0])):
            if value == old or value.startswith(old.rstrip("/") + "/"):
                return str(Path(new) / value[len(old):].lstrip("/"))
    return value


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def reference_overlap(references, predictions):
    """SQuAD-style answer F1/EM used by MultiDoc2Dial, aligned by task ID.

    Missing predictions score zero and remain visible in coverage.
    Metric definition: https://aclanthology.org/2021.emnlp-main.498/
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
        "scope": "answer token F1 and exact match on the original declared task units"}


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


# Keep BERTScore's empty input as a special-token-only sequence and use
# the paired DeBERTa model's position limit.
_ALGORITHMIC_RUNNER = """
import runpy
import sys
from bert_score import scorer, utils
from transformers import AutoConfig

original_get_tokenizer = scorer.get_tokenizer
original_sent_encode = utils.sent_encode

def get_tokenizer(model_type, use_fast=False):
    tokenizer = original_get_tokenizer(model_type, use_fast=use_fast)
    if model_type == "microsoft/deberta-xlarge-mnli":
        tokenizer.model_max_length = AutoConfig.from_pretrained(model_type).max_position_embeddings
        print(f"BERTScore {model_type} model_max_length={tokenizer.model_max_length}", flush=True)
    return tokenizer

def sent_encode(tokenizer, text):
    if not text.strip():
        return tokenizer.encode("", add_special_tokens=True)
    return original_sent_encode(tokenizer, text)

scorer.get_tokenizer = get_tokenizer
utils.sent_encode = sent_encode
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""


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
        return [python, "-c", _ALGORITHMIC_RUNNER, str(scripts / "run_algorithmic.py"), "-i", inp, "-o", out,
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
    completed = subprocess.run(command, cwd=root, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    receipt = {"mode": mode, "source_commit": DATASET_COMMIT, "command": command,
               "returncode": completed.returncode,
               "status": "completed" if completed.returncode == 0 else "failed",
               "judge_revision": judge_revision, "judge_revision_provenance": "operator_declared" if judge_revision else None}
    if completed.returncode:
        receipt["error"] = completed.stderr.strip().splitlines()[-1:]
    completed.check_returncode()
    return receipt


RAGAS_SCHEMA_MARKER = (
    "Please return the output in a JSON format that complies with the following schema "
    "as specified in JSON Schema:\n"
)
RAGAS_MAX_TOKENS = 4096


def with_ragas_output_schema(payload, *, ragas_max_tokens=None):
    """Use the original RAGAS Pydantic contract as the native output grammar."""
    for message in reversed(payload["messages"]):
        content = message["content"]
        if RAGAS_SCHEMA_MARKER in content:
            encoded = content.split(RAGAS_SCHEMA_MARKER, 1)[1].lstrip()
            schema, _ = json.JSONDecoder().raw_decode(encoded)
            structured = {**payload, "response_format": {"type": "json_schema", "json_schema": {
                "name": "ragas_prompt_schema", "schema": schema}}}
            if ragas_max_tokens is not None:
                structured["max_completion_tokens"] = ragas_max_tokens
            return structured
    return payload


def create_http_judge(model, endpoint, *, max_tokens=2048, enable_thinking=None,
                      ragas_json_schema=False, ragas_max_tokens=None):
    """Adapt the upstream string call and RAGAS interface to one cached HTTP client."""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    class OfficialJudge(LangchainLLMWrapper):
        def __call__(self, prompt, **kwargs):
            return self.langchain_llm.invoke(prompt, **kwargs)

    class RagasStructuredChatOpenAI(ChatOpenAI):
        def _get_request_payload(self, input_, *, stop=None, **kwargs):
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
            return with_ragas_output_schema(payload, ragas_max_tokens=ragas_max_tokens)

    extra = {} if enable_thinking is None else {
        "chat_template_kwargs": {"enable_thinking": enable_thinking}}
    client_type = RagasStructuredChatOpenAI if ragas_json_schema else ChatOpenAI
    client = client_type(model=model, base_url=endpoint,
        api_key=os.environ.get("OPENAI_API_KEY", "local"), temperature=0, top_p=1,
        max_tokens=max_tokens, max_retries=0, timeout=180, cache=False, extra_body=extra)
    judge = OfficialJudge(client)
    # RAGAS otherwise changes the declared greedy sampler to temperature 0.01.
    judge.bypass_temperature = True
    return judge


def _conversation_fields(messages):
    """Keep message boundaries when the current question itself contains newlines."""
    previous = "\n".join(f"{m['speaker']}: {m['text']}" for m in messages[:-1])
    return previous, messages[-1]["text"]


def _radbench_prompt(official, record):
    previous, question = _conversation_fields(record["input"])
    row = {"document": official.extract_document_texts(record["contexts"]),
           "response": official.extract_texts(record["predictions"]),
           "reference_answer": official.extract_reference(record["targets"]),
           "previous_conversation": previous, "current_question": question}
    # The author's formatter only consumes DataFrame.iterrows().
    return official.format_conversation_radbench(SimpleNamespace(iterrows=lambda: iter(((0, row),))))[0]


def _structured_radbench(official, operation, provider, model, input_path, output_path, *, workers, task_ids):
    extract, split = official.extract_conversation, official.split_conversation
    official.extract_conversation = lambda messages: messages
    official.split_conversation = _conversation_fields
    try:
        rows = load_jsonl(input_path)
        score = partial(operation, provider, model)
        shards = [rows[index::workers] for index in range(min(workers, len(rows)))]
        if len(shards) == 1:
            score(input_path, output_path)
        else:
            with TemporaryDirectory(prefix="rb-llm-", dir=Path(output_path).parent) as temporary:
                root = Path(temporary)
                inputs, outputs = [], []
                for index, shard in enumerate(shards):
                    source, target = root/f"input-{index}.jsonl", root/f"output-{index}.jsonl"
                    source.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in shard))
                    inputs.append(str(source))
                    outputs.append(str(target))
                with ThreadPoolExecutor(max_workers=len(shards)) as pool:
                    list(pool.map(score, inputs, outputs))
                scored = [row for target in outputs for row in load_jsonl(target)]
            order = {task_id: index for index, task_id in enumerate(task_ids)}
            scored.sort(key=lambda row: order[row["task_id"]])
            Path(output_path).write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in scored))
        return {"workers": len(shards), "shard_tasks": [len(shard) for shard in shards]}
    finally:
        official.extract_conversation, official.split_conversation = extract, split


def load_radbench_contract(checkout):
    """Reuse the frozen author's RB_llm prompt and parser without model dependencies."""
    root = Path(checkout).resolve()
    source = json.loads((root / "contextkv-source.json").read_text())
    if source["commit"] != DATASET_COMMIT:
        raise ValueError("official source checkout is not the frozen MTRAG revision")
    path = root / "scripts/evaluation/judge_utils.py"
    names = {"extract_document_texts", "extract_texts", "extract_reference",
             "format_conversation_radbench", "extract_rating"}
    parsed = ast.parse(path.read_text(), filename=str(path))
    functions = ast.Module(body=[node for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    namespace = {"ast": ast, "re": re}
    exec(compile(functions, str(path), "exec"), namespace)
    official = SimpleNamespace(**namespace)

    def prompt(record):
        return _radbench_prompt(official, record)

    identity = {"metric": "RB_llm", "source_commit": source["commit"],
                "conversation_format": "structured_message_boundaries_v1",
                "normalization": "official_extract_rating_divide_by_10"}
    return SimpleNamespace(prompt=prompt, parse=namespace["extract_rating"], identity=identity)


def load_conditioning_rules(checkout):
    """Load the pinned author's pure scoring rules without its model dependencies."""
    path = Path(checkout)/"scripts/evaluation/judge_wrapper.py"
    names = {"_get_answerability", "get_idk_score", "get_idk_underspec_score"}
    source = ast.parse(path.read_text(), filename=str(path))
    functions = ast.Module(body=[node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    namespace = {}
    exec(compile(functions, str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def condition_metrics(official, rows, metrics):
    """Apply the pinned author's answerability rule to every metric component."""
    has_underspecified = any(
        "UNDERSPECIFIED" in row.get("answerability", row.get("Answerability", [])) for row in rows)
    condition = official.get_idk_underspec_score if has_underspecified else official.get_idk_score
    suffix = "_idk_underspecified" if has_underspecified else "_idk"
    for row in rows:
        for metric in metrics:
            row["metrics"][metric+suffix] = [condition(row, metric)]
    return suffix


def score_http_stages(official, model, input_path, output_directory, *, answerability_adjustment=True,
                      metrics=("RB_llm", "RL_F"), rb_llm_workers=2):
    """Reuse upstream metrics with compact stage scores and temporary source text."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(input_path)
    task_ids = [row["task_id"] for row in rows]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("official scoring input contains duplicate task IDs")
    metrics = tuple(dict.fromkeys(metrics))
    operations = {"RB_llm": ("rb_llm", "run_radbench_judge"),
                  "RL_F": ("rl_f", "run_ragas_judges_local")}
    if not metrics or not set(metrics) <= operations.keys():
        raise ValueError("choose at least one of RB_llm and RL_F")
    has_underspecified = answerability_adjustment and any(
        "UNDERSPECIFIED" in row.get("answerability", row.get("Answerability", [])) for row in rows)
    stages = [("idk", "idk_eval", official.run_idk_judge)] if answerability_adjustment else []
    for metric in metrics:
        name, function = operations[metric]
        operation = getattr(official, function)
        if metric == "RB_llm":
            operation = partial(_structured_radbench, official, operation,
                                workers=rb_llm_workers, task_ids=task_ids)
        stages.append((name, metric, operation))
    if has_underspecified:
        stages.append(("underspecified", "underspecified_eval", lambda provider, name, inp, out:
                       official.underspecified_eval(name, inp, out)))
    scored_by_id = {row["task_id"]: row for row in rows}
    with TemporaryDirectory(prefix="judge-stage-", dir=output) as temporary:
        source, generated = Path(temporary)/"input.jsonl", Path(temporary)/"output.jsonl"
        for name, metric, operation in stages:
            target = output / f"{name}.jsonl"
            if target.exists():
                stage_rows = load_jsonl(target)
            else:
                source.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n"
                                          for row in scored_by_id.values()))
                operation("vllm", model, str(source), str(generated))
                stage_rows = load_jsonl(generated)
                if sorted(row["task_id"] for row in stage_rows) != sorted(task_ids):
                    raise ValueError(f"official {name} output changed the task cohort")
            stage_rows = [{"task_id": row["task_id"], "metrics": {metric: row["metrics"][metric]}}
                          for row in stage_rows]
            target.write_text("".join(json.dumps(row)+"\n" for row in stage_rows))
            for row in stage_rows:
                scored_by_id[row["task_id"]].setdefault("metrics", {})[metric] = row["metrics"][metric]
    scored = list(scored_by_id.values())
    suffix = condition_metrics(official, scored, metrics) if answerability_adjustment else None
    target = output / "scores.jsonl"
    target.write_text("".join(json.dumps({key: row[key] for key in (
        "task_id", "answerability", "Answerability", "metrics") if key in row})+"\n" for row in scored))
    return {"tasks": len(scored), "requested_metrics": list(metrics),
            "conditioned_suffix": suffix, "output": str(target)}


def run_http_evaluator(checkout, input_path, output_directory, *, model, judge_revision,
                       endpoint, max_tokens=2048, enable_thinking=None, answerability_adjustment=True,
                       metrics=("RB_llm", "RL_F"), ragas_json_schema=False, ragas_max_tokens=None,
                       rb_llm_workers=2):
    """Offline official RB_llm/RL_F/IDK over frozen answers and completed stage scores.

    Run with the configured Python interpreter. This function does not start a
    server, load weights, regenerate answers or alter official metric prompts.
    """
    root = Path(checkout).resolve()
    source = json.loads((root / "contextkv-source.json").read_text())
    if source["commit"] != DATASET_COMMIT:
        raise ValueError("official source checkout is not the frozen MTRAG revision")
    if not model or not judge_revision or not endpoint:
        raise ValueError("HTTP scoring requires model, judge revision and endpoint")
    if ragas_max_tokens is not None and (
            not ragas_json_schema or type(ragas_max_tokens) is not int or ragas_max_tokens <= 0):
        raise ValueError("ragas_max_tokens requires ragas_json_schema and a positive integer budget")
    if ragas_json_schema and "RL_F" in metrics and ragas_max_tokens is None:
        ragas_max_tokens = RAGAS_MAX_TOKENS
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = {"input": str(Path(input_path).resolve()),
                "source_commit": DATASET_COMMIT, "judge_model": model,
                "judge_revision": judge_revision, "temperature": 0, "top_p": 1,
                "max_tokens": max_tokens, "enable_thinking": enable_thinking,
                "conversation_format": "structured_message_boundaries_v1"}
    if not answerability_adjustment:
        identity["answerability_adjustment"] = False
    if ragas_json_schema:
        identity["ragas_json_schema"] = "original_prompt_Pydantic_contract"
    if ragas_max_tokens is not None:
        identity["ragas_max_tokens"] = ragas_max_tokens
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("scoring directory belongs to different answers or judge settings; use a new directory")
    identity_path.write_text(json.dumps(identity, indent=2)+"\n")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    os.environ.setdefault("LANGSMITH_TRACING", "false")
    from ragas.prompt import pydantic_prompt

    sys.path.insert(0, str(root / "scripts/evaluation"))
    official = importlib.import_module("judge_wrapper")
    loader = official.load_chat_openai_llm
    run_config = official.RunConfig
    extract_json = pydantic_prompt.extract_json
    if ragas_json_schema:
        # Native structured output is already JSON, including braces inside strings.
        pydantic_prompt.extract_json = str.strip
    official.load_chat_openai_llm = lambda name, max_new_tokens=None: create_http_judge(
        name, endpoint, max_tokens=max_tokens, enable_thinking=enable_thinking,
        ragas_json_schema=ragas_json_schema, ragas_max_tokens=ragas_max_tokens)
    # The upstream local-model path serializes RAGAS. HTTP requests can share
    # the same native batch scheduler as RB_llm, without changing its metric.
    official.RunConfig = lambda **kwargs: run_config(**{**kwargs, "max_workers": rb_llm_workers})
    try:
        result = score_http_stages(official, model, input_path, output,
                                   answerability_adjustment=answerability_adjustment, metrics=metrics,
                                   rb_llm_workers=rb_llm_workers)
    finally:
        official.load_chat_openai_llm = loader
        official.RunConfig = run_config
        pydantic_prompt.extract_json = extract_json
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "check", "retrieval", "algorithmic", "generation", "generation-http", "condition", "overlap"])
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
    parser.add_argument("--metrics", nargs="+", choices=["RB_llm", "RL_F", "RB_agg"], default=["RB_llm", "RL_F"],
                        help="Metrics to generate or condition; RB_agg is supplied by the algorithmic stage")
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--ragas-json-schema", action="store_true",
                        help="Constrain RAGAS output with its original prompt's Pydantic JSON schema")
    parser.add_argument("--ragas-max-tokens", type=int,
                        help="Output budget for requests matched by --ragas-json-schema")
    parser.add_argument("--rb-llm-workers", type=int, choices=(1, 2), default=2,
                        help="Parallel RB_llm shards for generation-http")
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
    elif args.mode == "condition":
        official = load_conditioning_rules(args.checkout)
        result = load_jsonl(args.input)
        condition_metrics(official, result, args.metrics)
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in result))
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
            answerability_adjustment=args.answerability_adjustment, metrics=args.metrics,
            ragas_json_schema=args.ragas_json_schema, ragas_max_tokens=args.ragas_max_tokens,
            rb_llm_workers=args.rb_llm_workers)
    else:
        if not args.checkout:
            parser.error("evaluation requires --checkout")
        run_evaluator(args.checkout, args.mode, args.input, args.output,
                      python=args.python, provider=args.provider, model=args.model, judge_revision=args.judge_revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
