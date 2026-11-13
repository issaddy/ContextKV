"""Score saved study answers with explicit benchmark and judge provenance."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from threading import Lock, Thread
import time

import httpx

from bench.official import (load_jsonl, merge_offline_labels, reference_overlap,
                            condition_metrics, load_conditioning_rules)
from bench.result_records import retain_metrics, write_task_metrics
from bench.study_metrics import (_metric, _required,
                                 split_metric_spec)


def _write(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False)+"\n")


def _settings(config, dataset_spec):
    return {**config.get("scoring", {}), **dataset_spec.get("scoring", {})}


def _judge_config(config, settings):
    declared = settings.get("judge")
    if declared is not None:
        return json.loads(Path(declared).read_text()) if isinstance(declared, str) else declared
    from controller.config import load_config
    run = load_config(config.get("generation", {}).get("run_config", "configs/qwen3_32b.yaml"))
    return {"provider": "local_qwen3", "model": run.model.name, "revision": run.model.revision,
        "base_url": config.get("generation", {}).get("teacher_url", "http://127.0.0.1:30001").rstrip("/")+"/v1",
        "max_tokens": settings.get("max_tokens", 2048), "enable_thinking": False}


def _cost_totals(records):
    costs = [record["judge_cost"] for record in records if "judge_cost" in record]
    return {"observed_input_tokens": sum(cost.get("input_tokens") or 0 for cost in costs),
        "observed_output_tokens": sum(cost.get("output_tokens") or 0 for cost in costs),
        "observed_elapsed_ms": sum(cost.get("elapsed_ms") or 0 for cost in costs),
        "unknown_input_token_records": sum(cost.get("input_tokens") is None for cost in costs),
        "unknown_output_token_records": sum(cost.get("output_tokens") is None for cost in costs),
        "unknown_elapsed_records": sum(cost.get("elapsed_ms") is None for cost in costs)}


@contextmanager
def record_judge_http(endpoint, totals_path):
    """Accumulate scoring cost without saving per-call messages or logs."""
    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    lock = Lock()
    totals = (json.loads(totals_path.read_text()) if totals_path.exists() else
              {"http_calls": 0, "input_tokens": 0, "output_tokens": 0, "elapsed_ms": 0.0})

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            path = self.path.removeprefix("/v1")
            started = time.perf_counter()
            with httpx.Client(timeout=180) as client:
                response = client.post(endpoint+path, json=payload)
            raw = response.json()
            usage = raw.get("usage", {})
            with lock:
                totals["http_calls"] += 1
                totals["elapsed_ms"] += (time.perf_counter()-started)*1000
                totals["input_tokens"] += usage.get("prompt_tokens", 0)
                totals["output_tokens"] += usage.get("completion_tokens", 0)
                totals_path.write_text(json.dumps(totals)+"\n")
            body = response.content
            self.send_response(response.status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", totals
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


async def _command(arguments, *, environment=None):
    started = time.perf_counter()
    result = await asyncio.to_thread(subprocess.run, arguments, text=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=environment)
    result.check_returncode()
    return {"elapsed_ms": (time.perf_counter()-started)*1000}


async def _official_scores(config, spec, directory, settings):
    generation = directory.parent
    predictions = load_jsonl(generation/"official.jsonl")
    references = load_jsonl(generation/"references.jsonl")
    merged = merge_offline_labels(predictions, references)
    prepared = directory/"official.with_labels.jsonl"
    _write(prepared, merged)
    primary = spec["primary_metric"]
    metrics = ("RB_llm", "RL_F") if primary == "mtrageval_harmonic" else ("RB_llm",)
    judge = _judge_config(config, settings)
    python = settings.get("python", sys.executable)
    checkout = settings.get("official_checkout", "data/mtrag-official")
    target = directory/"official"
    ragas_max_tokens = settings.get("ragas_max_tokens") if "RL_F" in metrics else None
    with record_judge_http(judge["base_url"], directory/"judge_cost.json") as (endpoint, call_cost):
        command = [python, "-m", "bench.official", "generation-http", "--checkout", checkout,
            "--input", str(prepared), "--output", str(target), "--model", judge["model"],
            "--judge-revision", judge["revision"], "--endpoint", endpoint,
            "--max-tokens", str(judge.get("max_tokens", 2048)), "--no-enable-thinking",
            "--metrics", *metrics]
        if primary != "mtrageval_harmonic":
            command.append("--no-answerability-adjustment")
        if settings.get("ragas_json_schema", False):
            command.append("--ragas-json-schema")
        if ragas_max_tokens is not None:
            command.extend(("--ragas-max-tokens", str(ragas_max_tokens)))
        receipt = await _command(command)
    records = load_jsonl(target/"scores.jsonl")
    if primary == "mtrageval_harmonic":
        algorithmic = directory/"official.algorithmic.jsonl"
        if not algorithmic.exists():
            pending = algorithmic.with_suffix(".pending.jsonl")
            command = [python, "-m", "bench.official", "algorithmic", "--checkout", checkout,
                "--input", str(prepared), "--output", str(pending), "--python", python]
            environment = dict(os.environ)
            if "gpu_devices" in settings:
                environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, settings["gpu_devices"]))
            await _command(command, environment=environment)
            if sorted(row["task_id"] for row in load_jsonl(pending)) != sorted(row["task_id"] for row in merged):
                raise ValueError("official algorithmic output changed the task cohort")
            _write(algorithmic, [{"task_id": row["task_id"], "metrics": {"RB_agg": row["metrics"]["RB_agg"]}}
                                 for row in load_jsonl(pending)])
            pending.unlink()
        algorithmic_rows = {row["task_id"]: row for row in load_jsonl(algorithmic)}
        for row in records:
            row["metrics"]["RB_agg"] = algorithmic_rows[row["task_id"]]["metrics"]["RB_agg"]
        condition_metrics(load_conditioning_rules(checkout), records, ("RL_F", "RB_llm", "RB_agg"))
    identity = {"protocol": "pinned_mtrag_official_http", "judge": judge,
        "official_source": {key: value for key, value in json.loads(
            (Path(checkout)/"contextkv-source.json").read_text()).items() if key in {"repository", "commit"}},
        "requested_metrics": metrics, "answerability_adjustment": primary == "mtrageval_harmonic",
        "metric_scope": "project_RB_llm" if primary == "RB_llm" else "MTRAGEval_generation_components"}
    scorer = json.loads((target/"identity.json").read_text())
    if "ragas_max_tokens" in scorer:
        identity["ragas_max_tokens"] = scorer["ragas_max_tokens"]
    call_cost["stage_wall_ms"] = receipt["elapsed_ms"]
    return records, identity, call_cost


def _overlap_scores(directory):
    started = time.perf_counter()
    report = reference_overlap(load_jsonl(directory.parent/"references.jsonl"),
                               load_jsonl(directory.parent/"official.jsonl"))
    records = [{"task_id": task_id, "metrics": metrics,
        "judge_cost": {"input_tokens": 0, "output_tokens": 0, "elapsed_ms": 0}}
        for task_id, metrics in report["tasks"].items()]
    identity = {"protocol": "multidoc2dial_author_token_f1_em", "score_range": [0, 1],
        "implementation": "bench.official.reference_overlap",
        "upstream": "https://aclanthology.org/2021.emnlp-main.498/"}
    cost = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0,
            "elapsed_ms": (time.perf_counter()-started)*1000, "cost_scope": "offline_cpu"}
    return records, identity, cost


async def _loong_scores(config, spec, directory, settings):
    from bench.judge import OpenAICompletionClient, JudgeConfig
    from bench.loong_score import COMMIT, score_answer
    generation = directory.parent
    references = load_jsonl(generation/"references.jsonl")
    predictions = {row["task_id"]: row["predictions"][0]["text"]
                   for row in load_jsonl(generation/"official.jsonl")}
    judge = _judge_config(config, settings)
    output_tokens = judge.get("max_tokens", 2048)
    client = OpenAICompletionClient(JudgeConfig(judge["base_url"], judge["model"], judge["revision"],
        max_new_tokens=output_tokens,
        timeout_seconds=settings.get("timeout_seconds", 180)))
    target = directory/"loong.scores.jsonl"
    records = load_jsonl(target) if target.exists() else []
    finished = {record["task_id"] for record in records}
    try:
        with target.open("a") as output:
            for reference in references:
                if reference["task_id"] in finished:
                    continue
                record = await asyncio.to_thread(score_answer, reference,
                    predictions.get(reference["task_id"]), client)
                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False)+"\n")
                output.flush()
    finally:
        client.close()
    return records, {"protocol": "loong_reference_rubric_local_judge", "upstream_commit": COMMIT,
        "judge": client.manifest, "score_range": [0, 100]}, _cost_totals(records)


async def _versionqa_scores(config, spec, directory, settings):
    from bench.judge import OpenAICompletionClient, JudgeConfig
    from bench.versionqa import SOURCE_COMMIT, score_answer

    references = load_jsonl(directory.parent/"references.jsonl")
    predictions = {row["task_id"]: row["predictions"][0]["text"]
                   for row in load_jsonl(directory.parent/"official.jsonl")}
    judge = _judge_config(config, settings)
    client = OpenAICompletionClient(JudgeConfig(judge["base_url"], judge["model"], judge["revision"],
        max_new_tokens=16,
        timeout_seconds=settings.get("timeout_seconds", 180)))
    target = directory/"versionqa.scores.jsonl"
    records = load_jsonl(target) if target.exists() else []
    finished = {record["task_id"] for record in records}
    try:
        with target.open("a") as output:
            for reference in references:
                if reference["task_id"] in finished:
                    continue
                record = await asyncio.to_thread(score_answer, reference,
                    predictions.get(reference["task_id"]), client)
                records.append(record)
                output.write(json.dumps(record, ensure_ascii=False)+"\n")
                output.flush()
    finally:
        client.close()
    return records, {"protocol": "versionqa_author_binary_accuracy", "upstream_commit": SOURCE_COMMIT,
        "judge": client.manifest, "score_range": [0, 1],
        "judge_view": "original question, reference answer and model answer",
        "adaptation": "author binary correctness rubric through the configured judge endpoint"}, _cost_totals(records)


def _record_metrics(record, spec, settings):
    def observed(value):
        if isinstance(value, (list, tuple)):
            return bool(value) and all(observed(item) for item in value)
        return isinstance(value, (int, float)) and math.isfinite(value)

    metrics = {key: value for key, value in (record.get("metrics") or {}).items() if observed(value)}
    primary = spec["primary_metric"]
    if primary == "mtrageval_harmonic":
        for suffix in ("", "_idk", "_idk_underspecified"):
            if "RB_agg"+suffix in metrics:
                metrics["RB_alg"+suffix] = metrics["RB_agg"+suffix]
        retained = {name+suffix for name in ("RL_F", "RB_llm", "RB_alg")
                    for suffix in ("", "_idk", "_idk_underspecified")}
        metrics = {key: statistics.fmean(value) if isinstance(value, (list, tuple)) else value
                   for key, value in metrics.items() if key in retained}
    if primary in {"loong_reference_score", "versionqa_accuracy"} and observed(record.get("score")):
        metrics[primary] = record["score"]
        if primary == "loong_reference_score":
            metrics["loong_perfect_rate"] = int(record["score"] == 100)
    return metrics


async def score_dataset(config, dataset_spec, rows, generation_dir, *, expected_task_ids=None, split=None):
    """Add benchmark metrics and a separate scoring ledger to saved result rows."""
    directory = Path(generation_dir)/"scoring"
    directory.mkdir(parents=True, exist_ok=True)
    generation_manifest = Path(generation_dir)/"generation_manifest.json"
    declaration = json.loads(generation_manifest.read_text()) if generation_manifest.exists() else {}
    split = split or declaration.get("split") or next((row["split"] for row in rows if "split" in row), None)
    dataset_spec = split_metric_spec(dataset_spec, split)
    if expected_task_ids is None and declaration:
        expected_task_ids = declaration["task_ids" if "task_ids" in declaration else "expected_tasks"]
    expected = list(expected_task_ids) if expected_task_ids is not None else [row["task_id"] for row in rows]
    settings = _settings(config, dataset_spec)
    started = time.perf_counter()
    primary = dataset_spec["primary_metric"]
    if not any(row["status"] == "completed" for row in rows):
        records = []
        identity = {"protocol": primary, "generated_answers": 0}
        judge_cost = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "elapsed_ms": 0}
    elif primary == "f1":
        records, identity, judge_cost = _overlap_scores(directory)
    elif primary in {"RB_llm", "mtrageval_harmonic"}:
        records, identity, judge_cost = await _official_scores(config, dataset_spec, directory, settings)
    elif primary == "loong_reference_score":
        records, identity, judge_cost = await _loong_scores(config, dataset_spec, directory, settings)
    elif primary == "versionqa_accuracy":
        records, identity, judge_cost = await _versionqa_scores(config, dataset_spec, directory, settings)
    else:
        raise ValueError(f"No declared scorer for {primary}")
    by_id = {str(record["task_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Scoring records contain duplicate task IDs")
    updated = []
    for row in rows:
        record = by_id.get(str(row["task_id"]))
        metrics = _record_metrics(record, dataset_spec, settings) if record is not None else {}
        observed_metrics = metrics
        has_primary = all(_metric({"metrics": observed_metrics}, name, dataset_spec)[0] is not None
                          for name in _required(dataset_spec))
        updated.append({**row, "metrics": observed_metrics,
            "scoring": {
                "status": "failed" if record is not None and record.get("status") == "failed" else "scored" if has_primary else "missing",
                "error": record.get("error") if record is not None else None,
                "judge_cost": record.get("judge_cost") if record else None}})
    manifest = {"identity": identity, "primary_metric": dataset_spec["primary_metric"], "split": split,
        "normalization": dataset_spec.get("normalization", 1.0), "requested": len(expected),
        "generated_answers": sum(row["status"] == "completed" for row in rows),
        "scored": sum(row["scoring"]["status"] == "scored" for row in updated),
        "judge_cost": judge_cost, "wall_elapsed_ms": (time.perf_counter()-started)*1000}
    (directory/"manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    write_task_metrics(generation_dir, updated)
    return updated


def main(argv=None):
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--config")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"))
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--rows")
    args = parser.parse_args(argv)
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    config = yaml.safe_load(Path(args.config or matrix["study_config"]).read_text())
    if args.rows:
        rows = load_jsonl(args.rows)
    else:
        from bench.study import saved_study_rows
        rows = [row for row in saved_study_rows(args.generation_dir) if row["status"] == "completed"]
    scored = asyncio.run(score_dataset(config, matrix["datasets"][args.dataset], rows, args.generation_dir,
        expected_task_ids=matrix.get("cohorts", {}).get(args.split, {}).get(args.dataset), split=args.split))
    if all(row["status"] != "completed" or row["scoring"]["status"] == "scored" for row in scored):
        retain_metrics(args.generation_dir)
    print((Path(args.generation_dir)/"scoring/manifest.json").read_text())


if __name__ == "__main__":
    main()
