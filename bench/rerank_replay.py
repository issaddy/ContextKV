"""Rerank frozen original-passage pools without new retrieval or answer generation."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import time

from adapters.hybrid_retrieval import query_text
from adapters.mtrag import OnlineContext
from adapters.qwen3_reranker import DEFAULT_INSTRUCTION, LocalQwen3Reranker
from bench.quality import load_qrels
from bench.retrieval_eval import file_source, summarize


def frozen_pools(rows, manifest, *, pool_method="current/hybrid", reference_method="current/hybrid_rerank", pool_k=50):
    """Require the same complete cohort/query/pool used by the saved BGE reranker."""
    if manifest.get("status") != "complete":
        raise ValueError("source retrieval run must be complete")
    mode, stage = pool_method.split("/")
    if stage != "hybrid" or pool_k < 5:
        raise ValueError("replay requires a hybrid pool covering the final top five")
    tasks = {row["task_id"]: row for row in manifest["tasks"]}
    if not tasks or len(tasks) != len(manifest["tasks"]):
        raise ValueError("frozen source manifest must contain unique nonempty tasks")
    methods = {}
    for name in (pool_method, reference_method):
        selected = [row for row in rows if row["method"] == name]
        mapped = {row["task_id"]: row for row in selected}
        if len(mapped) != len(selected) or set(mapped) != set(tasks):
            raise ValueError("pool/reference must cover the same complete frozen cohort")
        methods[name] = mapped
    result = []
    for task_id, task in sorted(tasks.items()):
        pool, reference = methods[pool_method][task_id], methods[reference_method][task_id]
        expected_query = query_text(task["user_questions"], mode, generated_history=task.get("generated_history"))
        for row in (pool, reference):
            if (row["status"] != "complete" or row["query_text"] != expected_query
                    or row["conversation_id"] != task["conversation_id"] or row["turn"] != task["turn"]):
                raise ValueError("source result differs from its frozen query/task")
        ids = pool["candidate_ids"][:pool_k]
        if len(ids) != pool_k or len(set(ids)) != pool_k:
            raise ValueError("source hybrid pool has fewer unique passages than requested")
        if (reference["rerank_candidate_ids"] != ids or len(reference["candidate_ids"]) != pool_k
                or set(reference["candidate_ids"]) != set(ids)):
            raise ValueError("saved reference reranker used a different candidate pool")
        result.append({"task_id": task_id, "conversation_id": task["conversation_id"], "turn": task["turn"],
            "query_mode": mode, "query_text": expected_query, "candidate_ids": list(ids),
            "reference": deepcopy(reference), "pool_elapsed_ms": pool["elapsed_ms"]})
    return result


def load_pool_passages(directory, source_manifest, pools):
    """Read original saved passages, without loading BM25 or embedding vectors."""
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    if manifest["corpus_sha256"] != source_manifest["sparse"]["corpus_sha256"]:
        raise ValueError("passage corpus differs from the frozen retrieval run")
    source = file_source(directory/"passages.jsonl")
    if source["sha256"] != manifest["files"]["passages.jsonl"]:
        raise ValueError("original passage file differs from the saved corpus manifest")
    wanted = {pid for pool in pools for pid in pool["candidate_ids"]}
    passages = {}
    with (directory/"passages.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["document_id"] in wanted:
                if row["document_id"] in passages:
                    raise ValueError("duplicate passage ID in frozen candidate pool")
                passages[row["document_id"]] = OnlineContext(**row)
    if set(passages) != wanted:
        raise ValueError("candidate passages are absent from the original corpus")
    return passages, source


def replay_pools(pools, passages, scorer, *, method="current/qwen3_rerank"):
    """No qrels, reference answers or generation outputs enter scorer.score."""
    for pool in pools:
        contexts = tuple(passages[pid] for pid in pool["candidate_ids"])
        started = time.perf_counter()
        base = {key: pool[key] for key in ("task_id", "conversation_id", "turn", "query_mode", "query_text")}
        base.update(method=method, stage="qwen3_rerank", rerank_candidate_ids=pool["candidate_ids"],
                    timing_scope="new_reranker_only_wall_time", source_pool_elapsed_ms=pool["pool_elapsed_ms"])
        try:
            scores = scorer.score(pool["query_text"], contexts)
            if len(scores) != len(contexts):
                raise ValueError("reranker scores do not align with the frozen candidates")
            if not all(math.isfinite(float(score)) for score in scores):
                raise ValueError("reranker returned non-finite scores")
            ordered = sorted(zip(contexts, scores), key=lambda item: (-float(item[1]), item[0].document_id))
            yield {**base, "status": "complete", "candidate_ids": [c.document_id for c, _ in ordered],
                "candidate_scores": [float(score) for _, score in ordered],
                "contexts": [asdict(replace(c, score=float(score))) for c, score in ordered[:5]],
                "elapsed_ms": (time.perf_counter()-started)*1000,
                "pair_diagnostics": deepcopy(scorer.last_pairs)}
        except Exception as exc:
            yield {**base, "status": "failed", "candidate_ids": [], "candidate_scores": [], "contexts": [],
                "elapsed_ms": (time.perf_counter()-started)*1000, "error": f"{type(exc).__name__}: {exc}"}


def reference_rows(pools):
    result = []
    for pool in pools:
        reference = deepcopy(pool["reference"])
        reference["source_stage_elapsed_ms"] = reference["elapsed_ms"]
        reference["source_pool_elapsed_ms"] = pool["pool_elapsed_ms"]
        reference["elapsed_ms"] -= pool["pool_elapsed_ms"]
        reference["timing_scope"] = "saved_BGE_reranker_stage_difference; separate earlier run"
        result.append(reference)
    return result


def analyze_replay(pools, rows, qrels):
    if len(rows) != len(pools) or {row["task_id"] for row in rows} != {pool["task_id"] for pool in pools}:
        raise ValueError("replay analysis requires the complete frozen cohort, including failures")
    result = summarize(reference_rows(pools)+rows, qrels, candidate_k=len(pools[0]["candidate_ids"]))
    result.update(expected_tasks=len(pools), completed_tasks=sum(r["status"] == "complete" for r in rows),
        failed_task_ids=[r["task_id"] for r in rows if r["status"] != "complete"],
        failure_metric_policy="failed reranks retain an empty ranking and count as zero on labelled queries",
        timing_scope="Qwen reranker-only wall time; BGE reranker derived from saved total-minus-shared-pool; "
                     "different runs/devices or load conditions may differ, not a controlled latency comparison",
        truncated_pairs=sum(item["truncated"] for row in rows for item in row.get("pair_diagnostics", [])),
        quality_scope="same original 50-candidate pool and query; same token cap, different tokenizer/prompt coverage")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--corpus-index", required=True, help="saved BM25 directory; only passages and manifest are read")
    parser.add_argument("--pool-method", default="current/hybrid")
    parser.add_argument("--reference-method", default="current/hybrid_rerank")
    parser.add_argument("--pool-k", type=int, default=50)
    parser.add_argument("--model-path", help="prepared fixed local Qwen3-Reranker-0.6B snapshot")
    parser.add_argument("--dry-run", action="store_true", help="export frozen pools without loading a model or qrels")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--qrels", help="opened only after all scoring completes")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if not args.dry_run and not args.model_path:
        parser.error("scoring requires a prepared --model-path; use --dry-run to inspect frozen pools")
    source = Path(args.source_run)
    original = json.loads((source/"manifest.json").read_text())
    if original["reranker"]["max_length"] != args.max_length:
        parser.error("max-length must match the saved BGE reference budget for this paired comparison")
    input_rows = [json.loads(line) for line in (source/"retrieval.jsonl").read_text().splitlines() if line]
    pools = frozen_pools(input_rows, original, pool_method=args.pool_method,
                         reference_method=args.reference_method, pool_k=args.pool_k)
    passages, passage_source = load_pool_passages(args.corpus_index, original, pools)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "status": "dry_run" if args.dry_run else "running", "config": vars(args),
        "sources": {"retrieval": file_source(source/"retrieval.jsonl"), "manifest": file_source(source/"manifest.json"),
                    "passages": passage_source, "corpus_manifest": file_source(Path(args.corpus_index)/"manifest.json")},
        "source_retrieval_manifest": original, "expected_tasks": len(pools), "pool_k": args.pool_k,
        "measurement_kind": "dry_run" if args.dry_run else "local_pretrained_rerank",
        "scope": "no new retrieval, qrels-guided candidate changes, answer generation, or online default changes"}
    path = output/"manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    (output/"pools.jsonl").write_text("".join(json.dumps(pool, ensure_ascii=False)+"\n" for pool in pools))
    if args.dry_run:
        print(json.dumps({"output": str(output), "tasks": len(pools), "status": "dry_run"}))
        return
    scorer, rows = None, []
    try:
        scorer = LocalQwen3Reranker(args.model_path, device=args.device, dtype=args.dtype,
                                   batch_size=args.batch_size, max_length=args.max_length, instruction=args.instruction)
        manifest["model"] = scorer.manifest
        with (output/"reranked.jsonl").open("w", encoding="utf-8") as stream:
            for row in replay_pools(pools, passages, scorer, method=args.pool_method.split("/")[0]+"/qwen3_rerank"):
                rows.append(row)
                stream.write(json.dumps(row, ensure_ascii=False)+"\n")
                stream.flush()
                print(json.dumps({"scored_tasks": len(rows), "expected_tasks": len(pools), "status": row["status"]}), flush=True)
        if args.qrels:
            summary = analyze_replay(pools, rows, load_qrels(args.qrels))
            manifest["sources"]["qrels"] = file_source(args.qrels)
            (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
        manifest["status"] = "complete" if all(r["status"] == "complete" for r in rows) else "completed_with_failures"
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["completed_stage_rows"] = len(rows)
        if scorer is not None:
            manifest["model_usage"] = {"pairs_processed": scorer.texts_processed, "truncated_pairs": scorer.truncated_pairs}
            scorer.close()
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
