"""Offline retrieval ablations on original passages; labels enter only analysis.

Build a local BGE index, then compare BM25/dense/RRF/reranking with frozen query
views. This performs no answer generation and does not change serving defaults.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
import time

from adapters.hybrid_retrieval import DenseIndex, LocalBGE, query_text, search_variants
from adapters.mtrag import CLOUD_COLLECTION, DATASET_COMMIT, OnlineContext, load_rows
from adapters.retrieval import BM25SRetriever, load_beir_corpus
from adapters.query_views import user_topic_manifest, user_topic_view
from bench.metrics import percentile
from bench.quality import load_qrels, retrieval_quality


@dataclass(frozen=True)
class RetrievalQuery:
    task_id: str
    conversation_id: str
    turn: int
    user_questions: tuple[str, ...]
    generated_history: tuple[dict, ...] | None = None


def prepare_queries(rows, split_manifest, *, split="dev", collection=CLOUD_COLLECTION,
                    history_audits=None):
    """Only user messages cross this boundary; official assistants/labels do not."""
    ids = set(split_manifest[split])
    histories = None
    if history_audits is not None:
        histories = {r["task_id"]: r for r in history_audits}
        if len(histories) != len(history_audits):
            raise ValueError("generated history audit has duplicate task IDs")
    queries = []
    for row in sorted(rows, key=lambda r: (str(r["conversation_id"]), int(r["turn"]))):
        if row["conversation_id"] not in ids or row["Collection"] != collection:
            continue
        inputs = row["input"]
        users = tuple(m["text"] for m in inputs if m["speaker"] in {"user", "human"})
        if not users or len(users) != int(row["turn"]) or inputs[-1]["text"] != users[-1]:
            raise ValueError("task must contain user questions only through its current turn")
        history = None
        if histories is not None:
            record = histories.get(row["task_id"])
            if record is None or record["protocol"] not in {"P2", "P3"}:
                raise ValueError("every task needs an actual P2/P3 history audit")
            if record["query"] != users[-1] or record["conversation_id"] != row["conversation_id"]:
                raise ValueError("generated history audit belongs to a different query")
            history = tuple({"speaker": m["speaker"], "text": m["text"]}
                            for m in record["actual_history"])
            if (len(history) != 2*(int(row["turn"])-1)
                    or any(m["speaker"] != ("user" if i % 2 == 0 else "assistant")
                           for i, m in enumerate(history))
                    or tuple(m["text"] for m in history if m["speaker"] == "user") != users[:-1]):
                raise ValueError("saved actual history must contain exactly prior complete turns")
        queries.append(RetrievalQuery(str(row["task_id"]), str(row["conversation_id"]),
                                     int(row["turn"]), users, history))
    if not queries:
        raise ValueError("selected retrieval cohort is empty")
    if len({q.task_id for q in queries}) != len(queries):
        raise ValueError("duplicate task IDs in retrieval cohort")
    return queries


def evaluate_queries(queries, sparse, dense=None, encoder=None, reranker=None, *,
                     modes=("last3",), candidate_k=50, rerank_k=50, rrf_k=60):
    """Yield per-stage results. Neither qrels nor reference answers are inputs."""
    if reranker is not None and dense is None:
        raise ValueError("hybrid reranking requires a dense index")
    for item in queries:
        for mode in modes:
            view_started = time.perf_counter()
            view = user_topic_view(item.user_questions) if mode == "user_topic" else None
            query = view["text"] if view else query_text(item.user_questions, mode, generated_history=item.generated_history)
            view_ms = (time.perf_counter()-view_started)*1000
            stages = search_variants(query, sparse, dense, encoder, reranker,
                                     candidate_k=candidate_k, rerank_k=rerank_k, rrf_k=rrf_k)
            for stage, values in stages.items():
                yield {"task_id": item.task_id, "conversation_id": item.conversation_id,
                    "turn": item.turn, "query_mode": mode, "query_text": query, "stage": stage,
                    "query_view_trace": view, "query_view_ms": view_ms,
                    "method": f"{mode}/{stage}", "status": "complete",
                    "contexts": [asdict(c) for c in values["contexts"][:5]],
                    "candidate_ids": [c.document_id for c in values["contexts"]],
                    "candidate_scores": [c.score for c in values["contexts"]],
                    "rerank_candidate_ids": values.get("rerank_candidate_ids"),
                    "elapsed_ms": view_ms+values["elapsed_ms"]}


def summarize(rows, qrels, *, candidate_k=50):
    """Post-hoc qrels analysis, with missing labels explicit and sessions retained."""
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
    methods = {}
    for method, items in sorted(by_method.items()):
        retrieved = {r["task_id"]: r["candidate_ids"] for r in items}
        if len(retrieved) != len(items):
            raise ValueError("duplicate task in retrieval method")
        quality = retrieval_quality(retrieved, qrels, k=5)
        pool = retrieval_quality(retrieved, qrels, k=candidate_k)
        sessions = defaultdict(list)
        for row in items:
            if row["task_id"] in quality["tasks"]:
                sessions[row["conversation_id"]].append(quality["tasks"][row["task_id"]])
        latencies = [r["elapsed_ms"] for r in items]
        methods[method] = {"top5": quality, "candidate_pool": pool,
            "sessions": {sid: {"turns": len(values),
                "mean_recall": mean(v["recall"] for v in values),
                "mean_ndcg": mean(v["ndcg"] for v in values)} for sid, values in sessions.items()},
            "elapsed_ms": {"mean": mean(latencies), "p50": percentile(latencies, .5),
                           "p95": percentile(latencies, .95)}}
    return {"schema_version": 1, "methods": methods,
        "answer_quality_evaluated": False,
        "timing_scope": "query-view formation plus sequential stage wall time; excludes model/index loading and answer generation; "
                        "shared stages are reused, not an independent latency benchmark",
        "interpretation": "Qrels measure labelled passage relevance, not answer correctness, "
                          "claim support, completeness or quality noninferiority."}


def file_source(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build dense vectors once from unchanged BEIR passages")
    build.add_argument("--corpus", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--snapshot", default=f"mtrag-{DATASET_COMMIT}")
    evaluate = commands.add_parser("evaluate", help="retrieve first, then read offline qrels")
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--split-manifest", required=True)
    evaluate.add_argument("--split", choices=["dev", "test"], default="dev")
    evaluate.add_argument("--collection", default=CLOUD_COLLECTION)
    evaluate.add_argument("--sparse-index", required=True)
    evaluate.add_argument("--dense-index")
    evaluate.add_argument("--reranker-path")
    evaluate.add_argument("--reranker-max-length", type=int, default=512)
    evaluate.add_argument("--qrels", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--query-modes", nargs="+", choices=["current", "last3", "all_user", "correction_aware", "user_topic", "generated_history"], default=["last3"])
    evaluate.add_argument("--history-audit")
    evaluate.add_argument("--candidate-k", type=int, default=50)
    evaluate.add_argument("--rerank-k", type=int, default=50)
    evaluate.add_argument("--rrf-k", type=int, default=60)
    evaluate.add_argument("--cpu-threads", type=int, default=1)
    for sub in (build, evaluate):
        sub.add_argument("--encoder-path", required=sub is build)
        sub.add_argument("--encoder-max-length", type=int, default=8192)
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
        sub.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error("output already exists; select a new directory to preserve prior results")
    if args.command == "evaluate":
        if args.dense_index and not args.encoder_path:
            parser.error("--dense-index requires --encoder-path")
        if args.reranker_path and not args.dense_index:
            parser.error("--reranker-path requires --dense-index")
        if "generated_history" in args.query_modes and not args.history_audit:
            parser.error("generated_history requires --history-audit from an actual P2/P3 run")
        if len(set(args.query_modes)) != len(args.query_modes):
            parser.error("query modes must be unique")
    encoder = (LocalBGE(args.encoder_path, device=args.device, dtype=args.dtype,
                       batch_size=args.batch_size, max_length=args.encoder_max_length)
               if args.encoder_path else None)
    if args.command == "build":
        contexts = [OnlineContext(**value) for value in load_beir_corpus(args.corpus).values()]
        index = DenseIndex.build(contexts, encoder, args.output, corpus_snapshot_id=args.snapshot,
                                 show_progress=True)
        print(json.dumps(index.manifest, indent=2))
        return
    output = Path(args.output)
    output.mkdir(parents=True)
    histories = ([json.loads(line) for line in Path(args.history_audit).read_text().splitlines() if line]
                 if args.history_audit else None)
    queries = prepare_queries(load_rows(args.input), json.loads(Path(args.split_manifest).read_text()),
                              split=args.split, collection=args.collection, history_audits=histories)
    sparse = BM25SRetriever.load(args.sparse_index, cpu_threads=args.cpu_threads)
    dense = DenseIndex.load(args.dense_index) if args.dense_index else None
    reranker = (LocalBGE(args.reranker_path, reranker=True, device=args.device, dtype=args.dtype,
                        batch_size=args.batch_size, max_length=args.reranker_max_length)
                if args.reranker_path else None)
    manifest = {"schema_version": 1, "status": "running", "config": vars(args),
        "sources": {"input": file_source(args.input), "split_manifest": file_source(args.split_manifest),
                    "history_audit": file_source(args.history_audit) if args.history_audit else None},
        "tasks": [asdict(q) for q in queries], "sparse": sparse.manifest,
        "dense": dense.manifest if dense else None, "encoder": encoder.manifest if encoder else None,
        "reranker": reranker.manifest if reranker else None,
        "measurement_kind": "local_pretrained_retrieval" if dense else "local_bm25_retrieval",
        "scope": "offline fixed-query retrieval exploration; no answer generation or online policy changes"}
    if "user_topic" in args.query_modes:
        manifest["query_view"] = user_topic_manifest()
    path = output/"manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    results = []
    try:
        with (output/"retrieval.jsonl").open("w") as stream:
            for row in evaluate_queries(queries, sparse, dense, encoder, reranker, modes=args.query_modes,
                                        candidate_k=args.candidate_k, rerank_k=args.rerank_k, rrf_k=args.rrf_k):
                results.append(row)
                stream.write(json.dumps(row, ensure_ascii=False)+"\n")
                stream.flush()
        # Ranking is finished before the qrels are opened, including missing-label accounting.
        summary = summarize(results, load_qrels(args.qrels), candidate_k=args.candidate_k)
        manifest["sources"]["qrels"] = file_source(args.qrels)
        (output/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
        manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", completed_stage_rows=len(results))
        raise
    finally:
        manifest["model_usage"] = {name: {"texts_processed": model.texts_processed,
            "token_limit_hits": model.token_limit_hits,
            "note": "at configured token cap; may include exact-length inputs, not proven truncation"}
            for name, model in (("encoder", encoder), ("reranker", reranker)) if model is not None}
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({name: {"recall@5": value["top5"]["mean_recall"], "ndcg@5": value["top5"]["mean_ndcg"]}
                      for name, value in summary["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
