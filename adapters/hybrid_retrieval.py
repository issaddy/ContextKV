"""Optional local BGE dense retrieval, RRF and cross-encoder reranking.

Original passages and the BM25 baseline remain unchanged. Heavy dependencies
are imported only by explicit model/index operations; no runtime downloads.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from threading import Lock
import time

from .mtrag import OnlineContext, json_hash
from .query_views import correction_query_mode, user_topic_manifest, user_topic_view
from .retrieval import deterministic_top_k, retrieval_query


ENCODER_ID = "BAAI/bge-m3"
ENCODER_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER_ID = "BAAI/bge-reranker-v2-m3"
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def query_text(user_questions, mode="last3", *, generated_history=None):
    """No answers/labels are accepted implicitly; history must be supplied explicitly."""
    if not user_questions:
        raise ValueError("retrieval requires the current user question")
    if mode == "correction_aware":
        mode = correction_query_mode(user_questions[-1])
    if mode == "user_topic":
        return user_topic_view(user_questions)["text"]
    if mode == "current":
        return user_questions[-1]
    if mode == "last3":
        return retrieval_query(user_questions)
    if mode == "all_user":
        return "\n".join(user_questions)
    if mode == "generated_history":
        if generated_history is None:
            raise ValueError("generated_history mode requires a saved actual P2/P3 history")
        # Current question comes first so length caps retain the retrieval target.
        history = "\n".join(f"{m['speaker']}: {m['text']}" for m in generated_history[-6:])
        return f"Current question: {user_questions[-1]}\nConversation context:\n{history}"
    raise ValueError("unknown query representation")


def reciprocal_rank_fusion(rankings, *, k=60):
    """Equal-weight RRF; duplicate IDs cannot vote twice within one ranking."""
    if k < 0:
        raise ValueError("RRF rank constant must be nonnegative")
    scores = {}
    for ranking in rankings:
        for rank, pid in enumerate(dict.fromkeys(ranking), 1):
            scores[pid] = scores.get(pid, 0.0)+1/(k+rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class LocalBGE:
    """Official Transformers-compatible checkpoints; explicit device/dtype/path."""
    def __init__(self, model_path, *, reranker=False, device="cpu", dtype="float32",
                 batch_size=16, max_length=None):
        if batch_size < 1 or (max_length is not None and max_length < 1):
            raise ValueError("model batch size and token limit must be positive")
        import torch
        from importlib.metadata import version
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
        self.torch, self.batch_size, self.reranker = torch, batch_size, reranker
        self.max_length = max_length if max_length is not None else (512 if reranker else 8192)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        loader = AutoModelForSequenceClassification if reranker else AutoModel
        self.model = loader.from_pretrained(str(model_path), local_files_only=True)
        self.model.to(device=device, dtype=getattr(torch, dtype)).eval()
        self.token_limit_hits = 0
        self.texts_processed = 0
        self.manifest = {"model_id": RERANKER_ID if reranker else ENCODER_ID,
            "revision": RERANKER_REVISION if reranker else ENCODER_REVISION,
            "model_path": str(Path(model_path).resolve()), "revision_provenance": "operator_supplied_local_snapshot",
            "license": "Apache-2.0" if reranker else "MIT", "device": str(self.model.device),
            "dtype": str(self.model.dtype), "max_length": self.max_length, "batch_size": batch_size,
            "packages": {name: version(name) for name in ("torch", "transformers", "numpy")},
            "implementation": "Transformers official checkpoint API",
            "pooling": None if reranker else "CLS then L2 normalization; BGE-M3 needs no query instruction"}

    def _forward(self, values):
        outputs = []
        for start in range(0, len(values), self.batch_size):
            batch = values[start:start+self.batch_size]
            encoded = self.tokenizer(batch, padding=True, truncation=True,
                                     max_length=self.max_length, return_tensors="pt")
            lengths = encoded["attention_mask"].sum(dim=1)
            self.token_limit_hits += int((lengths >= self.max_length).sum())
            self.texts_processed += len(batch)
            encoded = {name: value.to(self.model.device) for name, value in encoded.items()}
            with self.torch.inference_mode():
                result = self.model(**encoded, return_dict=True)
                value = (result.logits.reshape(-1) if self.reranker else
                         self.torch.nn.functional.normalize(result.last_hidden_state[:, 0], p=2, dim=1))
            outputs.append(value.float().cpu().numpy())
        import numpy as np
        return np.concatenate(outputs, axis=0) if outputs else np.array([])

    def encode(self, texts):
        if self.reranker:
            raise ValueError("cross-encoder is not a dense encoder")
        return self._forward(list(texts))

    def score(self, query, contexts):
        if not self.reranker:
            raise ValueError("dense encoder is not a cross-encoder")
        # Raw logits provide a ranking, not a calibrated relevance probability.
        return self._forward([(query, c.title+"\n"+c.text) for c in contexts])

    def close(self):
        """Release this process's model between matrix jobs; metadata stays readable."""
        device = self.model.device
        del self.model
        if device.type == "cuda":
            with self.torch.cuda.device(device):
                self.torch.cuda.empty_cache()


class DenseIndex:
    def __init__(self, contexts, vectors, manifest):
        self.contexts, self.vectors, self.manifest = tuple(contexts), vectors, manifest
        self.by_id = {context.document_id: context for context in self.contexts}
        if len(self.by_id) != len(self.contexts) or vectors.shape[0] != len(self.contexts):
            raise ValueError("dense vector rows must align with unique passage IDs")

    @classmethod
    def build(cls, contexts, encoder, directory, *, corpus_snapshot_id, batch_size=128,
              show_progress=False):
        import numpy as np
        if batch_size < 1:
            raise ValueError("index build batch size must be positive")
        contexts = tuple(sorted(contexts, key=lambda c: c.document_id))
        if not contexts:
            raise ValueError("dense corpus must not be empty")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {"schema_version": 1, "status": "building", "corpus_snapshot_id": corpus_snapshot_id,
            "corpus_sha256": json_hash([asdict(c) for c in contexts]),
            "passages": len(contexts), "encoder": encoder.manifest,
            "search": "exact normalized inner product", "passage_order": "ascending_document_id"}
        target = directory/"manifest.json"
        target.write_text(json.dumps(manifest, indent=2)+"\n")
        started, vectors = time.perf_counter(), None
        for start in range(0, len(contexts), batch_size):
            batch = contexts[start:start+batch_size]
            values = encoder.encode([c.title+"\n"+c.text for c in batch])
            if vectors is None:
                vectors = np.lib.format.open_memmap(directory/"vectors.npy", mode="w+", dtype="float32",
                                                   shape=(len(contexts), values.shape[1]))
            vectors[start:start+len(batch)] = values
            vectors.flush()
            completed = start+len(batch)
            if show_progress and (start//4096 != completed//4096 or completed == len(contexts)):
                print(json.dumps({"encoded_passages": completed, "total_passages": len(contexts),
                                  "elapsed_seconds": time.perf_counter()-started}), flush=True)
        (directory/"passages.jsonl").write_text("".join(json.dumps(asdict(c), ensure_ascii=False)+"\n" for c in contexts))
        manifest.update(status="complete", dimensions=vectors.shape[1], build_seconds=time.perf_counter()-started,
                        token_limit_hits=getattr(encoder, "token_limit_hits", None))
        target.write_text(json.dumps(manifest, indent=2)+"\n")
        return cls(contexts, vectors, manifest)

    @classmethod
    def load(cls, directory):
        import numpy as np
        directory = Path(directory)
        manifest = json.loads((directory/"manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError("dense index build has not completed")
        contexts = [OnlineContext(**json.loads(line)) for line in (directory/"passages.jsonl").read_text().splitlines() if line]
        return cls(contexts, np.load(directory/"vectors.npy", mmap_mode="r", allow_pickle=False), manifest)

    def search(self, query, encoder, *, top_k=50):
        import numpy as np
        for field in ("model_id", "revision", "max_length", "pooling"):
            if self.manifest["encoder"].get(field) != encoder.manifest.get(field):
                raise ValueError("query encoder differs from the saved dense index")
        vector = encoder.encode([query])[0]
        scores = np.asarray(self.vectors @ vector)
        indices = deterministic_top_k(scores, top_k, np)
        return tuple(replace(self.contexts[int(i)], score=float(scores[i])) for i in indices)


def search_variants(query, sparse, dense, encoder, reranker=None, *, candidate_k=50, rerank_k=50, rrf_k=60):
    """Return reusable stage outputs; no qrels or generated answer arguments."""
    if candidate_k < 5 or rerank_k < 5:
        raise ValueError("candidate/rerank budgets must cover the final top five")
    if dense is not None and sparse.manifest["corpus_sha256"] != dense.manifest["corpus_sha256"]:
        raise ValueError("sparse and dense stages must use the same original corpus")
    started = time.perf_counter()
    sparse_rows = sparse.retrieve([query], top_k=candidate_k)
    sparse_ms = (time.perf_counter()-started)*1000
    stages = {"bm25": {"contexts": sparse_rows, "elapsed_ms": sparse_ms}}
    if dense is None:
        return stages
    started = time.perf_counter()
    dense_rows = dense.search(query, encoder, top_k=candidate_k)
    dense_ms = (time.perf_counter()-started)*1000
    stages["dense"] = {"contexts": dense_rows, "elapsed_ms": dense_ms}
    started = time.perf_counter()
    fused = reciprocal_rank_fusion([[c.document_id for c in rows] for rows in (sparse_rows, dense_rows)], k=rrf_k)
    contexts = tuple(replace(dense.by_id[pid], score=score) for pid, score in fused)
    fusion_ms = (time.perf_counter()-started)*1000
    stages["hybrid"] = {"contexts": contexts, "elapsed_ms": sparse_ms+dense_ms+fusion_ms}
    if reranker is not None:
        started = time.perf_counter()
        candidates = contexts[:rerank_k]
        scores = reranker.score(query, candidates)
        if len(scores) != len(candidates):
            raise ValueError("reranker scores must align with candidate passages")
        reranked = tuple(replace(c, score=float(score)) for c, score in sorted(zip(candidates, scores),
                          key=lambda item: (-float(item[1]), item[0].document_id)))
        stages["hybrid_rerank"] = {"contexts": reranked,
            "elapsed_ms": sparse_ms+dense_ms+fusion_ms+(time.perf_counter()-started)*1000,
            "rerank_candidate_ids": [c.document_id for c in candidates]}
    return stages


def search_current_topic(current_query, topic_query, sparse, dense, encoder, reranker=None, *,
                         candidate_k=50, rerank_k=50, rrf_k=60):
    """One fixed equal-RRF pool, scored entirely with the topic query.

    Retain separate raw/topic retrieval when the texts differ. Fusion is one
    sum over their original BM25/dense lists, never fusion of fused scores.
    Per-view stages also provide an auditable cost and pool-size breakdown.
    Only hybrid stages are returned; no ambiguous combined BM25/dense stage.
    """
    if dense is None or encoder is None:
        raise ValueError("current_topic_rrf requires a dense index and encoder")
    same_query = current_query == topic_query
    started = time.perf_counter()
    if same_query:
        # Exact original path: one encode, sparse search, and (if requested) score.
        single = search_variants(topic_query, sparse, dense, encoder, reranker,
            candidate_k=candidate_k, rerank_k=rerank_k, rrf_k=rrf_k)
        source_stages = [("shared_current_topic", topic_query, single)]
        selected_stages = {key: value for key, value in single.items() if key in {"hybrid", "hybrid_rerank"}}
        fusion_ms, additional_current_ms = 0.0, 0.0
    else:
        raw = search_variants(current_query, sparse, dense, encoder,
            candidate_k=candidate_k, rerank_k=rerank_k, rrf_k=rrf_k)
        topic = search_variants(topic_query, sparse, dense, encoder,
            candidate_k=candidate_k, rerank_k=rerank_k, rrf_k=rrf_k)
        source_stages = [("current", current_query, raw), ("topic", topic_query, topic)]
        fusion_started = time.perf_counter()
        fused = reciprocal_rank_fusion([[c.document_id for c in stages[stage]["contexts"]]
            for _, _, stages in source_stages for stage in ("bm25", "dense")], k=rrf_k)
        contexts = tuple(replace(dense.by_id[pid], score=score) for pid, score in fused)
        fusion_ms = (time.perf_counter()-fusion_started)*1000
        selected_stages = {"hybrid": {"contexts": contexts, "elapsed_ms": (time.perf_counter()-started)*1000}}
        additional_current_ms = raw["hybrid"]["elapsed_ms"]
        if reranker is not None:
            candidates = contexts[:rerank_k]
            scores = reranker.score(topic_query, candidates)
            if len(scores) != len(candidates):
                raise ValueError("reranker scores must align with candidate passages")
            ranked = tuple(replace(c, score=float(score)) for c, score in sorted(zip(candidates, scores),
                key=lambda item: (-float(item[1]), item[0].document_id)))
            selected_stages["hybrid_rerank"] = {"contexts": ranked,
                "elapsed_ms": (time.perf_counter()-started)*1000,
                "rerank_candidate_ids": [c.document_id for c in candidates]}
    trace = {"strategy": "current_topic_rrf", "weights": "equal per unique query and retriever",
        "current_query": current_query, "topic_query": topic_query,
        "retrieval_query_count": len(source_stages), "duplicate_query_skipped": same_query,
        "candidate_k_per_query_retriever": candidate_k, "rerank_k_total": rerank_k, "rrf_k": rrf_k,
        "source_queries": [{"role": role, "query": query,
            "bm25_candidates": len(stages["bm25"]["contexts"]),
            "dense_candidates": len(stages["dense"]["contexts"]),
            "bm25_candidate_ids": [c.document_id for c in stages["bm25"]["contexts"]],
            "dense_candidate_ids": [c.document_id for c in stages["dense"]["contexts"]],
            "per_query_union_candidates": len(stages["hybrid"]["contexts"]),
            "retrieval_ms": stages["hybrid"]["elapsed_ms"]} for role, query, stages in source_stages],
        "fused_union_candidates": len(selected_stages["hybrid"]["contexts"]),
        "additional_current_retrieval_ms": additional_current_ms, "cross_query_fusion_ms": fusion_ms}
    for stage, values in selected_stages.items():
        reranked = stage == "hybrid_rerank"
        values["query_fusion_trace"] = {**trace, "rerank_query": topic_query if reranked else None,
            "rerank_candidate_ids": values.get("rerank_candidate_ids", []),
            "rerank_pairs": len(values["contexts"]) if reranked else 0,
            "rerank_ms": values["elapsed_ms"]-selected_stages["hybrid"]["elapsed_ms"] if reranked else 0.0}
    return selected_stages


class HybridRetriever:
    """Shared P3 retrieval; source-qualified history resolution is explicit opt-in."""
    def __init__(self, sparse, dense, encoder, reranker=None, *, query_mode="current",
                 stage="hybrid_rerank", candidate_k=50, rerank_k=50, rrf_k=60,
                 source_config=None, query_fusion="single", resolver=None):
        from adapters.query_resolution import SOURCE_MODES
        if query_mode not in {"current", "last3", "all_user", "correction_aware", "user_topic", *SOURCE_MODES}:
            raise ValueError("online query_mode must use user questions: current/last3/all_user/correction_aware/user_topic")
        if stage not in {"dense", "hybrid", "hybrid_rerank"}:
            raise ValueError("unknown online retrieval stage")
        if query_fusion not in {"single", "current_topic_rrf"}:
            raise ValueError("unknown query fusion")
        if query_fusion == "current_topic_rrf" and (query_mode not in {"user_topic", *SOURCE_MODES} or stage == "dense"):
            raise ValueError("current_topic_rrf requires user_topic/source_topic/assistant_topic_unverified and hybrid/hybrid_rerank")
        if query_mode in SOURCE_MODES and (resolver is None or query_fusion != "current_topic_rrf"):
            raise ValueError("topic selector modes require resolver and current_topic_rrf")
        if stage == "hybrid_rerank" and reranker is None:
            raise ValueError("hybrid_rerank requires a loaded reranker")
        if candidate_k < 5 or rerank_k < 5 or rrf_k < 0:
            raise ValueError("retrieval budgets must cover five passages and a nonnegative RRF constant")
        if sparse.manifest["corpus_sha256"] != dense.manifest["corpus_sha256"]:
            raise ValueError("sparse and dense stages must use the same original corpus")
        self.sparse, self.dense, self.encoder, self.reranker = sparse, dense, encoder, reranker
        self.corpus_snapshot_id = sparse.corpus_snapshot_id
        self.query_mode, self.stage = query_mode, stage
        self.query_fusion = query_fusion
        self.resolver = resolver
        self.requires_history_context = query_mode in SOURCE_MODES
        self.candidate_k, self.rerank_k, self.rrf_k = candidate_k, rerank_k, rrf_k
        self.source_config = source_config
        self._lock = Lock()
        self.requests = 0
        self.initialization_seconds = None
        self.query_view_traces = []

    @property
    def manifest(self):
        manifest = {"kind": "bge_hybrid", "corpus_snapshot_id": self.corpus_snapshot_id,
            "query_mode": self.query_mode, "stage": self.stage, "top_k": 5,
            "query_fusion": self.query_fusion,
            "candidate_k": self.candidate_k, "rerank_k": self.rerank_k, "rrf_k": self.rrf_k,
            "sparse": self.sparse.manifest, "dense": self.dense.manifest,
            "encoder": self.encoder.manifest, "reranker": self.reranker.manifest if self.reranker else None,
            "configuration": self.source_config,
            "initialization_seconds": self.initialization_seconds,
            "execution": "one loaded model pair per run; serialized retrieval, queue time included in P3 pipeline time",
            "completed_retrievals": self.requests,
            "model_usage": {name: {"texts_processed": model.texts_processed,
                "token_limit_hits": model.token_limit_hits}
                for name, model in (("encoder", self.encoder), ("reranker", self.reranker)) if model is not None},
            "token_limit_note": "at configured token cap; includes possible exact-length inputs"}
        if self.query_mode == "user_topic":
            manifest.update(query_view=user_topic_manifest(), query_view_traces=list(self.query_view_traces))
        if self.resolver is not None:
            manifest.update(query_view=self.resolver.manifest, query_view_traces=list(self.query_view_traces))
        return manifest

    def retrieve_turn(self, turn, *, prior_context=None):
        if turn.protocol != "P3":
            raise ValueError("hybrid retrieval requires a P3 turn")
        started = time.perf_counter()
        view = user_topic_view(turn.user_questions) if self.query_mode == "user_topic" else None
        # run_sessions dispatches retrieval via to_thread. Serialize this model
        # pair instead of allowing overlapping forwards and unbounded GPU batches.
        with self._lock:
            if self.resolver is not None:
                if prior_context is None:
                    raise ValueError("topic selector requires an explicit committed history context")
                view = self.resolver.resolve(turn, prior_context)
            fusion_trace = None
            retrieval_error = None
            try:
                query = view["text"] if view else query_text(turn.user_questions, self.query_mode)
                if self.stage == "dense":
                    contexts = self.dense.search(query, self.encoder, top_k=5)
                elif self.query_fusion == "current_topic_rrf":
                    stages = search_current_topic(turn.query, query, self.sparse, self.dense, self.encoder,
                        self.reranker if self.stage == "hybrid_rerank" else None,
                        candidate_k=self.candidate_k, rerank_k=self.rerank_k, rrf_k=self.rrf_k)
                    contexts = stages[self.stage]["contexts"][:5]
                    fusion_trace = stages[self.stage]["query_fusion_trace"]
                else:
                    stages = search_variants(query, self.sparse, self.dense, self.encoder,
                        self.reranker if self.stage == "hybrid_rerank" else None,
                        candidate_k=self.candidate_k, rerank_k=self.rerank_k, rrf_k=self.rrf_k)
                    contexts = stages[self.stage]["contexts"][:5]
                self.requests += 1
            except Exception as exc:
                retrieval_error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                if view is not None:
                    self.query_view_traces.append({"task_id": turn.task_id, "conversation_id": turn.conversation_id,
                        "turn": turn.turn, **view,
                        "retrieval_status": "failed" if retrieval_error else "completed",
                        "retrieval_error": retrieval_error,
                        # Full local wall interval, including lock wait and query resolution.
                        # The resolver's own elapsed_ms remains unchanged in view.
                        "retrieval_elapsed_ms": (time.perf_counter()-started)*1000,
                        **({"query_fusion_trace": fusion_trace} if fusion_trace else {})})
        return replace(turn, contexts=tuple(contexts))

    def close(self):
        with self._lock:
            self.encoder.close()
            if self.reranker is not None:
                self.reranker.close()
            if self.resolver is not None:
                self.resolver.close()
