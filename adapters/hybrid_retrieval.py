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
from .retrieval import deterministic_top_k, rerank_document, retrieval_query


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


def rerank_query_text(turn, resolved_query, mode="resolved"):
    """Choose only from pre-retrieval question views; no answers or labels."""
    if mode == "resolved":
        return resolved_query
    if mode == "current":
        return turn.query
    if mode == "user_topic":
        return user_topic_view(turn.user_questions)["text"]
    raise ValueError("unknown rerank query mode")


def reciprocal_rank_fusion(rankings, *, k=60):
    """Equal-weight RRF; duplicate IDs cannot vote twice within one ranking."""
    if k < 0:
        raise ValueError("RRF rank constant must be nonnegative")
    scores = {}
    for ranking in rankings:
        for rank, pid in enumerate(dict.fromkeys(ranking), 1):
            scores[pid] = scores.get(pid, 0.0)+1/(k+rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def select_ranked_evidence(ranked, *, top_k=5, anchor=None, group_key=None):
    """Select the highest-ranked representatives of distinct original content.

    This is ranked result collapsing, not semantic similarity or score tuning.
    A bound corpus adds version/dependency scope to the exact title/body key.
    An explicit task anchor is reserved before the remaining groups are chosen.
    Original sources/scores remain intact; no archive or evidence is deleted.
    """
    ranked = tuple(ranked)
    group_key = group_key or (lambda context: (context.title, context.text))
    selected, represented, collapsed = set(), {}, []
    if anchor is not None:
        context = next(context for context in ranked if context.document_id == anchor)
        selected.add(anchor)
        represented[group_key(context)] = anchor
    for context in ranked:
        if len(selected) >= top_k:
            break
        if context.document_id in selected:
            continue
        key = group_key(context)
        if key in represented:
            collapsed.append({"passage_id": context.document_id, "representative": represented[key]})
            continue
        selected.add(context.document_id)
        represented[key] = context.document_id
    result = tuple(context for context in ranked if context.document_id in selected)
    trace = ({"strategy": "ranked_exact_content_groups_v1", "budget": top_k,
              "original_top_ids": [c.document_id for c in ranked[:top_k]],
              "selected_ids": [c.document_id for c in result], "collapsed": collapsed,
              "anchor": anchor} if collapsed else None)
    return result, trace


def select_condition_evidence(stage, condition_query, reranker, *, top_k=5, rrf_k=60,
                              group_key=None):
    """Maximize fused rank score with the primary task's first passage retained.

    Both views score the same pool. The primary anchor protects the original
    task from query drift; it is not a guarantee of semantic support or complete
    answer coverage. Final group selection is shared with ordinary retrieval.
    """
    primary = stage["contexts"]
    if not condition_query or condition_query == stage["rerank_query"] or len(primary) <= top_k:
        return select_ranked_evidence(primary, top_k=top_k, group_key=group_key)
    by_id = {context.document_id: context for context in primary}
    candidates = tuple(by_id[pid] for pid in stage["rerank_candidate_ids"])
    started = time.perf_counter()
    scores = reranker.score(condition_query, candidates)
    if len(scores) != len(candidates):
        raise ValueError("reranker scores must align with candidate passages")
    condition_ids = [context.document_id for context, _ in sorted(zip(candidates, scores),
                     key=lambda item: (-float(item[1]), item[0].document_id))]
    primary_ids = [context.document_id for context in primary]
    fused = reciprocal_rank_fusion([primary_ids, condition_ids], k=rrf_k)
    anchor = primary_ids[0]
    # Keep genuine fusion scores/order; do not invent a larger anchor score.
    ranked = tuple(replace(by_id[pid], score=score) for pid, score in fused)
    contexts, selection = select_ranked_evidence(ranked, top_k=top_k, anchor=anchor, group_key=group_key)
    trace = {"strategy": "task_anchor_condition_rrf_v1", "condition_query": condition_query,
        "primary_query": stage["rerank_query"], "primary_anchor": anchor,
        "anchor_changed_selection": anchor not in {pid for pid, _ in fused[:top_k]},
        "primary_ranked_ids": primary_ids, "condition_ranked_ids": condition_ids,
        "selected_ids": [context.document_id for context in contexts],
        **({"evidence_selection": selection} if selection else {}),
        "rrf_k": rrf_k, "extra_rerank_pairs": len(candidates),
        "elapsed_ms": (time.perf_counter()-started)*1000,
        "score_semantics": "RRF scores within the primary-anchor-constrained selection"}
    return contexts, trace


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
        return self._forward([(query, rerank_document(c)) for c in contexts])

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


def search_variants(query, sparse, dense, encoder, reranker=None, *, candidate_k=50, rerank_k=50, rrf_k=60,
                    rerank_query=None, source_context=None):
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
        if source_context is not None:
            candidates = source_context(candidates)
        scoring_query = query if rerank_query is None else rerank_query
        scores = reranker.score(scoring_query, candidates)
        if len(scores) != len(candidates):
            raise ValueError("reranker scores must align with candidate passages")
        reranked = tuple(replace(c, score=float(score)) for c, score in sorted(zip(candidates, scores),
                          key=lambda item: (-float(item[1]), item[0].document_id)))
        stages["hybrid_rerank"] = {"contexts": reranked,
            "elapsed_ms": sparse_ms+dense_ms+fusion_ms+(time.perf_counter()-started)*1000,
            "rerank_query": scoring_query,
            "rerank_candidate_ids": [c.document_id for c in candidates]}
    return stages


def search_current_topic(current_query, topic_query, sparse, dense, encoder, reranker=None, *,
                         candidate_k=50, rerank_k=50, rrf_k=60, rerank_query=None, source_context=None):
    """One fixed equal-RRF pool, scored with topic unless explicitly overridden.

    Retain separate raw/topic retrieval when the texts differ. Fusion is one
    sum over their original BM25/dense lists, never fusion of fused scores.
    Per-view stages also provide an auditable cost and pool-size breakdown.
    Only hybrid stages are returned; no ambiguous combined BM25/dense stage.
    """
    if dense is None or encoder is None:
        raise ValueError("current_topic_rrf requires a dense index and encoder")
    same_query = current_query == topic_query
    scoring_query = topic_query if rerank_query is None else rerank_query
    started = time.perf_counter()
    if same_query:
        # Exact original path: one encode, sparse search, and (if requested) score.
        single = search_variants(topic_query, sparse, dense, encoder, reranker,
            candidate_k=candidate_k, rerank_k=rerank_k, rrf_k=rrf_k, rerank_query=scoring_query,
            source_context=source_context)
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
            if source_context is not None:
                candidates = source_context(candidates)
            scores = reranker.score(scoring_query, candidates)
            if len(scores) != len(candidates):
                raise ValueError("reranker scores must align with candidate passages")
            ranked = tuple(replace(c, score=float(score)) for c, score in sorted(zip(candidates, scores),
                key=lambda item: (-float(item[1]), item[0].document_id)))
            selected_stages["hybrid_rerank"] = {"contexts": ranked,
                "elapsed_ms": (time.perf_counter()-started)*1000,
                "rerank_query": scoring_query,
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
        values["query_fusion_trace"] = {**trace, "rerank_query": scoring_query if reranked else None,
            "rerank_candidate_ids": values.get("rerank_candidate_ids", []),
            "rerank_pairs": len(values["contexts"]) if reranked else 0,
            "rerank_ms": values["elapsed_ms"]-selected_stages["hybrid"]["elapsed_ms"] if reranked else 0.0}
    return selected_stages


class HybridRetriever:
    """Shared P3 retrieval; source-qualified history resolution is explicit opt-in."""
    def __init__(self, sparse, dense, encoder, reranker=None, *, query_mode="current",
                 stage="hybrid_rerank", candidate_k=50, rerank_k=50, rrf_k=60,
                 source_config=None, query_fusion="single", resolver=None, rerank_query_mode="resolved", trace_sink=None,
                 source_window=False):
        from adapters.query_resolution import SOURCE_MODES
        if query_mode not in {"current", "last3", "all_user", "correction_aware", "user_topic", "conversational", *SOURCE_MODES}:
            raise ValueError("online query_mode must use user questions: current/last3/all_user/correction_aware/user_topic")
        if stage not in {"dense", "hybrid", "hybrid_rerank"}:
            raise ValueError("unknown online retrieval stage")
        if query_fusion not in {"single", "current_topic_rrf"}:
            raise ValueError("unknown query fusion")
        if query_fusion == "current_topic_rrf" and (query_mode not in {"user_topic", "conversational", *SOURCE_MODES} or stage == "dense"):
            raise ValueError("current_topic_rrf requires user_topic/source_topic/assistant_topic_unverified and hybrid/hybrid_rerank")
        if query_mode in SOURCE_MODES and (resolver is None or query_fusion != "current_topic_rrf"):
            raise ValueError("topic selector modes require resolver and current_topic_rrf")
        if query_mode == "conversational" and (resolver is None or rerank_query_mode != "resolved"):
            raise ValueError("conversational retrieval needs resolver and resolved reranking")
        if stage == "hybrid_rerank" and reranker is None:
            raise ValueError("hybrid_rerank requires a loaded reranker")
        if rerank_query_mode not in {"resolved", "current", "user_topic"}:
            raise ValueError("unknown rerank query mode")
        if stage != "hybrid_rerank" and rerank_query_mode != "resolved":
            raise ValueError("rerank_query_mode requires hybrid_rerank")
        if candidate_k < 5 or rerank_k < 5 or rrf_k < 0:
            raise ValueError("retrieval budgets must cover five passages and a nonnegative RRF constant")
        if sparse.manifest["corpus_sha256"] != dense.manifest["corpus_sha256"]:
            raise ValueError("sparse and dense stages must use the same original corpus")
        self.sparse, self.dense, self.encoder, self.reranker = sparse, dense, encoder, reranker
        self.corpus_snapshot_id = sparse.corpus_snapshot_id
        self.query_mode, self.stage = query_mode, stage
        self.query_fusion = query_fusion
        self.rerank_query_mode = rerank_query_mode
        self.resolver = resolver
        self.requires_history_context = query_mode in SOURCE_MODES or query_mode == "conversational"
        self.candidate_k, self.rerank_k, self.rrf_k = candidate_k, rerank_k, rrf_k
        self.source_config = source_config
        self._lock = Lock()
        self.requests = 0
        self.initialization_seconds = None
        self.query_view_traces = []
        self.trace_sink = trace_sink
        self.source_context = None
        self.selection_group = None
        self.source_window = source_window
        self.expand_source_window = None

    def bind_corpus(self, corpus_index):
        """Share the generation metadata index with reranking, before serving starts."""
        self.source_context = corpus_index.contextualize
        self.selection_group = corpus_index.selection_group
        self.expand_source_window = corpus_index.source_window

    @property
    def manifest(self):
        manifest = {"kind": "bge_hybrid", "corpus_snapshot_id": self.corpus_snapshot_id,
            "query_mode": self.query_mode, "stage": self.stage, "top_k": 5,
            "query_fusion": self.query_fusion,
            "rerank_query_mode": self.rerank_query_mode,
            "rerank_document_view": "source address, title and original body",
            "pre_rerank_metadata": self.source_context is not None,
            "source_window": {"enabled": self.source_window, "radius": 1,
                "scope": "same source/document/version/declared section; explicit span or MT-RAG ingestion order",
                "selection": "one hop from ranked hits, complete originals, no invented rank or recursive expansion",
                "budget": "shared A/R required evidence; normal exact-token feasibility and native admission apply",
                "extra_model_calls": 0},
            "evidence_selection": {"strategy": "ranked_exact_content_groups_v1",
                "group": "exact title/body with known version, parent and dependency scope",
                "source_identity": "highest ranked original representative; task anchor retained",
                "budget": "at most five groups from the existing scored pool",
                "extra_model_calls": 0},
            "candidate_k": self.candidate_k, "rerank_k": self.rerank_k, "rrf_k": self.rrf_k,
            "sparse": self.sparse.manifest, "dense": self.dense.manifest,
            "encoder": self.encoder.manifest, "reranker": self.reranker.manifest if self.reranker else None,
            "configuration": self.source_config,
            "initialization_seconds": self.initialization_seconds,
            "execution": "one loaded model pair per run; serialized encoder/reranker; independent intent HTTP outside that lock; all queue time included in P3 pipeline time",
            "completed_retrievals": self.requests,
            "model_usage": {name: {"texts_processed": model.texts_processed,
                "token_limit_hits": model.token_limit_hits}
                for name, model in (("encoder", self.encoder), ("reranker", self.reranker)) if model is not None},
            "token_limit_note": "at configured token cap; includes possible exact-length inputs"}
        if self.query_mode == "user_topic":
            manifest.update(query_view=user_topic_manifest(), query_view_traces=list(self.query_view_traces))
        if self.resolver is not None:
            manifest.update(query_view=self.resolver.manifest, query_view_traces=list(self.query_view_traces))
        if self.query_mode == "conversational":
            manifest["condition_evidence_selection"] = {
                "strategy": "task_anchor_condition_rrf_v1",
                "activation": "literal named task continuation with a distinct condition view",
                "stage_enabled": self.stage == "hybrid_rerank",
                "pool": "same primary rerank candidates; no additional index search",
                "selection": "retain primary rank one; highest equal RRF score from distinct content groups in remaining slots",
                "extra_rerank_pairs_max": self.rerank_k, "rrf_k": self.rrf_k}
        if self.rerank_query_mode != "resolved" or self.query_view_traces:
            manifest["query_view_traces"] = list(self.query_view_traces)
        if self.rerank_query_mode == "user_topic":
            manifest["rerank_query_view"] = user_topic_manifest()
        if self.trace_sink is not None:
            manifest.pop("query_view_traces", None)
            manifest["query_view_trace_storage"] = "external_sink"
        return manifest

    def retrieve_turn(self, turn, *, prior_context=None):
        if turn.protocol != "P3":
            raise ValueError("hybrid retrieval requires a P3 turn")
        if self.source_window and self.expand_source_window is None:
            raise ValueError("source_window requires --evidence-corpus metadata")
        started = time.perf_counter()
        view = user_topic_view(turn.user_questions) if self.query_mode == "user_topic" else None
        if self.resolver is not None:
            if prior_context is None:
                raise ValueError("topic selector requires an explicit committed history context")
            view = self.resolver.resolve(turn, prior_context)
        # run_sessions dispatches retrieval via to_thread. Serialize this model
        # pair, but do not hold its lock while another service resolves intent.
        # An unrelated session can retrieve during that HTTP wait. Shutdown
        # drains complete retrieval workers before closing either client/model.
        with self._lock:
            fusion_trace = None
            condition_trace = None
            selection_trace = None
            supporting_contexts = ()
            retrieval_error = None
            scoring_query = None
            try:
                query = view["text"] if view else query_text(turn.user_questions, self.query_mode)
                scoring_query = rerank_query_text(turn, query, self.rerank_query_mode)
                if view is None and self.rerank_query_mode != "resolved":
                    view = {"mode": self.query_mode, "text": query}
                if self.stage == "dense":
                    ranked = self.dense.search(query, self.encoder, top_k=self.candidate_k)
                elif self.query_fusion == "current_topic_rrf":
                    stages = search_current_topic(turn.query, query, self.sparse, self.dense, self.encoder,
                        self.reranker if self.stage == "hybrid_rerank" else None,
                        candidate_k=self.candidate_k, rerank_k=self.rerank_k, rrf_k=self.rrf_k,
                        rerank_query=scoring_query, source_context=self.source_context)
                    ranked = stages[self.stage]["contexts"]
                    fusion_trace = stages[self.stage]["query_fusion_trace"]
                else:
                    stages = search_variants(query, self.sparse, self.dense, self.encoder,
                        self.reranker if self.stage == "hybrid_rerank" else None,
                        candidate_k=self.candidate_k, rerank_k=self.rerank_k, rrf_k=self.rrf_k,
                        rerank_query=scoring_query, source_context=self.source_context)
                    ranked = stages[self.stage]["contexts"]
                if self.stage == "hybrid_rerank" and view is not None and view.get("condition_query"):
                    contexts, condition_trace = select_condition_evidence(
                        stages[self.stage], view["condition_query"], self.reranker, rrf_k=self.rrf_k,
                        group_key=self.selection_group)
                else:
                    contexts, selection_trace = select_ranked_evidence(ranked, group_key=self.selection_group)
                if self.source_window:
                    supporting_contexts = self.expand_source_window(contexts)
                if view is None and (selection_trace is not None or self.source_window):
                    view = {"mode": self.query_mode, "text": query}
                self.requests += 1
            except Exception as exc:
                retrieval_error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                if view is not None:
                    trace = {"task_id": turn.task_id, "conversation_id": turn.conversation_id,
                        "turn": turn.turn, **view,
                        "rerank_query_mode": self.rerank_query_mode,
                        "rerank_query": scoring_query if self.stage == "hybrid_rerank" else None,
                        "retrieval_status": "failed" if retrieval_error else "completed",
                        "retrieval_error": retrieval_error,
                        # Full local wall interval, including lock wait and query resolution.
                        # The resolver's own elapsed_ms remains unchanged in view.
                        "retrieval_elapsed_ms": (time.perf_counter()-started)*1000,
                        **({"condition_evidence": condition_trace} if condition_trace else {}),
                        **({"evidence_selection": selection_trace} if selection_trace else {}),
                        **({"source_window": {"primary_ids": [c.document_id for c in contexts],
                            "supporting_ids": [c.document_id for c in supporting_contexts]}}
                           if self.source_window and not retrieval_error else {}),
                        **({"query_fusion_trace": fusion_trace} if fusion_trace else {})}
                    if self.trace_sink is None:
                        self.query_view_traces.append(trace)
                    else:
                        self.trace_sink(trace)
        from controller.schema import QueryContext
        from controller.serialization import decode
        intent = decode(QueryContext, view["context"]) if view is not None and view.get("context") is not None else None
        return replace(turn, contexts=tuple(contexts), query_context=intent,
                       supporting_contexts=supporting_contexts)

    def close(self):
        with self._lock:
            self.encoder.close()
            if self.reranker is not None:
                self.reranker.close()
            if self.resolver is not None:
                self.resolver.close()
