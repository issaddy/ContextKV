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


def reciprocal_rank_fusion(rankings, *, k=60, weights=None):
    """Rank fusion; duplicate IDs cannot vote twice within one ranking."""
    if k < 0:
        raise ValueError("RRF rank constant must be nonnegative")
    rankings = tuple(rankings)
    weights = (1.0,) * len(rankings) if weights is None else tuple(weights)
    scores = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, pid in enumerate(dict.fromkeys(ranking), 1):
            scores[pid] = scores.get(pid, 0.0)+weight/(k+rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def select_ranked_evidence(ranked, *, top_k=5, group_key=None):
    """Select distinct reading groups, retaining every retrieved source in them.

    This is ranked result collapsing, not semantic similarity or score tuning.
    A bound corpus adds version/dependency scope to the exact title/body key.
    Grouping budgets shared bodies, not citation identities. Only members of this
    ranked pool can join a selected group; generation shares exact bodies using
    its existing evidence aliases while preserving each source and score.
    """
    ranked = tuple(ranked)
    group_key = group_key or (lambda context: (context.title, context.text))
    selected, represented, collapsed = set(), {}, []
    for context in ranked:
        if context.document_id in selected:
            continue
        key = group_key(context)
        if key in represented:
            collapsed.append({"passage_id": context.document_id, "representative": represented[key]})
            selected.add(context.document_id)
            continue
        if len(represented) >= top_k:
            continue
        selected.add(context.document_id)
        represented[key] = context.document_id
    result = tuple(context for context in ranked if context.document_id in selected)
    trace = ({"strategy": "ranked_exact_content_groups_v1", "budget": top_k,
              "original_top_ids": [c.document_id for c in ranked[:top_k]],
              "selected_ids": [c.document_id for c in result], "collapsed": collapsed} if collapsed else None)
    return result, trace


def select_requirement_evidence(primary, rankings, *, top_k=5, group_key=None):
    """Greedy coverage of explicit information needs on a distinct-content pool.

    F(S) = sum_j max_{d in S} 1/rank_j(d), including the whole-task ranking.
    This nonnegative diminishing-return objective rewards complementarity,
    without treating logits/ranks as calibrated support probabilities. Reserve
    the primary task's first source, then maximize remaining marginal coverage.
    Ties retain primary relevance order. Return original scores and identities.
    """
    primary = tuple(primary)
    if not primary:
        return (), {"selected_ids": [], "marginal_gains": []}
    ids = [c.document_id for c in primary]
    group_key = group_key or (lambda context: (context.title, context.text))
    representatives, groups = {}, {}
    for context in primary:
        representative = representatives.setdefault(group_key(context), context.document_id)
        groups[context.document_id] = representative
    rows = []
    for ranking in (ids, *rankings):
        row = {}
        for rank, pid in enumerate(ranking, 1):
            representative = groups[pid]
            row[representative] = max(row.get(representative, 0.0), 1/rank)
        rows.append(row)
    selected = {ids[0]}
    covered = [row[ids[0]] for row in rows]
    gains = [{"passage_id": ids[0], "gain": sum(covered), "reason": "primary_task_anchor"}]
    while len(selected) < min(top_k, len(representatives)):
        remaining = [pid for pid in representatives.values() if pid not in selected]
        def gain(pid):
            return sum(max(0.0, row[pid]-best) for row, best in zip(rows, covered))
        winner = max(remaining, key=gain)
        gains.append({"passage_id": winner, "gain": gain(winner), "reason": "requirement_coverage"})
        selected.add(winner)
        covered = [max(best, row[winner]) for row, best in zip(rows, covered)]
    return tuple(c for c in primary if groups[c.document_id] in selected), {
        "selected_ids": [pid for pid in ids if groups[pid] in selected], "marginal_gains": gains,
        "best_selected_rank_per_view": [round(1/value) for value in covered]}


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

    def score_many(self, queries, contexts):
        """Batch query/passage pairs across views using the same checkpoint API."""
        if not self.reranker:
            raise ValueError("dense encoder is not a cross-encoder")
        queries, contexts = tuple(queries), tuple(contexts)
        documents = [rerank_document(c) for c in contexts]
        values = self._forward([(query, document) for query in queries for document in documents])
        return values.reshape(len(queries), len(contexts))

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
        return self.search_many([query], encoder, top_k=top_k)[0]

    def search_many(self, queries, encoder, *, top_k=50):
        """Encode the current task and its needs together; reuse the fixed vectors."""
        import numpy as np
        for field in ("model_id", "revision", "max_length", "pooling"):
            if self.manifest["encoder"].get(field) != encoder.manifest.get(field):
                raise ValueError("query encoder differs from the saved dense index")
        vectors = encoder.encode(list(queries))
        result = []
        for vector in vectors:
            scores = np.asarray(self.vectors @ vector)
            indices = deterministic_top_k(scores, top_k, np)
            result.append(tuple(replace(self.contexts[int(i)], score=float(scores[i])) for i in indices))
        return tuple(result)


def search_requirements(query, requirements, sparse, dense, encoder, reranker, *, candidate_k=50,
                        rerank_k=50, rrf_k=60, current_query=None, source_context=None,
                        source_neighbors=None, group_key=None, rank_coverage=True):
    """Retrieve complementary needs, then cover them in the same final budget.

    Subqueries come only from current committed/requested task state, not answer
    guesses, qrels or corpus facts. Common conditions qualify every subquery.
    One batched encode, equal-RRF seed union, and indexed source neighbors.
    Whole-task ranking bounds the distinct pool. The joint semantic selector can
    consume this pool directly, without repeated per-requirement scoring.
    """
    started = time.perf_counter()
    queries = tuple(dict.fromkeys([query, *requirements, *([current_query] if current_query else [])]))
    dense_rows = dense.search_many(queries, encoder, top_k=candidate_k)
    sparse_rows = [sparse.retrieve([q], top_k=candidate_k) for q in queries]
    fused = reciprocal_rank_fusion([[c.document_id for c in rows]
        for pair in zip(sparse_rows, dense_rows) for rows in pair], k=rrf_k)
    candidates = tuple(replace(dense.by_id[pid], score=score) for pid, score in fused[:rerank_k])
    recalled_ids = [c.document_id for c in candidates]
    neighbors = source_neighbors(candidates) if source_neighbors is not None else ()
    # Retrieval candidates belong to this indexed snapshot. Other metadata
    # revisions/headers remain accessible through explicit evidence dependencies.
    neighbor_ids = dict.fromkeys(c.document_id for c in neighbors if c.document_id not in recalled_ids
        and c.document_id in dense.by_id and dense.by_id[c.document_id].text == c.text)
    neighbors = tuple(dense.by_id[pid] for pid in neighbor_ids)
    candidates += neighbors
    if source_context is not None:
        candidates = source_context(candidates)
    ranked = candidates
    if reranker is not None:
        scores = reranker.score(query, candidates)
        if len(scores) != len(candidates):
            raise ValueError("reranker scores must align with candidate passages")
        ranked = tuple(replace(c, score=float(score)) for c, score in sorted(zip(candidates, scores),
            key=lambda item: (-float(item[1]), item[0].document_id)))
    primary, grouping = select_ranked_evidence(ranked, top_k=rerank_k, group_key=group_key)
    rankings = []
    for requirement in requirements if rank_coverage and reranker is not None else ():
        values = reranker.score(requirement, primary)
        if len(values) != len(primary):
            raise ValueError("reranker scores must align with candidate passages")
        rankings.append([c.document_id for c, _ in sorted(zip(primary, values),
            key=lambda item: (-float(item[1]), item[0].document_id))])
    if rankings:
        contexts, selection = select_requirement_evidence(primary, rankings, group_key=group_key)
    else:
        contexts, _ = select_ranked_evidence(primary, group_key=group_key)
        selection = {"selected_ids": [c.document_id for c in contexts]}
    trace = {"strategy": "task_requirement_coverage_v1" if rankings else "task_requirement_pool_v1", "query": query,
        "requirements": list(requirements), "retrieval_queries": queries,
        "recalled_ids": recalled_ids, "neighbor_candidate_ids": [c.document_id for c in neighbors],
        "candidate_ids": [c.document_id for c in candidates],
        "primary_ranked_ids": [c.document_id for c in primary], "requirement_ranked_ids": rankings,
        "source_rankings": [{"query": q, "sparse": [c.document_id for c in s],
            "dense": [c.document_id for c in d]} for q, s, d in zip(queries, sparse_rows, dense_rows)],
        "evidence_selection": grouping, **selection,
        "encoder_inputs": len(queries), "rerank_pairs": (len(candidates) if reranker is not None else 0)+len(primary)*len(rankings),
        "elapsed_ms": (time.perf_counter()-started)*1000,
        "objective": ("sum of best reciprocal rank per information need, including full task; primary anchor retained"
                      if rankings else "joint requirement recall; final selection uses the prepared reading catalog"),
        "semantics": "rank coverage surrogate, not proof of evidence sufficiency or answer correctness"}
    return contexts, trace


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


def _recall_query_views(queries, sparse, dense, encoder, *, candidate_k, rrf_k, weights=None):
    """One encoder batch and one fixed index per view; no generated answer input."""
    started = time.perf_counter()
    queries = tuple(queries)
    weights = (1.0,) * len(queries) if weights is None else tuple(weights)
    dense_rows = dense.search_many(queries, encoder, top_k=candidate_k)
    dense_ms = (time.perf_counter() - started) * 1000
    sparse_started = time.perf_counter()
    sparse_rows = [sparse.retrieve([query], top_k=candidate_k) for query in queries]
    sparse_ms = (time.perf_counter() - sparse_started) * 1000
    fused = reciprocal_rank_fusion([[c.document_id for c in rows]
        for pair in zip(sparse_rows, dense_rows, strict=True) for rows in pair], k=rrf_k,
        weights=[weight for weight in weights for _ in range(2)])
    return tuple(replace(dense.by_id[pid], score=score) for pid, score in fused), {
        "source_queries": [{"query": query, "weight": weight,
            "bm25_candidate_ids": [c.document_id for c in sparse_hits],
            "dense_candidate_ids": [c.document_id for c in dense_hits]}
            for query, weight, sparse_hits, dense_hits in zip(queries, weights, sparse_rows, dense_rows, strict=True)],
        "dense_batch_ms": dense_ms, "sparse_ms": sparse_ms,
        "elapsed_ms": (time.perf_counter() - started) * 1000, "encoder_batches": 1}


def search_dialogue_views(literal_query, resolved_query, requirements, sparse, dense, encoder,
                          reranker=None, *, candidate_k=50, rerank_k=50, rrf_k=60,
                          source_context=None, source_neighbors=None):
    """Keep literal conversation and derived needs through both retrieval stages.

    The literal view and the distinct derived-view family each receive one
    total vote. Splitting a task into more needs cannot multiply its family's
    influence. Every view reranks the same bounded pool; final fusion uses
    ranks, since cross-query logits are not calibrated probabilities.
    """
    started = time.perf_counter()
    queries = tuple(dict.fromkeys((literal_query, resolved_query, *requirements)))
    weights = (1.0,) if len(queries) == 1 else (1.0, *((1.0 / (len(queries)-1),) * (len(queries)-1)))
    contexts, recall = _recall_query_views(queries, sparse, dense, encoder,
        candidate_k=candidate_k, rrf_k=rrf_k, weights=weights)
    candidates = contexts[:rerank_k]
    seed_ids = {c.document_id for c in candidates}
    # Compound tasks may reuse the existing one-hop original-source recall.
    neighbors = source_neighbors(candidates) if requirements and source_neighbors is not None else ()
    by_id = {c.document_id: c for c in candidates}
    for neighbor in neighbors:
        original = dense.by_id.get(neighbor.document_id)
        if original is not None and original.text == neighbor.text:
            by_id.setdefault(neighbor.document_id, original)
    candidates = tuple(by_id.values())
    if source_context is not None:
        candidates = source_context(candidates)
    rankings = []
    ranked = contexts
    if reranker is not None:
        score_many = getattr(reranker, "score_many", None)
        scores = (score_many(queries, candidates) if score_many is not None else
                  [reranker.score(query, candidates) for query in queries])
        for values in scores:
            rankings.append([c.document_id for c, _ in sorted(zip(candidates, values, strict=True),
                key=lambda pair: (-float(pair[1]), pair[0].document_id))])
        fused = reciprocal_rank_fusion(rankings, k=rrf_k, weights=weights)
        originals = {c.document_id: c for c in candidates}
        ranked = tuple(replace(originals[pid], score=score) for pid, score in fused)
    trace = {"strategy": "dialogue_rrf", "literal_query": literal_query,
        "resolved_query": resolved_query, "requirements": list(requirements),
        "query_weights": list(weights), "family_weighting": "literal 1; all other distinct views share 1",
        "source_queries": recall["source_queries"], "encoder_batches": recall["encoder_batches"],
        "recall_ms": recall["elapsed_ms"], "dense_batch_ms": recall["dense_batch_ms"],
        "candidate_k_per_query_retriever": candidate_k, "rerank_k_seed_pool": rerank_k,
        "neighbor_candidate_ids": [pid for pid in by_id if pid not in seed_ids],
        "rerank_candidate_ids": [c.document_id for c in candidates] if reranker is not None else [],
        "rerank_queries": list(queries) if reranker is not None else [],
        "rerank_rankings": rankings, "rerank_pairs": len(queries)*len(candidates) if reranker is not None else 0,
        "elapsed_ms": (time.perf_counter()-started)*1000,
        "scope": "query-view fusion; original evidence bodies and task state unchanged"}
    return ranked, trace


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
        sparse_ids = [c.document_id for c in single["bm25"]["contexts"]]
        dense_ids = [c.document_id for c in single["dense"]["contexts"]]
        source_queries = [{"role": "shared_current_topic", "query": topic_query,
            "bm25_candidate_ids": sparse_ids, "dense_candidate_ids": dense_ids,
            "bm25_candidates": len(sparse_ids), "dense_candidates": len(dense_ids),
            "per_query_union_candidates": len(set(sparse_ids) | set(dense_ids)),
            "retrieval_ms": single["hybrid"]["elapsed_ms"]}]
        selected_stages = {key: value for key, value in single.items() if key in {"hybrid", "hybrid_rerank"}}
        fusion_ms, additional_current_ms = 0.0, 0.0
    else:
        contexts, recall = _recall_query_views((current_query, topic_query), sparse, dense, encoder,
            candidate_k=candidate_k, rrf_k=rrf_k)
        source_queries = [{**row, "role": role,
            "bm25_candidates": len(row["bm25_candidate_ids"]),
            "dense_candidates": len(row["dense_candidate_ids"]),
            "per_query_union_candidates": len(set(row["bm25_candidate_ids"]) | set(row["dense_candidate_ids"])),
            "retrieval_ms": None}
            for role, row in zip(("current", "topic"), recall["source_queries"], strict=True)]
        fusion_ms = recall["elapsed_ms"]-recall["dense_batch_ms"]-recall["sparse_ms"]
        selected_stages = {"hybrid": {"contexts": contexts, "elapsed_ms": (time.perf_counter()-started)*1000}}
        additional_current_ms = None  # Dense work is shared, not independently timed per query.
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
        "retrieval_query_count": len(source_queries), "encoder_batches": 1, "duplicate_query_skipped": same_query,
        "candidate_k_per_query_retriever": candidate_k, "rerank_k_total": rerank_k, "rrf_k": rrf_k,
        "source_queries": source_queries,
        "fused_union_candidates": len(selected_stages["hybrid"]["contexts"]),
        "additional_current_retrieval_ms": additional_current_ms, "cross_query_fusion_ms": fusion_ms}
    for stage, values in selected_stages.items():
        reranked = stage == "hybrid_rerank"
        values["query_fusion_trace"] = {**trace, "rerank_query": scoring_query if reranked else None,
            "rerank_candidate_ids": values.get("rerank_candidate_ids", []),
            "rerank_pairs": len(values["contexts"]) if reranked else 0,
            "rerank_ms": values["elapsed_ms"]-selected_stages["hybrid"]["elapsed_ms"] if reranked else 0.0}
    return selected_stages


def validate_retrieval_options(*, query_mode, stage, query_fusion, rerank_query_mode,
                               candidate_k, rerank_k, rrf_k, joint_selection=False):
    """One contract for file-based configuration and direct construction."""
    from adapters.query_resolution import SOURCE_MODES
    if query_mode not in {"current", "last3", "all_user", "correction_aware", "user_topic", "conversational", *SOURCE_MODES}:
        raise ValueError(f"unknown online query_mode: {query_mode}")
    if stage not in {"dense", "hybrid", "hybrid_rerank"}:
        raise ValueError("unknown retrieval stage")
    if query_fusion not in {"single", "current_topic_rrf", "dialogue_rrf"}:
        raise ValueError("unknown query fusion")
    if query_fusion == "current_topic_rrf" and (query_mode not in {"user_topic", "conversational", *SOURCE_MODES} or stage == "dense"):
        raise ValueError("current_topic_rrf requires user_topic/conversational/source_topic/assistant_topic_unverified and hybrid/hybrid_rerank")
    if query_fusion == "dialogue_rrf" and (query_mode != "conversational" or stage == "dense"):
        raise ValueError("dialogue_rrf requires conversational hybrid retrieval")
    if query_mode == "conversational" and rerank_query_mode != "resolved":
        raise ValueError("conversational retrieval needs resolved reranking")
    if joint_selection and (query_mode != "conversational" or query_fusion != "single"):
        raise ValueError("joint semantic selection uses conversational mode and its own literal/state recall; query_fusion must be single")
    if rerank_query_mode not in {"resolved", "current", "user_topic"}:
        raise ValueError("unknown rerank query mode")
    if stage != "hybrid_rerank" and rerank_query_mode != "resolved":
        raise ValueError("rerank_query_mode requires hybrid_rerank")
    if candidate_k < 5 or rerank_k < 5 or rrf_k < 0:
        raise ValueError("retrieval budgets must cover five passages and a nonnegative RRF constant")


def joint_recall_queries(turn, prior_context):
    """Raw dialogue stays a recall view; committed state is only another hint."""
    literal = retrieval_query(turn.user_questions)
    previous = prior_context.interpretations[-1] if prior_context.interpretations else None
    if previous is None or previous.task_frame is None or previous.relation == "ambiguous":
        return literal, literal
    from adapters.conversation import frame_query
    return literal, frame_query(previous.task_frame) + "\n" + turn.query


class HybridRetriever:
    """Shared P3 retrieval; source-qualified history resolution is explicit opt-in."""
    def __init__(self, sparse, dense, encoder, reranker=None, *, query_mode="current",
                 stage="hybrid_rerank", candidate_k=50, rerank_k=50, rrf_k=60,
                 source_config=None, query_fusion="single", resolver=None, rerank_query_mode="resolved", trace_sink=None,
                 source_window=False, evidence_selector=None, source_documents=None):
        from adapters.query_resolution import SOURCE_MODES
        validate_retrieval_options(query_mode=query_mode, stage=stage, query_fusion=query_fusion,
            rerank_query_mode=rerank_query_mode, candidate_k=candidate_k, rerank_k=rerank_k, rrf_k=rrf_k,
            joint_selection=evidence_selector is not None)
        if query_mode in SOURCE_MODES and (resolver is None or query_fusion != "current_topic_rrf"):
            raise ValueError("topic selector modes require resolver and current_topic_rrf")
        if query_mode == "conversational" and resolver is None:
            raise ValueError("conversational retrieval needs a resolver")
        if stage == "hybrid_rerank" and reranker is None:
            raise ValueError("hybrid_rerank requires a loaded reranker")
        if sparse.manifest["corpus_sha256"] != dense.manifest["corpus_sha256"]:
            raise ValueError("sparse and dense stages must use the same original corpus")
        self.sparse, self.dense, self.encoder, self.reranker = sparse, dense, encoder, reranker
        self.corpus_snapshot_id = sparse.corpus_snapshot_id
        self.query_mode, self.stage = query_mode, stage
        self.query_fusion = query_fusion
        self.rerank_query_mode = rerank_query_mode
        self.resolver = resolver
        self.requires_history_context = query_mode in SOURCE_MODES or query_mode == "conversational"
        self.requires_source_context = query_mode in SOURCE_MODES
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
        self.read_source_documents = None
        self.source_neighbors = None
        self.selection_support = None
        self.selection_record = None
        self.evidence_selector = evidence_selector
        self.source_documents = source_documents
        self.document_window_counts = None

    def bind_corpus(self, corpus_index):
        """Share the generation metadata index with reranking, before serving starts."""
        self.source_context = corpus_index.contextualize
        self.selection_group = corpus_index.selection_group
        self.expand_source_window = corpus_index.source_window
        if self.source_documents is not None:
            self.read_source_documents = lambda contexts, documents=self.source_documents: corpus_index.reading_contexts(contexts, documents)
            self.document_window_counts = corpus_index.document_window_counts
        self.source_neighbors = corpus_index.candidate_neighbors
        self.selection_support = corpus_index.selection_support
        self.selection_record = corpus_index.selection_record

    @property
    def manifest(self):
        manifest = {"kind": "bge_hybrid", "corpus_snapshot_id": self.corpus_snapshot_id,
            "query_mode": self.query_mode, "stage": self.stage, "top_k": 5,
            "query_fusion": self.query_fusion,
            "rerank_query_mode": "dialogue_views" if self.query_fusion == "dialogue_rrf" else self.rerank_query_mode,
            "rerank_document_view": "source address, title and original body",
            "pre_rerank_metadata": self.source_context is not None,
            "source_window": {"enabled": self.source_window, "radius": None if self.source_documents is not None else 1,
                "mode": "original_document_paragraphs" if self.source_documents is not None else "adjacent_passages",
                "document_alignment_counts": dict(self.document_window_counts or {}),
                "context_paragraphs_each_side": 2 if self.source_documents is not None else None,
                "scope": ("confirmed snapshot/version/source identity and literal whitespace-aligned passage" if self.source_documents is not None
                          else "same source/document/version/declared section; explicit span or MT-RAG ingestion order"),
                "selection": ("merge literal paragraph windows and replace unscoped slices; retain raw ranking separately" if self.source_documents is not None
                              else "one hop from ranked hits, complete originals, no invented rank or recursive expansion"),
                "semantic_selection": "prepare independent reading units and dependency closure before choice; no unselected output expansion",
                "budget": "shared A/R required evidence; normal exact-token feasibility and native admission apply",
                "extra_model_calls": 0},
            "semantic_evidence_selection": self.evidence_selector.manifest if self.evidence_selector else None,
            "evidence_selection": {"strategy": "ranked_exact_content_groups_v1",
                "group": "exact title/body with known version, parent and dependency scope",
                "source_identity": "all retrieved members of selected groups; exact body aliases retain separate references",
                "budget": "at most five groups from the existing scored pool",
                "extra_model_calls": 0},
            "requirement_selection": {"strategy": ("dialogue_view_rank_fusion_v1" if self.query_fusion == "dialogue_rrf"
                                                      else "task_requirement_coverage_v1"),
                "activation": ("conversational requests; distinct compound needs add derived views" if self.query_fusion == "dialogue_rrf"
                               else "resolved compound task with at least two explicit information needs; hybrid or hybrid_rerank"),
                "retrieval": ("batched query encoding, family-weighted RRF, and same-source neighbors for compound needs" if self.query_fusion == "dialogue_rrf"
                              else "batched query encoding, equal RRF, and available same-source neighbors as scored candidates"),
                "selection": ("joint task-conditioned selection of prepared reading units" if self.evidence_selector
                              else "literal and derived-family rank fusion within five distinct groups" if self.query_fusion == "dialogue_rrf"
                              else "primary task anchor plus diminishing-return coverage within five distinct groups"),
                "cost": ("batched query encoding and same-pool scoring across views" if self.query_fusion == "dialogue_rrf"
                         else "one batched encode and search per view; optional primary ranking; per-need scores only without joint semantic selection")},
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
        if self.evidence_selector is not None:
            manifest.update(query_fusion="literal_and_committed_state_recall", rerank_query_mode="literal_history",
                requirement_selection={"strategy": "joint_task_and_evidence",
                    "activation": "semantic selection",
                    "retrieval": "literal recent users plus previous committed state, batched recall and one initial ranking",
                    "selection": "current requirements and applicable sources inferred together from actual dialogue and complete original catalog",
                    "cost": "one helper call, no preceding interpretation or per-requirement reranking"})
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
        if self.evidence_selector is not None and self.source_context is None:
            raise ValueError("semantic evidence selection requires --evidence-corpus metadata")
        started = time.perf_counter()
        view = user_topic_view(turn.user_questions) if self.query_mode == "user_topic" else None
        if self.resolver is not None:
            if prior_context is None:
                raise ValueError("topic selector requires an explicit committed history context")
            if self.evidence_selector is None:
                view = self.resolver.resolve(turn, prior_context)
        from controller.schema import QueryContext
        from controller.serialization import decode
        intent = decode(QueryContext, view["context"]) if view is not None and view.get("context") is not None else None
        conversational = intent is not None and intent.relation == "conversational"
        semantic_selection = self.evidence_selector is not None
        requirements = ()
        if not conversational and intent is not None and intent.task_frame is not None and len(intent.task_frame.requirements) >= 2:
            from adapters.conversation import frame_query
            requirements = tuple(dict.fromkeys(frame_query(intent.task_frame, requirement=r)
                                               for r in intent.task_frame.requirements))
        fusion_trace = selection_trace = requirement_trace = semantic_trace = None
        contexts = ()
        supporting_contexts = ()
        retrieval_contexts = None
        retrieval_error = scoring_query = None
        try:
            if conversational:
                return replace(turn, contexts=(), query_context=intent,
                               supporting_contexts=(), retrieval_contexts=None)
            query = (retrieval_query(turn.user_questions) if semantic_selection else
                     view["text"] if view else query_text(turn.user_questions, self.query_mode))
            scoring_query = rerank_query_text(turn, query, self.rerank_query_mode)
            if view is None and self.rerank_query_mode != "resolved":
                view = {"mode": self.query_mode, "text": query}
            # Only the local encoder/reranker pair is serialized. Both semantic
            # HTTP operations run outside this lock; shutdown drains the worker.
            with self._lock:
                if semantic_selection:
                    literal, state = joint_recall_queries(turn, prior_context)
                    query = scoring_query = literal
                    if self.stage == "dense":
                        queries = tuple(dict.fromkeys((literal, state)))
                        if len(queries) == 1:
                            ranked = self.dense.search(literal, self.encoder, top_k=self.candidate_k)
                        else:
                            rows = self.dense.search_many(queries, self.encoder, top_k=self.candidate_k)
                            fused = reciprocal_rank_fusion([[c.document_id for c in row] for row in rows], k=self.rrf_k)
                            ranked = tuple(replace(self.dense.by_id[pid], score=score) for pid, score in fused)
                        fusion_trace = {"source_queries": list(queries), "encoder_batches": 1}
                    else:
                        stages = search_current_topic(literal, state, self.sparse, self.dense, self.encoder,
                            self.reranker if self.stage == "hybrid_rerank" else None,
                            candidate_k=self.candidate_k, rerank_k=self.rerank_k, rrf_k=self.rrf_k,
                            rerank_query=literal, source_context=self.source_context)
                        ranked = stages[self.stage]["contexts"]
                        fusion_trace = stages[self.stage]["query_fusion_trace"]
                    fusion_trace = {**fusion_trace, "strategy": "literal_and_committed_state_recall",
                        "literal_query": literal, "state_query": state,
                        "scope": "recall hints only; current interpretation and source selection follow together"}
                elif self.query_fusion == "dialogue_rrf":
                    ranked, fusion_trace = search_dialogue_views(retrieval_query(turn.user_questions), query,
                        requirements, self.sparse, self.dense, self.encoder,
                        self.reranker if self.stage == "hybrid_rerank" else None,
                        candidate_k=self.candidate_k, rerank_k=self.rerank_k, rrf_k=self.rrf_k,
                        source_context=self.source_context, source_neighbors=self.source_neighbors)
                    scoring_query = None  # Exact scoring views are recorded in fusion_trace.
                elif requirements and self.stage in {"hybrid", "hybrid_rerank"}:
                    contexts, requirement_trace = search_requirements(query, requirements,
                        self.sparse, self.dense, self.encoder, self.reranker,
                        candidate_k=self.candidate_k, rerank_k=self.rerank_k, rrf_k=self.rrf_k,
                        current_query=turn.query if self.query_fusion == "current_topic_rrf" else None,
                        source_context=self.source_context, source_neighbors=self.source_neighbors,
                        group_key=self.selection_group, rank_coverage=not semantic_selection)
                    orders = [requirement_trace["primary_ranked_ids"], *requirement_trace["requirement_ranked_ids"]]
                    ids = dict.fromkeys(pid for row in zip(*orders) for pid in row)
                    ranked = tuple(self.dense.by_id[pid] for pid in ids)
                elif self.stage == "dense":
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
                if requirement_trace is None:
                    contexts, selection_trace = select_ranked_evidence(ranked, group_key=self.selection_group)
                if semantic_selection:
                    from adapters.evidence_selection import reading_catalog
                    catalog, origins = reading_catalog(ranked, neighbors=self.source_neighbors,
                        indexed=self.dense.by_id, contextualize=self.source_context,
                        read=self.read_source_documents if self.source_window else None)
                    required_context = self.selection_support(catalog,
                        neighbors=self.source_window and self.read_source_documents is None)
                elif self.source_window:
                    if self.read_source_documents is not None:
                        reading = self.read_source_documents(contexts)
                        if reading != tuple(contexts):
                            retrieval_contexts = tuple(contexts)
                        contexts = reading
                    else:
                        supporting_contexts = self.expand_source_window(contexts)
                if view is None and (selection_trace is not None or self.source_window):
                    view = {"mode": self.query_mode, "text": query}
                self.requests += 1
            if semantic_selection:
                contexts, view, semantic_trace = self.evidence_selector.select(turn, prior_context, catalog, contexts,
                    required_context=required_context, source_record=self.selection_record)
                intent = decode(QueryContext, view["context"]) if view.get("context") is not None else None
                conversational = intent is not None and intent.relation == "conversational"
                if conversational:
                    contexts, supporting_contexts = (), ()
                elif semantic_trace["status"] == "selected":
                    selected_ids = {c.document_id for c in contexts}
                    supporting_contexts = tuple({s.document_id: s for c in contexts
                        for s in required_context.get(c.document_id, ()) if s.document_id not in selected_ids}.values())
                    selected_origins = tuple({hit.document_id: hit for c in contexts
                        for hit in origins[c.document_id]}.values())
                    semantic_trace["selected_origin_ids"] = [c.document_id for c in selected_origins]
                    if selected_origins != tuple(contexts):
                        retrieval_contexts = selected_origins
        except Exception as exc:
            retrieval_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if view is not None:
                trace = {"task_id": turn.task_id, "conversation_id": turn.conversation_id,
                    "turn": turn.turn, **view,
                    **({"context": asdict(intent)} if intent is not None else {}),
                    "rerank_query_mode": ("literal_history" if semantic_selection else
                                          "dialogue_views" if self.query_fusion == "dialogue_rrf" else self.rerank_query_mode),
                    "rerank_query": scoring_query if self.stage == "hybrid_rerank" else None,
                    "retrieval_status": ("failed" if retrieval_error else
                                         "completed_conversational" if conversational and semantic_selection else
                                         "skipped_conversational" if conversational else "completed"),
                    "retrieval_error": retrieval_error,
                    "retrieval_elapsed_ms": (time.perf_counter()-started)*1000,
                    **({"evidence_selection": selection_trace} if selection_trace else {}),
                    **({"requirement_evidence": requirement_trace} if requirement_trace else {}),
                    **({"semantic_evidence": semantic_trace} if semantic_trace else {}),
                    **({"source_window": {"primary_ids": [c.document_id for c in (retrieval_contexts if retrieval_contexts is not None else contexts)],
                        "reading_ids": [c.document_id for c in contexts],
                        "supporting_ids": [c.document_id for c in supporting_contexts]}}
                       if self.source_window and not retrieval_error else {}),
                    **({"query_fusion_trace": fusion_trace} if fusion_trace else {})}
                if self.trace_sink is None:
                    self.query_view_traces.append(trace)
                else:
                    self.trace_sink(trace)
        return replace(turn, contexts=tuple(contexts), query_context=intent,
                       supporting_contexts=supporting_contexts, retrieval_contexts=retrieval_contexts)

    def close(self):
        with self._lock:
            self.encoder.close()
            if self.reranker is not None:
                self.reranker.close()
            if self.resolver is not None:
                self.resolver.close()
