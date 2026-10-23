"""Optional local BGE dense retrieval, RRF and cross-encoder reranking.

Original passages and the BM25 baseline remain unchanged. Heavy dependencies
are imported only by explicit model/index operations; no runtime downloads.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from .mtrag import OnlineContext, json_hash
from .retrieval import deterministic_top_k, retrieval_query


ENCODER_ID = "BAAI/bge-m3"
ENCODER_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANKER_ID = "BAAI/bge-reranker-v2-m3"
RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def query_text(user_questions, mode="last3", *, generated_history=None):
    """No answers/labels are accepted implicitly; history must be supplied explicitly."""
    if not user_questions:
        raise ValueError("retrieval requires the current user question")
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
