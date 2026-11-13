"""Hybrid recall primitives: reciprocal rank fusion and frozen document vectors."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from controller.schema import OnlineContext
from .retrieval import deterministic_top_k


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
        if encoder.manifest.get("document_encoder") is not None:
            raise ValueError("build document vectors with the base document encoder, not a query view")
        contexts = tuple(sorted(contexts, key=lambda c: c.document_id))
        if not contexts:
            raise ValueError("dense corpus must not be empty")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {"schema_version": 1, "status": "building", "corpus_snapshot_id": corpus_snapshot_id,
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
        manifest.update(status="complete", dimensions=vectors.shape[1], build_seconds=time.perf_counter()-started)
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

    def search_vectors(self, vectors, *, top_k=50, legal_ids=None):
        """Search the legal source population using already computed query/Need vectors."""
        import numpy as np
        positions = np.asarray([index for index, context in enumerate(self.contexts)
                                if legal_ids is None or context.document_id in legal_ids], dtype=int)
        result = []
        for vector in vectors:
            scores = np.asarray(self.vectors[positions] @ vector)
            order = deterministic_top_k(scores, top_k, np)
            result.append(tuple(replace(self.contexts[int(positions[i])], score=float(scores[i])) for i in order))
        return result
