"""Shared, deterministic retrieval. BM25S is imported only when explicitly used."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from adapters.mtrag import OnlineContext, OnlineTurn, clean_context, json_hash

BM25S_VERSION = "0.3.15"
NUMPY_VERSION = "2.3.5"
RETRIEVAL_CONFIG = {"implementation": "bm25s", "version": BM25S_VERSION,
                    "numpy_version": NUMPY_VERSION, "method": "lucene", "k1": 1.5,
                    "b": .75, "backend": "numpy", "csc_backend": "numpy",
                    "stopwords": "en", "stemmer": None, "query_user_turns": 3, "top_k": 5,
                    "tie_break": "ascending_passage_id"}


def rerank_document(context: OnlineContext) -> str:
    """A scoring view with available source context; the original body is unchanged."""
    body = context.title + "\n" + context.text
    return f"Source: {context.source}\n{body}" if context.source else body


def bind_corpus_context(retriever, corpus_index):
    """Bind the already loaded metadata before model ranking, when supported."""
    if corpus_index is None and getattr(retriever, "evidence_selector", None) is not None:
        raise ValueError("semantic evidence selection requires --evidence-corpus metadata")
    bind = getattr(retriever, "bind_corpus", None)
    if corpus_index is not None and bind is not None:
        bind(corpus_index)


async def run_retrieval_worker(operation, /, *args, **kwargs):
    """Run one synchronous retrieval; cancellation drains it before returning.

    Threads cannot be cancelled by cancelling their await. Both serving and
    workload shutdown must wait before closing the shared retrieval models.
    """
    worker = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while True:
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                if not worker.done():
                    continue
            except Exception:
                pass  # Preserve the caller's cancellation, not a late error.
            break
        raise


def fixed_retrieve(turn: OnlineTurn) -> tuple[OnlineContext, ...]:
    if turn.protocol == "P3":
        raise ValueError("P3 requires real retrieval, not the official contexts")
    if len(turn.contexts) != 5:
        raise ValueError("fixed retrieval requires five contexts")
    return turn.contexts


def retrieval_query(user_questions: Sequence[str]) -> str:
    return "\n".join(user_questions[-3:])


def to_evidence(context: OnlineContext | Mapping[str, Any], corpus_snapshot_id: str,
                tokenizer: Any | None = None, *, corpus_id: str = "mtrag"):
    from controller.schema import EvidenceBlock, EvidenceKey
    c = clean_context(context)
    key = EvidenceKey(corpus_id, c.document_id, c.document_id, corpus_snapshot_id, c.content_sha256)
    tokens = tuple(tokenizer.encode(c.text)) if tokenizer is not None else ()
    uri = c.source or f"{corpus_id}://{corpus_snapshot_id}/{c.document_id}"
    return EvidenceBlock(key, uri, c.title, c.text, token_ids=tokens,
                         provenance=(uri, f"corpus_snapshot:{corpus_snapshot_id}"))


def load_beir_corpus(path: str | Path) -> dict[str, dict[str, Any]]:
    """Preserve BEIR or CorpusIndex passage identities and original bodies.

    Retrieval keys identify passages; a source document can contain several of
    them. Public metadata follows CorpusIndex's top-level-over-metadata rule.
    The historical document_id-only input remains a single-passage record.
    """
    corpus = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metadata = {**row.get("metadata", {}), **row}
        pid = str(metadata.get("passage_id", metadata.get("_id", metadata.get("document_id", ""))))
        if not pid or pid in corpus:
            raise ValueError(f"missing or duplicate passage ID {pid!r}")
        if not isinstance(row.get("text"), str):
            raise ValueError(f"passage {pid} has no original text")
        corpus[pid] = {"document_id": pid, "title": str(metadata.get("title") or ""),
                       "text": row["text"],
                       "source": str(metadata.get("source_uri") or metadata.get("source") or metadata.get("url") or "")}
    return corpus


class BM25SRetriever:
    """Pinned CPU BM25, deterministic score ties, immutable original passages.

    Keep retrieval in a separate environment from CUDA dependencies if necessary.
    The default constructor enforces both frozen package versions. No silent
    fallback to a different BM25 formula or tokenization is permitted.
    """
    def __init__(self, corpus: Mapping[str, Mapping[str, Any]], *, corpus_snapshot_id: str,
                 cpu_threads: int = 1):
        self._initialize_runtime(cpu_threads)
        if not corpus:
            raise ValueError("corpus must not be empty")
        self._ids = tuple(sorted(str(pid) for pid in corpus))
        self._contexts = tuple(OnlineContext(pid, str(corpus[pid]["text"]),
                              str(corpus[pid].get("source") or ""),
                              str(corpus[pid].get("title") or "")) for pid in self._ids)
        self.corpus_snapshot_id = corpus_snapshot_id
        self.manifest = {**RETRIEVAL_CONFIG, "corpus_snapshot_id": corpus_snapshot_id,
                         "corpus_sha256": json_hash([asdict(c) for c in self._contexts]),
                         "passage_order_sha256": json_hash(self._ids), "cpu_threads": cpu_threads,
                         "thread_environment": {k: os.environ[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
        texts = [c.title + "\n" + c.text for c in self._contexts]
        tokens = self._bm25s.tokenize(texts, stopwords="en", stemmer=None, return_ids=False, show_progress=False)
        self._index = self._bm25s.BM25(method="lucene", k1=1.5, b=.75, backend="numpy", csc_backend="numpy")
        self._index.index(tokens, show_progress=False)

    def _initialize_runtime(self, cpu_threads: int):
        if cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")
        for package, wanted in (("bm25s", BM25S_VERSION), ("numpy", NUMPY_VERSION)):
            try:
                installed = version(package)
            except Exception as exc:
                raise RuntimeError("P3 requires the retrieval extra: bm25s==0.3.15 numpy==2.3.5") from exc
            if installed != wanted:
                raise RuntimeError(f"{package} must be {wanted}; installed {installed}")
        # Record requested threading; callers should set these before importing BLAS.
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            if variable not in os.environ:
                os.environ[variable] = str(cpu_threads)
        import bm25s
        import numpy as np
        self._bm25s, self._np = bm25s, np

    @classmethod
    def load(cls, directory: str | Path, *, cpu_threads: int = 1, mmap: bool = True):
        """Load the saved index without retokenizing/rebuilding its fixed corpus.

        File digests are checked once at this persistence boundary, never per query.
        """
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if any(manifest.get(k) != v for k, v in RETRIEVAL_CONFIG.items()):
            raise ValueError("saved index uses a different retrieval configuration")
        for name, digest in manifest["files"].items():
            if sha256((directory / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"saved index file changed: {name}")
        result = cls.__new__(cls)
        result._initialize_runtime(cpu_threads)
        result._contexts = tuple(OnlineContext(**json.loads(line)) for line in
                                 (directory / "passages.jsonl").read_text().splitlines() if line)
        result._ids = tuple(c.document_id for c in result._contexts)
        result.corpus_snapshot_id = manifest["corpus_snapshot_id"]
        result.manifest = {**manifest, "cpu_threads": cpu_threads, "loaded_index": True,
                           "thread_environment": {k: os.environ[k] for k in
                           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
        result._index = result._bm25s.BM25.load(str(directory / "index"), mmap=mmap,
                                               show_progress=False)
        return result

    def retrieve(self, user_questions: Sequence[str], *, top_k: int = 5) -> tuple[OnlineContext, ...]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query = retrieval_query(user_questions)
        tokens = self._bm25s.tokenize([query], stopwords="en", stemmer=None,
                                      return_ids=False, show_progress=False)[0]
        if not tokens:
            return ()
        scores = self._index.get_scores(tokens)
        order = deterministic_top_k(scores, top_k, self._np)
        return tuple(replace(self._contexts[int(i)], score=float(scores[i])) for i in order)

    def retrieve_turn(self, turn: OnlineTurn) -> OnlineTurn:
        if turn.protocol != "P3":
            raise ValueError("real retrieval requires a P3 turn")
        return replace(turn, contexts=self.retrieve(turn.user_questions))

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._index.save(str(directory / "index"))
        (directory / "passages.jsonl").write_text("\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in self._contexts) + "\n", encoding="utf-8")
        files = {str(p.relative_to(directory)): sha256(p.read_bytes()).hexdigest()
                 for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "manifest.json"}
        (directory / "manifest.json").write_text(json.dumps({**self.manifest, "files": files}, indent=2), encoding="utf-8")


def deterministic_top_k(scores, k: int, np):
    """O(n) partition plus O(k log k) sorting, including boundary ties by ID.

    Corpus indices already follow ascending passage IDs. Only the first needed
    equal-score indices are included, so an all-zero query never sorts the corpus.
    """
    k = min(k, len(scores))
    if k == 0:
        return np.empty(0, dtype=int)
    cutoff = np.partition(scores, len(scores) - k)[len(scores) - k]
    greater = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)[:k - len(greater)]
    chosen = np.concatenate((greater, tied))
    return chosen[np.lexsort((chosen, -scores[chosen]))]


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "query"])
    parser.add_argument("--index", required=True)
    parser.add_argument("--corpus")
    parser.add_argument("--snapshot", default="mtrag-53fef6f5917c7e10c436f697e69a955dbc795d81")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--question", action="append")
    args = parser.parse_args(argv)
    if args.command == "build":
        if not args.corpus:
            parser.error("build requires --corpus")
        if Path(args.index).exists():
            parser.error("index already exists; reuse it with query/--retriever-index or select a new directory")
        retriever = BM25SRetriever(load_beir_corpus(args.corpus), corpus_snapshot_id=args.snapshot, cpu_threads=args.cpu_threads)
        retriever.save(args.index)
        print(json.dumps(retriever.manifest, indent=2))
    else:
        if not args.question:
            parser.error("query requires at least one --question")
        retriever = BM25SRetriever.load(args.index, cpu_threads=args.cpu_threads)
        print(json.dumps([asdict(c) for c in retriever.retrieve(args.question)], ensure_ascii=False))


if __name__ == "__main__":
    main()
