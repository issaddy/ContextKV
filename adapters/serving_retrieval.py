"""Configured retrieval from a committed controller snapshot, without dataset rows."""
from __future__ import annotations

import time

from adapters.mtrag import OnlineMessage, OnlineTurn
from adapters.query_resolution import snapshot_context
from adapters.retrieval import run_retrieval_worker, to_evidence
from controller.service import EvidenceSelection


class ServingRetrievalError(RuntimeError):
    code = "RETRIEVAL_FAILED"


class ServingRetriever:
    """One loaded retriever and optional public metadata index per application.

    The controller owns session serialization and idempotency. Only its immutable
    committed snapshot enters this adapter; callers cannot supply answer history.
    """
    def __init__(self, retriever, *, corpus_id: str, corpus_index=None, events=None):
        if not corpus_id or not corpus_id.strip():
            raise ValueError("serving retrieval requires an explicit corpus_id")
        if corpus_index is not None and (corpus_index.corpus_id != corpus_id
                or corpus_index.snapshot_id != retriever.corpus_snapshot_id):
            raise ValueError("metadata corpus/snapshot must match the configured retriever")
        self.retriever, self.corpus_index, self.events = retriever, corpus_index, events
        self.corpus_id = corpus_id
        self.corpus_snapshot_id = retriever.corpus_snapshot_id

    @property
    def manifest(self):
        return {"corpus_id": self.corpus_id, "corpus_snapshot_id": self.corpus_snapshot_id,
                "metadata_enabled": self.corpus_index is not None,
                "history_source": "committed_raw_history",
                "source_inventory_scope": "committed_required_keys",
                "retriever": self.retriever.manifest}

    async def retrieve(self, snapshot, query: str, request_id: str) -> EvidenceSelection:
        started = time.monotonic()
        return await run_retrieval_worker(self._retrieve, snapshot, query, request_id, started)

    def _retrieve(self, snapshot, query, request_id, started):
        history = tuple(message for turn in snapshot.raw_history for message in (
            OnlineMessage("user", turn.user_text), OnlineMessage("assistant", turn.assistant_text)))
        turn = OnlineTurn(snapshot.session_id, request_id, len(snapshot.raw_history)+1,
                          self.corpus_id, query, history, (), "P3")
        try:
            if getattr(self.retriever, "requires_history_context", False):
                retrieved = self.retriever.retrieve_turn(turn, prior_context=snapshot_context(turn, snapshot))
            else:
                retrieved = self.retriever.retrieve_turn(turn)
            if self.corpus_index is None:
                blocks = tuple(to_evidence(c, self.corpus_snapshot_id, corpus_id=self.corpus_id)
                               for c in retrieved.contexts)
                priority = tuple(b.key for b in blocks)
            else:
                priority = self.corpus_index.retrieved_keys(retrieved.contexts)
                blocks = self.corpus_index.resolve(priority)
        except Exception as error:
            if self.events is not None:
                self.events.emit("retrieval_backend_failed", business_session_id=snapshot.session_id,
                    request_id=request_id, snapshot_version=snapshot.version,
                    retrieval_elapsed_ms=(time.monotonic()-started)*1000,
                    error=f"{type(error).__name__}: {error}")
            raise ServingRetrievalError(f"{type(error).__name__}: {error}") from error
        if self.events is not None:
            self.events.emit("retrieval_backend_completed", business_session_id=snapshot.session_id,
                request_id=request_id, snapshot_version=snapshot.version,
                corpus_snapshot_id=self.corpus_snapshot_id,
                retrieval_elapsed_ms=(time.monotonic()-started)*1000,
                ranking=[{"passage_id": c.document_id, "score": c.score} for c in retrieved.contexts],
                retrieval_priority=priority, supplied_evidence_keys=tuple(b.key for b in blocks),
                query_context=retrieved.query_context)
        return EvidenceSelection(blocks, priority, retrieved.query_context)

    def close(self):
        """Application shutdown calls once; HybridRetriever.close waits on its lock."""
        if hasattr(self.retriever, "close"):
            self.retriever.close()
