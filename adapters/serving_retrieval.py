"""Configured retrieval from a committed controller snapshot, without dataset rows."""
from __future__ import annotations

import time

from controller.schema import OnlineMessage, OnlineTurn, history_messages
from controller.context_schema import RetrievalState
from adapters.conversation import snapshot_context
from adapters.retrieval import bind_corpus_context, run_retrieval_worker, to_evidence
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
        bind_corpus_context(retriever, corpus_index)
        self.source_directory = None

    def configure_study(self, task_interpreter, tokenizer, *, profile=None, reference_date=None):
        from adapters.evidence_pool import SourceDirectory
        self.source_directory = SourceDirectory(self.corpus_index, self.candidate_search, tokenizer,
            task_interpreter=task_interpreter, profile=profile, reference_date=reference_date,
            archive_encoder=task_interpreter.history_index.encode)
        return self.source_directory

    def retrieval_state(self, snapshot):
        if self.source_directory is not None:
            from adapters.history_state import retrieval_state
            return retrieval_state(snapshot, self.source_directory)
        return RetrievalState(snapshot.version, snapshot.raw_history,
            tuple(row.query_context.task_state for row in snapshot.raw_history
                  if row.query_context is not None and row.query_context.task_state is not None),
            tuple(block.key for block in snapshot.archive), self.source_directory)

    def candidate_search(self, query, *, top_k=50):
        return self.retriever.candidate_search(query, top_k=top_k)

    def source_fetch(self, keys):
        return self.corpus_index.resolve(keys)

    def archive_lookup(self, snapshot, keys):
        archive = {block.key: block for block in snapshot.archive}
        return tuple(archive[key] for key in keys)

    def retrieve_pool(self, snapshot, query, request_id):
        history = tuple(OnlineMessage(message.role, message.content)
                        for message in history_messages(snapshot.raw_history))
        turn = OnlineTurn(snapshot.session_id, request_id, len(snapshot.raw_history)+1,
                          self.corpus_id, query, history, (), "P3")
        memory = self.retrieval_state(snapshot)
        state = self.source_directory.task_interpreter.interpret(turn, memory)
        return state, self.source_directory.retrieve_pool(state, memory), memory

    @property
    def manifest(self):
        return {"corpus_id": self.corpus_id, "corpus_snapshot_id": self.corpus_snapshot_id,
                "metadata_enabled": self.corpus_index is not None,
                "history_source": "committed_raw_history",
                "retriever": self.retriever.manifest}

    async def retrieve(self, snapshot, query: str, request_id: str) -> EvidenceSelection:
        started = time.monotonic()
        return await run_retrieval_worker(self._retrieve, snapshot, query, request_id, started)

    async def retrieve_supplied(self, snapshot, query, request_id, blocks, *, source_metadata=None,
                                coverage_scope="selected_sources"):
        """The direct-answer baseline reads the caller's supplied original set."""
        blocks = tuple(blocks)
        return EvidenceSelection(blocks, tuple(block.key for block in blocks),
            auxiliary_usage={"input_tokens": 0, "output_tokens": 0, "compute_ms": 0, "models": []})

    def finish_request(self, session_id, task_id):
        """This retriever owns no request-local support predictions."""
        if self.events is not None:
            self.events.emit("retrieval_request_finished", task_id=task_id)

    def _retrieve(self, snapshot, query, request_id, started):
        history = tuple(OnlineMessage(message.role, message.content)
                        for message in history_messages(snapshot.raw_history))
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
                supporting = self.corpus_index.retrieved_keys(retrieved.supporting_contexts)
                blocks = self.corpus_index.resolve((*priority, *supporting))
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
                ranking=[{"passage_id": c.document_id, "score": c.score} for c in
                         (retrieved.retrieval_contexts if retrieved.retrieval_contexts is not None else retrieved.contexts)],
                retrieval_priority=priority, supplied_evidence_keys=tuple(b.key for b in blocks),
                query_context=retrieved.query_context)
        return EvidenceSelection(blocks, priority, retrieved.query_context)

    def close(self):
        """Application shutdown calls once; HybridRetriever.close waits on its lock."""
        if hasattr(self.retriever, "close"):
            self.retriever.close()
