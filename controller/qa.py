"""Document registration and one business path into the existing QA controller."""
from __future__ import annotations

import time
from functools import partial

from .session_store import DocumentSet


class DocumentQA:
    def __init__(self, controller, provider):
        self.controller, self.provider = controller, provider
        provider.committed_readings = controller.store.committed_readings

    async def start(self):
        """Start the controller lifecycle before an external queue admits requests."""
        return await self.controller.start()

    def register_documents(self, blocks, *, source_metadata=None):
        started, cpu_started = time.monotonic(), time.thread_time()
        documents = self.controller.store.register_document_set(blocks, source_metadata=source_metadata)
        document_count = len({(block.key.corpus_id, block.key.document_id, block.key.version)
                              for block in documents.blocks})
        input_chars = sum(len(block.text) for block in documents.blocks)
        ended, cpu_ended = time.monotonic(), time.thread_time()
        self.controller.events.emit("documents_registered", document_set_id=documents.document_set_id,
            corpus_snapshot_id=documents.corpus_snapshot_id, block_count=len(documents.blocks),
            document_count=document_count, input_chars=input_chars,
            registration_window={"start": started, "end": ended, "clock": "client_monotonic"},
            registration_cpu_ms=(cpu_ended-cpu_started)*1000, registration_ms=(ended-started)*1000)
        return documents

    async def ask(self, session_id, query, request_id, *, task_id=None, supplied_blocks=None,
                  source_metadata=None, coverage_scope="selected_sources",
                  from_corpus_snapshot_id=None, request_arrived=None, document_set_id=None):
        await self.start()
        arrived = time.monotonic() if request_arrived is None else request_arrived
        task_id = task_id or request_id
        try:
            snapshot_id = self.provider.corpus_snapshot_id
            source_request_identity = None
            if document_set_id is not None:
                if supplied_blocks is not None or source_metadata is not None:
                    raise ValueError("registered documents cannot be combined with inline documents or metadata")
                documents = self.controller.store.load_document_set(document_set_id)
                supplied_blocks = documents.blocks
                source_metadata = dict(documents.source_metadata)
                snapshot_id = documents.corpus_snapshot_id
                source_request_identity = documents.document_set_id, coverage_scope
            elif supplied_blocks is not None:
                supplied_blocks = tuple(supplied_blocks)
                documents = DocumentSet.create(supplied_blocks, source_metadata)
                source_request_identity = documents.document_set_id, coverage_scope

            async def retrieve(snapshot, retrieval_query=None):
                selected_query = query if retrieval_query is None else retrieval_query
                if supplied_blocks is None:
                    return await self.provider.retrieve(snapshot, selected_query, task_id)
                return await self.provider.retrieve_supplied(snapshot, selected_query, task_id, supplied_blocks,
                    source_metadata=source_metadata, coverage_scope=coverage_scope)

            extend_for_plan = getattr(self.provider, "extend_for_plan", None)
            if extend_for_plan is not None:
                extend_for_plan = partial(extend_for_plan, supplied_blocks=supplied_blocks,
                                          source_metadata=source_metadata, trace_id=task_id)
            return await self.controller.serve_turn(session_id, query, request_id=request_id,
                corpus_snapshot_id=snapshot_id, evidence_provider=retrieve,
                source_request_identity=source_request_identity,
                extend_for_plan=extend_for_plan,
                from_corpus_snapshot_id=from_corpus_snapshot_id, request_arrived=arrived,
                dialogue_budget=getattr(self.provider, "budget", None))
        finally:
            self.provider.finish_request(session_id, task_id)
