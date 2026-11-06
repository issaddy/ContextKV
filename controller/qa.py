"""Document registration and one business path into the existing QA controller."""
from __future__ import annotations

import time


class DocumentQA:
    def __init__(self, controller, provider):
        self.controller, self.provider = controller, provider
        provider.committed_readings = controller.store.committed_readings

    def register_documents(self, blocks, *, source_metadata=None):
        started, cpu_started = time.monotonic(), time.thread_time()
        documents = self.controller.store.register_document_set(blocks, source_metadata=source_metadata)
        self.controller.events.emit("documents_registered", document_set_id=documents.document_set_id,
            corpus_snapshot_id=documents.corpus_snapshot_id, block_count=len(documents.blocks),
            document_count=len({(block.key.corpus_id, block.key.document_id, block.key.version)
                                for block in documents.blocks}),
            input_chars=sum(len(block.text) for block in documents.blocks),
            registration_cpu_ms=(time.thread_time()-cpu_started)*1000,
            registration_ms=(time.monotonic()-started)*1000)
        return documents

    async def ask(self, session_id, query, request_id, *, task_id=None, supplied_blocks=None,
                  source_metadata=None, coverage_scope="selected_sources",
                  from_corpus_snapshot_id=None, request_arrived=None, document_set_id=None):
        arrived = time.monotonic() if request_arrived is None else request_arrived
        task_id = task_id or request_id
        memo_task_id = task_id
        try:
            previous = self.controller.store.find_request(session_id, request_id)
            if previous is not None:
                stored = self.controller.store.context_plan(previous.attempt_id) or previous.context_plan
                if stored is not None:
                    memo_task_id = stored.task_state.task_id
            snapshot_id = self.provider.corpus_snapshot_id
            if document_set_id is not None:
                if supplied_blocks is not None or source_metadata is not None:
                    raise ValueError("registered documents cannot be combined with inline documents or metadata")
                documents = self.controller.store.load_document_set(document_set_id)
                supplied_blocks = documents.blocks
                source_metadata = dict(documents.source_metadata)
                snapshot_id = documents.corpus_snapshot_id
            elif supplied_blocks is not None:
                supplied_blocks = tuple(supplied_blocks)

            async def retrieve(snapshot, retrieval_query=None):
                selected_query = query if retrieval_query is None else retrieval_query
                nonlocal memo_task_id
                if supplied_blocks is None:
                    selected = await self.provider.retrieve(snapshot, selected_query, task_id)
                else:
                    selected = await self.provider.retrieve_supplied(snapshot, selected_query, task_id, supplied_blocks,
                        source_metadata=source_metadata, coverage_scope=coverage_scope)
                if selected.query_context is not None and selected.query_context.task_state is not None:
                    memo_task_id = selected.query_context.task_state.task_id
                return selected

            return await self.controller.serve_turn(session_id, query, request_id=request_id,
                corpus_snapshot_id=snapshot_id, evidence_provider=retrieve,
                from_corpus_snapshot_id=from_corpus_snapshot_id, request_arrived=arrived,
                dialogue_budget=getattr(self.provider, "budget", None))
        finally:
            self.provider.finish_request(memo_task_id)
