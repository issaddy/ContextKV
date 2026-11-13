"""One entry point for registered and benchmark document question answering."""
from __future__ import annotations

import time


class DocumentQA:
    def __init__(self, controller, provider):
        self.controller, self.provider = controller, provider

    async def start(self):
        return await self.controller.start()

    def register_documents(self, blocks, *, source_metadata=None):
        return self.controller.store.register_document_set(blocks, source_metadata=source_metadata)

    async def ask(self, session_id, query, request_id, *, task_id=None, supplied_blocks=None,
                  source_metadata=None, coverage_scope="selected_sources", request_arrived=None,
                  document_set_id=None):
        await self.start()
        task_id = task_id or request_id
        snapshot_id = self.provider.corpus_snapshot_id
        if document_set_id is not None:
            documents = self.controller.store.load_document_set(document_set_id)
            supplied_blocks, source_metadata = documents.blocks, dict(documents.source_metadata)
            snapshot_id = documents.corpus_snapshot_id

        return await self.controller.serve_turn(session_id, query, request_id=request_id,
            corpus_snapshot_id=snapshot_id, provider=self.provider, task_id=task_id,
            supplied_blocks=supplied_blocks, source_metadata=source_metadata, coverage_scope=coverage_scope,
            request_arrived=request_arrived if request_arrived is not None else time.monotonic())
