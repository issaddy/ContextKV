"""HTTP entry point for the shared document-QA execution path."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import yaml

from .config import apply_study_config, create_controller, load_config
from .evidence import make_evidence
from .qa import DocumentQA
from .serialization import to_jsonable


class EvidenceInput(BaseModel):
    document_id: str
    text: str
    passage_id: str | None = None
    corpus_id: str = "api"
    version: str = "v1"
    source_uri: str | None = None
    title: str = ""
    metadata: dict = Field(default_factory=dict)


class RegistrationInput(BaseModel):
    documents: list[EvidenceInput]


class QueryInput(BaseModel):
    query: str
    request_id: str
    document_set_id: str | None = None
    coverage_scope: str = "selected_sources"


def create_app(controller=None, config=None, *, retrieval=None,
               study_config="configs/study.yaml"):
    study = yaml.safe_load(Path(study_config).read_text())
    settings = apply_study_config(config or load_config(study["generation"]["run_config"]), study)
    service = controller or create_controller(settings)

    @asynccontextmanager
    async def lifespan(app):
        from adapters.study_retrieval import create_study_retriever
        provider = retrieval or await asyncio.to_thread(create_study_retriever, study, service.template)
        app.state.qa = DocumentQA(service, provider)
        await app.state.qa.start()
        try:
            yield
        finally:
            provider.close()
            await service.aclose()
            service.store.close()

    app = FastAPI(title="ContextKV", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "engine": to_jsonable(service.engine_info)}

    @app.post("/documents")
    async def register(body: RegistrationInput):
        blocks, metadata = [], {}
        for item in body.documents:
            values = item.model_dump()
            fields = values.pop("metadata")
            block = make_evidence(**values)
            blocks.append(block)
            metadata[block.key] = fields
        registered = app.state.qa.register_documents(blocks, source_metadata=metadata)
        return {"document_set_id": registered.document_set_id, "documents": len(blocks)}

    @app.post("/sessions/{session_id}/query")
    async def query(session_id: str, body: QueryInput):
        result = await app.state.qa.ask(session_id, body.query, body.request_id,
            document_set_id=body.document_set_id, coverage_scope=body.coverage_scope)
        return {"answer": result.assistant_text,
                "sources": to_jsonable(result.visible_evidence_manifest),
                "computations": result.computations,
                "metrics": {name: getattr(result, name) for name in (
                    "complete_ms", "total_input_tokens", "total_output_tokens",
                    "total_device_hit_tokens", "total_host_loaded_tokens", "total_prefill_tokens")}}

    @app.get("/sessions/{session_id}")
    async def history(session_id: str):
        return to_jsonable(service.store.get_snapshot(session_id).raw_history)

    @app.post("/sessions/{session_id}/close")
    async def close(session_id: str):
        await service.close_session(session_id)
        return {"closed": True}

    @app.exception_handler(ValueError)
    async def invalid_input(request, error):
        return JSONResponse(status_code=422, content={"error": str(error)})

    return app
