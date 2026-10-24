"""Local controller HTTP API. Use one worker per state database in normal operation."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from adapters.serving_retrieval import ServingRetriever, ServingRetrievalError

from .config import RunConfig, create_controller
from .engine_client import EngineError
from .evidence import make_evidence
from .serialization import to_jsonable
from .service import ContextOverflow, Controller
from .session_store import StoreError


class EvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    passage_id: str | None = None
    corpus_id: str = "api"
    version: str = "v1"
    source_uri: str | None = None
    title: str = ""
    provenance: tuple[str, ...] = ()
    anchors: tuple[str, ...] = ()


class TurnInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    corpus_snapshot_id: str = "demo-v1"
    evidence: list[EvidenceInput] = Field(default_factory=list)


class RecoverInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    abandon: bool = False


class QueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1)
    request_id: str = Field(min_length=1)


def _load_retrieval(service, *, retriever_index, retriever_config, evidence_corpus, corpus_id):
    from adapters.corpus import CorpusIndex
    from adapters.retriever_factory import create_retriever

    retriever = create_retriever("P3", index_path=retriever_index, config_path=retriever_config,
        trace_sink=lambda trace: service.events.emit("retrieval_query_view", **trace))
    try:
        index = (CorpusIndex.from_jsonl(evidence_corpus, corpus_id=corpus_id,
                    snapshot_id=retriever.corpus_snapshot_id) if evidence_corpus else None)
        return ServingRetriever(retriever, corpus_id=corpus_id, corpus_index=index, events=service.events)
    except BaseException:
        if hasattr(retriever, "close"):
            retriever.close()
        raise


def create_app(controller: Controller | None = None, config: RunConfig | None = None, *,
               retrieval: ServingRetriever | None = None, retriever_index: str | None = None,
               retriever_config: str | None = None, evidence_corpus: str | None = None,
               corpus_id: str = "mtrag") -> FastAPI:
    if retriever_index and retriever_config:
        raise ValueError("choose retriever_index or retriever_config")
    configured = bool(retriever_index or retriever_config)
    if retrieval is not None and (configured or evidence_corpus):
        raise ValueError("choose an existing serving retriever or retrieval configuration")
    if evidence_corpus and not configured:
        raise ValueError("evidence_corpus requires a configured retriever")
    if not corpus_id.strip():
        raise ValueError("corpus_id must be nonempty")
    resolved_config = (config or RunConfig()) if controller is None else None
    service = controller if controller is not None else create_controller(resolved_config)

    async def retries():
        while True:
            await asyncio.sleep(2)
            await service.epochs.flush()

    @asynccontextmanager
    async def lifespan(app):
        worker = loader = None
        try:
            info = await service.engine.info()
            if info.fingerprint != service.template.fingerprint:
                raise RuntimeError("engine/controller fingerprint mismatch")
            if configured:
                loader = asyncio.create_task(asyncio.to_thread(_load_retrieval, service,
                    retriever_index=retriever_index, retriever_config=retriever_config,
                    evidence_corpus=evidence_corpus, corpus_id=corpus_id))
                app.state.retrieval = await asyncio.shield(loader)
            if app.state.retrieval is not None:
                service.events.emit("retrieval_configured", retrieval=app.state.retrieval.manifest)
            service.events.emit("service_started", engine=info, fingerprint=service.template.fingerprint,
                configuration_source="run_config" if resolved_config is not None else "injected_controller",
                config=resolved_config)
            worker = asyncio.create_task(retries())
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            try:
                if worker is not None:
                    worker.cancel()
                    with suppress(asyncio.CancelledError):
                        await worker
            finally:
                try:
                    if loader is not None and app.state.retrieval is None:
                        # A cancelled startup does not stop its loading thread.
                        # Reclaim its result before closing; preserve the original
                        # startup error/cancellation if loading itself failed.
                        with suppress(Exception):
                            app.state.retrieval = await asyncio.shield(loader)
                    if app.state.retrieval is not None:
                        await asyncio.to_thread(app.state.retrieval.close)
                finally:
                    try:
                        await service.aclose()
                    finally:
                        service.store.close()

    app = FastAPI(title="ContextKV", version="0.1.0", lifespan=lifespan)
    app.state.controller = service
    app.state.retrieval = retrieval
    app.state.ready = False

    @app.exception_handler(StoreError)
    async def store_error(request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(EngineError)
    async def engine_error(request, exc):
        return JSONResponse(status_code=503, content={"detail": str(exc), "code": exc.code})

    @app.exception_handler(ServingRetrievalError)
    async def retrieval_error(request, exc):
        return JSONResponse(status_code=503, content={"detail": str(exc), "code": exc.code})

    @app.exception_handler(ContextOverflow)
    async def overflow(request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc), "code": "CONTEXT_OVERFLOW"})

    @app.get("/health")
    async def health():
        info = await service.engine.info()
        adapter = app.state.retrieval
        return {"status": "ok", "engine": to_jsonable(info),
                "pending_retire_operations": len(service.store.outbox()),
                "retrieval": to_jsonable(adapter.manifest) if adapter is not None else None}

    @app.post("/sessions/{session_id}/query")
    async def query(session_id: str, body: QueryInput):
        adapter = app.state.retrieval
        if not app.state.ready or adapter is None:
            return JSONResponse(status_code=503, content={"code": "RETRIEVAL_NOT_READY",
                "detail": "query retrieval is not configured or not ready"})
        try:
            return to_jsonable(await service.serve_turn(session_id, body.query,
                request_id=body.request_id, corpus_snapshot_id=adapter.corpus_snapshot_id,
                evidence_provider=lambda snapshot: adapter.retrieve(snapshot, body.query, body.request_id)))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/sessions/{session_id}/turns")
    async def turn(session_id: str, body: TurnInput):
        try:
            blocks = tuple(make_evidence(**item.model_dump()) for item in body.evidence)
            return to_jsonable(await service.serve_turn(session_id, body.query, blocks,
                                  request_id=body.request_id, corpus_snapshot_id=body.corpus_snapshot_id))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}")
    async def state(session_id: str):
        try:
            return to_jsonable(service.store.get_snapshot(session_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc

    @app.post("/sessions/{session_id}/recover")
    async def recover(session_id: str, body: RecoverInput):
        result = await service.recover(session_id, abandon=body.abandon)
        return {"result": to_jsonable(result), "pending": to_jsonable(service.store.pending_attempt(session_id))}

    @app.post("/sessions/{session_id}/close")
    async def close(session_id: str):
        try:
            await service.close_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        return {"closed": True, "pending_retire_operations": len(service.store.outbox(session_id))}

    return app
