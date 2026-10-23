"""Local controller HTTP API. Use one worker per state database in normal operation."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

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


def create_app(controller: Controller | None = None, config: RunConfig | None = None) -> FastAPI:
    service = controller or create_controller(config or RunConfig())

    async def retries():
        while True:
            await asyncio.sleep(2)
            await service.epochs.flush()

    @asynccontextmanager
    async def lifespan(app):
        info = await service.engine.info()
        if info.fingerprint != service.template.fingerprint:
            raise RuntimeError("engine/controller fingerprint mismatch")
        worker = asyncio.create_task(retries())
        try:
            yield
        finally:
            worker.cancel()
            with suppress(asyncio.CancelledError):
                await worker
            await service.aclose()
            service.store.close()

    app = FastAPI(title="ContextKV", version="0.1.0", lifespan=lifespan)
    app.state.controller = service

    @app.exception_handler(StoreError)
    async def store_error(request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(EngineError)
    async def engine_error(request, exc):
        return JSONResponse(status_code=503, content={"detail": str(exc), "code": exc.code})

    @app.exception_handler(ContextOverflow)
    async def overflow(request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc), "code": "CONTEXT_OVERFLOW"})

    @app.get("/health")
    async def health():
        info = await service.engine.info()
        return {"status": "ok", "engine": to_jsonable(info), "pending_retire_operations": len(service.store.outbox())}

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
