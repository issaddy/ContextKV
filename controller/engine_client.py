"""Single-submission transport for SGLang evidence computation."""
from __future__ import annotations

from dataclasses import dataclass
import asyncio
import json
import time

import httpx

from .sampling import request_sampling


class EngineError(RuntimeError):
    def __init__(self, code, message=""):
        self.code = code
        super().__init__(f"{code}: {message}" if message else code)


@dataclass(frozen=True)
class EngineInfo:
    protocol_version: str
    engine_kind: str
    model_path: str
    page_size: int
    gpu_capacity_tokens: int
    host_capacity_tokens: int


class SGLangEngineClient:
    def __init__(self, base_url, *, timeout=180.0, sampling=None, transport=None):
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout,
                                         transport=transport)
        self._sampling_params = request_sampling(sampling)

    async def aclose(self):
        await self._client.aclose()

    async def _post(self, path, body):
        response = await self._client.post(path, json=body)
        response.raise_for_status()
        return response.json()

    async def info(self):
        return EngineInfo(**await self._post("/context/engine/info", {}))

    async def metrics(self):
        return await self._post("/context/metrics", {})

    async def profile_prepare(self, mode):
        return await self._post("/context/profile/prepare", {"mode": mode})

    async def close_session(self, session_id):
        return await self._post("/context/session/close", {"session_id": session_id})

    async def evidence_state(self, source_ids, *, cache_scope="shared", cache_namespace=None):
        return await self._post("/context/evidence/state", {"source_ids": list(source_ids),
            "cache_scope": cache_scope, "cache_namespace": cache_namespace})

    async def compute(self, request, *, session_id, request_id, max_new_tokens,
                      mode="auto", recompute_ratio=0.15, quality_loss=0.0,
                      quality_price_ms=0.0, response_schema=None, cache_namespace=None,
                      cache_scope="shared", costs=None):
        """One native submission carries the chosen input and future source demand."""
        sampling = {**self._sampling_params, "max_new_tokens": max_new_tokens}
        if response_schema is not None:
            sampling["json_schema"] = json.dumps(response_schema, ensure_ascii=False)
        started = time.perf_counter()
        body = {
            "rid": request_id, "input_ids": list(request.input_ids), "stream": False,
            "cache_salt": cache_namespace,
            "sampling_params": sampling,
            "contextkv_compute": {
                "session_id": session_id, "request_id": request_id, "mode": mode,
                "cache_scope": cache_scope,
                "blocks": list(request.blocks), "demand": list(request.demand),
                "recompute_ratio": recompute_ratio, "quality_loss": quality_loss,
                "quality_price_ms": quality_price_ms, "costs": costs or {},
            },
        }
        try:
            data = await self._post("/generate", body)
        except (httpx.TimeoutException, asyncio.CancelledError):
            response = await self._client.post("/abort_request", json={"rid": request_id})
            response.raise_for_status()
            raise
        return {"text": data["text"], "output_ids": tuple(data.get("output_ids", ())),
                "meta": data["meta_info"], "elapsed_ms": (time.perf_counter() - started) * 1000}
