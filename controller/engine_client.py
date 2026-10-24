"""Guarded engine protocol and capability-checked SGLang HTTP transport.

A stock /generate endpoint is deliberately insufficient: the research protocol
must be negotiated before any epoch guard fields are transmitted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time
import json
from typing import Protocol, Sequence
import uuid

from controller.schema import (CacheQuoteBatch, CompletionReceipt,
                               ConfigFingerprint, GenerationResult, PromptCandidate)
from controller.serialization import decode
from controller.templates import token_hash as token_hash

PROTOCOL_VERSION = "contextkv-v1"
SGLANG_COMMIT = "3f36fc08331f64cd63c4fbbdf39cc402f38bb38b"


class EngineError(RuntimeError):
    def __init__(self, code: str, message: str = ""):
        self.code = code
        super().__init__(f"{code}: {message}" if message else code)


class UnknownOutcomeError(EngineError):
    """Transport failed after submission; inspect status before another turn."""


@dataclass(frozen=True)
class EngineInfo:
    engine_restart_id: str
    protocol_version: str = PROTOCOL_VERSION
    engine_kind: str = "mock"
    host_quote_supported: bool = True
    completion_status_supported: bool = True
    guarded_generate_supported: bool = True
    scheduler_owned: bool = True
    cache_config_hash: str = "mock-v1"
    fingerprint: ConfigFingerprint | None = None
    sglang_commit: str | None = None
    metrics_supported: bool = False
    result_replay_supported: bool = False
    result_ttl_seconds: float | None = None
    profiling_enabled: bool = False
    streaming_generation_supported: bool = False
    business_session_identity_supported: bool = False
    admission_diagnostics_enabled: bool = False
    model_scope: dict | None = None


@dataclass(frozen=True)
class PrepareAck:
    engine_session_id: str
    session_generation: int
    engine_restart_id: str
    operation_id: str
    state: str = "PREPARED"
    business_session_id: str | None = None


@dataclass(frozen=True)
class RetireAck:
    engine_session_id: str
    session_generation: int
    engine_restart_id: str
    operation_id: str
    state: str = "RETIRED"
    acknowledged: bool = True
    freed_gpu_tokens: int = 0
    freed_host_tokens: int = 0


@dataclass(frozen=True)
class EpochStatus:
    state: str
    engine_restart_id: str
    engine_session_id: str | None = None
    session_generation: int | None = None
    attempt_id: str | None = None
    inflight: bool = False
    terminal: bool = False
    result: GenerationResult | None = None
    receipt: CompletionReceipt | None = None


class Engine(Protocol):
    async def info(self) -> EngineInfo: ...
    async def quote_batch(self, candidates: Sequence[PromptCandidate], fingerprint: ConfigFingerprint) -> CacheQuoteBatch: ...
    async def prepare(self, engine_session_id: str, operation_id: str, expected_restart_id: str,
                      *, business_session_id: str | None = None) -> PrepareAck: ...
    async def generate(self, *, input_ids: Sequence[int], session_id: str, expected_generation: int,
                       expected_restart_id: str, attempt_id: str, fingerprint: ConfigFingerprint,
                       max_new_tokens: int = 128, ignore_eos: bool = False) -> GenerationResult: ...
    async def retire(self, engine_session_id: str, expected_generation: int, operation_id: str,
                     expected_restart_id: str, reason: str = "replaced") -> RetireAck: ...
    async def status(self, engine_session_id: str | None = None, attempt_id: str | None = None,
                     operation_id: str | None = None, expected_restart_id: str | None = None) -> EpochStatus: ...


def decode_result(value: dict) -> GenerationResult:
    value = dict(value)
    value["output_ids"] = tuple(value["output_ids"])
    value["receipt"] = CompletionReceipt(**value["receipt"])
    return GenerationResult(**value)


class SGLangEngineClient:
    def __init__(self, base_url: str, fingerprint: ConfigFingerprint, *, timeout: float = 120.0,
                 transport=None, stream: bool = True, model_scope: dict | None = None):
        import httpx
        self.fingerprint = fingerprint
        self._expected_model_scope = ((model_scope["identity"], model_scope["record"]["model_path"])
                                      if model_scope is not None else None)
        self.stream = stream
        self._client = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout,
                                         transport=transport)
        self._info: EngineInfo | None = None

    async def aclose(self):
        await self._client.aclose()

    async def _post(self, path: str, body: dict) -> dict:
        import httpx
        try:
            response = await self._client.post(path, json=body)
        except httpx.TransportError as exc:
            raise UnknownOutcomeError("TRANSPORT_ERROR", str(exc)) from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise EngineError("INVALID_ENGINE_RESPONSE", response.text[:200]) from exc
        if response.is_error or data.get("error"):
            detail = data.get("error", data.get("detail"))
            if detail is None:
                # Native non-stream /generate HTTPException uses a top-level
                # ErrorResponse; context control uses the nested envelope below.
                # Preserve HTTP status/message without guessing a protocol code
                # from the human-readable message.
                raise EngineError("HTTP_ERROR", f"HTTP {response.status_code}: {data.get('message', str(data))}")
            if isinstance(detail, dict):
                raise EngineError(detail.get("code", "HTTP_ERROR"), detail.get("message", str(detail)))
            raise EngineError("HTTP_ERROR", str(detail))
        return data

    async def info(self) -> EngineInfo:
        data = await self._post("/context/engine/info", {})
        required = (data.get("protocol_version") == PROTOCOL_VERSION
                    and data.get("guarded_generate_supported") is True
                    and data.get("scheduler_owned") is True
                    and data.get("completion_status_supported") is True
                    and data.get("sglang_commit") == SGLANG_COMMIT)
        if not required:
            raise EngineError("UNSUPPORTED_ENGINE", "fixed-commit scheduler-owned contextkv-v1 patch required")
        if data.get("fingerprint") != asdict(self.fingerprint):
            raise EngineError("FINGERPRINT_MISMATCH", "engine model/tokenizer/template/namespace differ")
        if self._expected_model_scope is not None:
            actual = data.get("model_scope") or {}
            if (actual.get("identity"), actual.get("record", {}).get("model_path")) != self._expected_model_scope:
                raise EngineError("MODEL_SCOPE_MISMATCH", "engine local model identity/path differs from configured scope")
        data["fingerprint"] = ConfigFingerprint(**data["fingerprint"])
        self._info = EngineInfo(**data)
        return self._info

    async def _ready(self):
        if self._info is None:
            await self.info()

    async def quote_batch(self, candidates, fingerprint):
        await self._ready()
        self._check_fingerprint(fingerprint)
        data = await self._post("/cache/quote", {
            "request_id": uuid.uuid4().hex, "fingerprint": asdict(fingerprint),
            "candidates": [{"candidate_id": c.candidate_id, "input_ids": list(c.input_ids),
                            "input_hash": c.input_hash, "snapshot_version": c.snapshot_version}
                           for c in candidates]})
        return decode(CacheQuoteBatch, data)

    def _check_fingerprint(self, fingerprint):
        if fingerprint != self.fingerprint:
            raise EngineError("FINGERPRINT_MISMATCH")

    async def prepare(self, engine_session_id, operation_id, expected_restart_id, *, business_session_id=None):
        await self._ready()
        body = {"engine_session_id": engine_session_id, "operation_id": operation_id,
                "expected_restart_id": expected_restart_id}
        if business_session_id is not None:
            if not self._info.business_session_identity_supported:
                raise EngineError("BUSINESS_SESSION_IDENTITY_UNSUPPORTED")
            body["business_session_id"] = business_session_id
        return PrepareAck(**await self._post("/context/epoch/prepare", body))

    async def generate(self, *, input_ids, session_id, expected_generation, expected_restart_id,
                       attempt_id, fingerprint, max_new_tokens=128, ignore_eos=False):
        await self._ready()
        self._check_fingerprint(fingerprint)
        start = time.perf_counter()
        sampling = {"temperature": 0, "max_new_tokens": max_new_tokens}
        if ignore_eos:
            sampling["ignore_eos"] = True
        body = {
            "rid": attempt_id, "input_ids": list(input_ids), "session_id": session_id,
            "expected_generation": expected_generation, "expected_restart_id": expected_restart_id,
            "contextkv_attempt_id": attempt_id, "contextkv_fingerprint": asdict(fingerprint),
            "sampling_params": sampling, "stream": self.stream}
        first_token_ms = last_token_ms = None
        if self.stream:
            if not self._info.streaming_generation_supported:
                raise EngineError("UNSUPPORTED_STREAMING", "engine needs the guarded cumulative SSE patch")
            data, first_token_ms, last_token_ms = await self._stream_generate(body, start)
        else:
            data = await self._post("/generate", body)
        # Completion receipt comes through the native customized-info pipeline.
        meta = data.get("meta_info", {})
        receipt = meta.get("contextkv_receipt")
        if isinstance(receipt, list) and len(receipt) == 1:
            receipt = receipt[0]
        if not receipt:
            raise UnknownOutcomeError("MISSING_RECEIPT", "query attempt status before retry")
        parsed = CompletionReceipt(**receipt)
        timing = meta.get("contextkv_timing", {})
        if isinstance(timing, list):
            timing = timing[0] if timing else {}
        return GenerationResult(text=data["text"], output_ids=tuple(data["output_ids"]),
                                finish_reason=parsed.finish_reason, receipt=parsed,
                                generation_complete=parsed.generation_complete,
                                elapsed_ms=(time.perf_counter()-start)*1000,
                                ttft_ms=first_token_ms, last_token_ms=last_token_ms, **timing)

    async def _stream_generate(self, body, started):
        """Observe newly received tokens, not receipt-only/duplicate SSE frames.

        The guarded server contract normalizes native incremental output to
        cumulative snapshots. A partial body never becomes a GenerationResult.
        """
        import httpx
        first = last = None
        output_ids = []
        final = None
        done = False
        data_lines = []
        try:
            async with self._client.stream("POST", "/generate", json=body) as response:
                if response.is_error:
                    await response.aread()
                    try:
                        detail = response.json()
                    except ValueError:
                        detail = response.text[:200]
                    raise EngineError("HTTP_ERROR", str(detail))
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                        continue
                    if line or not data_lines:
                        continue
                    raw = "\n".join(data_lines)
                    data_lines.clear()
                    if raw == "[DONE]":
                        done = True
                        break
                    try:
                        item = json.loads(raw)
                    except ValueError as exc:
                        raise UnknownOutcomeError("INVALID_STREAM", "malformed SSE JSON; inspect attempt status") from exc
                    if item.get("error"):
                        raise UnknownOutcomeError("STREAM_ERROR", str(item["error"]))
                    ids = item.get("output_ids", [])
                    if len(ids) < len(output_ids):
                        raise UnknownOutcomeError("INVALID_STREAM", "cumulative token count decreased")
                    if len(ids) > len(output_ids):
                        observed = (time.perf_counter()-started)*1000
                        if first is None:
                            first = observed
                        last = observed
                        output_ids = ids
                    receipt = item.get("meta_info", {}).get("contextkv_receipt")
                    if receipt:
                        final = item
        except httpx.TransportError as exc:
            raise UnknownOutcomeError("TRANSPORT_ERROR", str(exc)) from exc
        if not done or final is None or data_lines:
            raise UnknownOutcomeError("INCOMPLETE_STREAM", "complete receipt and SSE DONE required; inspect attempt status")
        if final.get("output_ids") != output_ids:
            raise UnknownOutcomeError("INVALID_STREAM", "terminal snapshot disagrees with received tokens")
        return final, first, last

    async def retire(self, engine_session_id, expected_generation, operation_id,
                     expected_restart_id, reason="replaced"):
        await self._ready()
        return RetireAck(**await self._post("/context/epoch/retire", {
            "engine_session_id": engine_session_id, "expected_generation": expected_generation,
            "operation_id": operation_id, "expected_restart_id": expected_restart_id, "reason": reason}))

    async def status(self, engine_session_id=None, attempt_id=None, operation_id=None,
                     expected_restart_id=None):
        await self._ready()
        data = await self._post("/context/epoch/status", {
            "engine_session_id": engine_session_id, "attempt_id": attempt_id,
            "operation_id": operation_id, "expected_restart_id": expected_restart_id})
        if data.get("result"):
            data["result"] = decode_result(data["result"])
        if data.get("receipt"):
            data["receipt"] = CompletionReceipt(**data["receipt"])
        return EpochStatus(**data)

    async def metrics(self, after_seq=0):
        """Unique allocator counters and bounded, completed DMA/phase events."""
        await self._ready()
        return await self._post("/context/metrics", {"after_seq": after_seq})

    async def profile_prepare(self, mode, *, tier=None, tokens=None):
        """Only an explicitly enabled, idle profiling server accepts mutations."""
        await self._ready()
        body = {"mode": mode, "expected_restart_id": self._info.engine_restart_id}
        if tier is not None:
            body["tier"] = tier
        if tokens is not None:
            body["tokens"] = tokens
        result = await self._post("/context/profiling/prepare", body)
        # A cold fixture invalidates native sessions and changes restart identity.
        await self.info()
        return result
