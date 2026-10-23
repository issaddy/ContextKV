from __future__ import annotations

import asyncio
from dataclasses import asdict
import json

import httpx
import pytest

from controller.engine_client import EngineError, EngineInfo, SGLANG_COMMIT, SGLangEngineClient
from controller.schema import ConfigFingerprint
from scripts.prepare_cold_job import prepare_cold


def client(*, enabled=True, busy=0, error=None, retained=0):
    fingerprint = ConfigFingerprint()
    calls = []
    restart = "before"
    attempts = 0

    def info():
        return asdict(EngineInfo(restart, engine_kind="sglang", fingerprint=fingerprint,
                                sglang_commit=SGLANG_COMMIT, profiling_enabled=enabled, metrics_supported=True))

    def handler(request):
        nonlocal restart, attempts
        calls.append(request.url.path)
        if request.url.path == "/context/engine/info":
            return httpx.Response(200, json=info())
        if request.url.path == "/context/metrics":
            used = 128 if restart == "before" else retained
            return httpx.Response(200, json={"engine_restart_id": restart, "observed_seq": 1,
                "observed_time_ns": 100, "clock": "server_monotonic", "cache_config": {"host_layout": "layer_first"},
                "active_requests": 0, "queued_requests": 0,
                "pool": {"unique_gpu_tokens": used, "unique_host_tokens": used}})
        if request.url.path == "/context/profiling/prepare":
            assert json.loads(request.content) == {"mode": "cold", "expected_restart_id": "before"}
            attempts += 1
            if error or attempts <= busy:
                return httpx.Response(409, json={"error": {"code": error or "ENGINE_BUSY"}})
            restart = "after"
            return httpx.Response(200, json={"mode": "cold", "info": info(),
                                            "pool": {"unique_gpu_tokens": 0, "unique_host_tokens": 0}})
        pytest.fail(f"unexpected endpoint: {request.url.path}")

    return SGLangEngineClient("http://dedicated.test", fingerprint, transport=httpx.MockTransport(handler)), calls


def test_cold_job_rejects_service_without_profiling_before_reset():
    async def run():
        engine, calls = client(enabled=False)
        try:
            with pytest.raises(ValueError, match="CONTEXTKV_ENABLE_PROFILING"):
                await prepare_cold(engine)
            assert calls == ["/context/engine/info"]
        finally:
            await engine.aclose()
    asyncio.run(run())


def test_cold_job_records_actual_empty_pool_and_retries_only_busy():
    async def run():
        engine, calls = client(busy=1)
        try:
            report = await prepare_cold(engine)
            assert report["status"] == "ready" and report["cold_verified"]
            assert report["prepare_attempts"] == 2
            assert report["before"]["pool"]["unique_gpu_tokens"] == 128
            assert report["after"]["pool"]["unique_gpu_tokens"] == 0
            assert report["after"]["engine_restart_id"] == "after"
            assert report["answer_generation"] is False
            assert "/generate" not in calls
        finally:
            await engine.aclose()
    asyncio.run(run())


def test_cold_job_does_not_treat_reset_ack_as_empty_pool():
    async def run():
        engine, _ = client(retained=64)
        try:
            report = await prepare_cold(engine)
            assert report["status"] == "cold_not_verified" and not report["cold_verified"]
            assert report["after"]["pool"]["unique_gpu_tokens"] == 64
        finally:
            await engine.aclose()
    asyncio.run(run())


def test_cold_job_does_not_retry_other_engine_errors():
    async def run():
        engine, calls = client(error="FINGERPRINT_MISMATCH")
        try:
            with pytest.raises(EngineError, match="FINGERPRINT_MISMATCH"):
                await prepare_cold(engine)
            assert calls.count("/context/profiling/prepare") == 1
        finally:
            await engine.aclose()
    asyncio.run(run())
