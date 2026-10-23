"""Guarded cumulative SSE contracts, including transport and terminal loss."""
import asyncio
from dataclasses import asdict
import json

import httpx
import pytest

from controller.engine_client import SGLangEngineClient, SGLANG_COMMIT, UnknownOutcomeError, token_hash
from controller.schema import ConfigFingerprint
from engine_patches.contextkv_runtime import ContextKVStream


RECEIPT = dict(rid="a", attempt_id="a", engine_session_id="s", session_generation=1,
    engine_restart_id="r", input_hash=token_hash([1,2]), finish_reason="stop", generation_complete=True)
TIMING = dict(server_ttft_ms=20, server_elapsed_ms=60, decode_ms=40,
              timing_source="sglang_server_monotonic", tpot_ms=20)


def frame(ids, text="", *, terminal=False):
    return {"text":text, "output_ids":ids, "meta_info":{
        **({"contextkv_receipt":[RECEIPT], "contextkv_timing":[TIMING],
            "finish_reason":{"type":"stop"}} if terminal else {})}}


def info(fp):
    return dict(engine_restart_id="r", protocol_version="contextkv-v1", engine_kind="sglang",
        guarded_generate_supported=True, scheduler_owned=True, completion_status_supported=True,
        streaming_generation_supported=True, sglang_commit=SGLANG_COMMIT, fingerprint=asdict(fp))


class TimedSSE(httpx.AsyncByteStream):
    def __init__(self, events, clock):
        self.events, self.clock = events, clock
    async def __aiter__(self):
        for timestamp, value in self.events:
            self.clock[0] = timestamp
            if isinstance(value, Exception):
                raise value
            raw = ("data: "+(value if isinstance(value,str) else json.dumps(value))+"\n\n").encode()
            # Split inside field names and JSON; a network chunk is not an SSE event.
            for part in (raw[:3],raw[3:11],raw[11:]):
                yield part


async def collect(events, monkeypatch):
    import controller.engine_client as module
    clock = [10.0]
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    fp = ConfigFingerprint()
    def handler(request):
        if request.url.path == "/context/engine/info":
            return httpx.Response(200,json=info(fp))
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200,stream=TimedSSE(events,clock),
                              headers={"content-type":"text/event-stream"})
    engine = SGLangEngineClient("http://test",fp,transport=httpx.MockTransport(handler))
    try:
        return await engine.generate(input_ids=[1,2],session_id="s",expected_generation=1,
            expected_restart_id="r",attempt_id="a",fingerprint=fp)
    finally:
        await engine.aclose()


def test_cumulative_sse_tracks_only_new_tokens_excluding_receipt_latency(monkeypatch):
    result = asyncio.run(collect([
        (10.04,frame([])), (10.1,frame([1],"你")), (10.3,frame([1],"你")),
        (10.5,frame([1,2,3],"你好。")), (11,frame([1,2,3],"你好。",terminal=True)),
        (11.1,frame([1,2,3],"你好。",terminal=True)), (11.2,"[DONE]")],monkeypatch))
    assert result.output_ids == (1,2,3) and result.text == "你好。"
    assert result.ttft_ms == pytest.approx(100)
    assert result.last_token_ms == pytest.approx(500)
    assert result.elapsed_ms == pytest.approx(1200)
    assert result.decode_ms == 40 and result.tpot_ms == 20  # distinct native server metrics
    assert result.server_ttft_ms == 20 and result.generation_complete


@pytest.mark.parametrize("tail,code", [
    ([(10.5,"[DONE]")],"INCOMPLETE_STREAM"),
    ([(10.5,frame([1],"a",terminal=True))],"INCOMPLETE_STREAM"),
    ([(10.5,'{"text":')],"INVALID_STREAM"),
    ([(10.5,httpx.ReadError("stream disconnected"))],"TRANSPORT_ERROR"),
    ([(10.5,frame([]))],"INVALID_STREAM"),
    ([(10.5,{"error":{"code":"ABORTED"}})],"STREAM_ERROR"),
])
def test_partial_or_broken_sse_never_returns_a_committable_result(monkeypatch,tail,code):
    with pytest.raises(UnknownOutcomeError) as exc:
        asyncio.run(collect([(10.1,frame([1],"a")),*tail],monkeypatch))
    assert exc.value.code == code


def test_empty_and_receipt_only_frames_do_not_create_token_times(monkeypatch):
    result = asyncio.run(collect([(10.1,frame([])),
        (10.5,frame([],terminal=True)),(11,"[DONE]")],monkeypatch))
    assert result.ttft_ms is result.last_token_ms is None


def test_native_incremental_stream_is_normalized_before_result_publication():
    stream = ContextKVStream(incremental=True)
    assert stream.push(frame([1],"你"))["output_ids"] == [1]
    final = stream.push(frame([2,3],"好。",terminal=True))
    assert final["output_ids"] == [1,2,3] and final["text"] == "你好。"
    tokens = stream.tokens_before_receipt(final)
    assert tokens["text"] == final["text"] and tokens["output_ids"] == final["output_ids"]
    assert "contextkv_receipt" not in tokens["meta_info"]
    assert tokens["meta_info"]["finish_reason"] is None
    assert final["meta_info"]["contextkv_receipt"] == [RECEIPT]
    assert ContextKVStream(incremental=False).push(final) is final


def test_actual_native_stream_metadata_emits_terminal_receipt_after_prior_tokens():
    import textwrap
    from types import SimpleNamespace
    from test_engine_host_quote import native_source
    accumulator = SimpleNamespace(output_ids=[[7]], customized_info={},
        is_request_level_customized_info_key=lambda key: key.startswith("contextkv_"))
    req = SimpleNamespace(customized_info={"contextkv_receipt":[RECEIPT], "contextkv_timing":[TIMING]},
                          stream=True, finished=lambda: True)
    scope = dict(self=accumulator,req=req,send_token_offset=128,output_ids_=[7]*129)
    exec(textwrap.dedent(native_source("stream_meta_emit",patched=True)),scope)
    assert accumulator.customized_info["contextkv_receipt"] == [[RECEIPT]]
    assert accumulator.customized_info["contextkv_timing"] == [[TIMING]]


def test_actual_native_incremental_metadata_preserves_request_level_values():
    import textwrap
    from typing import Any, Dict, Iterable, Optional
    from test_engine_host_quote import native_source
    scope = dict(Any=Any,Dict=Dict,Iterable=Iterable,Optional=Optional,
                 _INCREMENTAL_STREAMING_META_INFO_KEYS=("output_token_logprobs",))
    exec(textwrap.dedent(native_source("stream_meta_slice",patched=True)),scope)
    metadata = {"contextkv_receipt":[RECEIPT],"contextkv_timing":[TIMING],
                "output_token_logprobs":[0,1,2]}
    scope["_slice_streaming_output_meta_info"](metadata,2,
        ["contextkv_receipt","contextkv_timing"])
    assert metadata["contextkv_receipt"] == [RECEIPT]
    assert metadata["contextkv_timing"] == [TIMING]
    assert metadata["output_token_logprobs"] == [2]
