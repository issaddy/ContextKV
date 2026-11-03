import json

import httpx
import pytest

from bench.study_scoring import score_dataset
from controller.templates import ByteTemplate


def prepare(tmp_path, actual_history, author_history, *, protocol="P3"):
    generation = tmp_path/"generation"
    generation.mkdir()
    source = tmp_path/"scores.jsonl"
    source.write_text(json.dumps({"task_id": "task", "metrics": {"RB_llm": .7}})+"\n")
    (generation/"audit.jsonl").write_text(json.dumps({"task_id": "task", "query": "What about that?",
        "actual_history": actual_history})+"\n")
    (generation/"references.jsonl").write_text(json.dumps({"task_id": "task", "input": [*author_history,
        {"speaker": "user", "text": "What about that?"}]})+"\n")
    rows = [{"task_id": "task", "metrics": {}, "status": "completed", "protocol": protocol,
             "cost": {"complete_ms": 10}}]
    spec = {"primary_metric": "RB_llm", "scoring": {"scores_path": str(source),
        "identity": {"protocol": "fixture"}, "history_audit": True,
        "judge": {"provider": "fixture", "model": "fixture", "revision": "v1", "base_url": "http://fixture/v1"}}}
    return generation, rows, spec


@pytest.mark.asyncio
async def test_p1_reference_history_gets_protocol_origin_without_model_call(tmp_path):
    history = [{"speaker": "user", "text": "Windows"}, {"speaker": "agent", "text": "Windows steps"}]
    generation, rows, spec = prepare(tmp_path, history, history, protocol="P1")
    result = await score_dataset({}, spec, rows, generation)
    assert result[0]["script_history_mismatch"] is False
    assert result[0]["history_audit"]["origin"] == "author_reference_history_protocol"
    manifest = json.loads((generation/"scoring/manifest.json").read_text())
    assert manifest["history_audit"]["calls"] == 0


@pytest.mark.asyncio
async def test_real_history_audit_flag_keeps_original_benchmark_grade(tmp_path, monkeypatch):
    actual = [{"speaker": "user", "text": "Windows"}, {"speaker": "assistant", "text": "Linux steps"}]
    author = [{"speaker": "user", "text": "Windows"}, {"speaker": "agent", "text": "Windows steps"}]
    generation, rows, spec = prepare(tmp_path, actual, author)
    monkeypatch.setattr("controller.templates.HuggingFaceTemplate", lambda *args, **kwargs: ByteTemplate())
    original = httpx.AsyncClient
    def respond(request):
        return httpx.Response(200, json={"text": json.dumps({"script_history_mismatch": True,
            "reason": "Actual answers changed the platform."}), "meta_info": {"completion_tokens": 11}})
    monkeypatch.setattr("bench.study_scoring.httpx.AsyncClient",
                        lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs))
    result = await score_dataset({}, spec, rows, generation)
    assert result[0]["script_history_mismatch"] is True
    assert result[0]["metrics"]["RB_llm"] == .7
    assert result[0]["cost"] == {"complete_ms": 10}
    manifest = json.loads((generation/"scoring/manifest.json").read_text())
    assert manifest["history_audit"]["judge_output_tokens"] == 11
    assert manifest["history_audit"]["benchmark_denominator"] == "unchanged"
