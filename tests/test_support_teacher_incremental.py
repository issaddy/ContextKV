import json
from threading import Lock

import httpx
import pytest

from controller.templates import ByteTemplate
from training.support_data import annotate_support_parallel


def candidate(index):
    return {"example_id": str(index), "task_id": str(index), "need_id": "need",
        "query": "question", "units": [{"text": "original", "source_spans": [{"span_id": str(index)}]}],
        "dependencies": [], "corpus_has_answer": None, "oracle_scope_complete": False}


def fixture_client(monkeypatch, *, fail_once=None):
    real_client = httpx.Client
    lock = Lock()
    calls = []

    def respond(request):
        schema = json.loads(json.loads(request.content)["sampling_params"]["json_schema"])
        span_id = schema["properties"]["span_ids"]["items"]["enum"][0]
        with lock:
            attempt = calls.count(span_id)
            calls.append(span_id)
        labels = {"applicability": 1, "direct_support": .2, "joint_support": .2,
            "bridge_value": .8, "quality": .3, "disposition": "partial",
            "answer_disposition": "partial", "span_ids": [span_id]}
        raw = "{" if span_id == fail_once and attempt == 0 else json.dumps(labels)
        return httpx.Response(200, json={"text": raw, "meta_info": {"completion_tokens": 7}})

    monkeypatch.setattr("training.support_data.httpx.Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs))
    return calls


def test_incremental_teacher_reuses_completed_ids_and_keeps_cumulative_calls(tmp_path, monkeypatch):
    import training.support_data
    monkeypatch.setattr(training.support_data, "httpx", httpx, raising=False)
    calls = fixture_client(monkeypatch)
    source = tmp_path/"candidates.jsonl"
    source.write_text("".join(json.dumps(candidate(index))+"\n" for index in range(5)))
    output = tmp_path/"labels.jsonl"
    first = annotate_support_parallel(source, output, template=ByteTemplate(), base_urls=["http://fixture"], workers=2, max_examples=3)
    second = annotate_support_parallel(source, output, template=ByteTemplate(), base_urls=["http://fixture"], workers=2, max_examples=5)
    third = annotate_support_parallel(source, output, template=ByteTemplate(), base_urls=["http://fixture"], workers=2, max_examples=5)
    assert first["calls"] == 3
    assert second["calls"] == 5 and second["new_calls"] == 2
    assert third["new_calls"] == 0
    assert len(calls) == 5
    assert third["output_tokens"] == 35
    assert third["cumulative_wall_elapsed_ms"] >= second["cumulative_wall_elapsed_ms"]
    assert len(output.read_text().splitlines()) == 5


def test_failed_parse_keeps_paid_call_and_completed_labels_for_resume(tmp_path, monkeypatch):
    import training.support_data
    monkeypatch.setattr(training.support_data, "httpx", httpx, raising=False)
    calls = fixture_client(monkeypatch, fail_once="1")
    source = tmp_path/"candidates.jsonl"
    source.write_text("".join(json.dumps(candidate(index))+"\n" for index in range(3)))
    output = tmp_path/"labels.jsonl"
    with pytest.raises(json.JSONDecodeError):
        annotate_support_parallel(source, output, template=ByteTemplate(), base_urls=["http://fixture"], workers=3)
    failed = json.loads(output.with_suffix(".manifest.json").read_text())
    assert failed["status"] == "incomplete"
    assert failed["calls"] == 3
    assert failed["output_tokens"] == 21
    completed = annotate_support_parallel(source, output, template=ByteTemplate(), base_urls=["http://fixture"], workers=3)
    assert completed["examples"] == 3
    assert completed["calls"] == 4
    assert completed["new_calls"] == 1
    assert calls.count("0") == 1 and calls.count("2") == 1 and calls.count("1") == 2
