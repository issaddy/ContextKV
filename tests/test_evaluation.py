from __future__ import annotations

import asyncio
import csv
from dataclasses import asdict
import json
from pathlib import Path

import httpx
import pytest

from adapters.mtrag import load_rows, online_turn
from bench.blind_review import export_blind, import_ratings
from bench.judge import (DIMENSIONS, JudgeConfig, JudgeResponseError, OfflineJudge,
                         parse_judge_response, rule_audit)
from bench.official import evaluator_command, merge_offline_labels
from bench.quality import export_evaluation, official_record
from bench.report import paired_quality, summarize_run, telemetry_series
from bench.run_sessions import run_sessions
from profiling.common import Measurement, summarize_measurements
from controller.templates import token_hash

FIXTURE = Path(__file__).parent / "fixtures/mtrag_small.jsonl"


def audit_record(n=12):
    def key(i):
        return {"document_id": str(i), "version": "v1"}
    return {"task_id": "s1::1", "conversation_id": "s1", "query": "What is the limit?",
            "answer": "30 seconds [E_abc]", "actual_history": [], "retrieved_evidence": [],
            "visible_evidence_manifest": [{"reference_id": "E_abc" if i == 0 else f"E_{i:x}", "key": key(i)} for i in range(n)],
            "audit_evidence": [{"key": key(i), "text": f"Full original evidence {i}"} for i in range(n)],
            "measurement_kind": "simulation", "truncated": False}


def test_offline_join_preserves_actual_history_and_does_not_mutate_online_record():
    source = load_rows(FIXTURE)[0]
    prediction = official_record(online_turn(source), "own answer")
    original = json.dumps(prediction)
    merged = merge_offline_labels([prediction], [source])[0]
    assert merged["predictions"] == [{"text": "own answer"}]
    assert merged["input"] == prediction["input"]
    assert json.dumps(prediction) == original
    assert "targets" not in prediction
    assert "targets" in merged


def test_actual_official_argument_names_and_no_secret_arguments():
    command = evaluator_command("/tmp/upstream", "generation", "/tmp/in.jsonl", "/tmp/out.jsonl",
                                provider="vllm", model="frozen-served-name", python="/env/python")
    assert command[-4:] == ["--provider", "vllm", "--judge_model", "frozen-served-name"]
    assert "-e" in command and "--input_file" not in command
    retrieval = evaluator_command("/tmp/upstream", "retrieval", "/tmp/in.jsonl", "/tmp/out.jsonl")
    assert "--input_file" in retrieval and "--output_file" in retrieval
    with pytest.raises(ValueError, match="hf/vllm"):
        evaluator_command("/tmp/upstream", "generation", "a", "b", provider="openai", model="name")


@pytest.mark.parametrize("thinking", [None,False])
@pytest.mark.parametrize("flat", [False,True])
def test_judge_http_retains_all_visible_bodies_and_freezes_config(thinking, flat):
    requests = []
    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        value = {"scores": {k: .8 for k in DIMENSIONS}, "script_history_mismatch": False,
                 "reason": "fake transport test only", "unsupported_claims": []}
        if flat:
            value.update(value.pop("scores"))
        return httpx.Response(200, json={"id": "fake", "model": "frozen", "choices": [{"finish_reason":"stop","message": {"content": json.dumps(value)}}]})
    async def check():
        judge = OfflineJudge(JudgeConfig("local", "frozen", "revision-1", "http://localhost/v1",enable_thinking=thinking),
                             transport=httpx.MockTransport(respond))
        try:
            record = audit_record() | {"input_ids": [11, 22, 33]}
            result = await judge.evaluate(record)
        finally:
            await judge.aclose()
        assert result["generation_measurement_kind"] == "simulation"
        assert result["judge"]["revision"] == "revision-1"
        assert result["source"] == "offline_judge"
        raw = json.loads(result["judge_response"]["choices"][0]["message"]["content"])
        assert ("scores" not in raw) == flat
        assert result["scores"] == {k:.8 for k in DIMENSIONS}
        assert result["query"] == record["query"] and result["input_hash"] == token_hash(record["input_ids"])
    asyncio.run(check())
    evidence = json.loads(requests[0]["messages"][1]["content"])["audit_evidence"]
    assert len(evidence) == 12 and evidence[-1]["text"] == "Full original evidence 11"
    assert requests[0]["temperature"] == 0
    if thinking is None:
        assert "chat_template_kwargs" not in requests[0]
    else:
        assert requests[0]["chat_template_kwargs"] == {"enable_thinking":False}


def test_truncated_judge_json_is_not_accepted_as_a_completed_grade():
    value = {"scores":{k:.8 for k in DIMENSIONS},"reason":"syntactically valid but budget-limited"}
    response = httpx.Response(200,json={"choices":[{"finish_reason":"length","message":{"content":json.dumps(value)}}]})
    async def check():
        judge = OfflineJudge(JudgeConfig("local","frozen","revision-1","http://localhost/v1"),
                             transport=httpx.MockTransport(lambda request:response))
        try:
            with pytest.raises(ValueError,match="did not finish normally"):
                await judge.evaluate(audit_record())
        finally:
            await judge.aclose()
    asyncio.run(check())


@pytest.mark.parametrize("values", [
    {k:.8 for k in DIMENSIONS if k != "correctness"},
    {k:True for k in DIMENSIONS}, {k:1.1 for k in DIMENSIONS},
])
def test_invalid_judge_values_retain_the_actual_response(values):
    raw = {"choices":[{"finish_reason":"stop", "message":{"content":json.dumps(values)}}]}
    with pytest.raises(JudgeResponseError) as caught:
        parse_judge_response(raw)
    assert caught.value.response is raw


def test_score_metadata_does_not_invalidate_named_measurements():
    scores = {k:None if k == "citations" else .75 for k in DIMENSIONS}
    raw = {"choices":[{"finish_reason":"stop", "message":{"content":json.dumps(
        {"scores":scores | {"note":"extra metadata"}, "reason":"task result"})}}]}
    result, parsed = parse_judge_response(raw)
    assert parsed == scores and result["reason"] == "task result"


def test_failed_file_assessment_keeps_raw_output_for_offline_repair(tmp_path, monkeypatch):
    from bench import judge as module
    raw = {"choices":[{"finish_reason":"stop", "message":{"content":'{"reason":"missing scores"}'}}]}
    monkeypatch.setattr(module, "OfflineJudge", lambda config: OfflineJudge(
        config, transport=httpx.MockTransport(lambda request:httpx.Response(200, json=raw))))
    source, target = tmp_path/"audit.jsonl", tmp_path/"scores.jsonl"
    source.write_text(json.dumps(audit_record())+"\n")
    asyncio.run(module.evaluate_file(source, JudgeConfig("local","frozen","revision-1","http://localhost/v1"), target))
    saved = json.loads(target.read_text())
    assert saved["status"] == "failed" and saved["scores"] is None
    assert saved["judge_response"] == raw


def test_task_protocol_sends_only_current_requirements_and_applicability(tmp_path, monkeypatch):
    from bench import judge as module
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        value = {"scores":{k:1 if k == "correctness" else None for k in DIMENSIONS}, "reason":"current task only"}
        return httpx.Response(200, json={"choices":[{"finish_reason":"stop", "message":{"content":json.dumps(value)}}]})
    monkeypatch.setattr(module, "OfflineJudge", lambda config: OfflineJudge(config, transport=httpx.MockTransport(respond)))
    source, protocol, target = tmp_path/"audit.jsonl", tmp_path/"protocol.json", tmp_path/"scores.jsonl"
    source.write_text(json.dumps(audit_record())+"\n")
    protocol.write_text(json.dumps({"dimensions":["correctness","conditions_versions"],
        "task_scope":{"conditions_versions":["future-task"]},
        "task_requirements":{"s1::1":"Answer only the current limit question.",
                             "future-task":"Private future product requirement."}}))
    asyncio.run(module.evaluate_file(source, JudgeConfig("local","frozen","revision-1","http://localhost/v1"),
                                     target, task_protocol_path=protocol))
    payload = json.loads(requests[0]["messages"][-1]["content"])
    expected = {"requirement":"Answer only the current limit question.", "dimensions":["correctness"]}
    assert payload["task_requirements"] == expected
    assert json.loads(target.read_text())["task_requirements"] == expected
    assert "future-task" not in json.dumps(requests) and "Private future" not in json.dumps(requests)


def test_rule_audit_flags_invalid_citation_without_inventing_quality():
    record = audit_record()
    record["answer"] = "Unsupported [E_deadbeef]"
    record["truncated"] = True
    result = rule_audit(record)
    assert result["unknown_citations"] == ["E_deadbeef"]
    assert result["semantic_quality_score"] is None
    assert set(result["flags"]) == {"unknown_citations", "truncated_answer"}


def test_export_has_current_question_and_more_than_ten_visible_bodies(tmp_path):
    record = audit_record()
    turn = online_turn(load_rows(FIXTURE)[0])
    export_evaluation(turn, "answer", official_path=tmp_path / "official.jsonl", audit_path=tmp_path / "audit.jsonl",
                      visible_evidence_manifest=record["visible_evidence_manifest"],
                      audit_evidence=record["audit_evidence"], input_ids=[1], output_ids=[2], measurement_kind="simulation",
                      citation_scope=("E_a",))
    audit = json.loads((tmp_path / "audit.jsonl").read_text())
    assert audit["query"] == turn.query and len(audit["audit_evidence"]) == 12
    assert audit["citation_scope"] == ["E_a"]
    assert len(json.loads((tmp_path / "official.jsonl").read_text())["contexts"]) == 5


def test_human_blinding_and_complete_ratings_roundtrip(tmp_path):
    export_blind({"method_A_secret": [audit_record()], "method_B_secret": [audit_record()]}, tmp_path)
    public = (tmp_path / "review.jsonl").read_text()
    assert "method_A_secret" not in public and "method_B_secret" not in public
    with (tmp_path / "ratings.csv").open() as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        row.update(reviewer_id="reviewer-1", script_history_mismatch="false", **{k: ".75" for k in DIMENSIONS})
    with (tmp_path / "ratings.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    result = import_ratings(tmp_path / "private-key.json", tmp_path / "ratings.csv")
    assert {r["method"] for r in result} == {"method_A_secret", "method_B_secret"}
    assert all(r["scores"]["correctness"] == .75 for r in result)


def test_report_missing_token_telemetry_remains_missing_and_failures_count():
    async def run():
        async def handle(turn):
            if turn.turn == 2:
                raise RuntimeError("injected")
            return {"assistant_text": "test", "output_ids": [2, 3]}
        return await run_sessions(load_rows(FIXTURE), handle, concurrency=2)
    report = asyncio.run(run())
    result = summarize_run({"engine_kind": "mock", "report": asdict(report)})
    assert result["timing"]["statuses"] == {"failed": 2, "success": 2}
    assert result["cohort"]["unfinished_turns"] == 2
    assert result["timing"]["ttft_seconds"]["count"] == 0
    assert result["occupancy"]["gpu_peak_slots"] is None
    assert result["transfers"] is None and result["quality"]["status"] == "not_evaluated"
    assert result["measurement_kind"] == "simulation"


def test_quality_pairing_refuses_partial_task_coverage_and_keeps_session_clusters():
    rows = [{"task_id": f"s{i}::{j}", "conversation_id": f"s{i}", "scores": {"correctness": .8}}
            for i in range(3) for j in range(2)]
    with pytest.raises(ValueError, match="identical task"):
        paired_quality(rows, rows[:-1])
    result = paired_quality(rows, rows, samples=30)
    assert result["clusters"] == 3 and result["difference"] == 0


def test_calibration_does_not_mix_host_sizes_or_first_token_clock_scopes():
    rows = [Measurement("first_token", 10, 5, 1, 1, 0, 1.0, "hardware", host_load_tokens=host, timing_scope=scope)
            for host, scope in [(0, "server_request"), (5, "server_request"), (0, "client_request")]]
    assert len(summarize_measurements(rows)) == 3


def test_engine_telemetry_uses_its_clock_filters_prior_events_and_marks_sampling():
    before = {"engine_restart_id": "a", "observed_time_ns": 10_000_000_000,
              "pool": {"unique_gpu_tokens": 2, "unique_host_tokens": 0},
              "events": [], "pool_sample_scope": "scheduler_boundaries"}
    after = {**before, "observed_time_ns": 12_000_000_000, "events": [
        {"event": "transfer", "observed_time_ns": 9_000_000_000, "transfer_id": "previous"},
        {"event": "pool_snapshot", "observed_time_ns": 11_000_000_000,
         "unique_gpu_tokens": 8, "unique_host_tokens": 3},
        {"event": "transfer", "observed_time_ns": 11_500_000_000, "transfer_id": "current",
         "direction": "H2D", "bytes": 1024, "status": "completed"}]}
    pools, transfers, info = telemetry_series({"before": before, "after": after})
    assert sorted(p["timestamp"] for p in pools) == [10, 11, 12]
    assert [t["transfer_id"] for t in transfers] == ["current"]
    assert info["peak_interpretation"] == "observed_samples_not_allocator_instantaneous_peak"


def test_report_uses_client_chunk_tpot_and_binds_quality_to_run_answers():
    turns = [{"conversation_id": "session", "task_id": f"task-{i}", "status": "success", "t_arrive": 1,
              "t_done": 9, "assistant_text": "answer", "result": {"output_ids": [1, 2, 3],
              "tpot_ms": 9000, "server_ttft_ms": 8000, "timing_source": "server",
              "timing": {"t_arrive": 2, "t_first": 3, "t_last_token": 5,
                         "t_response_done": 6, "t_commit": 7, "token_observation": "chunk"}}} for i in range(2)]
    payload = {"engine_kind": "sglang", "report": {"turns": turns, "started_at": 0, "arrival_window_end": 10,
               "completed_at": 10, "time_origin": "pipeline_arrival", "scheduled_sessions": 1, "arrived_sessions": 1,
               "unfinished_turns": 0, "timed_out_sessions": [], "session_makespans": [["session", 9]]}}
    for row, actual, gpu, branch in zip(turns, (12, 2), (0, 8), ("initial", "append")):
        row["result"].update(input_tokens=10, actual_prefill_tokens=actual, branch_reason=branch,
                              receipt={"actual_prefill_tokens": actual, "actual_device_hit_tokens": gpu,
                                       "actual_host_loaded_tokens": 0})
    quality = [{"task_id": "task-0", "conversation_id": "session", "answer": "answer", "scores": {"correctness": .8}}]
    summary = summarize_run(payload, quality=quality)
    assert summary["timing"]["ttft_seconds"]["mean"] == 2  # includes retrieval arrival=1, not controller arrival=2
    assert summary["timing"]["tpot_seconds"]["mean"] == 1
    assert summary["timing"]["turn_seconds"]["mean"] == 6  # max(response,commit)-arrival
    assert summary["server_metrics"]["tpot_ms"]["mean"] == 9000
    assert summary["receipt_work"]["prefill_to_prompt_ratio"]["mean"] == .7
    assert summary["receipt_work"]["prefill_to_prompt_ratio"]["p99"] > 1
    assert summary["receipt_work"]["prefill_to_prompt_ratio"]["count"] == 2
    assert summary["receipt_work"]["cold_prefill_fraction"] == .5
    assert summary["receipt_work"]["by_branch"]["initial"]["cold_prefill_fraction"] == 1
    assert summary["receipt_work"]["by_session"]["session"]["count"] == 2
    assert summary["quality"]["status"] == "partially_evaluated"
    assert summary["quality"]["missing_scored_task_ids"] == ["task-1"]
    with pytest.raises(ValueError, match="different answer"):
        summarize_run(payload, quality=[{**quality[0], "answer": "some other method's answer"}])
    with pytest.raises(ValueError, match="absent from"):
        summarize_run(payload, quality=[{**quality[0], "task_id": "another-run"}])


def test_official_un_labels_are_joined_only_after_generation():
    reference={'task_id':'un-1','input':[{'speaker':'user','text':'Current question',
        'metadata':{'author_type':'human'}}],'contexts':[], 'targets':[{'text':'Reference answer'}],
        'answerability':['UNDERSPECIFIED'],'Question Type':['Other'],'Multi-Turn':['Clarification']}
    prediction={'task_id':'un-1','input':[{'speaker':'user','text':'Current question'}],
                'contexts':[], 'predictions':[{'text':'Actual answer'}]}
    result,=merge_offline_labels([prediction],[reference])
    assert result['answerability']==['UNDERSPECIFIED']
    assert result['evaluation_history_matches_reference']
    assert result['predictions']==prediction['predictions'] and 'answerability' not in prediction


def test_official_retrieval_scores_ranked_hits_without_reading_expansion(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from adapters.mtrag import DATASET_COMMIT
    from bench import official
    root=tmp_path/'upstream'
    root.mkdir()
    (root/'contextkv-source.json').write_text(json.dumps({'commit':DATASET_COMMIT}))
    ranked=[{'document_id':'original','text':'Retrieved original','score':2}]
    rows=[{'task_id':'q','contexts':ranked+[{'document_id':'support','text':'Reading context','score':0}],
           'retrieval_contexts':ranked}]
    original=tmp_path/'answers.jsonl'
    original.write_text(json.dumps(rows[0])+'\n')
    captured=[]
    monkeypatch.setattr(official.subprocess,'check_output',lambda *a,**kw:'{"python":"fixture","packages":[]}')
    def run(command,**kwargs):
        path=Path(command[command.index('--input_file')+1])
        captured.append(json.loads(path.read_text()))
        return SimpleNamespace(returncode=0,stdout='',stderr='',check_returncode=lambda:None)
    monkeypatch.setattr(official.subprocess,'run',run)
    official.run_evaluator(root,'retrieval',original,tmp_path/'scores.jsonl')
    assert captured[0]['contexts']==ranked
    assert json.loads(original.read_text())['contexts']==rows[0]['contexts']
