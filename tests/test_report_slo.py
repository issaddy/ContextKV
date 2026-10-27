"""Report explicit SLO units and apply one protocol to both comparison arms."""
import json
from copy import deepcopy
from dataclasses import replace

import pytest

from bench.metrics import TurnTiming, summarize_timings
from bench.report import main, summarize_run, decision_summary, markdown_report


def payload():
    return {"engine_kind": "mock", "report": {"started_at": 0, "arrival_window_end": 10,
        "completed_at": 10, "time_origin": "pipeline_arrival", "scheduled_sessions": 1,
        "arrived_sessions": 1, "unfinished_turns": 0, "timed_out_sessions": [],
        "session_makespans": [["s", 3]], "turns": [{"task_id": "s:1", "conversation_id": "s",
            "status": "success", "assistant_text": "fixture", "t_arrive": 0, "t_done": 3,
            "result": {"output_ids": [1, 2, 3], "truncated": False, "timing": {"t_first": 1, "t_last_token": 1.1,
                "t_response_done": 2, "t_commit": 3}}}]}}


def test_buffered_default_needs_declared_deadline_and_keeps_engine_ttft_diagnostic():
    result = summarize_run(payload())
    assert result["timing"]["slo"] == {}
    assert result["timing"]["slo_status"] == "not_configured"
    assert result["timing"]["latency_slo_goodput_requests_per_second"] is None
    assert result["timing"]["first_token_role"] == "engine_client_diagnostic"
    assert summarize_run(payload(), complete_slo=3)["timing"]["latency_slo_goodput_requests_per_second"] == .1
    # Engine first-token is fast, but the committed response misses this deadline.
    assert summarize_run(payload(), complete_slo=2)["timing"]["latency_slo_goodput_requests_per_second"] == 0
    streaming = summarize_run(payload(), response_mode="streaming")
    assert streaming["timing"]["slo"] == {"ttft_seconds": 2.0, "tpot_seconds": .1}
    assert streaming["timing"]["latency_slo_goodput_requests_per_second"] == .1
    strict = summarize_run(payload(), response_mode="streaming", ttft_slo=.5, tpot_slo=.02)
    assert strict["timing"]["slo"] == {"ttft_seconds": .5, "tpot_seconds": .02}
    assert strict["timing"]["latency_slo_goodput_requests_per_second"] == 0
    assert strict["timing"]["turn_seconds"] == result["timing"]["turn_seconds"]


def test_decision_summary_separates_cost_selection_fallback_and_native_results():
    cases = [
        ("REBUILD", "only_feasible_candidate", None),
        ("APPEND", "uncalibrated_cost_fallback", {"calibrated": False}),
        ("REBUILD", "cost_saving", {"calibrated": True}),
        ("APPEND", "hysteresis", {"calibrated": False, "selection_eligible": True}),
        ("REBUILD", "native_kv_tie_break", {"calibrated": False, "selection_eligible": True}),
        ("REBUILD", "context_size_tie_break", {"calibrated": False, "selection_eligible": True}),
        ("APPEND", "expected_cost", {"calibrated": False, "selection_eligible": True}),
        ("REBUILD", "always_rebuild", None),
    ]
    turns = [{"status": "success", "result": {"action": action, "reason": reason,
                                               "prediction": estimate}}
             for action, reason, estimate in cases]
    turns += [{"status": "failed", "result": None},
              {"status": "success", "result": {"output_ids": [1]}}]
    result = decision_summary(turns)
    assert result["observed_turns"] == 8
    assert result["actions"] == {"REBUILD": 5, "APPEND": 3}
    assert result["cost_based_selection_turns"] == 5
    assert result["fallback_turns"] == 1
    assert result["selected_prediction_turns"] == 6
    assert result["selected_prediction_calibrated_turns"] == 1
    assert result["selected_prediction_eligible_turns"] == 5
    assert summarize_run(payload())["decisions"]["observed_turns"] == 0


@pytest.mark.parametrize("field", ["actual_prefill_tokens", "actual_device_hit_tokens", "actual_host_loaded_tokens"])
def test_invalid_cache_counter_is_diagnostic_without_changing_success_or_raw_result(field):
    value = payload()
    result = value["report"]["turns"][0]["result"]
    result.update(input_tokens=10, receipt={"actual_prefill_tokens": 20,
        "actual_device_hit_tokens": 0, "actual_host_loaded_tokens": 0})
    result["receipt"][field] = -1
    original = deepcopy(value)
    summary = summarize_run(value)
    assert value == original
    assert summary["timing"]["statuses"] == {"success": 1}
    assert summary["timing"]["turn_seconds"]["mean"] == 3
    assert summary["receipt_work"]["invalid_counters"] == [{"task_id": "s:1", "values": {field: -1}}]
    if field == "actual_prefill_tokens":
        assert summary["actual_prefill_tokens"] == {"observed_turns": 0, "total": None}
        assert summary["receipt_work"]["prefill_to_prompt_ratio"]["count"] == 0
    else:
        assert summary["actual_prefill_tokens"]["total"] == 20
        assert summary["receipt_work"]["prefill_to_prompt_ratio"]["mean"] == 2
        assert summary["receipt_work"]["cold_prefill_fraction"] is None


@pytest.mark.parametrize("cli,expected", [([], {}),
    (["--complete-slo-seconds", "3"], {"complete_seconds": 3}),
    (["--response-mode", "streaming"], {"ttft_seconds": 2.0, "tpot_seconds": .1}),
    (["--response-mode", "streaming", "--ttft-slo-seconds", ".5", "--tpot-slo-seconds", ".02"], {"ttft_seconds": .5, "tpot_seconds": .02})])
def test_cli_persists_thresholds_and_applies_them_to_reference(tmp_path, monkeypatch, cli, expected):
    import bench.report as report
    input_path, reference_path = tmp_path/"input.json", tmp_path/"reference.json"
    input_path.write_text(json.dumps(payload()))
    reference_path.write_text(json.dumps(payload()))
    captured = []
    original = report.summarize_run

    def track(value, **kwargs):
        result = original(value, **kwargs)
        captured.append(result["timing"]["slo"])
        return result

    monkeypatch.setattr(report, "summarize_run", track)
    output = tmp_path/"summary.json"
    main([str(input_path), "--reference-run", str(reference_path), "--output", str(output), *cli])
    assert captured == [expected, expected]
    assert json.loads(output.read_text())["timing"]["slo"] == expected


@pytest.mark.parametrize("invalid", [-1, float("nan"), float("inf")])
def test_invalid_slo_is_rejected_once_at_summary_boundary(invalid):
    turn = TurnTiming("s", "1", "success", 1, 0, 1, 1, 2, 2)
    for ttft, tpot in ((invalid, .1), (2, invalid)):
        with pytest.raises(ValueError, match="SLO"):
            summarize_timings([turn], window_start=0, window_end=3, ttft_slo=ttft, tpot_slo=tpot)
    with pytest.raises(ValueError, match="SLO"):
        summarize_timings([turn], window_start=0, window_end=3, ttft_slo=2, tpot_slo=.1, complete_slo=invalid)


@pytest.mark.parametrize("mode,met", [("buffered", 2), ("streaming", 1)])
def test_goodput_excludes_truncated_unknown_empty_failed_and_out_of_window(mode, met):
    good = TurnTiming("s", "1", "success", 3, 0, .1, .2, .3, .4, truncated=False)
    turns = [good, replace(good, task_id="truncated", truncated=True),
        replace(good, task_id="unknown", truncated=None),
        replace(good, task_id="empty", output_tokens=0),
        replace(good, task_id="failed", status="failed"),
        replace(good, task_id="late_commit", t_commit=5),
        replace(good, task_id="no_stream", t_first=None, t_last_token=None)]
    result = summarize_timings(turns, window_start=0, window_end=5, ttft_slo=1, tpot_slo=.1,
                               complete_slo=1, response_mode=mode)
    assert result["completed_in_window"] == 5
    assert result["slo_met_in_window"] == met
    assert result["latency_slo_goodput_requests_per_second"] == met/5
    assert result["truncated_in_window"] == result["unknown_truncation_in_window"] == 1
    assert not result["semantic_quality_included_in_goodput"]


@pytest.mark.parametrize("finish,truncated,goodput", [("length", 1, 0), ("stop", 0, .1), (None, 0, 0)])
def test_report_recovers_only_known_finish_metadata(finish, truncated, goodput):
    data = payload()
    result = data["report"]["turns"][0]["result"]
    result.pop("truncated")
    result["receipt"] = {"finish_reason": finish}
    summary = summarize_run(data, complete_slo=3)
    assert summary["truncated_answers"]["count"] == truncated
    assert summary["timing"]["latency_slo_goodput_requests_per_second"] == goodput
    assert summary["timing"]["unknown_truncation_in_window"] == int(finish is None)


def test_session_equal_latency_excludes_think_time_and_resource_units_are_explicit():
    data = payload()
    first = data["report"]["turns"][0]
    first["result"]["timing"].update(t_response_done=1, t_commit=1)
    for i in range(3):
        row = deepcopy(first)
        row.update(task_id=f"long:{i}", conversation_id="long")
        row["result"]["timing"].update(t_response_done=3, t_commit=3)
        data["report"]["turns"].append(row)
    data["report"]["session_makespans"] = [["s", 8], ["long", 100]]
    samples = [{"timestamp": 100, "gpu_slots": 8, "host_slots": 12},
               {"timestamp": 104, "gpu_slots": 8, "host_slots": 12}]
    summary = summarize_run(data, occupancy=samples)
    assert summary["timing"]["turn_seconds"]["mean"] == 2.5
    assert summary["session_latency"]["equal_session_mean_turn_seconds"] == 2
    assert summary["session_latency"]["cumulative_system_wait_seconds"]["mean"] == 5
    assert summary["resource_efficiency"]["gpu_slot_seconds_per_nontruncated_turn"] == 8
    assert summary["resource_efficiency"]["host_slot_seconds_per_nontruncated_turn"] == 12
    assert summarize_run(data)["resource_efficiency"]["gpu_slot_seconds_per_nontruncated_turn"] is None


def test_report_primary_quality_keeps_reference_eligibility_separate_from_semantics():
    data = payload()
    turn = data["report"]["turns"][0]
    turn["assistant_text"] = "Answer [E_bbb, E_dead]."
    turn["result"].update(visible_evidence_manifest=[{"reference_id": "E_aaa"}, {"reference_id": "E_bbb"}],
                          citation_scope=["E_aaa"])
    summary = summarize_run(data)
    refs = summary["citation_validity"]
    assert refs["unknown_reference_task_ids"] == refs["out_of_scope_task_ids"] == ["s:1"]
    assert summary["quality"]["status"] == "not_evaluated"
    assert summary["evaluation_scope"] == "single_run_observations"
    assert "performance_claims_validated" not in summary
    assert not summary["quality"]["quality_noninferiority_established"]
    text = markdown_report(summary)
    assert "完整响应 P95" in text and "会话等权" in text and "未测量/未配置" in text
    assert "引用可见性违规：1/1" in text and "本轮范围违规：1/1" in text
    assert "Controller decision reasons" not in text and "GPU peak unique slots" not in text
