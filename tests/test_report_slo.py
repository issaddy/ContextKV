"""Report explicit SLO units and apply one protocol to both comparison arms."""
import json

import pytest

from bench.metrics import TurnTiming, summarize_timings
from bench.report import main, summarize_run, decision_summary


def payload():
    return {"engine_kind": "mock", "report": {"started_at": 0, "arrival_window_end": 10,
        "completed_at": 10, "time_origin": "pipeline_arrival", "scheduled_sessions": 1,
        "arrived_sessions": 1, "unfinished_turns": 0, "timed_out_sessions": [],
        "session_makespans": [["s", 3]], "turns": [{"task_id": "s:1", "conversation_id": "s",
            "status": "success", "assistant_text": "fixture", "t_arrive": 0, "t_done": 3,
            "result": {"output_ids": [1, 2, 3], "timing": {"t_first": 1, "t_last_token": 1.1,
                "t_response_done": 2, "t_commit": 3}}}]}}


def test_default_thresholds_are_recorded_without_changing_goodput():
    result = summarize_run(payload())
    assert result["timing"]["slo"] == {"ttft_seconds": 2.0, "tpot_seconds": .1}
    assert result["timing"]["slo_goodput_requests_per_second"] == .1
    strict = summarize_run(payload(), ttft_slo=.5, tpot_slo=.02)
    assert strict["timing"]["slo"] == {"ttft_seconds": .5, "tpot_seconds": .02}
    assert strict["timing"]["slo_goodput_requests_per_second"] == 0
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


@pytest.mark.parametrize("cli,expected", [([], {"ttft_seconds": 2.0, "tpot_seconds": .1}),
    (["--ttft-slo-seconds", ".5", "--tpot-slo-seconds", ".02"], {"ttft_seconds": .5, "tpot_seconds": .02})])
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
