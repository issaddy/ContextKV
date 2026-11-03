from copy import deepcopy
import json

import pytest

from bench.official import condition_metrics, load_conditioning_rules
from bench.reassess_saved import history_pairs, reassess
from bench.study_metrics import declared_queue_summary


def saved(task, *, method="R0", status="completed", score=.8, history=()):
    return {"task_id": task, "method": method, "dataset": "qa", "split": "dev",
        "conversation_id": "conversation", "question": "Which version applies?",
        "actual_history": list(history), "answer": "Version 2" if status == "completed" else "",
        "status": status, "metrics": {"RB_llm": score} if score is not None else {},
        "cost": {"total_input_tokens": 100, "total_output_tokens": 10,
                 "complete_ms": 20, "gpu_gib_seconds": 2, "host_gib_seconds": 1}}


def test_declared_queue_keeps_failed_absent_tasks_and_all_paid_costs():
    rows = [saved("answer"), saved("failure", status="failed", score=None)]
    result = declared_queue_summary(rows, {"primary_metric": "RB_llm"},
                                    ["answer", "failure", "not_started"])
    assert result["queue_primary_score"] == pytest.approx(.8/3)
    assert result["scored_subset_primary"] == .8
    assert result["completion_rate"] == pytest.approx(1/3)
    assert result["missing_execution_task_ids"] == ["not_started"]
    assert result["cost"]["online_tokens"] == 220
    assert result["paid_cost"]["total_input_tokens"]["failed_request_total"] == 100


def test_unscored_completed_answer_retains_unknown_quality_and_cost_coverage():
    rows = [saved("a"), saved("b", score=None)]
    del rows[1]["cost"]["total_input_tokens"]
    result = declared_queue_summary(rows, {"primary_metric": "RB_llm"}, ["a", "b"])
    assert result["queue_primary_score"] is None
    assert result["queue_score_bounds"] == pytest.approx([.4, .9])
    assert result["paid_cost"]["total_input_tokens"]["observed_total"] == 100
    assert result["paid_cost"]["total_input_tokens"]["unmeasured_task_ids"] == ["b"]
    assert result["cost"]["online_tokens"] is None


def test_harmonic_queue_uses_conditioned_component_means_and_declared_denominator():
    rows = [saved("a"), saved("b")]
    rows[0]["metrics"] = {"RL_F_idk": [1], "RB_llm_idk": [.2], "RB_agg_idk": [.2]}
    rows[1]["metrics"] = {"RL_F_idk": [.2], "RB_llm_idk": [1], "RB_agg_idk": [1]}
    result = declared_queue_summary(rows, {"primary_metric": "mtrageval_harmonic"}, ["a", "b", "c"])
    assert result["scored_subset_primary"] == pytest.approx(.6)
    assert result["queue_primary_score"] == pytest.approx(.4)


def test_author_functions_load_without_importing_model_runtime(tmp_path):
    scripts = tmp_path/"scripts/evaluation"
    scripts.mkdir(parents=True)
    (scripts/"judge_wrapper.py").write_text('''
import nonexistent_model_runtime
raise RuntimeError("model runtime must not run")
def _get_answerability(row):
    return row["answerability"]
def get_idk_score(row, metric):
    return 0 if _get_answerability(row) == "UNANSWERABLE" else row["metrics"][metric][0]
def get_idk_underspec_score(row, metric):
    return 1 if _get_answerability(row) == "UNDERSPECIFIED" else get_idk_score(row, metric)
''')
    rules = load_conditioning_rules(tmp_path)
    rows = [{"answerability": "UNANSWERABLE", "metrics": {"RB_agg": [.9]}}]
    condition_metrics(rules, rows, ("RB_agg",))
    assert rows[0]["metrics"] == {"RB_agg": [.9], "RB_agg_idk": [0]}


def test_history_attribution_compares_actual_text_and_preserves_own_p3():
    original = [{"turn_id": 1, "user_text": "Previous question", "assistant_text": "Version 1"}]
    changed = [{**original[0], "assistant_text": "Version 2"}]
    rows = [saved("a", history=original), saved("a", method="R1", history=changed)]
    pair, = history_pairs(rows, "R0", ["R0", "R1"])
    assert pair["same_question"]
    assert not pair["same_actual_history"]
    assert not pair["fixed_input_pair"]
    assert rows[1]["actual_history"] == changed


def test_offline_reference_retains_answers_costs_and_missing_queue_items(tmp_path):
    spec = {"datasets": {"qa": {"primary_metric": "RB_llm", "weight": 1}},
        "cohorts": {"dev": {"qa": ["a", "b", "c"]}}}
    rows = [saved("a"), saved("b", status="failed", score=None)]
    original = deepcopy(rows)
    report = reassess(spec, rows, tmp_path/"new-reference", split="dev",
                      baseline="R0", history_protocol="own_p3")
    assert rows == original
    assert report["model_calls"] == 0
    assert report["methods"]["R0"]["queue_quality"] == pytest.approx(.8/3)
    saved_rows = [json.loads(line) for line in
        (tmp_path/"new-reference/results.scored.jsonl").read_text().splitlines()]
    assert saved_rows == original
