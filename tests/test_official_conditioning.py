from types import SimpleNamespace

from bench.official import condition_metrics


def test_all_three_generation_components_use_upstream_answerability_conditioning():
    calls = []

    def condition(row, metric):
        calls.append((row["task_id"], metric))
        return 0

    api = SimpleNamespace(get_idk_score=condition)
    records = [{"task_id": "unanswerable", "Answerability": ["UNANSWERABLE"],
                "metrics": {"RL_F": [.8], "RB_llm": [.7], "RB_agg": [.6], "idk_eval": [0]}}]
    assert condition_metrics(api, records, ("RL_F", "RB_llm", "RB_agg")) == "_idk"
    assert calls == [("unanswerable", name) for name in ("RL_F", "RB_llm", "RB_agg")]
    assert records[0]["metrics"]["RB_agg"] == [.6]
    assert records[0]["metrics"]["RB_agg_idk"] == [0]


def test_underspecified_cohort_conditions_algorithmic_and_judge_components_together():
    api = SimpleNamespace(get_idk_underspec_score=lambda row, name: 1)
    records = [{"task_id": "underspecified", "answerability": ["UNDERSPECIFIED"],
                "metrics": {"RL_F": [.8], "RB_llm": [.7], "RB_agg": [.6]}}]
    assert condition_metrics(api, records, ("RL_F", "RB_llm", "RB_agg")) == "_idk_underspecified"
    assert all(records[0]["metrics"][name+"_idk_underspecified"] == [1]
               for name in ("RL_F", "RB_llm", "RB_agg"))
