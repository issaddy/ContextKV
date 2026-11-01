"""Offline scoring must resume without repeating completed model work."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from adapters.mtrag import DATASET_COMMIT
from bench.official import load_jsonl, run_http_evaluator, score_http_stages


def upstream(calls, *, fail=None):
    def metric(name):
        def run(provider, model, inp, out):
            calls.append(name)
            assert provider == "vllm" and model == "fixed-judge"
            if name == fail:
                raise RuntimeError("interrupted transport")
            rows = load_jsonl(inp)
            for row in rows:
                row.setdefault("metrics", {})[name] = [0.5]
            with open(out, "w") as stream:
                stream.write("".join(json.dumps(row)+"\n" for row in rows))
        return run
    return SimpleNamespace(run_idk_judge=metric("idk_eval"),
        run_radbench_judge=metric("RB_llm"), run_ragas_judges_local=metric("RL_F"),
        extract_conversation=lambda messages: str(messages),
        split_conversation=lambda text: ("", text),
        underspecified_eval=lambda model, inp, out: metric("underspecified_eval")("vllm", model, inp, out),
        get_idk_score=lambda row, name: row["metrics"][name][0],
        get_idk_underspec_score=lambda row, name: row["metrics"][name][0])


def predictions(path, label="ANSWERABLE"):
    rows = [{"task_id": "conversation::1", "Answerability": [label],
             "input": [{"speaker": "user", "text": "Original question"}],
             "predictions": [{"text": "Original answer"}],
             "contexts": [{"text": "Actual original evidence"}]}]
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    return rows


def test_resume_reuses_completed_stage_and_preserves_actual_answers(tmp_path):
    source = tmp_path / "answers.jsonl"
    original = predictions(source)
    original_bytes = source.read_bytes()
    calls = []
    output = tmp_path / "scores"
    with pytest.raises(RuntimeError, match="interrupted"):
        score_http_stages(upstream(calls, fail="RB_llm"), "fixed-judge", source, output)
    assert (output / "idk.jsonl").exists()
    assert not (output / "rb_llm.jsonl").exists()
    calls.clear()
    result = score_http_stages(upstream(calls), "fixed-judge", source, output)
    assert calls == ["RB_llm", "RL_F"]
    assert result["reused_stages"] == ["idk"]
    scored = load_jsonl(result["output"])
    assert {k: v for k, v in scored[0].items() if k != "metrics"} == original[0]
    assert source.read_bytes() == original_bytes
    calls.clear()
    result = score_http_stages(upstream(calls), "fixed-judge", source, output)
    assert calls == []
    assert result["reused_stages"] == ["idk", "rb_llm", "rl_f"]


def test_clarification_is_only_scored_when_the_cohort_requires_it(tmp_path):
    source = tmp_path / "answers.jsonl"
    predictions(source, "UNDERSPECIFIED")
    calls = []
    result = score_http_stages(upstream(calls), "fixed-judge", source, tmp_path / "scores")
    assert calls == ["idk_eval", "RB_llm", "RL_F", "underspecified_eval"]
    assert "RL_F_idk_underspecified" in load_jsonl(result["output"])[0]["metrics"]


def test_other_public_tasks_use_raw_quality_without_invented_answerability_labels(tmp_path):
    source = tmp_path / "answers.jsonl"
    rows = predictions(source)
    del rows[0]["Answerability"]
    source.write_text(json.dumps(rows[0]) + "\n")
    calls = []
    api = upstream(calls)
    del api.run_idk_judge, api.get_idk_score, api.get_idk_underspec_score, api.underspecified_eval
    result = score_http_stages(api, "fixed-judge", source, tmp_path / "raw-scores",
                               answerability_adjustment=False)
    assert calls == ["RB_llm", "RL_F"] and result["conditioned_suffix"] is None
    assert load_jsonl(result["output"])[0]["metrics"] == {"RB_llm": [0.5], "RL_F": [0.5]}


@pytest.mark.parametrize("first,other,unused", [
    ("RB_llm", "RL_F", "run_ragas_judges_local"),
    ("RL_F", "RB_llm", "run_radbench_judge"),
])
def test_selected_metric_does_not_load_unrequested_judge_and_can_expand_without_repeating(tmp_path,first,other,unused):
    source = tmp_path / "answers.jsonl"
    predictions(source)
    calls = []
    api = upstream(calls)
    delattr(api,unused)
    output = tmp_path / "scores"
    result = score_http_stages(api, "fixed-judge", source, output,
                               answerability_adjustment=False, metrics=(first,))
    assert calls == [first] and result["requested_metrics"] == [first]
    assert load_jsonl(result["output"])[0]["metrics"] == {first: [0.5]}
    calls.clear()
    result = score_http_stages(upstream(calls), "fixed-judge", source, output, answerability_adjustment=False)
    assert calls == [other]
    assert load_jsonl(result["output"])[0]["metrics"] == {"RB_llm": [0.5], "RL_F": [0.5]}


def test_changed_input_cannot_reuse_another_scoring_cache(tmp_path):
    checkout = tmp_path / "upstream"
    checkout.mkdir()
    (checkout / "contextkv-source.json").write_text(json.dumps({"commit": DATASET_COMMIT}))
    source = tmp_path / "answers.jsonl"
    predictions(source)
    output = tmp_path / "scores"
    output.mkdir()
    (output / "identity.json").write_text(json.dumps({"input_sha256": "another-answer-set"}))
    with pytest.raises(ValueError, match="different answers or judge settings"):
        run_http_evaluator(checkout, source, output, model="fixed-judge", judge_revision="frozen",
                           endpoint="http://localhost:30002/v1")


def test_upstream_dropped_tasks_are_not_committed_as_a_completed_stage(tmp_path):
    source = tmp_path / "answers.jsonl"
    predictions(source)
    api = deepcopy(upstream([]))
    api.run_idk_judge = lambda provider, model, inp, out: open(out, "w").close()
    output = tmp_path / "scores"
    with pytest.raises(ValueError, match="changed the task cohort"):
        score_http_stages(api, "fixed-judge", source, output)
    assert not (output / "idk.jsonl").exists()


def test_radbench_receives_complete_multiline_questions_and_actual_message_history(tmp_path):
    source = tmp_path / "answers.jsonl"
    rows = [
        {"task_id": "initial", "input": [{"speaker": "user", "text": "Help with credit.\n\nWhere can I download the documents?"}]},
        {"task_id": "followup", "input": [
            {"speaker": "user", "text": "Explain the forms.\nInclude costs."},
            {"speaker": "agent", "text": "The form has two fields:\nuser: account owner\nNo fee is stated."},
            {"speaker": "user", "text": "What changes for a student?\nAnd for an employee?"},
        ]},
    ]
    source.write_text("".join(json.dumps(row)+"\n" for row in rows))
    api = upstream([])
    original = api.extract_conversation, api.split_conversation
    observed = []

    def score(provider, model, inp, out):
        values = load_jsonl(inp)
        for row in values:
            observed.append(api.split_conversation(api.extract_conversation(row["input"])))
            row["metrics"] = {"RB_llm": [0.9]}
        with open(out, "w") as stream:
            stream.write("".join(json.dumps(row)+"\n" for row in values))

    api.run_radbench_judge = score
    score_http_stages(api, "fixed-judge", source, tmp_path / "scores",
                      answerability_adjustment=False, metrics=("RB_llm",))
    assert observed == [
        ("", "Help with credit.\n\nWhere can I download the documents?"),
        ("user: Explain the forms.\nInclude costs.\nagent: The form has two fields:\nuser: account owner\nNo fee is stated.",
         "What changes for a student?\nAnd for an employee?"),
    ]
    assert (api.extract_conversation, api.split_conversation) == original
    assert [row["input"] for row in load_jsonl(source)] == [row["input"] for row in rows]
