from hashlib import sha256

from adapters.query_selector import CompletionResult
from bench.loong_score import RUBRIC, evaluation_prompt, parse_score, score_answer, summarize


def reference():
    return {"task_id": "case", "source_task": {"type": "legal", "level": 4, "set": 1,
            "question": "Match cases.", "instruction": "Use every fact.",
            "prompt_template": "{docs}\n{instruction}\n{question}"},
            "reference_value": {"Case 1": ["Result B"]}}


def test_pinned_rubric_and_original_reference_objects():
    assert sha256(RUBRIC.encode()).hexdigest() == "9d67b3da27c5eb80bb94160f2958c744d305cfe5c83e111a30c0ef828087b5bf"
    prompt = evaluation_prompt(reference(), "Saved answer unchanged")
    assert "{docs}" not in prompt and "Use every fact.\nMatch cases." in prompt
    assert "{'Case 1': ['Result B']}" in prompt and "Saved answer unchanged" in prompt


def test_invalid_scores_stay_missing_and_zero_is_valid():
    assert parse_score("Rating: [[0]]") == 0
    assert parse_score("Rating: [87.5]") == 87.5
    assert all(parse_score(t) is None for t in ("No grade", "[[101]]", "[[80]] then [[90]]"))
    result = summarize([{"task_id": "a", "score": 0}, {"task_id": "b", "score": None}])
    assert result["observed_mean_0_100"] == 0 and result["full_cohort_mean_bounds"] == [0, 50]
    assert result["missing_task_ids"] == ["b"]


def test_truncated_judge_does_not_count_an_early_rating_and_no_answer_is_regenerated():
    class Client:
        calls = 0
        def complete(self, messages):
            self.calls += 1
            assert messages == [{"role": "user", "content": evaluation_prompt(reference(), "Answer")}]
            return CompletionResult("completed", "Rating: [[100]]\nunfinished", "length")
    client = Client()
    assert score_answer(reference(), "Answer", client)["score"] is None
    assert score_answer(reference(), None, client)["status"] == "missing_prediction"
    assert client.calls == 1
