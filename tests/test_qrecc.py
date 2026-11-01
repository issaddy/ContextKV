import pytest

from bench.qrecc import evaluate, prepare, term_scores


def test_qrecc_uses_raw_history_and_keeps_rewrite_labels_offline():
    rows = []
    conversation = 0
    for source in ("nq", "quac", "trec"):
        for length in (2, 3, 4, 5, 6):
            for _ in range(5):
                conversation += 1
                for turn in range(1, length + 1):
                    rows.append({"Conversation_source": source, "Conversation_no": conversation,
                        "Turn_no": turn, "Question": f"raw-question-{conversation}-{turn}",
                        "Answer": f"previous-answer-{conversation}-{turn}",
                        "Rewrite": f"private-rewrite-{conversation}-{turn}"})
    cases, references, manifest = prepare(rows, seed=27)
    assert len(cases) == len(references) == 36
    assert manifest["dev_cases"] == 12 and manifest["test_cases"] == 24
    assert len({case["conversation_id"] for case in cases}) == 36
    for case in cases:
        assert len(case["history"]) == case["turn"] - 1
        assert all("raw-question" in item["question"] and "private-rewrite" not in item["question"]
                   for item in case["history"])
        assert "private-rewrite" not in str(case)
    assert all("private-rewrite" in row["rewrite"] for row in references)
    assert prepare(rows, seed=27) == (cases, references, manifest)
    excluded = {cases[0]["conversation_id"]}
    other, _, metadata = prepare(rows, seed=27, excluded_conversations=excluded)
    assert not excluded & {item["conversation_id"] for item in other}
    assert metadata["excluded_conversations"] == 1


def test_qrecc_scoring_counts_missing_predictions_and_matches_labels():
    cases = [{"case_id": "a", "conversation_id": "s1", "split": "dev"},
             {"case_id": "b", "conversation_id": "s2", "split": "test"}]
    refs = [{"case_id": "a", "rewrite": "The blue car"}, {"case_id": "b", "rewrite": "red car"}]
    assert term_scores("blue car", refs[0]["rewrite"])["f1"] == 1
    result = evaluate(cases, refs, [{"case_id": "a", "method": "incremental", "query": "blue car",
                                    "status": "resolved"}])
    assert result["methods"]["incremental"]["dev"]["f1"] == 1
    assert result["methods"]["incremental"]["test"]["statuses"] == {"missing": 1}
    assert result["methods"]["incremental"]["test"]["f1"] == 0
    with pytest.raises(ValueError, match="references"):
        evaluate(cases, refs[:1], [])
    with pytest.raises(ValueError, match="one unique case"):
        evaluate([cases[0], {**cases[1], "conversation_id": "s1"}], refs, [])
