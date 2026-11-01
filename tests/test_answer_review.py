import hashlib
import json

import pytest

from bench.answer_review import main, prepare_review, score_review


def examples():
    rows = [
        {"task_id": "a1", "conversation_id": "a", "query": "And the free plan?", "changed_selection": True,
         "modes": {"base": {"status": "completed", "answer": "Seven days."},
                   "new": {"status": "completed", "answer": "Seven days, no restore."}}},
        {"task_id": "a2", "conversation_id": "a", "query": "What about Canada?", "changed_selection": True,
         "modes": {"base": {"status": "completed", "answer": "Not stated."},
                   "new": {"status": "completed", "answer": "Thirty days."}}},
        {"task_id": "b1", "conversation_id": "b", "query": "And Berlin?", "changed_selection": False,
         "modes": {"base": {"status": "completed", "answer": "Not stated."},
                   "new": {"status": "completed", "answer": "Not stated."}}},
    ]
    rubric = {"a1": {"required": ["seven days", "no restore"], "contradictions": ["business rule"]},
              "a2": {"required": ["not stated"], "contradictions": ["US rule in Canada"]},
              "b1": {"required": ["not stated"], "contradictions": []}}
    return rows, rubric


def test_blinded_review_weights_each_task_and_keeps_controls_separate():
    rows, rubric = examples()
    histories = {row["task_id"]: [{"speaker": "user", "text": "Previous question"}] for row in rows}
    review, key = prepare_review(rows, rubric, ("base", "new"), histories=histories, seed=3)
    assert (review, key) == prepare_review(rows, rubric, ("base", "new"), histories=histories, seed=3)
    assert all("base" not in row and "new" not in row and "group" not in row for row in review)
    judgments = []
    for row in review:
        task, mapping = row["task_id"], key[row["task_id"]]
        answer = {"task_id": task}
        for label in ("A", "B"):
            method = mapping[label]
            facts = {"a1": [1, 0] if method == "base" else [1, 1],
                     "a2": [1] if method == "base" else [0], "b1": [1]}[task]
            conflicts = [] if task == "b1" else [task == "a2" and method == "new"]
            answer[label] = {"requirements": facts, "contradictions": conflicts}
        judgments.append(answer)
    result = score_review(key, judgments, ("base", "new"))
    changed = result["groups"]["changed"]["dimensions"]
    assert changed["coverage"]["mean"] == {"base": .75, "new": .5}
    assert changed["contradiction"]["mean"] == {"base": 0, "new": .5}
    control = result["groups"]["control"]["dimensions"]["contradiction"]
    assert control == {"assessed_tasks": 0, "mean": None, "paired": None}
    assert result["groups"]["all"]["dimensions"]["contradiction"]["assessed_tasks"] == 2


def test_incomplete_or_invalid_judgments_cannot_be_aggregated():
    rows, rubric = examples()
    review, key = prepare_review(rows, rubric, ("base", "new"))
    with pytest.raises(ValueError, match="every reviewed task"):
        score_review(key, [], ("base", "new"))
    judgments = [{"task_id": row["task_id"], **{label: {"requirements": [1] * len(row["requirements"]),
        "contradictions": [False] * len(row["contradictions"])} for label in ("A", "B")}} for row in review]
    judgments[0]["A"]["requirements"][0] = True
    with pytest.raises(ValueError, match="requirement scores"):
        score_review(key, judgments, ("base", "new"))
    judgments[0]["A"]["requirements"][0] = 1
    judgments[0]["A"]["contradictions"] = [None]
    with pytest.raises(ValueError, match="contradiction checks"):
        score_review(key, judgments, ("base", "new"))


def test_review_keeps_each_answer_with_its_own_history_and_evidence():
    rows, rubric = examples()
    for row in rows:
        for method in ("base", "new"):
            row["modes"][method]["history"] = [{"speaker": "assistant", "text": method + " history"}]
            row["modes"][method]["evidence"] = [{"document_id": method + " source"}]
    review, key = prepare_review(rows, rubric, ("base", "new"), seed=8)
    for row in review:
        for label in ("A", "B"):
            method = key[row["task_id"]][label]
            assert row[label]["history"][0]["text"] == method + " history"
            assert row[label]["evidence"][0]["document_id"] == method + " source"
    del rows[0]["modes"]["new"]["history"]
    with pytest.raises(ValueError, match="both answers need method-specific history"):
        prepare_review(rows, rubric, ("base", "new"))


def test_cli_requires_frozen_predictions_and_creates_fillable_blind_review(tmp_path):
    rows, rubric = examples()
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "freeze.json"
    manifest.write_text(json.dumps({"sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(), "rows": len(rows)}))
    rubric_path = tmp_path / "rubric.json"
    rubric_path.write_text(json.dumps({"cases": rubric}))
    output = tmp_path / "review"
    main(["prepare", "--predictions", str(predictions), "--freeze", str(manifest), "--rubric", str(rubric_path),
          "--methods", "base", "new", "--output", str(output)])
    judgments = [json.loads(line) for line in (output / "judgments.jsonl").read_text().splitlines()]
    assert all(all(value is None for value in row["A"]["requirements"]) for row in judgments)
    predictions.write_text(predictions.read_text() + "\n")
    with pytest.raises(ValueError, match="freeze manifest"):
        main(["prepare", "--predictions", str(predictions), "--freeze", str(manifest), "--rubric", str(rubric_path),
              "--methods", "base", "new", "--output", str(tmp_path / "invalid")])
