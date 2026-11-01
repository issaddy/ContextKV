import json
from pathlib import Path

import pytest

from bench.intent_review import main, prepare_review, score_review


def cases():
    return [
        {"task_id": task_id, "conversation_id": session, "question": "And now?",
         "base": {"status": "resolved", "text": "First interpretation"},
         "candidate": {"status": "resolved", "text": "Second interpretation"}}
        for task_id, session in (("a1", "a"), ("a2", "a"), ("b1", "b"))
    ]


def test_prepare_and_score_pairs_by_conversation():
    rows = cases()
    rubric = {row["task_id"]: "Expected task and conditions" for row in rows}
    review, key = prepare_review(rows, rubric, ("base", "candidate"), seed=3)
    assert (review, key) == prepare_review(rows, rubric, ("base", "candidate"), seed=3)
    assert all("base" not in row and "candidate" not in row for row in review)
    judgments = []
    for row in review:
        mapping = key[row["task_id"]]
        judgment = {"task_id": row["task_id"]}
        for label in ("A", "B"):
            judgment[label] = {"task": 1 if mapping[label] == "base" else .5, "conditions": 1}
        judgments.append(judgment)
    result = score_review(key, judgments, ("base", "candidate"))
    assert result["tasks"] == 3 and result["sessions"] == 2
    assert result["dimensions"]["task"]["mean"] == {"base": 1, "candidate": .5}
    assert result["dimensions"]["combined"]["paired"]["difference"] == -.25
    assert "decision" not in result["dimensions"]["combined"]["paired"]


def test_missing_or_invalid_judgments_do_not_score():
    rows = cases()
    _, key = prepare_review(rows, {row["task_id"]: "Expected" for row in rows}, ("base", "candidate"))
    with pytest.raises(ValueError, match="every reviewed task"):
        score_review(key, [], ("base", "candidate"))
    judgments = [{"task_id": row["task_id"], "A": {"task": 1, "conditions": 1},
                  "B": {"task": 1, "conditions": 1}} for row in rows]
    judgments[0]["A"]["task"] = True
    with pytest.raises(ValueError, match="must be 0, 0.5 or 1"):
        score_review(key, judgments, ("base", "candidate"))


def test_fictional_rubric_covers_the_intended_followups():
    root = Path(__file__).parents[1] / "bench/assets"
    fixture = json.loads((root / "conversation_quality_stress.json").read_text())
    rubric = json.loads((root / "intent_quality_stress.json").read_text())["cases"]
    expected = {f"{session['id']}<::>{turn}" for session in fixture["sessions"]
                if session["id"] in {"sora", "lark", "nova", "kestrel"}
                for turn in range(2, len(session["turns"]) + 1)}
    assert set(rubric) == expected


def test_cli_creates_a_fillable_review_and_scores_it(tmp_path):
    rows = cases()
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("".join(json.dumps(row) + "\n" for row in rows))
    rubric = tmp_path / "rubric.json"
    rubric.write_text(json.dumps({"cases": {row["task_id"]: "Expected" for row in rows}}))
    out = tmp_path / "review"
    main(["prepare", "--predictions", str(predictions), "--rubric", str(rubric),
          "--methods", "base", "candidate", "--output", str(out)])
    template = [json.loads(line) for line in (out / "judgments.jsonl").read_text().splitlines()]
    assert all(row["A"]["task"] is None for row in template)
    for row in template:
        row["A"] = row["B"] = {"task": 1, "conditions": .5}
    (out / "judgments.jsonl").write_text("".join(json.dumps(row) + "\n" for row in template))
    report = tmp_path / "report.json"
    main(["score", "--key", str(out / "key.json"), "--judgments", str(out / "judgments.jsonl"),
          "--methods", "base", "candidate", "--output", str(report)])
    assert json.loads(report.read_text())["dimensions"]["combined"]["mean"] == {"base": .75, "candidate": .75}
