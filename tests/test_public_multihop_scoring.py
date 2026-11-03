import json
from pathlib import Path
from types import SimpleNamespace

from bench.public_multihop import score
from training.reader_data import rows, write_rows


def cohort(tmp_path):
    source = tmp_path/"author"
    source.mkdir()
    (source/"evaluate.original.py").write_text("# author evaluator fixture\n")
    tasks = [{"task_id": task, "offline_reference": {"original_id": task,
              "answer": answer, "supporting_facts": [], "evidences": [], "sentence_spans": []}}
             for task, answer in (("valid", "Alpha"), ("format", "Beta"),
                                  ("failed", "Gamma"), ("absent", "Delta"))]
    task_path = tmp_path/"tasks.jsonl"
    write_rows(task_path, tasks)
    generation = tmp_path/"generation"
    original = [{"task_id": task, "status": status, "answer": answer,
                 "metrics": {}, "response": {"truncated": False},
                 "cost": {"complete_ms": latency}}
                for task, status, answer, latency in (
                    ("valid", "completed", "Final answer: Alpha", 10),
                    ("format", "completed", "Beta", 20),
                    ("failed", "failed", "", 30))]
    write_rows(generation/"results.unscored.jsonl", original)
    return source, task_path, generation, original


def test_official_denominator_includes_format_generation_and_absent_predictions(tmp_path, monkeypatch):
    source, task_path, generation, original = cohort(tmp_path)
    exact = lambda answer, reference: float(answer == reference)
    monkeypatch.setattr("runpy.run_path", lambda path: {
        "exact_match_score": exact, "f1_score": lambda answer, reference: (exact(answer, reference), 0, 0)})
    executions = []

    def author_command(arguments, **kwargs):
        predictions = json.loads(Path(arguments[2]).read_text())
        gold = json.loads(Path(arguments[3]).read_text())
        executions.append({"prediction_ids": sorted(predictions["answer"]),
                           "gold_ids": sorted(row["_id"] for row in gold)})
        # The author evaluator divides by the whole gold list and prints missing IDs.
        correct = sum(predictions["answer"].get(row["_id"]) == row["answer"] for row in gold)
        return SimpleNamespace(stdout="missing answer format\nmissing answer failed\nmissing answer absent\n"+
            json.dumps({"em": 100*correct/len(gold), "f1": 100*correct/len(gold)}))

    monkeypatch.setattr("bench.public_multihop.subprocess.run", author_command)
    manifest = score(source, task_path, generation, projection="final_line")
    assert executions == [{"prediction_ids": ["valid"], "gold_ids": ["absent", "failed", "format", "valid"]}]
    assert manifest["official"]["f1"] == 25
    assert manifest["scored"] == manifest["planned"] == 4
    assert manifest["valid_prediction_metrics"]["2wiki_answer_f1"] == 100
    assert manifest["valid_answer_predictions"] == 1
    assert manifest["missing_prediction_task_ids"] == ["absent", "failed", "format"]
    assert manifest["missing_generation_task_ids"] == ["absent"]
    assert manifest["projection_failed_task_ids"] == ["format"]
    scored = {row["task_id"]: row for row in rows(generation/"results.scored.jsonl")}
    assert scored["format"]["metrics"]["2wiki_answer_f1"] == 0
    assert scored["format"]["scoring"]["raw_full_answer_metrics"]["raw_full_answer_F1"] == 100
    assert scored["failed"]["status"] == "failed"
    assert scored["failed"]["cost"] == original[2]["cost"]
    assert scored["failed"]["metrics"]["2wiki_answer_f1"] == 0
    assert rows(generation/"results.unscored.jsonl") == original
