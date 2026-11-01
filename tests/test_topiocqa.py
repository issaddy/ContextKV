import csv
from hashlib import sha256
import json

import pytest

from adapters.mtrag import OnlineSession, load_rows
import bench.topiocqa as topiocqa
from bench.topiocqa import convert_wiki, materialize, prepare


def fixtures():
    questions, labels = [], []
    for conversation in (1, 2):
        history = []
        for turn in (1, 2):
            question, answer = f"question {conversation}-{turn}", f"answer {conversation}-{turn}"
            questions.append({"Conversation_no": conversation, "Turn_no": turn, "Context": list(history),
                "Question": question, "Answer": answer, "Topic": "sample topic", "Topic_section": "part",
                "Additional_answers": [{"Answer": "alternate"}, {"Answer": "alternate"}]})
            labels.append({"conv_id": conversation, "turn_id": turn, "answers": [answer],
                "positive_ctxs": [{"passage_id": str(conversation * 10 + turn),
                                   "title": "gold title", "text": "private gold evidence"}]})
            history.extend((question, answer))
    return questions, labels


def test_topiocqa_prepares_disjoint_sessions_and_keeps_current_gold_offline(tmp_path):
    questions, labels = fixtures()
    source, gold = tmp_path / "questions.json", tmp_path / "gold.json"
    source.write_text(json.dumps(questions))
    gold.write_text(json.dumps(labels))
    out = tmp_path / "prepared"
    manifest = materialize(source, gold, out, seed=7, dev_conversations=1)
    tasks = load_rows(out / "tasks.jsonl")
    split = json.loads((out / "split.json").read_text())
    assert manifest["turns"] == 4 and set(split["dev"]).isdisjoint(split["test"])
    assert all(row["contexts"] == [] and "targets" not in row and "topic" not in row for row in tasks)
    assert all("private gold evidence" not in str(row) for row in tasks)
    assert "topiocqa:1<::>1\t11\t1" in (out / "qrels.tsv").read_text()
    oracle = load_rows(out / "oracle-tasks.jsonl")
    assert oracle[0]["contexts"][0]["text"] == "private gold evidence"
    refs = [json.loads(line) for line in (out / "references.jsonl").read_text().splitlines()]
    assert refs[0]["targets"] == [{"text": "answer 1-1"}, {"text": "alternate"}, {"text": "alternate"}]

    session = OnlineSession([r for r in tasks if r["conversation_id"] == split["dev"][0]], "P3")
    first = session.current()
    session.commit(first.task_id, "actual generated answer")
    assert session.current().history[-1].text == "actual generated answer"
    assert prepare(questions, labels, seed=7, dev_conversations=1) == prepare(
        questions, labels, seed=7, dev_conversations=1)

    with pytest.raises(ValueError, match="missing or duplicate"):
        prepare(questions, labels[:-1], dev_conversations=1)


def test_topiocqa_streams_original_wikipedia_rows(tmp_path):
    source, target = tmp_path / "wiki.tsv", tmp_path / "corpus.jsonl"
    with source.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("id", "text", "title"))
        writer.writerow(("11", 'Original "quote"\nand second line.', "Article [SEP] Section"))
    oracle = tmp_path / "oracle.jsonl"
    oracle.write_text(json.dumps({"contexts": [{"document_id": "11", "title": "Article [SEP] Section",
                                              "text": 'Original "quote"\nand second line.'}]}) + "\n")
    assert convert_wiki(source, target, oracle_tasks=oracle) == 1
    row = json.loads(target.read_text())
    assert row["passage_id"] == row["document_id"] == "11"
    assert row["text"] == 'Original "quote"\nand second line.'
    assert row["title"] == "Article [SEP] Section"
    oracle.write_text(json.dumps({"contexts": [{"document_id": "11", "title": "Article [SEP] Section",
                                              "text": "changed evidence"}]}) + "\n")
    with pytest.raises(ValueError, match="different original content"):
        convert_wiki(source, tmp_path / "mismatch.jsonl", oracle_tasks=oracle)


def test_topiocqa_scoring_keeps_missing_turns_in_denominator(tmp_path, monkeypatch):
    evaluator = tmp_path / "evaluate_reader.py"
    evaluator.write_text("class TopiOCQAEvaluator:\n"
        "    @staticmethod\n"
        "    def _compute_turn_score(answers, prediction):\n"
        "        value = float(prediction in answers)\n"
        "        return {'em': value, 'f1': value}\n")
    monkeypatch.setattr(topiocqa, "EVALUATOR_SHA256", sha256(evaluator.read_bytes()).hexdigest())
    references = [{"task_id": "a", "conversation_id": "s", "targets": [{"text": "yes"}]},
                  {"task_id": "b", "conversation_id": "s", "targets": [{"text": "no"}]}]
    predictions = [{"task_id": "a", "predictions": [{"text": "yes"}]}]
    result = topiocqa.score_official(references, predictions, evaluator, conversations={"s"})
    assert (result["em"], result["f1"], result["turns"]) == (50., 50., 2)
    assert result["missing_predictions"] == ["b"]
    with pytest.raises(ValueError, match="unique"):
        topiocqa.score_official(references, predictions * 2, evaluator)
