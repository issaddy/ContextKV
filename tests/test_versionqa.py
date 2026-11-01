import csv
import hashlib
import json

import pytest

from bench.versionqa import prepare


class CharacterTokenizer:
    def __call__(self, text, **_):
        return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_versionqa_materialization_keeps_labels_separate_and_spans_literal(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    for number in range(28):
        (source / f"manual {number}.1.0.md").write_text(f"manual {number} " + "x" * 700)
    pdfs = []
    for number in range(6):
        path = source / f"release {number}.2.0.pdf"
        path.write_bytes(f"pdf-{number}".encode())
        pdfs.append({"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "extractor": "test", "text": f"extracted source {number}"})
    pdf_text = tmp_path / "pdf.json"
    pdf_text.write_text(json.dumps(pdfs))
    questions = tmp_path / "questions.csv"
    with questions.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("Type", "Question", "Answer"))
        writer.writeheader()
        writer.writerows({"Type": "content", "Question": f"question {i}?", "Answer": f"answer {i}"}
                         for i in range(100))

    output = tmp_path / "prepared"
    manifest = prepare(source, questions, output, CharacterTokenizer(), pdf_text=pdf_text)
    tasks = [json.loads(line) for line in (output / "tasks.jsonl").read_text().splitlines()]
    refs = [json.loads(line) for line in (output / "references.jsonl").read_text().splitlines()]
    passages = [json.loads(line) for line in (output / "corpus.jsonl").read_text().splitlines()]
    assert len(tasks) == len(refs) == 100 and all("targets" not in task for task in tasks)
    assert refs[0]["targets"][0]["text"] == "answer 0"
    assert manifest["passages"] == len(passages)
    for passage in passages:
        if passage["document_id"].endswith(".md"):
            text = (source / passage["document_id"]).read_text()
            start, end = passage["span"]
            assert passage["text"] == text[start:end]

    pdfs[0]["sha256"] = "wrong"
    pdf_text.write_text(json.dumps(pdfs))
    with pytest.raises(ValueError, match="PDF text does not match"):
        prepare(source, questions, tmp_path / "invalid", CharacterTokenizer(), pdf_text=pdf_text)
