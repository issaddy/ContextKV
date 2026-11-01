import json

import pytest

from adapters.mtrag import OnlineSession, load_rows
from adapters.retrieval import load_beir_corpus
from bench.quality import load_qrels
from bench.recor import prepare, write_prepared
from bench.retrieval_eval import prepare_queries


def test_recor_exports_public_labels_separately_from_generated_history(tmp_path):
    cases = [{"id": "c1", "task": "drones", "turns": [
        {"turn_id": 1, "query": "Why?", "answer": "gold one", "gold_doc_ids": ["d1"]},
        {"turn_id": 2, "query": "And how?", "answer": "gold two", "gold_doc_ids": ["absent"]},
    ]}, {"id": "c2", "task": "drones", "turns": [
        {"turn_id": 1, "query": "Other?", "answer": "gold three", "gold_doc_ids": ["d1"]},
    ]}]
    prepared = prepare(cases, [{"doc_id": "d1", "content": "original document"}],
                       revision="frozen", dev_fraction=.5)
    output = tmp_path / "export"
    write_prepared(output, prepared)
    rows = load_rows(output / "tasks.jsonl")
    assert "gold" not in (output / "tasks.jsonl").read_text()
    assert load_beir_corpus(output / "corpus.jsonl")["d1"]["text"] == "original document"
    assert load_qrels(output / "qrels.tsv")["c1:2"] == {"absent": 1}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["missing_gold_document_ids"] == ["absent"]
    split = json.loads((output / "split.json").read_text())
    assert set(split["dev"]).isdisjoint(split["test"])
    all_ids = set(split["dev"] + split["test"])
    views = prepare_queries(rows, {"test": list(all_ids)}, split="test", collection="drones")
    assert next(q for q in views if q.task_id == "c1:2").user_questions == ("Why?", "And how?")
    session = OnlineSession([r for r in rows if r["conversation_id"] == "c1"], "P3")
    assert session.current().query == "Why?"
    session.commit("c1:1", "actual generated answer")
    second = session.current()
    assert second.history[-1].text == "actual generated answer" and not second.contexts


def test_recor_rejects_duplicate_documents_and_nonconsecutive_turns():
    case = {"id": "c", "task": "drones", "turns": [
        {"turn_id": 2, "query": "q", "answer": "a", "gold_doc_ids": ["d"]}]}
    with pytest.raises(ValueError, match="unique IDs"):
        prepare([case], [{"doc_id": "d", "content": "a"}, {"doc_id": "d", "content": "b"}], revision="r")
    with pytest.raises(ValueError, match="consecutive"):
        prepare([case], [{"doc_id": "d", "content": "a"}], revision="r")
