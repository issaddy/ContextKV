"""Custom corpus passage/source identity survives the shared retrieval boundary."""
import json
from dataclasses import replace

import pytest

from adapters.corpus import CorpusIndex
from adapters.mtrag import OnlineContext
from adapters.retrieval import BM25SRetriever, load_beir_corpus, to_evidence
from controller.evidence import EvidenceError, evidence_index


def write_corpus(tmp_path, rows):
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    return path


def test_custom_namespace_is_used_only_when_source_is_absent():
    missing = to_evidence(OnlineContext("p1", "original"), "snapshot", corpus_id="manuals")
    assert missing.key.corpus_id == "manuals"
    assert missing.source_uri == "manuals://snapshot/p1"
    assert missing.provenance[0] == missing.source_uri
    explicit = to_evidence(OnlineContext("p1", "original", "https://example.org/page"),
                           "snapshot", corpus_id="manuals")
    assert explicit.source_uri == "https://example.org/page"
    assert to_evidence(OnlineContext("p1", "original"), "snapshot").source_uri == "mtrag://snapshot/p1"


def test_passage_metadata_roundtrips_bm25_and_corpus_closure(tmp_path):
    rows = [
        {"passage_id": "value", "document_id": "manual", "title": "Timeout",
         "text": "ORCHID timeout is 7 seconds.\n原文不重写。",
         "metadata": {"version": "r2", "source_uri": "manuals://manual/value",
                      "condition_passage_ids": ["condition"]}},
        {"text": "Only ORCHID preview requests use this timeout.",
         "metadata": {"passage_id": "condition", "document_id": "manual", "version": "r2",
                      "source_uri": "manuals://manual/condition"}},
    ]
    path = write_corpus(tmp_path, rows)
    corpus = load_beir_corpus(path)
    assert set(corpus) == {"value", "condition"}
    assert corpus["value"]["text"] == rows[0]["text"]
    assert corpus["condition"]["source"] == "manuals://manual/condition"
    retriever = BM25SRetriever(corpus, corpus_snapshot_id="manual-snapshot")
    hits = retriever.retrieve(["ORCHID timeout"], top_k=2)
    metadata = CorpusIndex.from_jsonl(path, corpus_id="manuals", snapshot_id="manual-snapshot")
    blocks = metadata.enrich(hits)
    assert {b.key.passage_id for b in blocks} == {"value", "condition"}
    assert {b.key.document_id for b in blocks} == {"manual"}
    assert all(b.key.version == "r2" for b in blocks)
    assert all(c.source == f"manuals://manual/{c.document_id}" for c in hits)
    value = next(b for b in blocks if b.key.passage_id == "value")
    assert value.text == rows[0]["text"]
    assert value.dependencies == (next(b.key for b in blocks if b.key.passage_id == "condition"),)


def test_explicit_fields_follow_corpus_index_precedence(tmp_path):
    row = {"_id": "beir-alias", "passage_id": "explicit", "document_id": "document",
           "text": "original", "source_uri": "custom://top", "source": "custom://old",
           "metadata": {"passage_id": "nested", "source_uri": "custom://nested"}}
    corpus = load_beir_corpus(write_corpus(tmp_path, [row]))
    assert list(corpus) == ["explicit"]
    assert corpus["explicit"]["source"] == "custom://top"
    index = CorpusIndex.from_records([row], corpus_id="custom", snapshot_id="snapshot")
    assert index.blocks[0].key.passage_id == "explicit"
    assert index.blocks[0].source_uri == corpus["explicit"]["source"]


def test_beir_and_document_only_legacy_rows_keep_their_identity(tmp_path):
    rows = [{"_id": "beir", "text": "a", "source": "custom://a"},
            {"document_id": "single", "text": "b"}]
    corpus = load_beir_corpus(write_corpus(tmp_path, rows))
    assert list(corpus) == ["beir", "single"]
    assert corpus["beir"]["source"] == "custom://a"
    assert corpus["single"]["source"] == ""


def test_duplicate_passage_does_not_infer_a_revision(tmp_path):
    rows = [{"passage_id": "same", "document_id": "document", "text": "old", "version": "v1"},
            {"passage_id": "same", "document_id": "document", "text": "new", "version": "v2"}]
    with pytest.raises(ValueError, match="duplicate passage ID"):
        load_beir_corpus(write_corpus(tmp_path, rows))


def test_existing_wrong_custom_source_is_not_silently_migrated():
    block = to_evidence(OnlineContext("p1", "original"), "snapshot", corpus_id="manuals")
    old_uri = "mtrag://snapshot/p1"
    legacy = replace(block, source_uri=old_uri, provenance=(old_uri, "corpus_snapshot:snapshot"))
    with pytest.raises(EvidenceError, match="inconsistent source/provenance"):
        evidence_index((block,), committed=(legacy,))
