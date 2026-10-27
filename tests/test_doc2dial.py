"""Synthetic author-format inputs; no real dataset text or model calls."""
from copy import deepcopy
from hashlib import sha256
import json

import pytest

from adapters.corpus import CorpusIndex
from adapters.doc2dial import convert, main
from adapters.retrieval import load_beir_corpus


def sources():
    text = "序🙂\nOnly batch clients.\nOnly batch clients."
    start = text.rindex("Only batch clients.")
    document = {"domain": "manual", "doc_id": "doc/a", "title": "Original title", "doc_text": text,
        "spans": {"scope": {"id_sp": "scope", "start_sp": start, "end_sp": len(text),
                            "text_sp": text[start:]},
                  "unused-bad": {"start_sp": -8}}}
    docs = {"doc_data": {"manual": {"doc/a": document,
        "unused": {"domain": "manual", "doc_id": "unused", "title": "Another title",
                   "doc_text": "Complete unrelated original document.", "spans": {}}}}}
    turns = [
        {"turn_id": 4, "role": "user", "utterance": "Can I retry?", "da": "query_condition",
         "references": [{"sp_id": "scope", "label": "precondition"}]},
        {"turn_id": 9, "role": "user", "utterance": "I mean batch mode.", "da": "clarify",
         "references": [{"sp_id": "scope", "label": "solution_fuzzy"}]},
        {"turn_id": 20, "role": "agent", "utterance": "Only in that mode.", "da": "respond",
         "references": []},
    ]
    dialogs = {"dial_data": {"manual": {"doc/a": [{"domain": "manual", "doc_id": "doc/a",
                                                 "dial_id": "conversation", "turns": turns}]}}}
    return docs, dialogs


def converted(documents, dialogues, source_format="v1"):
    return convert(documents, dialogues, dataset_version="author-v1", source_uri="dataset://author/archive",
                   split="train", source_format=source_format)


def test_original_order_same_roles_ids_and_unicode_addresses_are_preserved():
    docs, dialogs = sources()
    before = deepcopy((docs, dialogs))
    corpus, supervision, summary = converted(docs, dialogs)
    turns = supervision[0]["turns"]
    original_turns = dialogs["dial_data"]["manual"]["doc/a"][0]["turns"]
    assert [t["turn_id"] for t in turns] == [4, 9, 20]
    assert [t["role"] for t in turns] == ["user", "user", "agent"]
    assert [t["utterance"] for t in turns] == [t["utterance"] for t in original_turns]
    assert [t["da"] for t in turns] == [t["da"] for t in original_turns]
    ref = turns[0]["grounding"][0]
    text = docs["doc_data"]["manual"]["doc/a"]["doc_text"]
    assert ref["start"] == text.rindex(ref["quote"]) and ref["start"] != text.index(ref["quote"])
    assert text[ref["start"]:ref["end"]] == ref["quote"]
    assert ref["start"] != len(text[:ref["start"]].encode("utf-8"))
    assert turns[1]["grounding"][0]["labels"] == ["solution_fuzzy"]
    assert [t["references"] for t in turns] == [t["references"] for t in original_turns]
    assert turns[2]["references"] == turns[2]["grounding"] == []
    assert summary["reference_occurrences"] == 2 and summary["unique_referenced_spans"] == 1
    assert summary["conversations_with_consecutive_same_role"] == 1
    assert len(corpus) == 2  # The unreferenced document is not discarded.
    assert (docs, dialogs) == before


def test_corpus_has_no_dialogue_gold_and_loads_through_existing_identity(tmp_path):
    docs, dialogs = sources()
    corpus, supervision, _ = converted(docs, dialogs)
    serialized = json.dumps(corpus)
    for private in ("query_condition", "solution_fuzzy", "precondition", "Can I retry?", "Only in that mode."):
        assert private not in serialized
    assert all("dependencies" not in record and "spans" not in record for record in corpus)
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in corpus))
    loaded = load_beir_corpus(path)
    index = CorpusIndex.from_jsonl(path, corpus_id="doc2dial", snapshot_id="author-v1")
    assert set(loaded) == {block.key.passage_id for block in index.blocks}
    assert all(not b.dependencies and b.parent_section is None for b in index.blocks)
    key = supervision[0]["passage_id"]
    assert json.loads(key) == ["manual", "doc/a"]
    assert loaded[key]["text"] == docs["doc_data"]["manual"]["doc/a"]["doc_text"]
    assert loaded[key]["source"].startswith("dataset://author/archive#")
    assert {block.key.version for block in index.blocks} == {"author-v1"}


@pytest.mark.parametrize("mutation", ["unknown", "offset", "quote", "missing_references"])
def test_bad_current_reference_fails_without_find_clamp_or_missing_fill(mutation):
    docs, dialogs = sources()
    turn = dialogs["dial_data"]["manual"]["doc/a"][0]["turns"][0]
    if mutation == "unknown":
        turn["references"][0]["sp_id"] = "not-present"
    elif mutation == "offset":
        docs["doc_data"]["manual"]["doc/a"]["spans"]["scope"]["end_sp"] += 100
    elif mutation == "quote":
        docs["doc_data"]["manual"]["doc/a"]["spans"]["scope"]["text_sp"] = "Onlybatchclients."
    else:
        del turn["references"]
    with pytest.raises((ValueError, KeyError)):
        converted(docs, dialogs)


def test_cli_binds_sources_and_rebuilds_derived_files(tmp_path):
    docs, dialogs = sources()
    documents, dialogues = tmp_path / "documents.json", tmp_path / "dialogues.json"
    documents.write_text(json.dumps(docs, ensure_ascii=False))
    dialogues.write_text(json.dumps(dialogs, ensure_ascii=False))
    output = tmp_path / "converted"
    argv = ["--documents", str(documents), "--dialogues", str(dialogues), "--dataset-version", "author-v1",
            "--source-format", "v1", "--source-uri", "dataset://author/archive", "--split", "train", "--output", str(output)]
    manifest = main(argv)
    assert manifest["dataset"] == "doc2dial" and manifest["split_provenance"] == "operator_declared"
    assert manifest["sources"]["documents"]["sha256"] == sha256(documents.read_bytes()).hexdigest()
    assert manifest["sources"]["dialogues"]["sha256"] == sha256(dialogues.read_bytes()).hexdigest()
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    main(argv)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_v09_multilabel_and_empty_ood_reference_are_preserved_only_offline():
    docs, dialogs = sources()
    original_turns = dialogs["dial_data"]["manual"]["doc/a"][0]["turns"]
    for turn in original_turns:
        del turn["references"]
        turn["reference"] = {"scope": ["reference", "precondition_fuzzy"]}
    original_turns[-1]["da"] = "respond/ood"
    original_turns[-1]["reference"] = {}
    corpus, supervision, summary = converted(docs, dialogs, source_format="v0.9")
    turns = supervision[0]["turns"]
    assert [t["reference"] for t in turns] == [t["reference"] for t in original_turns]
    assert turns[0]["grounding"][0]["labels"] == ["reference", "precondition_fuzzy"]
    assert turns[-1]["reference"] == {} and turns[-1]["grounding"] == []
    assert turns[-1]["da"] == "respond/ood" and "no_support" not in turns[-1]
    assert summary["reference_occurrences"] == 2 and summary["reference_label_occurrences"] == 4
    assert summary["unique_referenced_spans"] == 1 and summary["turns_with_empty_references"] == 1
    assert "respond/ood" not in json.dumps(corpus) and "precondition_fuzzy" not in json.dumps(corpus)
    assert all("dependencies" not in row for row in corpus)
    with pytest.raises((ValueError, KeyError)):
        converted(docs, dialogs, source_format="v1")


def test_source_format_is_explicit_not_inferred_from_version_or_fields():
    docs, dialogs = sources()
    with pytest.raises((ValueError, KeyError)):
        converted(docs, dialogs, source_format="v0.9")
    with pytest.raises(ValueError):
        converted(docs, dialogs, source_format="auto")


def test_multidoc_exchanges_preserve_cross_document_grounding_and_actual_history():
    from adapters.doc2dial import benchmark_rows
    from adapters.mtrag import OnlineSession, dataset_identity

    docs, single = sources()
    old = single["dial_data"]["manual"]["doc/a"][0]["turns"]
    second = docs["doc_data"]["manual"]["unused"]
    second["spans"] = {"1": {"id_sp": "1", "start_sp": 0,
        "end_sp": len(second["doc_text"]), "text_sp": second["doc_text"]}}
    turns = [deepcopy(old[0]), deepcopy(old[1]), deepcopy(old[2]),
             {"turn_id": 21, "role": "agent", "utterance": "Additional original detail.", "da": "respond"},
             {"turn_id": 25, "role": "user", "utterance": "And the other document?", "da": "query"},
             {"turn_id": 26, "role": "agent", "utterance": "The second original answer.", "da": "respond"},
             {"turn_id": 30, "role": "user", "utterance": "Unanswered closing question.", "da": "query"}]
    for turn in turns:
        turn["references"] = [{"doc_id": "unused", "id_sp": "1", "label": "solution"}]
    turns[2]["references"].append({"doc_id": "doc/a", "id_sp": "scope", "label": "precondition"})
    dialogs = {"dial_data": {"manual": [{"dial_id": "multi", "turns": turns}]}}
    corpus, supervision, summary = converted(docs, dialogs, "multidoc2dial")
    rows = list(benchmark_rows(supervision))
    assert summary["dataset"] == "multidoc2dial" and len(corpus) == 2
    assert supervision[0]["turns"][-1]["utterance"] == turns[-1]["utterance"]
    assert len(rows) == 2 and rows[0]["source_turn_ids"] == {"user": [4, 9], "agent": [20, 21]}
    assert rows[0]["targets"][0]["text"] == "Only in that mode.\n\nAdditional original detail."
    assert {g["passage_id"] for g in rows[0]["grounding"]} == {c["passage_id"] for c in corpus}
    assert dataset_identity(rows) == {"name": "multidoc2dial", "revision": "author-v1"}
    gate = OnlineSession(rows, "P3")
    first = gate.current()
    assert first.query == "Can I retry?\n\nI mean batch mode." and not first.contexts
    gate.commit(first.task_id, "Actual generated answer, not author target.")
    following = gate.current()
    assert following.history[-1].text == "Actual generated answer, not author target."
    assert "grounding" not in following.__dict__ and "targets" not in following.__dict__
    assert "Unanswered closing" not in following.query


def test_structural_retrieval_preserves_full_original_text_and_parent_identity():
    from adapters.doc2dial import structural_corpus
    text = "Preamble.\n\nEligibility\nOnly residents qualify.\n\nDocuments\nBring identification.\nTail."
    first, second = text.index("Eligibility"), text.index("Documents")
    document = {"doc_text": text, "spans": {
        "1": {"start_sp": first, "id_sec": "one", "title": "Eligibility", "parent_titles": [{"text": "Application"}]},
        "2": {"start_sp": first + 12, "id_sec": "one", "title": "Eligibility", "parent_titles": [{"text": "Application"}]},
        "3": {"start_sp": second, "id_sec": "two", "title": "Documents", "parent_titles": [{"text": "Application"}]}}}
    original = {"passage_id": "parent", "document_id": "parent", "text": text,
        "title": "Application", "source_uri": "source://public", "version": "v1",
        "metadata": {"domain": "service", "source_document_id": "doc"}}
    rows = list(structural_corpus({"doc_data": {"service": {"doc": document}}}, [original]))
    assert len(rows) == 2 and ''.join(row["text"] for row in rows) == text
    assert [row["span"] for row in rows] == [[0, second], [second, len(text)]]
    assert all(row["text"] == text[slice(*row["span"])] for row in rows)
    assert all(row["document_id"] == "parent" and row["source_uri"] == "source://public" for row in rows)
    assert [row["title"] for row in rows] == ["Application / Eligibility", "Application / Documents"]
    index = CorpusIndex.from_records(rows, corpus_id="public", snapshot_id="snapshot")
    assert {b.key.document_id for b in index.blocks} == {"parent"}
    assert {b.parent_section for b in index.blocks} == {row["title"] for row in rows}
