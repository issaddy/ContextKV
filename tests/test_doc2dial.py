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


def test_cli_binds_sources_and_refuses_overwrite(tmp_path):
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
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2
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
    with pytest.raises(ValueError, match="source_format"):
        converted(docs, dialogs, source_format="auto")
