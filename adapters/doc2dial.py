"""Offline author documents/dialogues export, not oracle retrieval or direct SFT.

Complete documents are public corpus inputs. Dialogue acts and reference labels
remain only in a separate supervision file; they never create corpus dependencies.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import quote


def _identity(*parts):
    """Reversible source identity without hashing or ambiguous separators."""
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty original string")
    return value


def convert(documents, dialogues, *, dataset_version, source_uri, split, source_format):
    """Return full corpus records, role-preserving offline supervision and counts.

    Split is an operator declaration, not inferred from filenames. Span offsets
    are Python string character positions [start,end), exactly as in the author
    text. Only referenced spans are validated, once per source address.
    """
    _text(dataset_version, "dataset_version")
    _text(source_uri, "source_uri")
    _text(split, "split")
    if source_format not in {"v1", "v0.9"}:
        raise ValueError("source_format must explicitly be v1 or v0.9")
    docs, corpus, supervision, spans, seen = {}, [], [], {}, set()
    for domain, by_document in sorted(documents["doc_data"].items()):
        _text(domain, "domain")
        for doc_id, document in sorted(by_document.items()):
            _text(doc_id, "doc_id")
            if document["domain"] != domain or document["doc_id"] != doc_id:
                raise ValueError("document identity differs from its author container")
            text = _text(document["doc_text"], "doc_text")
            passage = _identity(domain, doc_id)
            docs[domain, doc_id] = document
            corpus.append({"passage_id": passage, "document_id": passage,
                "text": text, "title": document["title"], "version": dataset_version,
                "source_uri": source_uri + "#" + quote(passage, safe=""),
                "provenance": [source_uri, "dataset:doc2dial", "dataset_version:" + dataset_version],
                "metadata": {"domain": domain, "source_document_id": doc_id,
                             "version_scope": "dataset_release_not_inferred_document_revision"}})

    def address(domain, doc_id, span_id):
        key = (domain, doc_id, str(span_id))
        if key in spans:
            return spans[key]
        document = docs[domain, doc_id]
        if str(span_id) not in document["spans"]:
            raise ValueError(f"unknown grounding span: {key}")
        span = document["spans"][str(span_id)]
        start, end, selected = span["start_sp"], span["end_sp"], span["text_sp"]
        if str(span["id_sp"]) != str(span_id):
            raise ValueError(f"span identity differs from its author key: {key}")
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(document["doc_text"])
                or not isinstance(selected, str) or document["doc_text"][start:end] != selected):
            raise ValueError(f"grounding span does not equal exact original text: {key}")
        spans[key] = {"passage_id": _identity(domain, doc_id), "start": start, "end": end, "quote": selected}
        return spans[key]

    turns_count = refs_count = labels_count = empty_refs = same_role = 0
    for domain, by_document in sorted(dialogues["dial_data"].items()):
        for doc_id, conversations in sorted(by_document.items()):
            if (domain, doc_id) not in docs:
                raise ValueError(f"dialogue references unknown document: {(domain, doc_id)}")
            for conversation in conversations:
                if conversation["domain"] != domain or conversation["doc_id"] != doc_id:
                    raise ValueError("dialogue identity differs from its author container")
                dial_id = _text(conversation["dial_id"], "dial_id")
                identifier = _identity(domain, doc_id, dial_id)
                if identifier in seen:
                    raise ValueError(f"duplicate conversation identity: {identifier}")
                seen.add(identifier)
                turns = []
                for turn in conversation["turns"]:
                    role = _text(turn["role"], "role")
                    utterance = _text(turn["utterance"], "utterance")
                    field = "references" if source_format == "v1" else "reference"
                    other = "reference" if source_format == "v1" else "references"
                    if other in turn:
                        raise ValueError("turn mixes reference fields from different source formats")
                    original = turn[field]
                    if source_format == "v1":
                        if not isinstance(original, list):
                            raise ValueError("v1 references must be an explicit author list")
                        references = [(ref["sp_id"], [ref["label"]]) for ref in original]
                    else:
                        if not isinstance(original, dict):
                            raise ValueError("v0.9 reference must be an explicit span-to-labels object")
                        references = list(original.items())
                    if any(not isinstance(labels, list) or any(not isinstance(label, str) for label in labels)
                           for _, labels in references):
                        raise ValueError("reference labels must retain their original string list")
                    # Original IDs, roles, consecutive speakers and DA remain
                    # untouched. List position is not a fabricated QA turn ID.
                    turns.append({"turn_id": turn["turn_id"], "role": role, "utterance": utterance,
                        "da": deepcopy(turn["da"]), field: deepcopy(original), "grounding": [
                            {"span_id": span_id, "labels": deepcopy(labels), **address(domain, doc_id, span_id)}
                            for span_id, labels in references]})
                    refs_count += len(references)
                    labels_count += sum(len(labels) for _, labels in references)
                    empty_refs += not references
                same_role += any(a["role"] == b["role"] for a, b in zip(turns, turns[1:]))
                turns_count += len(turns)
                supervision.append({"dataset": "doc2dial", "dataset_version": dataset_version, "source_format": source_format,
                    "split": split, "conversation_id": identifier, "source_dialogue_id": dial_id,
                    "domain": domain, "source_document_id": doc_id, "passage_id": _identity(domain, doc_id),
                    "turns": turns})
    summary = {"dataset": "doc2dial", "dataset_version": dataset_version, "source_format": source_format, "split": split,
        "split_provenance": "operator_declared", "source_uri": source_uri,
        "documents": len(corpus), "conversations": len(supervision), "turns": turns_count,
        "reference_occurrences": refs_count, "unique_referenced_spans": len(spans),
        "reference_label_occurrences": labels_count, "turns_with_empty_references": empty_refs,
        "conversations_with_consecutive_same_role": same_role,
        "offset_unit": "python_unicode_character_half_open",
        "corpus_scope": "all_provided_complete_documents_no_gold_selection",
        "supervision_scope": "offline_original_dialogue_and_author_grounding_not_direct_sft",
        "reference_labels": "preserved_author_labels_not_verified_entailment",
        "empty_references": "preserved_not_inferred_as_general_no_support",
        "static_dependencies_from_grounding": False}
    return corpus, supervision, summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--dialogues", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--source-format", choices=("v1", "v0.9"), required=True)
    parser.add_argument("--source-uri", required=True, help="Dataset/archive source; not an inferred government webpage")
    parser.add_argument("--split", required=True, help="Explicit source partition declaration, e.g. train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; preserve earlier exports")
    raw_documents, raw_dialogues = args.documents.read_bytes(), args.dialogues.read_bytes()
    corpus, supervision, manifest = convert(json.loads(raw_documents), json.loads(raw_dialogues),
        dataset_version=args.dataset_version, source_uri=args.source_uri, split=args.split, source_format=args.source_format)
    manifest["sources"] = {
        "documents": {"path": str(args.documents.resolve()), "sha256": sha256(raw_documents).hexdigest()},
        "dialogues": {"path": str(args.dialogues.resolve()), "sha256": sha256(raw_dialogues).hexdigest()}}
    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("corpus.jsonl", corpus), ("supervision.jsonl", supervision)):
        (args.output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    manifest["status"] = "completed"
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("status", "documents", "conversations", "turns", "reference_occurrences")}))
    return manifest


if __name__ == "__main__":
    main()
