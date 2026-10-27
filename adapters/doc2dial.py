"""Offline author documents/dialogues export, not oracle retrieval or direct SFT.

Complete documents are public corpus inputs. Dialogue acts and reference labels
remain only in a separate supervision file; they never create corpus dependencies.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
from itertools import groupby
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
    if source_format not in {"v1", "v0.9", "multidoc2dial"}:
        raise ValueError("unknown Doc2Dial source format")
    dataset = "multidoc2dial" if source_format == "multidoc2dial" else "doc2dial"
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
                "provenance": [source_uri, "dataset:" + dataset, "dataset_version:" + dataset_version],
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
        groups = [(None, by_document)] if dataset == "multidoc2dial" else sorted(by_document.items())
        for doc_id, conversations in groups:
            if doc_id is not None and (domain, doc_id) not in docs:
                raise ValueError(f"dialogue references unknown document: {(domain, doc_id)}")
            for conversation in conversations:
                if doc_id is not None and (conversation["domain"] != domain or conversation["doc_id"] != doc_id):
                    raise ValueError("dialogue identity differs from its author container")
                dial_id = _text(conversation["dial_id"], "dial_id")
                identifier = _identity(domain, doc_id, dial_id) if doc_id is not None else _identity(domain, dial_id)
                if identifier in seen:
                    raise ValueError(f"duplicate conversation identity: {identifier}")
                seen.add(identifier)
                turns = []
                for turn in conversation["turns"]:
                    role = _text(turn["role"], "role")
                    utterance = _text(turn["utterance"], "utterance")
                    field = "reference" if source_format == "v0.9" else "references"
                    other = "references" if source_format == "v0.9" else "reference"
                    if other in turn:
                        raise ValueError("turn mixes reference fields from different source formats")
                    original = turn[field]
                    if source_format != "v0.9":
                        if not isinstance(original, list):
                            raise ValueError("v1 references must be an explicit author list")
                        references = [(ref["doc_id"], ref["id_sp"], [ref["label"]])
                                      if dataset == "multidoc2dial" else (doc_id, ref["sp_id"], [ref["label"]])
                                      for ref in original]
                    else:
                        if not isinstance(original, dict):
                            raise ValueError("v0.9 reference must be an explicit span-to-labels object")
                        references = [(doc_id, span_id, labels) for span_id, labels in original.items()]
                    if any(not isinstance(labels, list) or any(not isinstance(label, str) for label in labels)
                           for _, _, labels in references):
                        raise ValueError("reference labels must retain their original string list")
                    # Original IDs, roles, consecutive speakers and DA remain
                    # untouched. List position is not a fabricated QA turn ID.
                    turns.append({"turn_id": turn["turn_id"], "role": role, "utterance": utterance,
                        "da": deepcopy(turn["da"]), field: deepcopy(original), "grounding": [
                            {"span_id": span_id, "labels": deepcopy(labels), **address(domain, source_doc, span_id)}
                            for source_doc, span_id, labels in references]})
                    refs_count += len(references)
                    labels_count += sum(len(labels) for _, _, labels in references)
                    empty_refs += not references
                same_role += any(a["role"] == b["role"] for a, b in zip(turns, turns[1:]))
                turns_count += len(turns)
                supervision.append({"dataset": dataset, "dataset_version": dataset_version, "source_format": source_format,
                    "split": split, "conversation_id": identifier, "source_dialogue_id": dial_id,
                    "domain": domain,
                    **({"source_document_id": doc_id, "passage_id": _identity(domain, doc_id)} if doc_id is not None else {}),
                    "turns": turns})
    summary = {"dataset": dataset, "dataset_version": dataset_version, "source_format": source_format, "split": split,
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


def benchmark_rows(supervision):
    """Project complete MultiDoc2Dial exchanges into the shared P3 runner.

    Consecutive messages from one speaker form one exchange side, joined by a
    blank line. Original message IDs and targets remain in offline fields and
    supervision. A terminal user message without an author answer stays in the
    source export but has no scored exchange. No gold document is preselected.
    """
    for conversation in supervision:
        groups = [(role, list(turns)) for role, turns in
                  groupby(conversation["turns"], key=lambda turn: turn["role"])]
        if not groups or groups[0][0] != "user" or any(role not in {"user", "agent"} for role, _ in groups):
            raise ValueError("benchmark conversation must start with a user and use user/agent roles")
        history = []
        for index in range(0, len(groups) - 1, 2):
            users, agents = groups[index][1], groups[index + 1][1]
            question = "\n\n".join(t["utterance"] for t in users)
            answer = "\n\n".join(t["utterance"] for t in agents)
            number = index // 2 + 1
            yield {"dataset": {"name": conversation["dataset"], "revision": conversation["dataset_version"]},
                "conversation_id": conversation["conversation_id"],
                "task_id": f"{conversation['conversation_id']}<::>{number}", "turn": number,
                "Collection": conversation["dataset"] + "/" + conversation["domain"],
                "input": [*history, {"speaker": "user", "text": question}], "contexts": [],
                "targets": [{"text": answer}],
                "source_turn_ids": {"user": [t["turn_id"] for t in users], "agent": [t["turn_id"] for t in agents]},
                "grounding": [g for t in agents for g in t["grounding"]]}
            history = [*history, {"speaker": "user", "text": question}, {"speaker": "agent", "text": answer}]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--dialogues", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--source-format", choices=("v1", "v0.9", "multidoc2dial"), required=True)
    parser.add_argument("--source-uri", required=True, help="Dataset/archive source; not an inferred government webpage")
    parser.add_argument("--split", required=True, help="Explicit source partition declaration, e.g. train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    raw_documents, raw_dialogues = args.documents.read_bytes(), args.dialogues.read_bytes()
    corpus, supervision, manifest = convert(json.loads(raw_documents), json.loads(raw_dialogues),
        dataset_version=args.dataset_version, source_uri=args.source_uri, split=args.split, source_format=args.source_format)
    manifest["sources"] = {
        "documents": {"path": str(args.documents.resolve()), "sha256": sha256(raw_documents).hexdigest()},
        "dialogues": {"path": str(args.dialogues.resolve()), "sha256": sha256(raw_dialogues).hexdigest()}}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("corpus.jsonl", corpus), ("supervision.jsonl", supervision)):
        (args.output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    if args.source_format == "multidoc2dial":
        rows = list(benchmark_rows(supervision))
        (args.output / "tasks.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        partition = "dev" if args.split in {"dev", "validation"} else args.split
        splits = {"dev": [], "test": [], partition: sorted({row["conversation_id"] for row in rows})}
        (args.output / "split_manifest.json").write_text(json.dumps(splits, indent=2) + "\n")
        (args.output / "qrels.tsv").write_text("query-id\tcorpus-id\tscore\n" + "".join(
            f"{row['task_id']}\t{pid}\t1\n" for row in rows
            for pid in sorted({g["passage_id"] for g in row["grounding"]})))
        manifest["benchmark"] = {"protocols": ["P3"], "exchanges": len(rows),
            "grouping": "consecutive_same_speaker_messages_joined_with_blank_line",
            "terminal_unanswered_user_groups": sum(c["turns"][-1]["role"] == "user" for c in supervision),
            "reference_scope": "author_exchange_targets_and_grounding_offline_only"}
    manifest["status"] = "completed"
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("status", "documents", "conversations", "turns", "reference_occurrences")}))
    return manifest


if __name__ == "__main__":
    main()
