"""Loong's complete allowed documents as independent P1 tasks, never P3 gold retrieval.

Source assembly follows MozerWang/Loong at 6d2115b8b48a3d19412ccb52a0c9c4ee37869af4,
src/utils/prompt.py. Declared adaptations: fixed published document order; reject
ambiguous financial filenames; recognize Link the Links in the question field,
so its separately supplied judgments are not attached to their source facts.
No document is removed to fit a model budget. Original structured answers remain
offline labels, serialized as JSON only at the existing text-reference interface.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.parse import quote
import zipfile


COMMIT = "6d2115b8b48a3d19412ccb52a0c9c4ee37869af4"
QUESTIONS_URL = f"https://raw.githubusercontent.com/MozerWang/Loong/{COMMIT}/data/loong.jsonl"
DOCUMENTS_URL = "https://alibaba-research.oss-cn-beijing.aliyuncs.com/loong/doc.zip"
PROTOCOL = "loong-complete-allowed-documents-p1-v2"
LINK_RESULTS = "阅读以上判决文书，我将给你若干份判决结果："


class SourceResolutionError(ValueError):
    def __init__(self, document, candidates):
        self.document, self.candidates = document, list(candidates)
        super().__init__(f"expected one original source for {document!r}; found {len(candidates)}")


class Documents:
    """Read the author's archive directly, with its actual UTF-8 ZIP filenames."""
    def __init__(self, path, *, source_uri=DOCUMENTS_URL):
        self.path, self.source_uri = Path(path), source_uri
        # The published archive has UTF-8 names without the ZIP UTF-8 flag.
        self.archive = zipfile.ZipFile(self.path, metadata_encoding="utf-8")
        self.names = tuple(i.filename for i in self.archive.infolist()
                           if not i.is_dir() and i.filename.startswith("doc/")
                           and not i.filename.endswith(".DS_Store"))
        if len(set(self.names)) != len(self.names):
            self.archive.close()
            raise ValueError("duplicate document members in archive")
        self.legal = None
        self._texts = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        self.archive.close()
        self._texts.clear()

    def read(self, name):
        if name not in self._texts:
            self._texts[name] = self.archive.read(name).decode("utf-8")
        return self._texts[name]

    def context(self, row, document, position):
        domain, level = row["type"], row["level"]
        mode, title = "complete", ""
        if domain == "paper":
            path = "doc/paper/" + document
            if path not in self.names:
                raise SourceResolutionError(document, [])
            body = self.read(path)
            title = body.split("\n", 1)[0].strip("#").strip()
            text = title + "\n" + body + "\n\n"
        elif domain == "financial":
            marker = document if level == 4 else "2024-" + document
            candidates = [p for p in self.names if p.startswith("doc/financial/")
                          and p.endswith(".txt") and marker in p.rsplit("/", 1)[1]]
            if len(candidates) != 1:
                raise SourceResolutionError(document, candidates)
            path, = candidates
            body = self.read(path)
            # Preserve the author's display label, even when it is only a suffix.
            title = "《" + Path(path).stem.split("-")[-1] + "》"
            text = title + "\n" + body + "\n\n"
        elif domain == "legal":
            path = "doc/legal/legal.json"
            if self.legal is None:
                self.legal = json.loads(self.read(path))
            if document not in self.legal:
                raise SourceResolutionError(document, [])
            entry = self.legal[document]
            mode = ("facts_without_attached_result" if level == 4 and
                    LINK_RESULTS in row["question"] else "facts_and_result")
            body = entry["content"] + (entry["result"] if mode == "facts_and_result" else "")
            title = f"《判决文书{position + 1}》"
            text = title + "\n" + body + "\n\n"
        else:
            raise ValueError("unknown Loong domain")
        if not body.strip():
            raise ValueError(f"empty original document: {document}")
        # Position preserves the published occurrence order under stable-key
        # assembly, including repeated documents and legal numbered references.
        document_identity = sha256(document.encode()).hexdigest() if domain == "legal" else document
        identifier = f"{position:06d}:" + json.dumps([domain, document_identity, mode], ensure_ascii=False,
                                                     separators=(",", ":"))
        # Legal record names contain case categories asked by the benchmark.
        # Keep their reversible association offline; model-visible provenance
        # must not reveal that metadata in either plain or percent-encoded form.
        address = quote(path, safe="/") + (
            "?case_sha256=" + document_identity if domain == "legal" else "")
        return {"document_id": identifier, "text": text, "title": "",
                "source": self.source_uri + "#" + address,
                "metadata": {"original_document": document, "archive_member": path,
                             "published_position": position, "author_title": title,
                             "reading_mode": mode}}


def task(row, documents, *, revision):
    """Externalize only the docs placeholder; keep the requested task verbatim."""
    if (not isinstance(row["id"], str) or not row["id"] or not isinstance(row["doc"], list)
            or not row["doc"] or any(not isinstance(s, str) or not s for s in row["doc"])):
        raise ValueError("Loong tasks need an ID and a nonempty original document list")
    if row["type"] not in {"paper", "legal", "financial"} or row["level"] not in {1, 2, 3, 4}:
        raise ValueError("unsupported Loong task type/level")
    if not all(isinstance(row[k], str) for k in ("question", "instruction", "prompt_template")):
        raise ValueError("original task text must be strings")
    template = row["prompt_template"]
    if template.count("{docs}") != 1 or template.count("{instruction}") != 1:
        raise ValueError("task template must contain the document and instruction placeholders once")
    placeholder = ("[Documents are supplied in the current evidence sources.]" if row["language"] == "en"
                   else "【资料见本轮证据来源。】")
    values = {"docs": placeholder, "instruction": row["instruction"], "question": row["question"]}
    # One substitution pass cannot expand braces embedded in original user text.
    question = re.sub(r"\{(docs|instruction|question)\}", lambda m: values[m[1]], template)
    contexts = [documents.context(row, name, index) for index, name in enumerate(row["doc"])]
    answer = row["answer"]
    if not isinstance(answer, (str, list, dict)):
        raise ValueError("unsupported reference answer type")
    target = answer if isinstance(answer, str) else json.dumps(answer, ensure_ascii=False)
    return {"dataset": {"name": "loong", "revision": revision},
            "conversation_id": row["id"], "task_id": row["id"], "turn": 1,
            "Collection": "loong/" + row["type"],
            "input": [{"speaker": "user", "text": question}], "contexts": contexts,
            "targets": [{"text": target}], "reference_value": answer,
            "source_task": {k: row[k] for k in ("id", "level", "set", "length", "type", "language",
                                                "question", "instruction", "prompt_template", "doc", "shuffle_doc")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length-set", type=int, nargs="+", choices=(1, 2, 3, 4))
    args = parser.parse_args(argv)
    rows = [json.loads(s) for s in args.questions.read_text(encoding="utf-8").splitlines() if s.strip()]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate published task IDs")
    selected = [r for r in rows if not args.length_set or r["set"] in args.length_set]
    args.output.mkdir(parents=True, exist_ok=True)
    revision = COMMIT + ":" + PROTOCOL
    errors, ids, counts = [], [], Counter()
    with Documents(args.documents) as documents, (args.output/"tasks.jsonl").open("w", encoding="utf-8") as output:
        for row in selected:
            try:
                value = task(row, documents, revision=revision)
            except SourceResolutionError as exc:
                errors.append({"task_id": row["id"], "domain": row["type"], "length_set": row["set"],
                               "document": exc.document, "candidates": exc.candidates, "error": str(exc)})
                continue
            output.write(json.dumps(value, ensure_ascii=False) + "\n")
            ids.append(row["id"])
            counts[row["type"]] += 1
    manifest = {"status": "partial_source_resolution" if errors else "completed", "protocol": PROTOCOL,
                "upstream_commit": COMMIT, "sources": {"questions": str(args.questions.resolve()), "documents": str(args.documents.resolve())}, "published_tasks": len(rows),
                "requested_tasks": len(selected), "exported_tasks": len(ids), "unresolved_tasks": len(errors),
                "length_sets": args.length_set, "domains": dict(counts), "model_calls": 0,
                "document_order": "published list order, including repeated occurrences; no runtime shuffle",
                "allowed_documents": "all original documents for each exported task; no model-budget filtering",
                "legal_matching": "question-field Link the Links withholds attached result; result options remain in the original question; differs from upstream instruction-field check",
                "financial_resolution": "unique matching archive member required; ambiguous tasks recorded, never picked using reference answers",
                "reference_serialization": "original string or JSON rendering of the original list/dict; original value retained offline",
                "scope": "single-turn given-document P1 supplement; not multi-turn, P3 retrieval or service-capacity evidence"}
    for name, value in (("manifest.json", manifest), ("source_issues.json", errors),
                        ("split_manifest.json", {"dev": [], "test": ids,
                                                 "scope": "exported public tasks; unresolved tasks remain in source_issues.json"})):
        (args.output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    result = main()
    raise SystemExit(0 if result["status"] == "completed" else 2)
