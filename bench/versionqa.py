"""Prepare the published VersionQA questions and literal versioned corpus.

Reference answers are written to a separate offline file and never included in
the serving tasks. PDF offsets refer to the extracted text, not PDF byte offsets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote


SOURCE_COMMIT = "2a2cbe8f285557f99dc7a79d6df7134e1e3eccff"


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def extract_pdfs(source):
    """Extract text once so its exact PDF input and parser version are recorded."""
    try:
        import pypdf
    except ImportError as exc:
        raise RuntimeError("PDF extraction needs pypdf, or supply a recorded --pdf-text file") from exc
    rows = []
    for path in sorted(Path(source).glob("*.pdf")):
        reader = pypdf.PdfReader(path)
        rows.append({"filename": path.name, "sha256": _digest(path.read_bytes()),
                     "pages": len(reader.pages), "extractor": f"pypdf {pypdf.__version__}",
                     "text": "\n\n".join(page.extract_text() or "" for page in reader.pages)})
    return rows


def prepare(source, questions, output, tokenizer, *, pdf_text=None):
    """Create 512-token passages, independent tasks, and offline references."""
    source, output = Path(source), Path(output)
    paths = sorted(path for path in source.iterdir() if path.suffix in {".md", ".pdf"})
    pdf_rows = extract_pdfs(source) if pdf_text is None else json.loads(Path(pdf_text).read_text(encoding="utf-8"))
    pdfs = {row["filename"]: row for row in pdf_rows}
    if len(paths) != 34 or len(pdfs) != 6 or len(pdf_rows) != 6:
        raise ValueError("the pinned VersionQA corpus requires 34 documents, including six PDFs")
    passages, documents = [], []
    for path in paths:
        raw = path.read_bytes()
        digest = _digest(raw)
        if path.suffix == ".pdf":
            record = pdfs.get(path.name)
            if record is None or record["sha256"] != digest:
                raise ValueError(f"PDF text does not match its source: {path.name}")
            text = record["text"]
        else:
            text = raw.decode("utf-8")
        version = re.search(r"(?:v)?(\d+\.\d+\.\d+)", path.name)
        if not text or not version:
            raise ValueError(f"document needs text and its published version: {path.name}")
        version = version.group(1)
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        source_uri = (f"https://github.com/danielhuwiler/versionrag/blob/{SOURCE_COMMIT}/data/raw/"
                      + quote(path.name))
        doc_id = _digest(path.name.encode())[:16]
        count = 0
        for start in range(0, len(offsets), 462):
            stop = min(start + 512, len(offsets))
            left, right = offsets[start][0], offsets[stop - 1][1]
            if right <= left:
                continue
            passages.append({"passage_id": f"{doc_id}-{start:06d}", "document_id": path.name,
                             "version": version, "title": path.stem, "source_uri": source_uri,
                             "span": [left, right], "text": text[left:right]})
            count += 1
            if stop == len(offsets):
                break
        documents.append({"file": path.name, "sha256": digest, "version": version,
                          "chars": len(text), "tokens": len(offsets), "passages": count,
                          "extracted_from_pdf": path.suffix == ".pdf"})

    with Path(questions).open(encoding="utf-8-sig", newline="") as stream:
        source_tasks = list(csv.DictReader(stream))
    if len(source_tasks) != 100 or any(not all(row.get(name) for name in ("Type", "Question", "Answer"))
                                       for row in source_tasks):
        raise ValueError("the pinned VersionQA question file requires 100 complete rows")
    tasks, references = [], []
    for number, row in enumerate(source_tasks, 1):
        session = f"versionqa-{number:03d}"
        task = {"dataset": {"name": "versionqa", "revision": SOURCE_COMMIT},
                "conversation_id": session, "task_id": session + "<::>1", "turn": 1,
                "Collection": "versionqa", "input": [{"speaker": "user", "text": row["Question"]}],
                "contexts": []}
        tasks.append(task)
        references.append({**task, "targets": [{"text": row["Answer"]}], "category": row["Type"]})

    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / "corpus.jsonl", passages)
    _write_jsonl(output / "tasks.jsonl", tasks)
    _write_jsonl(output / "references.jsonl", references)
    (output / "split.json").write_text(json.dumps({"dev": [], "test": [r["conversation_id"] for r in tasks]},
                                                  indent=2) + "\n", encoding="utf-8")
    manifest = {"declared_source_commit": SOURCE_COMMIT, "documents": documents, "passages": len(passages),
                "questions_sha256": _digest(Path(questions).read_bytes()),
                "corpus_sha256": _digest((output / "corpus.jsonl").read_bytes()),
                "chunking": "tokenizer offsets; 512 tokens with 50-token overlap",
                "pdf_extractors": sorted({row["extractor"] for row in pdf_rows}),
                "scope": "original corpus files and separate post-generation references"}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="VersionQA data/raw directory")
    parser.add_argument("--questions", required=True, help="VersionQA data/test/evaluation_set.csv")
    parser.add_argument("--tokenizer", required=True, help="local Qwen3 tokenizer directory")
    parser.add_argument("--output", required=True, help="new output directory")
    parser.add_argument("--pdf-text", help="optional JSON from extract_pdfs with source SHA-256 values")
    args = parser.parse_args(argv)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    print(json.dumps(prepare(args.source, args.questions, args.output, tokenizer, pdf_text=args.pdf_text),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
