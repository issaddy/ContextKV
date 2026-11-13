"""Fetch small pinned evaluator/qrels metadata by default; corpus is opt-in."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import urllib.request
import zipfile

from adapters.mtrag import DATASET_COMMIT

BASE = f"https://raw.githubusercontent.com/IBM/mt-rag-benchmark/{DATASET_COMMIT}/"


def fetch_metadata(destination: str | Path, *, cloud_corpus: bool = False) -> dict:
    root = Path(destination)
    tree_url = f"https://api.github.com/repos/IBM/mt-rag-benchmark/git/trees/{DATASET_COMMIT}?recursive=1"
    with urllib.request.urlopen(tree_url) as stream:
        tree = json.load(stream)["tree"]
    files = [r for r in tree if r["type"] == "blob" and
             (r["path"].startswith("scripts/evaluation/") or
              (r["path"].startswith("mtrag-human/retrieval_tasks/") and r["path"].endswith("qrels/dev.tsv")) or
              r["path"] in {"LICENSE", "README.md", "corpora/README.md"} or
              (cloud_corpus and r["path"] == "corpora/passage_level/cloud.jsonl.zip"))]
    for entry in files:
        relative = entry["path"]
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urllib.request.urlopen(BASE + relative) as response:
                data = response.read()
            path.write_bytes(data)
    manifest = {"repository": "IBM/mt-rag-benchmark", "commit": DATASET_COMMIT, "files": len(files),
                "cloud_corpus_downloaded": cloud_corpus,
                "cloud_corpus_url": BASE + "corpora/passage_level/cloud.jsonl.zip"}
    (root / "contextkv-source.json").write_text(json.dumps(manifest, indent=2))
    if cloud_corpus:
        with zipfile.ZipFile(root / "corpora/passage_level/cloud.jsonl.zip") as archive:
            members = [n for n in archive.namelist() if n.endswith(".jsonl") and not n.startswith("__MACOSX/")]
            if len(members) != 1:
                raise ValueError("expected one original Cloud passage JSONL in the official archive")
            (root / "cloud.jsonl").write_bytes(archive.read(members[0]))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/mtrag-official")
    parser.add_argument("--cloud-corpus", action="store_true", help="explicitly download the full Cloud passage archive")
    args = parser.parse_args(argv)
    manifest = fetch_metadata(args.output, cloud_corpus=args.cloud_corpus)
    print(json.dumps({"output": str(Path(args.output).resolve()), "files": manifest["files"],
                      "cloud_corpus_downloaded": manifest["cloud_corpus_downloaded"]}))


if __name__ == "__main__":
    main()
