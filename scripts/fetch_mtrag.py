#!/usr/bin/env python3
"""Download the published MT-RAG questions and record their source."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.mtrag import DATASET_COMMIT, DATASET_URL, MTRAG_UN_URL


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mtrag", "mtrag-un"), default="mtrag")
    parser.add_argument("--output", help="Defaults to data/<dataset>")
    args = parser.parse_args(argv)
    root = Path(args.output or "data/" + args.dataset)
    root.mkdir(parents=True, exist_ok=True)
    is_un = args.dataset == "mtrag-un"
    url = MTRAG_UN_URL if is_un else DATASET_URL
    target = root / ("reference.jsonl" if is_un else "RAG.jsonl")
    if not target.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            target.write_bytes(response.read())
    source = {"dataset": args.dataset, "source_url": url, "commit": DATASET_COMMIT}
    (root / "source.json").write_text(json.dumps(source, indent=2)+"\n")
    print(json.dumps({"output": str(target), **source}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
