#!/usr/bin/env python3
"""Fetch immutable MTRAG data and write a reproducible profile/split manifest."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.mtrag import (DATASET_COMMIT, DATASET_SHA256, DATASET_URL, MTRAG_UN_SHA256, MTRAG_UN_URL,
                            dataset_profile, load_rows, split_overlap, stratified_split)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mtrag", "mtrag-un"), default="mtrag")
    parser.add_argument("--output", help="Defaults to data/<dataset>")
    parser.add_argument("--offline", action="store_true", help="Profile the existing pinned data without downloading")
    args = parser.parse_args(argv)
    root = Path(args.output or "data/" + args.dataset)
    root.mkdir(parents=True, exist_ok=True)
    is_un = args.dataset == "mtrag-un"
    url, expected_sha = (MTRAG_UN_URL, MTRAG_UN_SHA256) if is_un else (DATASET_URL, DATASET_SHA256)
    target = root / ("reference.jsonl" if is_un else "RAG.jsonl")
    if not args.offline and not target.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read()
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
    actual_sha = sha256(target.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(f"pinned {target.name} SHA256 mismatch: {actual_sha}; expected {expected_sha}")
    rows = load_rows(target)
    report = dataset_profile(rows)
    report.update(dataset=args.dataset, source_url=url, source_sha256=actual_sha,
                  protocols=["P1"] if is_un else ["P1", "P2", "P3"])
    # A file digest identifies the pinned release; derived counts are useful
    # diagnostics, not another set of acceptance constants for the same bytes.
    if is_un:
        manifest = {"dataset_commit": DATASET_COMMIT, "source_sha256": actual_sha,
                    "dev": [], "test": sorted({r["conversation_id"] for r in rows}),
                    "scope": "published MTRAG-UN reference tasks; one selected turn per conversation"}
    else:
        manifest = stratified_split(rows, source_sha256=actual_sha)
    report["split_overlap"] = split_overlap(rows, manifest)
    for name, payload in (("profile.json", report), ("split_manifest.json", manifest)):
        path = root / name
        if name == "split_manifest.json" and path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"refusing to replace frozen manifest/profile {path}; use a new output directory")
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
