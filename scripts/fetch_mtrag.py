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
from adapters.mtrag import DATASET_SHA256, DATASET_URL, dataset_profile, load_rows, split_overlap, stratified_split


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/mtrag")
    parser.add_argument("--offline", action="store_true", help="Profile an existing RAG.jsonl without downloading")
    args = parser.parse_args(argv)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "RAG.jsonl"
    if not args.offline and not target.exists():
        with urllib.request.urlopen(DATASET_URL, timeout=120) as response:
            content = response.read()
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
    actual_sha = sha256(target.read_bytes()).hexdigest()
    if actual_sha != DATASET_SHA256:
        raise ValueError(f"pinned RAG.jsonl SHA256 mismatch: {actual_sha}; expected {DATASET_SHA256}")
    rows = load_rows(target)
    report = dataset_profile(rows)
    report["source_url"] = DATASET_URL
    report["source_sha256"] = actual_sha
    expected = {"all": {"tasks": 842, "sessions": 110, "adjacent_repeat": 657, "ever_repeat": 951},
                "cloud": {"tasks": 205, "sessions": 26, "adjacent_repeat": 188, "ever_repeat": 277}}
    for domain, fields in expected.items():
        for field, value in fields.items():
            if report[domain][field] != value:
                raise ValueError(f"pinned data mismatch {domain}.{field}: {report[domain][field]} != {value}")
    if report["reference_markers"] != 1053:
        raise ValueError("pinned data reference-marker count differs from 1053")
    manifest = stratified_split(rows, source_sha256=report["source_sha256"])
    report["split_overlap"] = split_overlap(rows, manifest)
    for name, payload in (("profile.json", report), ("split_manifest.json", manifest)):
        path = root / name
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"refusing to replace frozen manifest/profile {path}; use a new output directory")
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
