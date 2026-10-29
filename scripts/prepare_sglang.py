#!/usr/bin/env python3
"""Fetch the pinned SGLang source, apply and syntax-check the integration patch.

--source-only downloads the audited files for CPU patch verification. It does NOT
create an installable engine. Omit it to prepare a complete git checkout on Linux.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "engine_patches/source_manifest.json").read_text())
PATCH = ROOT / "engine_patches/0001-contextkv-protocol.patch"


def run(args, cwd, *, check=True):
    return subprocess.run(args, cwd=cwd, check=check, capture_output=True, text=True)


def prepare(checkout: Path, source_only: bool):
    checkout.mkdir(parents=True, exist_ok=True)
    commit = MANIFEST["commit"]
    if not (checkout / ".git").exists():
        if any(checkout.iterdir()):
            raise RuntimeError("Refusing to initialize a nonempty checkout directory")
        run(["git", "init", "-q"], checkout)
        if source_only:
            (checkout / ".contextkv-source-only").write_text(commit + "\n")
        else:
            run(["git", "remote", "add", "origin", MANIFEST["repository"] + ".git"], checkout)
            run(["git", "fetch", "--depth", "1", "origin", commit], checkout)
            run(["git", "checkout", "--detach", "FETCH_HEAD"], checkout)
    elif source_only != (checkout / ".contextkv-source-only").exists():
        raise RuntimeError("Source-only validation and full checkout destinations must differ")
    if not source_only:
        head = run(["git", "rev-parse", "HEAD"], checkout).stdout.strip()
        if head != commit:
            raise RuntimeError(f"Wrong SGLang commit: {head}; expected {commit}")
    reverse = run(["git", "apply", "--reverse", "--check", str(PATCH)], checkout, check=False)
    already_applied = reverse.returncode == 0
    if source_only and not already_applied:
        def fetch(rel):
            dest = checkout / rel
            if dest.exists():
                raw = dest.read_bytes()
            else:
                url = f"https://raw.githubusercontent.com/sgl-project/sglang/{commit}/{rel}"
                raw = urllib.request.urlopen(url, timeout=60).read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fetch, MANIFEST["original_files"]))
    if not already_applied:
        run(["git", "apply", "--check", str(PATCH)], checkout)
        run(["git", "apply", str(PATCH)], checkout)
    for rel in MANIFEST["changed_files"]:
        compile((checkout / rel).read_text(), str(checkout / rel), "exec")
    return {"base_commit": commit, "patch_applies": True, "already_applied": already_applied,
            "syntax_checked_files": MANIFEST["changed_files"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=ROOT / "vendor/sglang")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = prepare(args.checkout.resolve(), args.source_only)
    rendered = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
