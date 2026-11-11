#!/usr/bin/env python3
"""Prepare the current SGLang 0.5.18 source integration, reusing installed packages.

The source revision identifies this patch's applicability, not an environment
upgrade policy. No torch, CUDA, kernel, or Python dependency is installed here.
--source-only downloads just the files used for CPU patch/syntax verification.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import hashlib
import importlib.metadata
import sys
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "engine_patches/integrations/sglang-0.5.18"
MANIFEST = json.loads((INTEGRATION / "manifest.json").read_text())
PATCH = INTEGRATION / MANIFEST["patch"]


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
            raise RuntimeError(f"SGLang patch applies to a different source revision: {head}; expected {commit}")
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
        if not source_only:
            original = set(MANIFEST["original_files"])
            tracked = [rel for rel in MANIFEST["changed_files"] if rel in original]
            run(["git", "restore", "--source", commit, "--worktree", "--", *tracked], checkout)
            for rel in set(MANIFEST["changed_files"])-original:
                (checkout / rel).unlink(missing_ok=True)
        run(["git", "apply", "--check", str(PATCH)], checkout)
        run(["git", "apply", str(PATCH)], checkout)
    for rel in MANIFEST["changed_files"]:
        compile((checkout / rel).read_text(), str(checkout / rel), "exec")
    packages = {}
    for name in ("sglang", "torch", "sglang-kernel", "flashinfer-python", "transformers",
                 "tokenizers", "apache-tvm-ffi"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"repository": MANIFEST["repository"], "source_tag": MANIFEST["source_tag"],
            "base_commit": commit, "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(),
            "patch_applies": True, "already_applied": already_applied, "source_only": source_only,
            "checkout": str(checkout), "python": sys.executable, "installed_packages": packages,
            "dependency_installation": "none; reuse the selected environment",
            "syntax_checked_files": MANIFEST["changed_files"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=ROOT / "vendor/sglang-0.5.18")
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
