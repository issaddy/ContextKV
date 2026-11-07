#!/usr/bin/env python3
"""Fetch fixed ContextPilot author source without installing its package or hooks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "b4c80327f6cfae2286839413731fcb6f53578b2f"
REPOSITORY = "https://github.com/EfficientContext/ContextPilot.git"


def run(args, checkout):
    return subprocess.run(args, cwd=checkout, check=True, capture_output=True, text=True)


def prepare(checkout):
    checkout.mkdir(parents=True, exist_ok=True)
    if not (checkout / ".git").exists():
        if any(checkout.iterdir()):
            raise RuntimeError("Refusing to initialize a nonempty checkout directory")
        run(["git", "init", "-q"], checkout)
        run(["git", "remote", "add", "origin", REPOSITORY], checkout)
        run(["git", "fetch", "--depth", "1", "origin", COMMIT], checkout)
        run(["git", "checkout", "--detach", "FETCH_HEAD"], checkout)
    head = run(["git", "rev-parse", "HEAD"], checkout).stdout.strip()
    if head != COMMIT:
        raise RuntimeError(f"Wrong ContextPilot commit: {head}; expected {COMMIT}")
    if run(["git", "status", "--porcelain", "--untracked-files=no"], checkout).stdout.strip():
        raise RuntimeError("ContextPilot checkout has modified tracked source")
    return {"repository": REPOSITORY, "commit": head, "checkout": str(checkout),
            "package_installed": False, "inference_hooks_installed": False,
            "index_cpu_dependencies": ["numpy", "scipy"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=ROOT / "vendor/contextpilot")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = prepare(args.checkout.resolve())
    rendered = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
