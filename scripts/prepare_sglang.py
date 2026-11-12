#!/usr/bin/env python3
"""Apply the SGLang source integration using the installed Python/GPU runtime."""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / 'engine_patches/integrations/sglang-0.5.23'
MANIFEST = json.loads((INTEGRATION / 'manifest.json').read_text())
PATCH = INTEGRATION / MANIFEST['patch']


def run(args, cwd, *, check=True):
    return subprocess.run(args, cwd=cwd, check=check, capture_output=True, text=True)


def prepare(checkout: Path, source_only: bool):
    checkout.mkdir(parents=True, exist_ok=True)
    commit = MANIFEST['commit']
    applied = run(['git', 'apply', '--reverse', '--check', str(PATCH)], checkout, check=False).returncode == 0
    if not applied:
        if source_only:
            def fetch(rel):
                target = checkout / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(urllib.request.urlopen(
                    f'https://raw.githubusercontent.com/sgl-project/sglang/{commit}/{rel}', timeout=60).read())
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(fetch, MANIFEST['changed_files']))
        run(['git', 'apply', '--check', str(PATCH)], checkout)
        run(['git', 'apply', str(PATCH)], checkout)
    for module in MANIFEST['modules']:
        source = ROOT / 'engine_patches' / module
        shutil.copyfile(source, checkout / 'python/sglang/srt/managers' / module)
        compile(source.read_text(), str(source), 'exec')
    for rel in MANIFEST['changed_files']:
        compile((checkout / rel).read_text(), rel, 'exec')
    return dict(checkout=str(checkout), source_version=MANIFEST['source_version'], source_commit=commit, source_only=source_only,
                protocol='contextkv-compute-v2', patch_applies=True,
                checked_files=len(MANIFEST['changed_files']) + len(MANIFEST['modules']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, help='Installed SGLang source directory; defaults to the active Python package')
    parser.add_argument('--source-only', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    checkout = args.checkout or Path(importlib.util.find_spec('sglang').origin).resolve().parents[2]
    report = prepare(checkout.resolve(), args.source_only)
    rendered = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + '\n')
    print(rendered)


if __name__ == '__main__':
    main()
