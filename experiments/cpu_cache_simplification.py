"""CPU construction cost and equivalence of EvidenceCosts.

Run: python -m experiments.cpu_cache_simplification
Synthetic sources and cache snapshots exercise the production constructor.
Measurements cover CPU construction only, with no model or GPU execution.
"""
from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
import statistics
import subprocess
import time

import controller.cooperative_request as request_module
from controller.cooperative_request import EvidenceCosts
from controller.schema import EvidenceBlock, EvidenceKey, EvidenceUnit, SourceSpan

ROOT = Path(__file__).resolve().parents[1]
REVISION = "8f9fd328458d7df23e63da7f99b01822c2236679"


class Tokenizer:
    """One character per synthetic token; never used for text-quality claims."""
    def encode(self, text, **kwargs):
        return [1] * len(text)


def constructor_profile(policy):
    source = subprocess.check_output(
        ["git", "show", f"{REVISION}:controller/cooperative_request.py"], cwd=ROOT, text=True)
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == "EvidenceCosts")
    namespace = dict(vars(request_module))
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "<baseline-EvidenceCosts>", "exec"), namespace)
    baseline = namespace["EvidenceCosts"]

    def fixture(n, overlapping=False):
        units, sources = [], []
        for i in range(n):
            key = EvidenceKey("fixture", str(i), str(i), "v1", str(i))
            text = "x"*256
            sources.append(EvidenceBlock(key, str(i), "", text))
            span = SourceSpan(str(i), key, 64 if overlapping and i % 2 else 0, 256)
            spans = (span, span, SourceSpan("extra", key, 0, 64)) if overlapping and i % 2 else (span,)
            units.append(EvidenceUnit(str(i), spans, text))
        return units, sources

    rows = []
    for n in (128, 512, 1024):
        units, sources = fixture(n)
        snapshot = dict(costs={}, prefixes=[], sources={}, page_size=64, available_tokens=1_000_000)
        kwargs = dict(prompt_tokens=256, output_tokens=1024, policy=policy)
        before = baseline(units, sources, Tokenizer(), snapshot, **kwargs)
        after = EvidenceCosts(units, sources, Tokenizer(), snapshot, **kwargs)
        assert before.__dict__ == after.__dict__
        row = dict(sources=n, units=n)
        for name, constructor in (("baseline_ms", baseline), ("patched_ms", EvidenceCosts)):
            samples = []
            for _ in range(7):
                start = time.perf_counter()
                constructor(units, sources, Tokenizer(), snapshot, **kwargs)
                samples.append((time.perf_counter()-start)*1000)
            row[name] = statistics.median(samples)
        row["speedup"] = row["baseline_ms"]/row["patched_ms"]
        rows.append(row)

    units, sources = fixture(8, overlapping=True)
    headers = tuple(replace(block, text=block.text[:32], span=(0, 32)) for block in sources[:2])
    checks = states = order_checks = 0
    for use_headers in (False, True):
        for warm in (False, True):
            for mode in ("auto", "exact", "infoflow"):
                snapshot = dict(costs={}, prefixes=[], sources={}, page_size=64, available_tokens=2048)
                kwargs = dict(prompt_tokens=256, output_tokens=1024, policy=policy, mode=mode,
                              source_headers=headers if use_headers else ())
                before = baseline(units, sources, Tokenizer(), snapshot, **kwargs)
                if warm:
                    snapshot["sources"] = {address: dict(gpu_tokens=length if i%2 else 0,
                        host_tokens=0 if i%2 else length, unpinned_nodes={i:length} if i%2 else {},
                        host_nodes={} if i%2 else {i:length})
                        for i, (address, length) in enumerate(before.lengths.items())}
                    snapshot["prefixes"] = [dict(source_ids=before.addresses(("3", "1", "2")),
                        gpu_tokens=256, host_tokens=128, unpinned_nodes={"prefix":256})]
                after = EvidenceCosts(units, sources, Tokenizer(), snapshot, **kwargs)
                assert before.__dict__ == after.__dict__
                states += 1
                for ids in ((), ("0",), ("1", "3", "2"), ("3", "1", "2"),
                            tuple(str(i) for i in range(8)), tuple(str(i) for i in reversed(range(8)))):
                    assert before.order(ids) == after.order(ids)
                    order_checks += 1
                    for preserve in (False, True):
                        for method in ("quote", "estimate", "fits"):
                            assert getattr(before, method)(ids, preserve_order=preserve) == getattr(after, method)(ids, preserve_order=preserve)
                        checks += 1
    return dict(method="Actual baseline git-show and patched constructor; 7 repetitions, median CPU wall time",
        tokenizer="Deterministic one-character/token fixture, 256-token sources; identical tokenizer calls, no model inference",
        timings=rows, equivalence=dict(states=states, order_checks=order_checks,
            quote_estimate_fits_checks=checks, headers_overlap_duplicates_gpu_host=True))


def main():
    policy = json.loads((ROOT / "results/mechanism.json").read_text())["infoflow"]["optimized"]
    result = dict(revision=REVISION,
        method="Baseline git-show versus current production EvidenceCosts; synthetic sources and cache snapshots",
        constructor=constructor_profile(policy))
    target = ROOT / "results/cpu_cache_simplification_20261116.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(output=str(target), **result), indent=2))


if __name__ == "__main__":
    main()
