"""Compare baseline population selection with single-quote selection on CPU.

Synthetic vectors and the production EvidenceCosts are used; these measurements
exclude model inference and are not end-to-end or answer-quality estimates.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from adapters.shared_encoding import EvidenceEncoding
from controller.cooperative_request import EvidenceCosts
from controller.schema import EvidenceBlock, EvidenceKey, EvidenceUnit, SourceSpan

ROOT = Path(__file__).resolve().parents[1]
BASE = "8f9fd328458d7df23e63da7f99b01822c2236679"


def baseline_population():
    source = subprocess.check_output(
        ["git", "show", BASE + ":adapters/shared_encoding.py"], cwd=ROOT, text=True)
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == "EvidenceEncoding")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "project_population")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "baseline/shared_encoding.py", "exec"), namespace)
    return namespace["project_population"]


def corpus(n, documents, seed, *, dependencies=False, dimension=512):
    generator = torch.Generator().manual_seed(seed)
    centers = torch.nn.functional.normalize(torch.randn(8, dimension, generator=generator), dim=-1)
    labels = torch.randint(8, (n,), generator=generator)
    vectors = torch.nn.functional.normalize(
        centers[labels] + .04 * torch.randn(n, dimension, generator=generator), dim=-1)
    relevance = (.95 - labels.float() * .07 + .025 * torch.rand(n, generator=generator)).clamp_min(0)
    units, blocks = [], []
    for i in range(n):
        key = EvidenceKey("cpu", f"doc{i % documents}", f"p{i}", "v1", f"hash{i}")
        text = "x " * 256
        span = SourceSpan(f"s{i}", key, 0, len(text))
        deps = (f"u{i - 1}",) if dependencies and i > 0 and i % 7 == 0 else ()
        units.append(EvidenceUnit(f"u{i}", (span,), text, dependencies=deps, body_tokens=256))
        blocks.append(EvidenceBlock(key, "cpu://synthetic", "source", text))
    return EvidenceEncoding(tuple(u.unit_id for u in units), vectors, relevance, (), None), tuple(units), tuple(blocks)


class Words:
    def encode(self, text, **_):
        return text.split()


def arguments(units, blocks, budget):
    calibration = subprocess.check_output(["git", "show", f"{BASE}:results/hotpath.json"], cwd=ROOT)
    policy = json.loads(calibration)["infoflow"]["optimized"]
    snapshot = {"costs": {}, "sources": {}, "prefixes": [], "page_size": 64,
                "available_tokens": 65536}
    price = EvidenceCosts(units, blocks, Words(), snapshot, prompt_tokens=256,
                         output_tokens=256, policy=policy, mode="exact")
    return dict(body_tokens=budget, cost_model=price, seconds_price=.001)


def timed_pair(baseline, encoding, units, args):
    calls = (lambda: baseline(encoding, units, **args),
             lambda: encoding.project_population(units, **args))
    selected = [call() for call in calls]
    assert selected[0] == selected[1]
    samples = [[], []]
    for repeat in range(3):
        for side in ((0, 1) if repeat % 2 == 0 else (1, 0)):
            start = time.perf_counter()
            result = calls[side]()
            samples[side].append((time.perf_counter()-start)*1000)
            assert result == selected[0]
    before, after = (statistics.median(values) for values in samples)
    return dict(selected_tokens=len(selected[0])*256, selected_ids_equal=True,
                baseline_ms=before, single_quote_ms=after, speedup=before/after)


def main():
    baseline = baseline_population()
    for seed in (0, 7, 13, 23):
        encoding, units, blocks = corpus(48, 4, seed, dependencies=True, dimension=128)
        args = arguments(units, blocks, 4096)
        assert baseline(encoding, units, **args) == encoding.project_population(units, **args)

    encoding, units, blocks = corpus(32, 4, 44, dimension=32)
    generator = torch.Generator().manual_seed(44)
    vectors = torch.nn.functional.normalize(torch.randn(8, 32, generator=generator), dim=-1)
    encoding.embeddings[:] = vectors.repeat_interleave(4, dim=0)
    encoding.relevance[:] = 1.
    args = arguments(units, blocks, 1280)
    before = baseline(encoding, units, **args)
    after = encoding.project_population(units, **args)
    assert before == after

    timings = []
    for n in (64, 256, 1024):
        encoding, units, blocks = corpus(n, 8, 9123)
        row = dict(candidates=n, documents=8, dimension=512,
                   **timed_pair(baseline, encoding, units, arguments(units, blocks, 8192)))
        timings.append(row)
        print(json.dumps(row), flush=True)
    result = dict(base_commit=BASE, torch=torch.__version__, threads=1,
        measurement="Synthetic source-labelled vectors and actual EvidenceCosts; no model inference",
        timing="One warm-up and three alternating-order repetitions; median CPU wall time",
        equivalence=dict(cross_document_dependency_cases=4, identical_selections=4,
                         near_tie_seed=44, near_tie_selected_ids=after, near_tie_equal=True),
        timings=timings)
    (ROOT / "results/cpu_evidence_simplification_20261116.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    main()
