"""CPU checks of evidence selection; synthetic vectors, never answer-quality estimates.

Run from the repository with a CPU PyTorch interpreter. Production selectors are
called directly. The alternative keeps the population objective but computes
only within-source affinities, batching marginal coverage updates.
"""
from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json
import statistics
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import torch

from adapters.shared_encoding import EvidenceEncoding
from controller.cooperative_request import EvidenceCosts
from controller.schema import EvidenceBlock, EvidenceKey, EvidenceUnit, SourceSpan

ROOT = Path(__file__).resolve().parents[1]
BASE = "8f9fd328458d7df23e63da7f99b01822c2236679"


def topology(units):
    positions = {u.unit_id: i for i, u in enumerate(units)}
    groups = {}
    closures = []
    for i, unit in enumerate(units):
        key = unit.source_spans[0].source_key
        groups.setdefault((key.corpus_id, key.document_id, key.version), []).append(i)
        closure, pending = set(), [i]
        while pending:
            j = pending.pop()
            if j not in closure:
                closure.add(j)
                pending.extend(positions[d] for d in units[j].dependencies)
        closures.append(closure)
    return list(groups.values()), closures


def baseline_population():
    """Load the exact original method, including after this checkout is optimized."""
    source = subprocess.check_output(
        ["git", "show", BASE + ":adapters/shared_encoding.py"], cwd=ROOT, text=True)
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == "EvidenceEncoding")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "project_population")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "baseline/shared_encoding.py", "exec"), namespace)
    return namespace["project_population"]


def quote_variants():
    """Change only the duplicated quote block; usable before or after integration."""
    source = textwrap.dedent(inspect.getsource(EvidenceEncoding.project_population))
    duplicate = textwrap.indent(
        "if cost_model and not cost_model.fits(candidate, preserve_order=True):\n"
        "    continue\n"
        "extra_price = (cost_model.estimate(candidate, preserve_order=True)-price) if cost_model else 0.",
        " " * 12)
    single = textwrap.indent(
        "quote = cost_model.quote(candidate, preserve_order=True) if cost_model else None\n"
        "if quote and quote[\"new_tokens\"]+quote[\"pin_tokens\"] > cost_model.snapshot[\"available_tokens\"]:\n"
        "    continue\n"
        "extra_price = quote[\"milliseconds\"]-price if quote else 0.",
        " " * 12)
    assert duplicate in source or single in source
    methods = []
    for old, new in ((single, duplicate), (duplicate, single)):
        namespace = {}
        exec(compile(source.replace(old, new), "quote_variant/shared_encoding.py", "exec"), namespace)
        methods.append(namespace["project_population"])
    return tuple(methods)


def quote_experiment():
    baseline = baseline_population()
    blocked, single_quote = quote_variants()
    encoding, units, blocks, _ = corpus(1024, 8, 9123, dimension=512)
    arguments = dict(body_tokens=8192, cost_model=costs(units, blocks), seconds_price=.001)
    original_ids, original_ms = timed(lambda: baseline(encoding, units, **arguments))
    blocked_ids, blocked_ms = timed(lambda: blocked(encoding, units, **arguments))
    single_ids, single_ms = timed(lambda: single_quote(encoding, units, **arguments))
    assert original_ids == blocked_ids == single_ids
    result = dict(n=1024, documents=8, dimension=512, actual_cost_function=True,
                  selected_tokens=len(single_ids)*256, selected_ids_equal=True,
                  original_ms=original_ms, blocked_ms=blocked_ms,
                  blocked_single_quote_ms=single_ms,
                  quote_dedup_speedup=blocked_ms/single_ms,
                  total_speedup=original_ms/single_ms)
    path = ROOT / "results/cpu_evidence_simplification_20261116.json"
    output = json.loads(path.read_text())
    output["single_quote_comparison"] = result
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"single_quote_comparison": result}), flush=True)


def simple_select(encoding, units, body_tokens, *, novelty):
    """Untrained query-weighted deduplication, same source seeds and token budget."""
    groups, closures = topology(units)
    selected = set().union(*(closures[max(g, key=lambda i: float(encoding.relevance[i]))]
                             for g in groups))
    sizes = [u.body_tokens for u in units]
    group_id = torch.empty(len(units), dtype=torch.long)
    for i, members in enumerate(groups):
        group_id[members] = i
    represented = encoding.relevance.new_zeros(len(units))
    used = sum(sizes[i] for i in selected)
    if not novelty:
        for i in encoding.relevance.argsort(descending=True).tolist():
            extra = closures[i] - selected
            tokens = sum(sizes[j] for j in extra)
            if used + tokens <= body_tokens:
                selected.update(extra)
                used += tokens
        return tuple(units[i].unit_id for i in sorted(selected))

    def update(indexes):
        for i in sorted(indexes):
            scores = (encoding.embeddings @ encoding.embeddings[i]).clamp(0, 1)
            scores *= group_id == group_id[i]
            represented.copy_(torch.maximum(represented, scores))

    update(selected)
    while True:
        marginal = (encoding.relevance.clamp_min(0) * (1 - represented)).tolist()
        best, best_score = None, -float("inf")
        for i, closure in enumerate(closures):
            extra = closure - selected
            tokens = sum(sizes[j] for j in extra)
            if not extra or used + tokens > body_tokens:
                continue
            score = sum(marginal[j] for j in extra) / tokens
            if score > best_score:
                best, best_score = extra, score
        if best is None:
            return tuple(units[i].unit_id for i in sorted(selected))
        selected.update(best)
        used += sum(sizes[i] for i in best)
        update(best)


def corpus(n, documents, seed, *, dependencies=False, dimension=128):
    generator = torch.Generator().manual_seed(seed)
    topics = 8
    centers = torch.nn.functional.normalize(torch.randn(topics, dimension, generator=generator), dim=-1)
    labels = torch.randint(topics, (n,), generator=generator)
    vectors = torch.nn.functional.normalize(centers[labels] + .04 * torch.randn(n, dimension, generator=generator), dim=-1)
    relevance = (.95 - labels.float() * .07 + .025 * torch.rand(n, generator=generator)).clamp_min(0)
    units, blocks = [], []
    for i in range(n):
        key = EvidenceKey("cpu", f"doc{i % documents}", f"p{i}", "v1", f"hash{i}")
        text = "x " * 256
        span = SourceSpan(f"s{i}", key, 0, len(text))
        deps = (f"u{i - 1}",) if dependencies and i > 0 and i % 7 == 0 else ()
        units.append(EvidenceUnit(f"u{i}", (span,), text, dependencies=deps, body_tokens=256))
        blocks.append(EvidenceBlock(key, "cpu://synthetic", "source", text))
    return EvidenceEncoding(tuple(u.unit_id for u in units), vectors, relevance, (), None), tuple(units), tuple(blocks), labels


class Words:
    def encode(self, text, **_):
        return text.split()


def costs(units, blocks, capacity=65536):
    policy = json.loads((ROOT / "results/hotpath.json").read_text())["infoflow"]["optimized"]
    snapshot = {"costs": {}, "sources": {}, "prefixes": [], "page_size": 64,
                "available_tokens": capacity}
    return EvidenceCosts(units, blocks, Words(), snapshot, prompt_tokens=256,
                         output_tokens=256, policy=policy, mode="exact")


def objective(encoding, units, chosen):
    selected = {u.unit_id for u in units if u.unit_id in chosen}
    total = 0.
    groups, _ = topology(units)
    relevance = encoding.relevance.clamp_min(0)
    for group in groups:
        present = [i for i in group if units[i].unit_id in selected]
        if present:
            affinity = (encoding.embeddings[group] @ encoding.embeddings[present].T).clamp(0, 1)
            total += float((affinity * relevance[present][None, :]).amax(1) @ relevance[group].square())
    return total


def timed(function, repeats=3):
    function()
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        durations.append((time.perf_counter() - start) * 1000)
    return result, statistics.median(durations)


def main():
    baseline = baseline_population()
    blocked, _ = quote_variants()
    equivalence = []
    for seed in range(24):
        encoding, units, blocks, _ = corpus(48, 4, seed, dependencies=True)
        price = costs(units, blocks, 8192)
        arguments = dict(body_tokens=4096, cost_model=price, seconds_price=.001)
        before = baseline(encoding, units, **arguments)
        after = encoding.project_population(units, **arguments)
        equivalence.append(before == after)
    if not all(equivalence):
        raise AssertionError("Block execution changed selected identities")

    timings = []
    for n, documents, dimension in [(128, 1, 128), (128, 8, 128), (512, 1, 128),
                                     (512, 8, 128), (1024, 8, 128), (1024, 8, 512)]:
        encoding, units, blocks, _ = corpus(n, documents, 9123, dimension=dimension)
        budget = 8192
        for include_cost in (False, True):
            price = costs(units, blocks) if include_cost else None
            arguments = dict(body_tokens=budget, cost_model=price, seconds_price=.001)
            before, baseline_ms = timed(lambda: baseline(encoding, units, **arguments))
            after, blocked_ms = timed(lambda: blocked(encoding, units, **arguments))
            if before != after:
                raise AssertionError("Block execution changed selected identities")
            timings.append(dict(n=n, documents=documents, dimension=dimension, actual_cost_function=include_cost,
                                selected_tokens=len(before)*256, selected_ids_equal=before == after,
                                original_ms=baseline_ms, blocked_ms=blocked_ms,
                                speedup=baseline_ms/blocked_ms,
                                original_similarity_elements=n*n,
                                blocked_similarity_elements=sum(len(g)**2 for g in topology(units)[0])))
            print(json.dumps(timings[-1]), flush=True)

    simple = {name: [] for name in ("production", "deduplicate", "ranked")}
    for seed in range(16):
        encoding, units, _, labels = corpus(128, 4, seed)
        for name in simple:
            if name == "production":
                chosen = encoding.project_population(units, body_tokens=4096)
            else:
                chosen = simple_select(encoding, units, 4096, novelty=name == "deduplicate")
            indexes = [i for i, u in enumerate(units) if u.unit_id in chosen]
            # Topic counts are synthetic diagnostic coverage, not factual support or QA quality.
            covered = {(units[i].source_spans[0].source_key.document_id, int(labels[i])) for i in indexes}
            simple[name].append({"tokens": len(chosen)*256, "facility_objective": objective(encoding, units, chosen),
                                 "document_topic_pairs": len(covered)})
    simple_means = {name: {key: statistics.mean(x[key] for x in rows) for key in rows[0]}
                    for name, rows in simple.items()}
    encoding, units, _, _ = corpus(128, 4, 9123)
    simple_times = {}
    for name, call in {
        "baseline": lambda: baseline(encoding, units, body_tokens=4096),
        "optimized": lambda: encoding.project_population(units, body_tokens=4096),
        "deduplicate": lambda: simple_select(encoding, units, 4096, novelty=True),
        "ranked": lambda: simple_select(encoding, units, 4096, novelty=False),
    }.items():
        selected, elapsed = timed(call)
        simple_times[name] = {"median_ms": elapsed, "selected_tokens": len(selected)*256}

    encoding, units, _, _ = corpus(4, 1, 1)
    encoding.relevance = torch.tensor([.95, .94, .93, .8])
    encoding.embeddings = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.]])
    candidates = [row for row in encoding.frontier(units) if len(row)*256 <= 512]
    dedup = simple_select(encoding, units, 512, novelty=True)
    prefix = {"budget_tokens": 512, "two_required_groups": [["u0", "u1", "u2"], ["u3"]],
              "production_feasible_candidates": candidates, "deduplicate_choice": dedup,
              "any_frontier_covers_both": any("u3" in row and any(i in row for i in ("u0", "u1", "u2")) for row in candidates)}

    paired = list(units)
    for index, version, other in ((2, "v1", "u3"), (3, "v2", "u2")):
        span = paired[index].source_spans[0]
        span = replace(span, source_key=replace(span.source_key, version=version))
        paired[index] = replace(paired[index], source_spans=(span,), dependencies=(other,))
    feasible = [row for row in encoding.frontier(paired) if len(row)*256 <= 512]
    version_pair = {"budget_tokens": 512, "required_version_pair": ["u2", "u3"],
                    "required_pair_tokens": 512, "production_feasible_candidates": feasible,
                    "pair_in_any_candidate": any({"u2", "u3"} <= set(row) for row in feasible)}

    encoding, units, blocks, _ = corpus(4, 4, 2)
    price = costs(units, blocks, capacity=512)
    seeds = encoding.project_population(units, body_tokens=512, cost_model=price)
    budget_case = {"body_budget_tokens": 512, "selected_body_tokens": len(seeds)*256,
                   "physical_capacity_tokens": 512, "production_cost_fits": price.fits(seeds, preserve_order=True)}
    output = {"base_commit": BASE, "torch": torch.__version__,
              "threads": 1, "seed": 9123,
              "data": "synthetic source-labelled vectors; no model weights or benchmark traces available",
              "exact_equivalence": {"cases": len(equivalence), "identical_selections": sum(equivalence),
                                    "includes_cross_document_dependencies": True,
                                    "uses_production_EvidenceCosts": True},
              "timings": timings, "equal_4096_token_proxy_means_16_seeds": simple_means,
              "equal_4096_token_cpu_timing": simple_times,
              "prefix_counterexample": prefix, "version_pair_counterexample": version_pair,
              "mandatory_seed_capacity_counterexample": budget_case}
    path = ROOT / "results/cpu_evidence_simplification_20261116.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": simple_means, "prefix": prefix, "budget": budget_case}), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if sys.argv[1:] != ["--quote-only"]:
        main()
    quote_experiment()
