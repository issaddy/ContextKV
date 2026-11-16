"""CPU execution of cache decisions plus replay of committed GPU measurements.

Run: python -m experiments.cpu_cache_simplification
No GPU timing or answer quality is simulated. Synthetic cases exercise the
production scalar admission code with explicit residency and request metadata.
"""
from __future__ import annotations

import ast
from array import array
from dataclasses import replace
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace as NS

import controller.cooperative_request as request_module
from controller.cooperative_request import (
    CompiledRequest, EvidenceCosts, with_source_demand,
)
from controller.schema import EvidenceBlock, EvidenceKey, EvidenceUnit, SourceSpan

ROOT = Path(__file__).resolve().parents[1]
REVISION = "8f9fd328458d7df23e63da7f99b01822c2236679"


def admission_decision(source):
    """Execute unchanged admit statements, stopping before GPU state creation."""
    path = ROOT / "engine_patches/contextkv_compute.py"
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == "ComputeRuntime")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "admit")
    stop = next(i for i, node in enumerate(method.body)
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "RequestState")
    credit = "build_value" if any(isinstance(node, ast.Name) and node.id == "build_value"
                                  for node in ast.walk(method)) else "0.0"
    method.body = method.body[:stop] + ast.parse(
        "return dict(mode=mode, exact=exact, infoflow=approximate, "
        f"future_credit={credit}, quality_cost=quality_loss*quality_price_ms)"
    ).body
    namespace = dict(math=math, array=array, MatchPrefixParams=NS,
                     RadixKey=lambda *args, **kwargs: args)
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["admit"]


ADMIT = admission_decision(subprocess.check_output(
    ["git", "show", f"{REVISION}:engine_patches/contextkv_compute.py"], cwd=ROOT, text=True))
PATCHED_ADMIT = admission_decision((ROOT / "engine_patches/contextkv_compute.py").read_text())


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


def run_case(policy, body=4096, prompt=256, output=1024, uses=1.0, warm=False, admit=ADMIT):
    # Only the second block is demanded. with_source_demand consequently does
    # not mark it as a reusable exact prefix: the preceding block is absent.
    lengths = (body // 8, body - body // 8)
    units, sources = [], []
    for i, length in enumerate(lengths):
        key = EvidenceKey("synthetic", str(i), str(i), "v1", str(i))
        text = "x" * length
        sources.append(EvidenceBlock(key, "synthetic://" + str(i), "", text))
        units.append(EvidenceUnit(str(i), (SourceSpan(str(i), key, 0, length),), text))
    snapshot = dict(costs={}, prefixes=[], sources={}, page_size=64,
                    available_tokens=1_000_000)
    prices = EvidenceCosts(units, sources, Tokenizer(), snapshot,
                           prompt_tokens=prompt - 40, output_tokens=output, policy=policy)
    addresses = prices.addresses(("0", "1"))
    if warm:
        snapshot["sources"] = {address: dict(gpu_tokens=length, host_tokens=0,
            unpinned_nodes={}, host_nodes={}) for address, length in zip(addresses, lengths)}
    quote = prices.quote(("0", "1"), preserve_order=True)
    first = prompt // 2
    blocks = tuple(dict(source_id=address, start=start, end=start+length)
                   for address, length, start in zip(addresses, lengths,
                       (first, first+lengths[0])))
    request = CompiledRequest(tuple(range(body+prompt)), (), (), (), (), blocks, ())
    request = with_source_demand(request, ({"source_id": addresses[1], "uses": uses},))
    metadata = dict(session_id="cpu", mode="auto", blocks=blocks,
                    demand=request.demand, costs=policy["costs"],
                    recompute_ratio=policy["recompute_ratio"],
                    quality_loss=policy["quality_loss"], quality_price_ms=1_000_000)
    # Match results are the only mocked cache facts. Native prefix hit is zero
    # for both cold and warm non-prefix requests; page capacity is nonbinding.
    native = NS(device_prefix_len=0, host_hit_length=0)
    cache = NS(page_size=64, match_prefix=lambda _: native,
               evictable_size=lambda: 0,
               tree_core=NS(node_by_id=lambda _: NS(parent=None)))
    runtime = NS(costs={}, demands={}, cache=cache,
                 scheduler=NS(max_req_input_len=1_000_000),
                 allocator=NS(available_size=lambda: 1_000_000),
                 _refresh_priorities=lambda: None,
                 _match=lambda identity, ids: NS(device_prefix_len=len(ids) if warm else 0,
                                                   host_hit_length=0, last_device_node=0))
    req = NS(origin_input_ids=request.input_ids, sampling_params=NS(max_new_tokens=output),
             extra_key=None, cache_salt="cpu")
    decision = admit(runtime, req, metadata)
    return dict(quote_mode=quote["mode"], quote_ms=quote["milliseconds"], **decision)


def main():
    policy = json.loads((ROOT / "results/mechanism.json").read_text())["infoflow"]["optimized"]
    pair_data = json.loads((ROOT / "results/infoflow_pair_20261116.json").read_text())
    replay = []
    for case in pair_data["cases"]:
        native, ordered, info = case["results"]
        replay.append(dict(task_id=case["task_id"],
            native_ms=native["complete_ms"], infoflow_ms=info["complete_ms"],
            native_including_seed_ms=native["complete_ms"]+native["seed"]["engine_ms"],
            infoflow_including_seed_ms=info["complete_ms"]+info["seed"]["engine_ms"],
            infoflow_minus_native_f1=100*(info["quality"]-native["quality"]),
            cache_order_ms=ordered["complete_ms"],
            cache_order_prefill_saved=native["prefill_tokens"]-ordered["prefill_tokens"],
            cache_order_output_delta=ordered["output_tokens"]-native["output_tokens"]))

    counterexample = [dict(uses=uses, **run_case(policy, uses=uses),
                          patched_mode=run_case(policy, uses=uses, admit=PATCHED_ADMIT)["mode"])
                      for uses in (0, .25, .5, 1, 2, 4, 8)]
    cold, warm = run_case(policy), run_case(policy, warm=True)
    amortization = [dict(future_revisits=n,
                        native_ms=(n+1)*cold["exact"],
                        forced_infoflow_ms=cold["infoflow"]+n*warm["infoflow"],
                        existing_auto_after_initial_infoflow_ms=cold["infoflow"]+n*min(warm["exact"], warm["infoflow"]))
                    for n in (0, 1, 2, 4, 8)]
    def empty_counts():
        return {name: dict(cases=0, infoflow_choices=0, quote_worker_disagreements=0,
            slower_current_request=0, never_amortizable=0, loses_complete_modeled_chain=0,
            maximum_current_penalty_ms=0.0)
            for name in ("existing", "A_current_request_only", "B_complete_future_credit")}
    counts = {"production_uses_0_to_1": empty_counts(), "including_uses_2_and_4": empty_counts()}
    for body in (512, 1024, 2048, 4096, 8192, 16384):
        for prompt in (128, 256, 512, 1024):
            for output in (64, 128, 256, 512, 1024):
                for uses in (0, .5, 1, 2, 4):
                    case = run_case(policy, body, prompt, output, uses)
                    hot = run_case(policy, body, prompt, output, uses, warm=True)
                    patched = run_case(policy, body, prompt, output, uses, admit=PATCHED_ADMIT)
                    assert patched["future_credit"] == 0
                    assert patched["exact"] == case["exact"] and patched["infoflow"] == case["infoflow"]
                    info = case["infoflow"]+case["quality_cost"]
                    warm_info = hot["infoflow"]+hot["quality_cost"]
                    full_credit = uses*max(0, hot["exact"]-warm_info)
                    choices = dict(existing=case["mode"],
                        A_current_request_only=patched["mode"],
                        B_complete_future_credit="infoflow" if info-full_credit < case["exact"] else "exact")
                    groups = ["including_uses_2_and_4"]
                    if uses <= 1:
                        groups.append("production_uses_0_to_1")
                    for group in groups:
                        for name, mode in choices.items():
                            c = counts[group][name]
                            selected, slower = mode == "infoflow", info > case["exact"]
                            c["cases"] += 1
                            c["infoflow_choices"] += selected
                            c["quote_worker_disagreements"] += mode != case["quote_mode"]
                            c["slower_current_request"] += selected and slower
                            c["never_amortizable"] += selected and slower and warm_info >= hot["exact"]
                            c["loses_complete_modeled_chain"] += selected and info+uses*warm_info > (uses+1)*case["exact"]
                            if selected and slower:
                                c["maximum_current_penalty_ms"] = max(c["maximum_current_penalty_ms"], info-case["exact"])
    qdiff = [row["exact"]-row["infoflow"] for row in policy["pairs"]]
    result = dict(
        revision=REVISION,
        method="Production EvidenceCosts plus baseline git-show and patched working-tree ComputeRuntime.admit AST; mocked residency, no GPU execution",
        calibration_source="results/mechanism.json:infoflow.optimized (four MultiDoc training states, C4)",
        assumptions=["Synthetic geometry, not production trace replay", "Native has no prefix hits on revisits; this favors InfoFlow",
                     "Expected decode lengths and phase costs use the recorded calibration; no new latency/quality measurement"],
        recorded_gpu_pair_replay=replay,
        measured_pair_total=dict(
            native_ms=sum(r["native_ms"] for r in replay),
            infoflow_ms=sum(r["infoflow_ms"] for r in replay),
            native_including_seed_ms=sum(r["native_including_seed_ms"] for r in replay),
            infoflow_including_seed_ms=sum(r["infoflow_including_seed_ms"] for r in replay)),
        synthetic_counterexample=dict(body_tokens=4096, prompt_tokens=256, output_cap=1024,
            demanded_nonprefix_block_tokens=3584, cold_decisions=counterexample,
            warm_decision=warm, modeled_revisit_costs=amortization,
            break_even_revisits=None if warm["infoflow"] >= warm["exact"] else
                max(0, math.ceil((cold["infoflow"]-cold["exact"])/(warm["exact"]-warm["infoflow"])))),
        synthetic_sweep=counts,
        alternatives=dict(A="Delete build_value from admission; choose using this request's full cost, matching EvidenceCosts.quote",
                          B="Credit uses * max(0, future Native full cost - future warm InfoFlow full cost); future geometry is assumed equal to this request",
                          scope="A executes the patched production admission function; B is an optimistic scalar counterfactual, not a production change or observed future requests"),
        quality_calibration=dict(states=len(qdiff), states_with_loss=sum(x>0 for x in qdiff),
            mean_exact_minus_infoflow=sum(qdiff)/len(qdiff),
            charged_mean_loss=max(0, sum(qdiff)/len(qdiff)),
            note="This is an expected-quality target, not a per-request guarantee; averaging before clipping is not inherently an implementation bug."),
        constructor=constructor_profile(policy))
    target=ROOT / "results/cpu_cache_simplification_20261116.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(dict(output=str(target), measured=result["measured_pair_total"],
                         production_domain=counts["production_uses_0_to_1"],
                         constructor=result["constructor"]), indent=2))


if __name__ == "__main__":
    main()
