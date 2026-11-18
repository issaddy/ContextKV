"""Current-request costs used by evidence selection and scheduler admission."""

import math


def page_tokens(tokens, page):
    return math.ceil(tokens / page) * page


def decode_cost(costs, mode, output_limit):
    # Compare two implementations of the same answer using one length forecast.
    tokens = min(output_limit, costs.get("answer_output_tokens", costs.get("exact_output_tokens", output_limit)))
    return tokens * costs.get(mode + "_decode_token_ms", costs.get("exact_decode_token_ms", 0.0))


def exact_cost(costs, tokens, gpu_hit, host_hit, output_limit):
    prefill = costs.get("prefill_token_ms", 1.0)
    transfer = costs.get("transfer_token_ms", prefill * .05)
    return (max(0, tokens - gpu_hit - host_hit) * prefill + host_hit * transfer
            + decode_cost(costs, "exact", output_limit))


def infoflow_cost(costs, prompt, body, cold, host, ratio, output_limit, quality_ms):
    prefill = costs.get("prefill_token_ms", 1.0)
    return (cold * costs.get("build_token_ms", prefill)
            + host * costs.get("transfer_token_ms", prefill * .05)
            + prompt * costs.get("score_token_ms", prefill)
            + (prompt + ratio * body) * costs.get("recompute_token_ms", prefill)
            + decode_cost(costs, "infoflow", output_limit) + quality_ms)


def infoflow_work(cold_pages, prompt, body, ratio, page):
    return cold_pages + page_tokens(prompt, page) + prompt + math.ceil(body * ratio)


def select_route(exact, approximate, *, mode, available_tokens, capacity_tokens, max_input_tokens):
    """Prefer executable routes; keep a feasible queued route when the pool is busy."""
    routes = ([exact] if mode == "exact" or approximate is None else
              [approximate] if mode == "infoflow" else [exact, approximate])
    feasible = [route for route in routes
                if route["work_tokens"] <= max_input_tokens
                and route["new_tokens"] + route["resident_tokens"] <= capacity_tokens]
    ready = [route for route in feasible
             if route["new_tokens"] + route["pin_tokens"] <= available_tokens]
    chosen = min(ready or feasible or routes, key=lambda route: route["milliseconds"])
    return {**chosen, "feasible": chosen in feasible, "fits": chosen in ready}


def retention_value(costs, *, mode, on_gpu, host_copy, ratio):
    """Marginal milliseconds saved per retained token relative to its next tier."""
    prefill = costs.get("prefill_token_ms", 1.0)
    transfer = costs.get("transfer_token_ms", prefill * .05)
    rebuild = prefill
    if mode == "infoflow":
        replay = ratio * costs.get("recompute_token_ms", prefill)
        rebuild = min(costs.get("build_token_ms", prefill), max(0.0, prefill - replay))
    if on_gpu and host_copy:
        return min(transfer, rebuild)
    return max(0.0, rebuild - (0.0 if on_gpu else transfer))
