"""Current-request costs used by evidence selection and scheduler admission."""

import math


def page_tokens(tokens, page):
    return math.ceil(tokens / page) * page


def _output_tokens(costs, output_limit):
    return min(output_limit, costs.get("answer_output_tokens", costs.get("exact_output_tokens", output_limit)))


def decode_cost(costs, mode, output_limit):
    # Compare two implementations of the same answer using one length forecast.
    return _output_tokens(costs, output_limit) * costs.get(mode + "_decode_token_ms", costs.get("exact_decode_token_ms", 0.0))


def exact_cost(costs, tokens, gpu_hit, host_hit, output_limit):
    prefill = costs.get("prefill_token_ms", 1.0)
    transfer = costs.get("transfer_token_ms", prefill * .05)
    return (max(0, tokens - gpu_hit - host_hit) * prefill + host_hit * transfer
            + decode_cost(costs, "exact", output_limit))


def infoflow_cost(costs, prompt, body, cold, host, ratio, output_limit, quality_ms, *,
                  active_exact_reqs=0):
    prefill = costs.get("prefill_token_ms", 1.0)
    # Cold independent-block build executes the same transformer layers in Python/SDPA
    # and cannot physically beat fused Native prefill per token.
    build = max(costs.get("build_token_ms", prefill), prefill)
    exact_decode = costs.get("exact_decode_token_ms", 0.0)
    infoflow_decode = costs.get("infoflow_decode_token_ms", exact_decode)
    interference = 0.0
    if active_exact_reqs > 0 and exact_decode > 0.0:
        graph_penalty = max(infoflow_decode - exact_decode, exact_decode * 0.15)
        interference = active_exact_reqs * _output_tokens(costs, output_limit) * graph_penalty
    return (cold * build
            + host * costs.get("transfer_token_ms", prefill * .05)
            + prompt * costs.get("score_token_ms", prefill)
            + (prompt + ratio * body) * costs.get("recompute_token_ms", prefill)
            + decode_cost(costs, "infoflow", output_limit) + quality_ms + interference)
