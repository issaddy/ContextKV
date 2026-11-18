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
