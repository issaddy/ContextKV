"""Paged causal attention over immutable block KV and sparse request updates."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention(Q, K, V, ROWS, LOCAL, POSITIONS, ROPE, OUT,
                     queries, length, qs0: tl.constexpr, qs1: tl.constexpr,
                     ks0: tl.constexpr, ks1: tl.constexpr,
                     vs0: tl.constexpr, vs1: tl.constexpr,
                     os0: tl.constexpr, os1: tl.constexpr,
                     rope_stride: tl.constexpr, GROUPS: tl.constexpr,
                     HEAD_DIM: tl.constexpr, ROTARY_DIM: tl.constexpr,
                     NEOX: tl.constexpr, SCALE: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    query_block, head = tl.program_id(0), tl.program_id(1)
    kv_head = head // GROUPS
    rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    dimensions = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + rows[:, None] * qs0 + head * qs1 + dimensions[None, :],
                rows[:, None] < queries, other=0)
    positions = tl.load(POSITIONS + rows, rows < queries, other=-1)
    maximum = tl.full((BLOCK_M,), -float('inf'), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulated = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    if NEOX:
        pair = (dimensions + ROTARY_DIM // 2) % ROTARY_DIM
        frequency = dimensions % (ROTARY_DIM // 2)
        sign = tl.where(dimensions < ROTARY_DIM // 2, -1., 1.)
    else:
        pair = dimensions ^ 1
        frequency = dimensions // 2
        sign = tl.where(dimensions % 2 == 0, -1., 1.)
    for first in range(0, tl.cdiv(length, BLOCK_N)):
        columns = first * BLOCK_N + tl.arange(0, BLOCK_N)
        slots = tl.load(ROWS + columns, columns < length, other=0)
        local = tl.load(LOCAL + columns, columns < length, other=0)
        key = tl.load(K + slots[:, None] * ks0 + kv_head * ks1 + dimensions[None, :],
                      columns[:, None] < length, other=0).to(tl.float32)
        paired = tl.load(K + slots[:, None] * ks0 + kv_head * ks1 + pair[None, :],
                         columns[:, None] < length, other=0).to(tl.float32)
        valid = (columns[:, None] < length) & (dimensions[None, :] < ROTARY_DIM)
        old_cos = tl.load(ROPE + local[:, None] * rope_stride + frequency[None, :], valid, other=1).to(tl.float32)
        old_sin = tl.load(ROPE + local[:, None] * rope_stride + ROTARY_DIM // 2 + frequency[None, :], valid, other=0).to(tl.float32)
        new_cos = tl.load(ROPE + columns[:, None] * rope_stride + frequency[None, :], valid, other=1).to(tl.float32)
        new_sin = tl.load(ROPE + columns[:, None] * rope_stride + ROTARY_DIM // 2 + frequency[None, :], valid, other=0).to(tl.float32)
        norm = old_cos * old_cos + old_sin * old_sin
        cosine = (new_cos * old_cos + new_sin * old_sin) / norm
        sine = (new_sin * old_cos - new_cos * old_sin) / norm
        key = (key * cosine + paired * sign[None, :] * sine).to(q.dtype)
        logits = tl.dot(q, tl.trans(key)) * (SCALE * 1.4426950408889634)
        valid_attention = (columns[None, :] < length) & (columns[None, :] <= positions[:, None])
        logits = tl.where(valid_attention, logits, -float('inf'))
        updated_maximum = tl.maximum(maximum, tl.max(logits, axis=1))
        correction = tl.exp2(maximum - updated_maximum)
        probabilities = tl.exp2(logits - updated_maximum[:, None])
        value = tl.load(V + slots[:, None] * vs0 + kv_head * vs1 + dimensions[None, :],
                        columns[:, None] < length, other=0)
        accumulated = accumulated * correction[:, None] + tl.dot(probabilities.to(value.dtype), value)
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        maximum = updated_maximum
    result = accumulated / denominator[:, None]
    tl.store(OUT + rows[:, None] * os0 + head * os1 + dimensions[None, :],
             result, rows[:, None] < queries)


def paged_attention(query, key_pool, value_pool, rows, local_positions, query_positions,
                    rotary, length):
    output = torch.empty_like(query)
    _paged_attention[(triton.cdiv(len(query), 16), query.shape[1])](
        query, key_pool, value_pool, rows, local_positions, query_positions,
        rotary.cos_sin_cache, output, len(query), length,
        query.stride(0), query.stride(1), key_pool.stride(0), key_pool.stride(1),
        value_pool.stride(0), value_pool.stride(1), output.stride(0), output.stride(1),
        rotary.cos_sin_cache.stride(0), query.shape[1] // key_pool.shape[1],
        query.shape[-1], rotary.cos_sin_cache.shape[-1], rotary.is_neox_style,
        query.shape[-1] ** -.5, 16, 64, num_warps=4)
    return output
