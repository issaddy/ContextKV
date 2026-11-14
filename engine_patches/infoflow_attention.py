"""Paged causal attention over immutable block KV and sparse request updates."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention(Q, K, V, ROWS, COSINE, SINE, POSITIONS, OUT, LSE,
                     queries, length, qs0: tl.constexpr, qs1: tl.constexpr,
                     ks0: tl.constexpr, ks1: tl.constexpr,
                     vs0: tl.constexpr, vs1: tl.constexpr,
                     os0: tl.constexpr, os1: tl.constexpr,
                     position_stride: tl.constexpr, GROUPS: tl.constexpr, HEADS: tl.constexpr,
                     HEAD_DIM: tl.constexpr, ROTARY_DIM: tl.constexpr,
                     NEOX: tl.constexpr, SCALE: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SPLITS: tl.constexpr):
    query_block, head = tl.program_id(0), tl.program_id(1)
    split = tl.program_id(2)
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
    split_tiles = tl.cdiv(tl.cdiv(length, SPLITS), BLOCK_N)
    available = tl.minimum(length, tl.max(positions, 0)+1)
    for first in range(split*split_tiles, tl.minimum((split+1)*split_tiles, tl.cdiv(available, BLOCK_N))):
        columns = first * BLOCK_N + tl.arange(0, BLOCK_N)
        slots = tl.load(ROWS + columns, columns < available, other=0)
        key = tl.load(K + slots[:, None] * ks0 + kv_head * ks1 + dimensions[None, :],
                      columns[:, None] < available, other=0).to(tl.float32)
        paired = tl.load(K + slots[:, None] * ks0 + kv_head * ks1 + pair[None, :],
                         columns[:, None] < available, other=0).to(tl.float32)
        valid = (columns[:, None] < available) & (dimensions[None, :] < ROTARY_DIM)
        cosine = tl.load(COSINE + columns[:, None] * position_stride + frequency[None, :], valid, other=1)
        sine = tl.load(SINE + columns[:, None] * position_stride + frequency[None, :], valid, other=0)
        key = (key * cosine + paired * sign[None, :] * sine).to(q.dtype)
        logits = tl.dot(q, tl.trans(key)) * (SCALE * 1.4426950408889634)
        valid_attention = (columns[None, :] < length) & (columns[None, :] <= positions[:, None])
        logits = tl.where(valid_attention, logits, -float('inf'))
        updated_maximum = tl.maximum(maximum, tl.max(logits, axis=1))
        normalizer = tl.where(updated_maximum == -float('inf'), 0., updated_maximum)
        correction = tl.exp2(maximum - normalizer)
        probabilities = tl.exp2(logits - normalizer[:, None])
        value = tl.load(V + slots[:, None] * vs0 + kv_head * vs1 + dimensions[None, :],
                        columns[:, None] < available, other=0)
        accumulated = accumulated * correction[:, None] + tl.dot(probabilities.to(value.dtype), value)
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        maximum = updated_maximum
    result = tl.where(denominator[:, None] > 0, accumulated / denominator[:, None], 0.)
    tl.store(OUT + rows[:, None] * os0 + head * os1 + split*HEAD_DIM + dimensions[None, :],
             result, rows[:, None] < queries)
    if SPLITS > 1:
        tl.store(LSE + (rows*HEADS+head)*SPLITS+split,
                 (maximum+tl.log2(denominator))*0.6931471805599453, rows < queries)


def paged_attention(query, key_pool, value_pool, rows, cosine, sine, query_positions,
                    rotary, length, workspace):
    output = torch.empty_like(query)
    splits = min(32, triton.cdiv(length, 1024)) if len(query) == 1 else 1
    if splits > 1:
        from sglang.kernels.ops.attention.decode_attention import _fwd_kernel_stage2
        shape = (1, query.shape[1], splits, query.shape[-1])
        if workspace.get("shape") != shape:
            workspace.update(shape=shape, partial=torch.empty(shape, dtype=torch.float32, device=query.device),
                             lse=torch.empty(shape[:-1], dtype=torch.float32, device=query.device),
                             indptr=torch.tensor([0, length], device=query.device, dtype=torch.int32), length=length)
        if workspace["length"] != length:
            workspace["indptr"][1] = length
            workspace["length"] = length
        partial, lse = workspace["partial"], workspace["lse"]
    else:
        partial, lse = output, output
    _paged_attention[(triton.cdiv(len(query), 16), query.shape[1], splits)](
        query, key_pool, value_pool, rows, cosine, sine, query_positions,
        partial, lse, len(query), length,
        query.stride(0), query.stride(1), key_pool.stride(0), key_pool.stride(1),
        value_pool.stride(0), value_pool.stride(1), partial.stride(0), partial.stride(1),
        cosine.stride(0), query.shape[1] // key_pool.shape[1], query.shape[1],
        query.shape[-1], rotary.cos_sin_cache.shape[-1], rotary.is_neox_style,
        query.shape[-1] ** -.5, 16, 64, splits, num_warps=4)
    if splits > 1:
        _fwd_kernel_stage2[(1, query.shape[1])](partial, lse, output, 1., workspace["indptr"],
            workspace["indptr"], None, partial.stride(0), partial.stride(1), partial.stride(2),
            output.stride(0), output.stride(1), MAX_KV_SPLITS=splits, MIN_BLOCK_KV=64,
            BLOCK_DV=query.shape[-1], Lv=query.shape[-1], HAS_SINK=False, FORCED_KV_SPLITS=splits,
            num_warps=4, num_stages=2)
    return output
