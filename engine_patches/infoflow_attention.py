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
    dimensions = tl.arange(0, HEAD_DIM)
    kv_head = head // GROUPS
    rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    valid_rows = rows < queries
    q = tl.load(Q + rows[:, None] * qs0 + head * qs1 + dimensions[None, :],
                valid_rows[:, None], other=0)
    positions = tl.load(POSITIONS + rows, valid_rows, other=-1)
    output_offsets = rows[:, None] * os0 + head * os1 + split*HEAD_DIM + dimensions[None, :]
    lse_offsets = (rows*HEADS+head)*SPLITS+split
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
    tl.store(OUT + output_offsets, result, valid_rows[:, None])
    if SPLITS > 1:
        tl.store(LSE + lse_offsets, (maximum+tl.log2(denominator))*0.6931471805599453, valid_rows)


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


@triton.jit
def _pack_decode_rows(ROWS, ORDER, POSITIONS, INDPTR, INPUT_LENGTHS, LENGTHS, INDICES,
                      WIDTH: tl.constexpr, SEGMENTS: tl.constexpr, TILE: tl.constexpr):
    request = tl.program_id(0)
    offsets = tl.program_id(1)*TILE + tl.arange(0, TILE)
    valid = offsets <= tl.load(POSITIONS+request)
    positions = tl.load(ORDER+request*WIDTH+offsets, valid, other=0)
    slots = tl.load(ROWS+request*WIDTH+positions, valid, other=0)
    first = tl.load(INDPTR+request*SEGMENTS)
    original = tl.load(INPUT_LENGTHS+request)
    private = tl.load(LENGTHS+request*SEGMENTS)
    generated = tl.load(POSITIONS+request)+1-original
    destination = tl.where(offsets >= original, private+offsets-original,
                           tl.where(offsets >= private, offsets+generated, offsets))
    tl.store(INDICES+first+destination, slots, valid)


def pack_decode_rows(rows, order, positions, indptr, input_lengths, lengths, segments):
    indices = torch.empty(rows.numel(), device=rows.device, dtype=rows.dtype)
    _pack_decode_rows[(len(rows), triton.cdiv(rows.shape[1], 256))](
        rows, order, positions, indptr, input_lengths, lengths, indices, rows.shape[1], segments, 256)
    return indices


@triton.jit
def _segment_queries(Q, COSINE, SINE, OUT, HEADS: tl.constexpr, DIM: tl.constexpr,
                     ROTARY_DIM: tl.constexpr, SEGMENTS: tl.constexpr, NEOX: tl.constexpr,
                     QS0: tl.constexpr, QS1: tl.constexpr, CS0: tl.constexpr, SS0: tl.constexpr):
    segment, head = tl.program_id(0), tl.program_id(1)
    dims = tl.arange(0, DIM)
    if NEOX:
        pair = (dims+ROTARY_DIM//2) % ROTARY_DIM
        frequency = dims % (ROTARY_DIM//2)
        sign = tl.where(dims < ROTARY_DIM//2, -1., 1.)
    else:
        pair = dims ^ 1
        frequency = dims//2
        sign = tl.where(dims % 2 == 0, -1., 1.)
    first = segment//SEGMENTS*QS0+head*QS1
    q = tl.load(Q+first+dims).to(tl.float32)
    paired = tl.load(Q+first+pair).to(tl.float32)
    c = tl.load(COSINE+segment*CS0+frequency, dims < ROTARY_DIM, other=1.)
    s = tl.load(SINE+segment*SS0+frequency, dims < ROTARY_DIM, other=0.)
    tl.store(OUT+(segment*HEADS+head)*DIM+dims, q*c+paired*sign*s)


@triton.jit
def _merge_segments(PARTIAL, LSE, OUT, HEADS: tl.constexpr, DIM: tl.constexpr,
                    SEGMENTS: tl.constexpr, SPLITS: tl.constexpr):
    request, head = tl.program_id(0), tl.program_id(1)
    terms = tl.arange(0, SEGMENTS*SPLITS)
    offsets = ((request*SEGMENTS+terms//SPLITS)*HEADS+head)*SPLITS + terms%SPLITS
    scores = tl.load(LSE+offsets)
    weights = tl.exp(scores-tl.max(scores, 0))
    dims = tl.arange(0, DIM)
    values = tl.load(PARTIAL+offsets[:, None]*DIM+dims[None, :])
    result = tl.sum(weights[:, None]*values, 0)/tl.sum(weights, 0)
    tl.store(OUT+(request*HEADS+head)*DIM+dims, result)


def segmented_decode(query, key_pool, value_pool, indptr, indices, cosine, sine,
                     segments, rotary, workspace):
    """Rotate each block's query once, then reuse native GQA attention and LSE merging.

    Shared blocks retain local RoPE. Private replay and generated tokens form a
    global-position segment. Each context token belongs to exactly one segment.
    """
    from sglang.kernels.ops.attention.decode_attention import _decode_grouped_att_m_fwd
    batch, heads, width = query.shape
    splits = 4
    shape = (batch*segments, heads, splits, width)
    if not workspace:
        workspace.update(partial=torch.zeros(shape, dtype=torch.float32, device=query.device),
            lse=torch.full(shape[:-1], -torch.inf, dtype=torch.float32, device=query.device),
            splits=torch.full((batch*segments,), splits, dtype=torch.int32, device=query.device))
    rotated = query.new_empty((batch*segments, heads, width))
    _segment_queries[(batch*segments, heads)](query, cosine, sine, rotated,
        heads, width, rotary.cos_sin_cache.shape[-1], segments, rotary.is_neox_style,
        query.stride(0), query.stride(1), cosine.stride(0), sine.stride(0), num_warps=4)
    partial, lse = workspace['partial'], workspace['lse']
    _decode_grouped_att_m_fwd(rotated, key_pool, value_pool, partial, lse,
        indptr, indices, workspace['splits'], splits, width**-.5, 0.)
    output = torch.empty_like(query)
    _merge_segments[(batch, heads)](partial, lse, output, heads, width, segments, splits, num_warps=4)
    return output
