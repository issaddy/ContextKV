"""InfoFlow execution over SGLang Qwen3 weights and native KV pages.

Independent source blocks keep chunk-local KV. A request reads those pages at
its global positions and writes recomputed tokens into its own native pages.
The attention-norm selection and sparse replay follow InfoFlow KV (ICML 2026).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
import math

import torch
import torch.nn.functional as F
from sglang.srt.managers.infoflow_attention import paged_attention


def relocate_keys(keys, local_positions, global_positions, rotary):
    """Undo the stored local RoPE and apply the actual request's RoPE."""
    cache = rotary.cos_sin_cache.to(device=keys.device, dtype=torch.float32)
    old_cos, old_sin = cache[local_positions].chunk(2, dim=-1)
    new_cos, new_sin = cache[global_positions].chunk(2, dim=-1)
    denominator = old_cos.square() + old_sin.square()
    cosine = (new_cos * old_cos + new_sin * old_sin) / denominator
    sine = (new_sin * old_cos - new_cos * old_sin) / denominator
    cosine, sine = cosine[:, None, :], sine[:, None, :]
    rotary_dim = cosine.shape[-1] * 2
    value, tail = keys[..., :rotary_dim], keys[..., rotary_dim:]
    if rotary.is_neox_style:
        first, second = value.chunk(2, dim=-1)
        value = torch.cat((first * cosine - second * sine,
                           second * cosine + first * sine), dim=-1)
    else:
        first, second = value[..., ::2], value[..., 1::2]
        value = torch.stack((first * cosine - second * sine,
                             second * cosine + first * sine), dim=-1).flatten(-2)
    return torch.cat((value, tail), dim=-1).to(keys.dtype)


def causal_attention(query, key, value):
    """Contiguous block prefill uses the optimized right-aligned causal backend."""
    from torch.nn.attention.bias import causal_lower_right
    k, v = key.transpose(0, 1)[None], value.transpose(0, 1)[None]
    q = query.transpose(0, 1)[None]
    return F.scaled_dot_product_attention(q, k, v,
        attn_mask=causal_lower_right(len(query), len(key)), enable_gqa=q.shape[1] != k.shape[1])[0].transpose(0, 1)


def attention_norm_squared(query, key, positions, *, query_chunk=32):
    """Accumulate squared head-mean attention; prompt chunks share the L2 norm."""
    from sglang.srt.distributed import tensor_model_parallel_all_reduce
    from sglang.srt.runtime_context import get_parallel
    groups = query.shape[1] // key.shape[1]
    k = key.repeat_interleave(groups, dim=1).transpose(0, 1).float()
    result = torch.zeros(len(key), device=key.device, dtype=torch.float32)
    columns = torch.arange(len(key), device=key.device)
    for start in range(0, len(query), query_chunk):
        q = query[start:start + query_chunk].transpose(0, 1).float()
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(query.shape[-1])
        logits.masked_fill_(columns[None, :] > positions[start:start + query_chunk, None],
                            -torch.inf)
        head_sum = logits.softmax(-1).sum(0)
        world = get_parallel().attn_tp_size
        if world > 1:
            head_sum = tensor_model_parallel_all_reduce(head_sum)
        result += (head_sum / (query.shape[1] * world)).square().sum(0)
    return result


@dataclass
class SourceBlock:
    source_id: tuple[str, str]
    start: int
    end: int
    slots: torch.Tensor | None = None
    ready: int = 0
    node: object = None
    lock: object = None
    owned: bool = False
    work_start: int = 0
    work_end: int = 0
    host_lock: object = None
    host_tokens: int = 0


@dataclass
class RequestState:
    request_id: str
    metadata: dict
    input_ids: tuple[int, ...]
    blocks: list[SourceBlock]
    mode: str
    request: object
    pool: object
    row: torch.Tensor | None = None
    local_positions: torch.Tensor | None = None
    position_cosine: torch.Tensor | None = None
    position_sine: torch.Tensor | None = None
    attention_workspace: dict = field(default_factory=dict)
    selected: torch.Tensor | None = None
    replayed: torch.Tensor | None = None
    prompt: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    complete: bool = False
    counters: dict = field(default_factory=lambda: dict(
        cold_tokens=0, reused_tokens=0, recomputed_tokens=0, replay_tokens=0, score_tokens=0,
        host_load_tokens=0, device_hit_tokens=0, private_tokens=0, new_compute_kv_pages=0))

    def layer_kv(self, layer_id, rotary, length, *, overlay=True):
        """Materialize one layer, keeping base pages immutable across requests."""
        keys, values = self.pool.get_kv_buffer(layer_id)
        k = keys[self.row[:length]].clone()
        v = values[self.row[:length]].clone()
        for block in self.blocks:
            end = min(block.end, length)
            if end <= block.start:
                continue
            count = end - block.start
            local = torch.arange(count, device=k.device)
            global_positions = local + block.start
            k[block.start:end] = relocate_keys(keys[block.slots[:count]], local,
                                               global_positions, rotary)
            v[block.start:end] = values[block.slots[:count]]
        if overlay and self.replayed is not None:
            selected = self.replayed[self.replayed < length]
            k[selected] = keys[self.row[selected]]
            v[selected] = values[self.row[selected]]
        return k, v


class QwenInfoFlow:
    """Run chunk construction, query scoring, sparse replay and decoding."""

    def __init__(self, model, forward_batch):
        self.model, self.forward_batch = model, forward_batch
        self.layers = model.layers[model.start_layer:model.end_layer]

    @staticmethod
    def _qkv(layer, hidden, positions):
        attn = layer.self_attn
        normalized = layer.input_layernorm(hidden)
        q, k, v = attn.forward_prepare_native(positions, normalized)
        return (q.view(-1, attn.num_heads, attn.head_dim),
                k.view(-1, attn.num_kv_heads, attn.head_dim),
                v.view(-1, attn.num_kv_heads, attn.head_dim))

    def _finish_layer(self, layer, hidden, attention):
        from sglang.srt.distributed import tensor_model_parallel_all_reduce
        from sglang.srt.runtime_context import get_parallel

        out, _ = layer.self_attn.o_proj(attention.flatten(1))
        if get_parallel().attn_tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        hidden = hidden + out
        out = layer.mlp(layer.post_attention_layernorm(hidden), forward_batch=self.forward_batch)
        if get_parallel().tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return hidden + out

    def _attend(self, state, layer, query, positions, length):
        keys, values = state.pool.get_kv_buffer(layer.self_attn.attn.layer_id)
        if state.position_cosine is None:
            cache = layer.self_attn.rotary_emb.cos_sin_cache
            local_cos, local_sin = cache[state.local_positions].float().chunk(2, -1)
            global_cos, global_sin = cache[:len(state.row)].float().chunk(2, -1)
            norm = local_cos.square()+local_sin.square()
            state.position_cosine = (global_cos*local_cos+global_sin*local_sin)/norm
            state.position_sine = (global_sin*local_cos-global_cos*local_sin)/norm
        return paged_attention(query, keys, values, state.row, state.position_cosine, state.position_sine,
                               positions, layer.self_attn.rotary_emb, length, state.attention_workspace)

    @torch.inference_mode()
    def build(self, state, segments):
        """Share Qwen projections/MLPs across the independent blocks ready in this chunk."""
        device = segments[0][0].slots.device
        spans = [(block, block.ready, last) for block, last in segments]
        ids = torch.tensor([token for block, first, last in spans
                            for token in state.input_ids[block.start+first:block.start+last]],
                           device=device, dtype=torch.int64)
        positions = torch.cat([torch.arange(first, last, device=device) for _, first, last in spans])
        slots = torch.cat([block.slots[first:last] for block, first, last in spans])
        hidden = self.model.embed_tokens(ids)
        for layer in self.layers:
            layer_id = layer.self_attn.attn.layer_id
            q, k, v = self._qkv(layer, hidden, positions)
            keys, values = state.pool.get_kv_buffer(layer_id)
            keys[slots], values[slots] = k, v
            attention, offset = [], 0
            for block, first, last in spans:
                attention.append(causal_attention(q[offset:offset+last-first],
                    keys[block.slots[:last]], values[block.slots[:last]]))
                offset += last-first
            hidden = self._finish_layer(layer, hidden, torch.cat(attention))
        for block, first, last in spans:
            block.ready = last
            state.counters['cold_tokens'] += last-first

    @torch.inference_mode()
    def score(self, state, prompt):
        device, count = state.row.device, int(prompt[-1]) + 1
        query_start, query_end = state.metadata['query_span']
        query_rows = (prompt >= query_start) & (prompt < query_end)
        ids = torch.tensor(state.input_ids, device=device, dtype=torch.int64)
        hidden = self.model.embed_tokens(ids[prompt])
        # The author's default aggregates the last two layers.
        for index, layer in enumerate(self.layers):
            layer_id = layer.self_attn.attn.layer_id
            q, k, v = self._qkv(layer, hidden, prompt)
            keys, values = state.pool.get_kv_buffer(layer_id)
            keys[state.row[prompt]], values[state.row[prompt]] = k, v
            if index >= len(self.layers) - 2 and count > query_start:
                all_k, _ = state.layer_kv(layer_id, layer.self_attn.rotary_emb,
                                          count, overlay=False)
                state.scores[index-len(self.layers)+2, :count] += attention_norm_squared(
                    q[query_rows], all_k, prompt[query_rows])
            hidden = self._finish_layer(layer, hidden,
                self._attend(state, layer, q, prompt, count))
        state.counters['score_tokens'] += len(prompt)

    @torch.inference_mode()
    def select(self, state):
        count, device = len(state.input_ids), state.row.device
        context = torch.zeros(count, dtype=torch.bool, device=device)
        for block in state.blocks:
            context[block.start:block.end] = True
        scores = state.scores.sqrt().mean(0)
        context_positions = context.nonzero().flatten()
        ratio = float(state.metadata.get('recompute_ratio', .15))
        amount = min(len(context_positions), math.ceil(len(context_positions) * ratio))
        selected = context_positions[scores[context_positions].topk(amount).indices]
        state.selected = torch.cat((state.prompt, selected)).sort().values
        state.position_cosine[state.selected] = 1
        state.position_sine[state.selected] = 0
        state.replayed = state.selected[:0]
        state.counters['private_tokens'] = len(state.selected)

    @torch.inference_mode()
    def replay(self, state, positions):
        ids = torch.tensor(state.input_ids, device=positions.device, dtype=torch.int64)
        hidden = self.model.embed_tokens(ids[positions])
        for layer in self.layers:
            layer_id = layer.self_attn.attn.layer_id
            q, k, v = self._qkv(layer, hidden, positions)
            keys, values = state.pool.get_kv_buffer(layer_id)
            keys[state.row[positions]], values[state.row[positions]] = k, v
            hidden = self._finish_layer(layer, hidden,
                self._attend(state, layer, q, positions, int(positions[-1]) + 1))
        state.replayed = torch.cat((state.replayed, positions))
        state.counters['replay_tokens'] += len(positions)
        state.counters['recomputed_tokens'] += int((~torch.isin(positions, state.prompt)).sum())
        state.complete = len(state.replayed) == len(state.selected)
        return self.model.norm(hidden)[-1]

    @torch.inference_mode()
    def decode(self, state, input_id, position):
        """Reuse one decode graph per length bucket, updating only row/position maps."""
        size = len(state.row)
        bucket = 1 << (size-1).bit_length()
        graphs = state.runtime.decode_graphs
        if bucket not in graphs:
            width = state.position_cosine.shape[1]
            graphs[bucket] = SimpleNamespace(pool=state.pool,
                row=state.row.new_zeros(bucket),
                position_cosine=state.position_cosine.new_ones((bucket, width)),
                position_sine=state.position_sine.new_zeros((bucket, width)), attention_workspace={},
                input=input_id.reshape(1).clone(), position=position.reshape(1).clone(), graph=None)
        fixed = graphs[bucket]
        fixed.row[:size].copy_(state.row)
        fixed.position_cosine[:size].copy_(state.position_cosine)
        fixed.position_sine[:size].copy_(state.position_sine)
        fixed.input.copy_(input_id.reshape(1))
        fixed.position.copy_(position.reshape(1))
        if fixed.graph is None:
            self._decode(fixed, fixed.input, fixed.position)
            fixed.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(fixed.graph):
                fixed.output = self._decode(fixed, fixed.input, fixed.position)
        fixed.graph.replay()
        return fixed.output

    def _decode(self, state, input_id, position):
        positions = position.reshape(1)
        hidden = self.model.embed_tokens(input_id.reshape(1))
        length = len(state.row)
        for layer in self.layers:
            layer_id = layer.self_attn.attn.layer_id
            q, k, v = self._qkv(layer, hidden, positions)
            keys, values = state.pool.get_kv_buffer(layer_id)
            keys[state.row[positions]], values[state.row[positions]] = k, v
            hidden = self._finish_layer(layer, hidden,
                self._attend(state, layer, q, positions, length))
        return self.model.norm(hidden)[0]
