"""One-submit evidence execution, owned by the SGLang scheduler and worker."""
from __future__ import annotations

from copy import copy
from array import array
import math
import time

import torch

from sglang.srt.managers.infoflow import QwenInfoFlow, RequestState, SourceBlock
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey


class ComputeRuntime:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.cache = scheduler.tree_cache
        self.allocator = scheduler.token_to_kv_pool_allocator
        self.pool = self.allocator.get_kvcache()
        self.costs = {}
        self.output_tokens_mean = 0.0
        self.demands = {}
        self.source_nodes = {}
        self.prioritized_nodes = set()
        self.gpu_token_seconds = self.host_token_seconds = 0.0
        self.observed_at = time.monotonic()
        self.pool_used = self._pool_used()
        keys, values = self.pool.get_kv_buffer(0)
        layers = scheduler.tp_worker.model_runner.model.config.num_hidden_layers
        self.bytes_per_token = (keys[0].numel() * keys.element_size() +
                                values[0].numel() * values.element_size()) * layers

    def _pool_used(self):
        host = self.cache.cache_controller.mem_pool_host if self.cache.cache_controller else None
        return (self.allocator.size - self.allocator.available_size(),
                host.size - host.available_size() if host else 0)

    def observe_pool(self):
        now = time.monotonic()
        elapsed = now - self.observed_at
        self.gpu_token_seconds += elapsed * self.pool_used[0]
        self.host_token_seconds += elapsed * self.pool_used[1]
        self.observed_at, self.pool_used = now, self._pool_used()

    def control(self, action, data):
        if action == 'info':
            host = self.cache.cache_controller.mem_pool_host if self.cache.cache_controller else None
            return dict(protocol_version='contextkv-compute-v2', engine_kind='sglang',
                model_path=self.scheduler.server_args.model_path, page_size=self.cache.page_size,
                gpu_capacity_tokens=self.allocator.size, host_capacity_tokens=host.size if host else 0)
        if action == 'profile_prepare':
            flushed = self.scheduler.flush_cache()
            if flushed:
                self.demands.clear()
                self.source_nodes.clear()
                self.prioritized_nodes.clear()
                self.gpu_token_seconds = self.host_token_seconds = 0.0
                self.observed_at, self.pool_used = time.monotonic(), self._pool_used()
            return {'flushed': flushed}
        self.observe_pool()
        return dict(pool=dict(gpu_used_tokens=self.pool_used[0], host_used_tokens=self.pool_used[1],
                              bytes_per_token=self.bytes_per_token),
            gpu_token_seconds=self.gpu_token_seconds, host_token_seconds=self.host_token_seconds,
            gpu_kv_gib_seconds=self.gpu_token_seconds * self.bytes_per_token / 2**30,
            host_kv_gib_seconds=self.host_token_seconds * self.bytes_per_token / 2**30,
            costs=dict(self.costs),
            observed_time_ns=time.monotonic_ns())

    def _key(self, source_id, ids):
        return RadixKey(array('q', ids), cache_salt='contextkv:independent-block:' + source_id[0])

    def _match(self, source_id, ids):
        return self.cache.match_prefix(MatchPrefixParams(key=self._key(source_id, ids)))

    def _indices(self, match):
        return self.cache.path_device_indices(match.last_device_node)[:match.device_prefix_len]

    def _lock(self, node):
        return self.cache.inc_lock_ref(node).to_dec_params()

    def _unlock(self, block):
        self.cache.dec_lock_ref(block.node, block.lock)

    def admit(self, req, metadata):
        """Select work using current residency; no external cache-control call."""
        metadata = dict(metadata)
        session = metadata['session_id']
        scope = metadata.get('cache_scope', 'shared')
        self.demands[session] = {(scope, item['source_id']): item for item in metadata.get('demand', ())}
        self._refresh_priorities()
        ids = tuple(req.origin_input_ids)
        page = self.cache.page_size
        blocks, warm, cold, host_pages, gpu_nodes = [], 0, 0, set(), set()
        new_gpu_pins = 0
        for spec in metadata.get('blocks', ()):
            start = int(spec['start'])
            end = int(spec['end'])
            if end == start:
                continue
            block_ids = list(ids[start:end])
            block_ids += [0] * (-len(block_ids) % page)
            source_id = scope, spec['source_id']
            match = self._match(source_id, block_ids)
            full = match.device_prefix_len + match.host_hit_length == len(block_ids)
            warm += end - start if full else 0
            cold += 0 if full else end - start
            if full and match.host_hit_length:
                transfer, _ = self.cache.tree_core.build_load_back_spec(match.best_match_node, req=req)
                host_pages.update(transfer.host_indices.tolist())
            if full:
                node = self.cache.tree_core.node_by_id(match.last_device_node)
                while node.parent is not None and node.id not in gpu_nodes:
                    gpu_nodes.add(node.id)
                    component = node.component_data[0]
                    if component.value is not None and component.lock_ref == 0:
                        new_gpu_pins += len(component.value)
                    node = node.parent
            blocks.append((source_id, start, end, match, block_ids))
        costs = dict(metadata.get('costs', {}))
        costs.update(self.costs)
        prefill = costs.get('prefill_token_ms', 1.0)
        build = costs.get('build_token_ms', prefill)
        score = costs.get('score_token_ms', prefill)
        replay = costs.get('recompute_token_ms', prefill)
        transfer = costs.get('transfer_token_ms', prefill * .05)
        ratio = float(metadata.get('recompute_ratio', .15))
        prompt = len(ids) - warm - cold
        approximate = cold * build + prompt * score + (prompt + (warm + cold) * ratio) * replay + len(host_pages) * transfer
        native = self.cache.match_prefix(MatchPrefixParams(key=RadixKey(
            array('q', ids), req.extra_key, cache_salt=req.cache_salt)))
        exact = (len(ids) - native.device_prefix_len - native.host_hit_length) * prefill + native.host_hit_length * transfer
        expected_output = min(req.sampling_params.max_new_tokens,
                              self.output_tokens_mean or req.sampling_params.max_new_tokens)
        exact_decode = costs.get('exact_decode_token_ms', 0.0)
        approximate += expected_output * costs.get('infoflow_decode_token_ms', exact_decode)
        exact += expected_output * exact_decode
        # A future consumer saves each source block's work once. Demand for
        # another block never finances this block's independent construction.
        cold_consumers = {}
        for source_id, start, end, match, block_ids in blocks:
            item = self.demands[session].get(source_id)
            if item is not None and match.device_prefix_len+match.host_hit_length < len(block_ids):
                consumers = cold_consumers.setdefault(tuple(block_ids), (end-start, set()))[1]
                consumers.update(item.get('consumer_ids', ()) or range(int(item.get('uses', 0))))
        build_value = sum(length * len(consumers) * build for length, consumers in cold_consumers.values())
        quality_loss = float(metadata.get('quality_loss', 0.0))
        quality_price_ms = float(metadata.get('quality_price_ms', 0.0))
        mode = metadata.get('mode', 'auto')
        if mode == 'auto':
            mode = ('infoflow' if 'prefill_token_ms' in costs and blocks
                    and approximate + quality_loss * quality_price_ms - build_value < exact
                    else 'exact')
            cold_work = sum(len(tokens) for _, _, _, match, tokens in blocks
                            if match.device_prefix_len+match.host_hit_length < len(tokens))
            required = cold_work + math.ceil(prompt/page)*page + prompt + math.ceil((warm + cold)*ratio)
            if required > min(self.scheduler.max_req_input_len,
                    self.allocator.available_size() + self.cache.evictable_size()
                    - req.sampling_params.max_new_tokens - len(host_pages) - new_gpu_pins):
                mode = 'exact'
        state = RequestState(req.rid, metadata, ids, [], mode, req, self.pool)
        state.runtime = self
        state.events = []
        state.published = False
        state.native_work = 0
        state.transferred_ranges = []
        state.host_pages = host_pages if mode == 'infoflow' else set()
        req.contextkv_compute_state = state
        if mode == 'infoflow':
            # Approximate request pages never enter the exact prefix namespace.
            req.cache_salt = 'contextkv:request:' + req.rid
            req.skip_radix_cache_insert = True
            work = []
            source_positions = set()
            for source_id, start, end, match, block_ids in blocks:
                source_positions.update(range(start, end))
                block = SourceBlock(source_id, start, end)
                block.cache_ids = block_ids
                full = match.device_prefix_len + match.host_hit_length == len(block_ids)
                if full:
                    block.node = match.best_match_node
                    self.source_nodes[source_id] = block.node
                    if match.host_hit_length:
                        block.host_tokens = match.host_hit_length
                    else:
                        block.slots = self._indices(match)
                        block.ready = end - start
                        state.counters['device_hit_tokens'] += end - start
                    state.counters['reused_tokens'] += end - start
                else:
                    block.owned = True
                    block.work_start = len(work)
                    work.extend(block_ids)
                    block.work_end = len(work)
                state.blocks.append(block)
            state.prompt_positions = [index for index in range(len(ids)) if index not in source_positions]
            state.score_start = len(work)
            work.extend(ids[index] for index in state.prompt_positions)
            work.extend([0] * (-len(work) % page))
            state.score_end = len(work)
            state.replay_start = len(work)
            replay_count = len(state.prompt_positions) + math.ceil(len(source_positions) * ratio)
            work.extend([0] * replay_count)
            state.replay_end = len(work)
            state.work_length = len(work)
            # The scheduler batches actual construction/scoring/replay work.
            # Source addresses and global positions remain in state.input_ids.
            req.origin_input_ids = array('q', work)
            req.full_untruncated_fill_ids = array('q', work)
            self._refresh_priorities()
        return state

    def pin(self, state):
        for block in state.blocks:
            if not block.owned:
                block.lock = self._lock(block.node)
                if block.host_tokens:
                    block.host_lock = self.cache.inc_host_lock_ref(block.node).to_dec_params()

    def unpin(self, state):
        for block in state.blocks:
            if block.lock is not None:
                self._unlock(block)
                block.lock = None
            if block.host_lock is not None:
                self.cache.dec_host_lock_ref(block.node, block.host_lock)
                block.host_lock = None

    def prepare(self, state):
        for block in state.blocks:
            if block.owned or block.slots is not None:
                continue
            if block.host_tokens:
                original_node = block.node
                current = self._match(block.source_id, block.cache_ids)
                if current.device_prefix_len == len(block.cache_ids) and not current.host_hit_length:
                    self._unlock(block)
                    block.node, block.slots = current.last_device_node, self._indices(current)
                    block.lock = self._lock(block.node)
                    block.ready = block.end - block.start
                    self.cache.dec_host_lock_ref(original_node, block.host_lock)
                    block.host_lock = None
                    state.counters['device_hit_tokens'] += block.end - block.start
                    continue
                started, loaded = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                started.record()
                self.cache.load_back(block.node, req=state.request)
                self.cache.cache_controller.start_loading()
                for ack in self.cache.cache_controller.ack_load_queue:
                    if block.node in ack.node_ids:
                        torch.cuda.current_stream().wait_event(ack.finish_event)
                loaded.record()
                state.events.append(('transfer', started, loaded, block.host_tokens))
                match = self._match(block.source_id, block.cache_ids)
                block.slots = self._indices(match)
                block.ready = block.end - block.start
                self._unlock(block)
                block.node = match.last_device_node
                block.lock = self._lock(block.node)
                self.cache.dec_host_lock_ref(original_node, block.host_lock)
                block.host_lock = None
                state.counters['host_load_tokens'] += block.host_tokens
        state.host_pages.clear()

    def publish_blocks(self, state):
        for block in state.blocks:
            if (not block.owned or block.ready != block.end - block.start
                    or block.work_end > state.native_work):
                continue
            for layer in state.layer_ids:
                keys, values = self.pool.get_kv_buffer(layer)
                keys[block.slots[block.ready:]] = 0
                values[block.slots[block.ready:]] = 0
            result = self.cache.insert(InsertParams(
                key=self._key(block.source_id, block.cache_ids),
                value=block.slots.clone(), priority=self._priority(block.source_id)))
            block.node = result.last_device_node
            block.lock = self._lock(block.node)
            block.slots = self._indices(self._match(block.source_id, block.cache_ids))
            state.row[block.start:block.end] = block.slots[:block.end-block.start]
            block.owned = False
            state.transferred_ranges.append((block.work_start, block.work_end))
            self.source_nodes[block.source_id] = block.node

    def _priority(self, source_id):
        consumers, anonymous = set(), 0
        for session, scope in self.demands.items():
            item = scope.get(source_id)
            if item is None:
                continue
            ids = item.get('consumer_ids', ())
            consumers.update((session, consumer) for consumer in ids)
            if not ids:
                anonymous += int(item.get('uses', 0))
        avoided_ms = self.costs.get('build_token_ms', 0.0)
        return int(1000 * (len(consumers) + anonymous) * avoided_ms)

    def _refresh_priorities(self):
        self.demands = {session: scope for session, scope in self.demands.items() if scope}
        for node_id in self.prioritized_nodes:
            node = self.cache.tree_core._node_arena.get(node_id)
            if node is not None:
                node.priority = 0
        self.prioritized_nodes.clear()
        nodes = {}
        for source_id, node_id in tuple(self.source_nodes.items()):
            node = self.cache.tree_core._node_arena.get(node_id)
            if node is None:
                del self.source_nodes[source_id]
            else:
                consumers = set()
                for session, scope in self.demands.items():
                    item = scope.get(source_id)
                    if item:
                        consumers.update((session, consumer) for consumer in
                            (item.get('consumer_ids', ()) or range(int(item.get('uses', 0)))))
                while node.parent is not None:
                    nodes.setdefault(node.id, set()).update(consumers)
                    node = node.parent
        for node_id, consumers in nodes.items():
            priority = int(1000 * len(consumers) * self.costs.get('build_token_ms', 0.0))
            self.cache.tree_core.node_by_id(node_id).priority = priority
            if priority:
                self.prioritized_nodes.add(node_id)

    def ready_tokens(self, state):
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

        if state.mode == 'infoflow':
            nodes = [block.node for block in state.blocks if not block.owned]
        else:
            match = self.cache.match_prefix(MatchPrefixParams(key=RadixKey(
                array('q', state.input_ids), state.request.extra_key, cache_salt=state.request.cache_salt)))
            nodes = [match.best_match_node]
        seen, gpu, host = set(), 0, 0
        for node_id in nodes:
            node = self.cache.tree_core.node_by_id(node_id)
            while node is not None and node.id not in seen:
                seen.add(node.id)
                full = node.component_data[ComponentType.FULL]
                gpu += len(full.value) if full.value is not None else 0
                host += (len(full.host_value) if full.host_value is not None
                         and node.write_through_pending_id is None else 0)
                node = node.parent
        return gpu, host

    def finish(self, req):
        state = req.contextkv_compute_state
        if state.published:
            return
        state.published = True
        self.publish_blocks(state)
        self.unpin(state)
        phase_ms = {}
        for phase, first, last, tokens in state.events:
            elapsed = first.elapsed_time(last)
            phase_ms[phase] = phase_ms.get(phase, 0.0) + elapsed
            if tokens:
                name = phase + '_token_ms'
                measured = elapsed / tokens
                self.costs[name] = .8 * self.costs.get(name, measured) + .2 * measured
        state.events.clear()
        stats = req.time_stats
        elapsed = (stats.completion_time - stats.scheduler_recv_time) * 1000
        output_count = len(req.output_ids_through_stop)
        self.output_tokens_mean = (.8 * (self.output_tokens_mean or output_count) + .2 * output_count)
        if output_count > 1:
            name = state.mode + '_decode_token_ms'
            measured = (stats.completion_time - stats.prefill_finished_time) * 1000 / (output_count - 1)
            self.costs[name] = .8 * self.costs.get(name, measured) + .2 * measured
        if state.mode == 'exact' and (prefill_tokens := len(state.input_ids) - req.cached_tokens):
            measured = (stats.prefill_finished_time - stats.forward_entry_time) * 1000 / prefill_tokens
            self.costs['prefill_token_ms'] = .8 * self.costs.get('prefill_token_ms', measured) + .2 * measured
        self._refresh_priorities()
        host_pool = self.cache.cache_controller.mem_pool_host if self.cache.cache_controller else None
        receipt = dict(mode=state.mode, input_tokens=len(state.input_ids),
            output_tokens=output_count, elapsed_ms=elapsed,
            **state.counters, **{name + '_ms': value for name, value in phase_ms.items()},
            gpu_available_tokens=self.allocator.available_size(),
            host_available_tokens=host_pool.available_size() if host_pool else 0,
            actual_prefill_tokens=(state.counters['cold_tokens'] + state.counters['score_tokens'] +
                state.counters['replay_tokens'] if state.mode == 'infoflow'
                else len(state.input_ids) - req.cached_tokens),
            cached_tokens=state.counters['reused_tokens'] if state.mode == 'infoflow' else req.cached_tokens)
        host_loaded = state.counters['host_load_tokens'] if state.mode == 'infoflow' else req.cached_tokens_host
        gpu_ready, host_ready = self.ready_tokens(state)
        receipt.update(prefill_tokens=receipt['actual_prefill_tokens'],
            device_hit_tokens=state.counters['device_hit_tokens'] if state.mode == 'infoflow' else req.cached_tokens_device,
            host_loaded_tokens=host_loaded,
            queue_depth=len(self.scheduler.waiting_queue),
            gpu_ready_tokens=gpu_ready, host_ready_tokens=host_ready, **self.costs)
        req.customized_info = req.customized_info or {}
        req.customized_info['contextkv_compute'] = [receipt]


def get_compute_runtime(scheduler):
    runtime = getattr(scheduler, 'contextkv_compute_runtime', None)
    if runtime is None:
        runtime = ComputeRuntime(scheduler)
        scheduler.contextkv_compute_runtime = runtime
    return runtime


def admit_compute(scheduler, recv, req):
    req.contextkv_compute_metadata = recv.contextkv_compute
    req.contextkv_native_cache_salt = req.cache_salt


def prepare_compute_request(scheduler, req):
    metadata = getattr(req, 'contextkv_compute_metadata', None)
    if metadata is None:
        return
    runtime = get_compute_runtime(scheduler)
    previous = getattr(req, 'contextkv_compute_state', None)
    if previous is not None:
        runtime.unpin(previous)
        req.origin_input_ids = array('q', previous.input_ids)
        req.full_untruncated_fill_ids = req.origin_input_ids + req.output_ids
    req.cache_salt = req.contextkv_native_cache_salt
    req.skip_radix_cache_insert = False
    state = runtime.admit(req, metadata)
    if previous is not None and previous.events:
        state.events = previous.events
        for name in state.counters:
            state.counters[name] += previous.counters[name]
    runtime.pin(state)


def pending_host_tokens(adder, req):
    state = getattr(req, 'contextkv_compute_state', None)
    return len(state.host_pages - getattr(adder, 'contextkv_host_pages', set())) if state else 0


def commit_host_reserve(adder, req):
    state = getattr(req, 'contextkv_compute_state', None)
    if state is not None:
        amount = pending_host_tokens(adder, req)
        adder.memory_budget.total_offset += amount
        adder.memory_budget.current_offset += amount
        adder.contextkv_host_pages = getattr(adder, 'contextkv_host_pages', set()) | state.host_pages


def finish_compute_admission(adder, req):
    state = getattr(req, 'contextkv_compute_state', None)
    if state is None:
        return
    if adder.can_run_list and adder.can_run_list[-1] is req:
        commit_host_reserve(adder, req)
    else:
        state.runtime.unpin(state)


def attach_compute_batch(forward_batch, batch, model_runner):
    forward_batch.contextkv_translator = model_runner.kv_index_translator
    forward_batch.contextkv_states = [getattr(req, 'contextkv_compute_state', None) for req in batch.reqs]
    forward_batch.contextkv_rows = batch.req_to_token_pool.req_to_token[batch.req_pool_indices]


def _measure(state, phase, tokens, operation):
    first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    first.record()
    result = operation()
    last.record()
    state.events.append((phase, first, last, tokens))
    return result


def _single_batch(batch, index, offset, length):
    """The native backend receives the same one-request metadata it normally builds."""
    result = copy(batch)
    result.batch_size = 1
    result.input_ids = batch.input_ids[offset:offset + length]
    result.positions = batch.positions[offset:offset + length]
    result.out_cache_loc = batch.out_cache_loc[offset:offset + length]
    result.req_pool_indices = batch.req_pool_indices[index:index + 1]
    result.seq_lens = batch.seq_lens[index:index + 1]
    result.orig_seq_lens = batch.orig_seq_lens[index:index + 1]
    result.seq_lens_cpu = batch.seq_lens_cpu[index:index + 1]
    result.seq_lens_sum = int(result.seq_lens_cpu[0])
    result.max_seq_len_host = result.seq_lens_sum
    result.num_token_non_padded_cpu = length
    if batch.forward_mode.is_extend():
        result.extend_num_tokens = length
        result.extend_seq_lens = batch.extend_seq_lens[index:index + 1]
        result.extend_prefix_lens = batch.extend_prefix_lens[index:index + 1]
        result.extend_start_loc = torch.zeros_like(result.extend_seq_lens)
        result.extend_seq_lens_cpu = [length]
        result.extend_prefix_lens_cpu = [batch.extend_prefix_lens_cpu[index]]
        result.extend_logprob_start_lens_cpu = [batch.extend_logprob_start_lens_cpu[index]]
    from sglang.srt.model_executor.forward_context import get_attn_backend
    translator = batch.contextkv_translator
    cols = batch.kv_loc_plan.cols_slice(batch.kv_loc_cols, slice(offset, offset + length))
    result.out_cache_loc = batch.kv_loc_plan.virtual_write_ids(cols=cols)
    translator.own_plan(result).bind(result, translator)
    get_attn_backend().init_forward_metadata(result)
    return result


def forward_qwen(model, native_forward, input_ids, positions, batch, input_embeds=None,
                 pp_proxy_tensors=None):
    states = getattr(batch, 'contextkv_states', ())
    if not any(state is not None and state.mode == 'infoflow' for state in states):
        return native_forward(input_ids, positions, batch, input_embeds, pp_proxy_tensors)
    lengths = batch.extend_seq_lens_cpu if batch.forward_mode.is_extend() else [1] * batch.batch_size
    result = torch.zeros((len(input_ids), model.config.hidden_size),
                         dtype=model.embed_tokens.weight.dtype, device=input_ids.device)
    offset = 0
    driver = QwenInfoFlow(model, batch)
    for index, (state, length) in enumerate(zip(states, lengths)):
        if state is None or state.mode == 'exact':
            from sglang.srt.model_executor.forward_context import forward_context, get_forward_context
            one = _single_batch(batch, index, offset, length)
            with forward_context(get_forward_context(), forward_batch=one):
                result[offset:offset + length] = native_forward(
                    one.input_ids, one.positions, one, None, pp_proxy_tensors)
        else:
            work_row = batch.contextkv_rows[index]
            state.runtime.prepare(state)
            state.layer_ids = [layer.self_attn.attn.layer_id for layer in driver.layers]
            if state.replayed is not None:
                count = len(state.replayed)
                state.row[state.replayed] = work_row[state.replay_start:state.replay_start + count]
                decoded = max(0, int(batch.seq_lens_cpu[index]) - state.work_length)
                state.row[len(state.input_ids):len(state.input_ids) + decoded] = work_row[
                    state.work_length:state.work_length + decoded]
            prefilling = (batch.forward_mode.is_extend()
                          and batch.extend_prefix_lens_cpu[index] < state.work_length)
            if prefilling:
                first = batch.extend_prefix_lens_cpu[index]
                end = first + length
                if first == 0:
                    state.row = torch.zeros(len(state.input_ids) + state.request.sampling_params.max_new_tokens + 1,
                                            device=input_ids.device, dtype=torch.int64)
                    state.local_positions = torch.arange(len(state.row), device=input_ids.device)
                    state.prompt = torch.tensor(state.prompt_positions, device=input_ids.device)
                    state.scores = torch.zeros(len(state.input_ids), device=input_ids.device)
                    state.selected = state.replayed = None
                    state.complete = False
                    state.transferred_ranges = []
                state.native_work = end
                for block in state.blocks:
                    if block.owned:
                        block.slots = work_row[block.work_start:block.work_end]
                        amount = min(end - block.work_start, block.end - block.start)
                        if amount > block.ready:
                            _measure(state, 'build', amount - block.ready,
                                lambda b=block, n=amount: driver.build(state, b, b.start + n))
                    if state.selected is None:
                        state.row[block.start:block.end] = block.slots[:block.end-block.start]
                        state.local_positions[block.start:block.end] = torch.arange(
                            block.end-block.start, device=input_ids.device)
                state.runtime.publish_blocks(state)
                score_first = max(0, first - state.score_start)
                score_end = min(len(state.prompt), end - state.score_start)
                if score_first < score_end:
                    prompt = state.prompt[score_first:score_end]
                    state.row[prompt] = work_row[state.score_start + score_first:state.score_start + score_end]
                    _measure(state, 'score', len(prompt), lambda: driver.score(state, prompt))
                if end >= state.score_end and state.selected is None:
                    driver.select(state)
                replay_first = max(0, first - state.replay_start)
                replay_end = min(state.replay_end, end) - state.replay_start
                if replay_first < replay_end:
                    selected = state.selected[replay_first:replay_end]
                    state.row[selected] = work_row[state.replay_start + replay_first:state.replay_start + replay_end]
                    state.local_positions[selected] = selected
                    last_hidden = _measure(state, 'recompute', len(selected),
                                           lambda: driver.replay(state, selected))
                    if end >= state.work_length:
                        result[offset + length - 1] = last_hidden
                for position in range(max(first, state.work_length), end):
                    global_position = positions.new_tensor(position - state.work_length + len(state.input_ids))
                    state.row[global_position] = work_row[position]
                    result[offset + position - first] = driver.decode(
                        state, input_ids[offset + position - first], global_position)
            else:
                for local in range(length):
                    global_position = positions[offset + local] - state.work_length + len(state.input_ids)
                    state.row[global_position] = work_row[positions[offset + local]]
                    result[offset + local] = driver.decode(state, input_ids[offset + local], global_position)
        offset += length
    return result


def private_release_segments(req, start, end):
    """Native insert owns the adopted pages and has already freed duplicates."""
    state = getattr(req, 'contextkv_compute_state', None)
    if state is None or state.mode == 'exact':
        return [(start, end)]
    segments, cursor = [], start
    for first, last in sorted(state.transferred_ranges):
        if cursor < min(first, end):
            segments.append((cursor, min(first, end)))
        cursor = max(cursor, last)
    if cursor < end:
        segments.append((cursor, end))
    return segments
