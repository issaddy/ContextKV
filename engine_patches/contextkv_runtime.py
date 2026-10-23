"""Scheduler-owned ContextKV protocol, for the pinned SGLang Python backend.

This module deliberately has no background thread or GPU synchronizations. It is
called only from scheduler dispatch and its terminal output callback. The FULL
GPU/host read view uses observation-only transfer markers; it never processes
ACKs, takes locks, splits nodes or treats queued transfers as ready GPU copies.
"""
from __future__ import annotations

from array import array
from collections import OrderedDict, deque
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

COMMIT = "3f36fc08331f64cd63c4fbbdf39cc402f38bb38b"
PROTOCOL = "contextkv-v1"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
MODEL_CONFIG_HASH = "b6891f5037cc0853d4325d96957f76765bb3ec11b01eee59670e33b825694a06"


def snapshot_extend_work(batch):
    """Snapshot actual scheduled extend lengths; mixed decode rows are excluded."""
    if not batch.forward_mode.is_extend():
        return ()
    decode_rids = {r.rid for r in (batch.decoding_reqs or ())}
    return tuple((req, int(length)) for req, length in zip(batch.reqs, batch.extend_lens)
                 if hasattr(req, "contextkv_runtime") and req.rid not in decode_rids)


def complete_extend_work(work):
    # Called once after a successful worker forward. Chunking and retraction
    # replays each add their real work; no inference from prompt-hit counts.
    for req, length in work:
        req.contextkv_prefill_tokens += length



def native_timing(stats, output_count):
    """Native scheduler completion timestamps, never CPU launch/CUDA timings.

    forward_entry follows cache preparation in PrefillAdder, so it cannot split
    queue from restoration. Keep those unavailable instead of mislabeling them.
    """
    def interval(start, end):
        return (end-start)*1000 if start > 0 and end >= start else None
    received = getattr(stats, "scheduler_recv_time", 0)
    first = getattr(stats, "prefill_finished_time", 0)
    last = getattr(stats, "completion_time", 0)
    forward = getattr(stats, "forward_entry_time", 0)
    decode = interval(first, last)
    return dict(timing_source="sglang_server_monotonic",
        server_ttft_ms=interval(received, first),
        server_elapsed_ms=interval(received, last),
        forward_to_first_token_ms=interval(forward, first),
        decode_ms=decode,
        tpot_ms=decode/(output_count-1) if decode is not None and output_count > 1 else None)


class ContextKVStream:
    """Normalize only guarded native SSE snapshots; keep exact terminal text."""
    def __init__(self, incremental):
        self.incremental = incremental
        self.output_ids = []
        self.text_parts = []

    def push(self, out):
        if not self.incremental or "error" in out:
            return out
        self.output_ids.extend(out.get("output_ids", ()))
        self.text_parts.append(out.get("text") or "")
        return {**out, "output_ids": list(self.output_ids), "text": "".join(self.text_parts)}

    @staticmethod
    def tokens_before_receipt(out):
        # The final tokens leave HTTP before terminal-publication IPC. The final
        # frame repeats their cumulative snapshot with the receipt afterwards.
        meta = dict(out.get("meta_info", {}))
        meta.pop("contextkv_receipt", None)
        meta.pop("contextkv_timing", None)
        meta["finish_reason"] = None
        return {**out, "meta_info": meta}


class ContextKVError(ValueError):
    def __init__(self, code, message=""):
        self.code = code
        super().__init__(message or code)


def token_hash(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def peek_full_prefix(tree, key, full_component_type=0):
    """Pure FULL-only tree walk; counts matched fragments without splitting.

    A transfer may have published indices before its completion ACK. Device and
    host readiness are therefore classified independently from index presence.
    A hole terminates each contiguous prefix permanently; pending counters are
    observations along the logically matching path, not usable prefix lengths.
    """
    key = key.page_aligned(tree.page_size)
    node = tree.root_node
    counts = dict(raw_match_tokens=0, ready_gpu_prefix_tokens=0,
                  recoverable_prefix_tokens=0, host_only_load_tokens=0,
                  inflight_h2d_tokens=0, inflight_d2h_tokens=0)
    gpu_contiguous = recoverable_contiguous = True
    while len(key) > 0:
        child = node.children.get(key.child_key(tree.page_size))
        if child is None:
            break
        matched = child.key.match(key, page_size=tree.page_size)
        if matched == 0:
            break
        full = child.component_data[full_component_type]
        h2d_pending = child.contextkv_h2d_pending_id is not None
        d2h_pending = child.write_through_pending_id is not None
        gpu_ready = full.value is not None and not h2d_pending
        host_ready = full.host_value is not None and not d2h_pending
        counts["raw_match_tokens"] += matched
        counts["inflight_h2d_tokens"] += matched if h2d_pending else 0
        counts["inflight_d2h_tokens"] += matched if d2h_pending else 0
        gpu_contiguous = gpu_contiguous and gpu_ready
        recoverable_contiguous = recoverable_contiguous and (gpu_ready or host_ready)
        if gpu_contiguous:
            counts["ready_gpu_prefix_tokens"] += matched
        if recoverable_contiguous:
            counts["recoverable_prefix_tokens"] += matched
            if host_ready and full.value is None and not h2d_pending:
                counts["host_only_load_tokens"] += matched
        if matched < len(child.key):
            break
        node = child
        key = key[matched:]
    return counts


class ContextKVRuntime:
    def __init__(self, cache, fingerprint, receipt_limit=1024, *, scheduler=None,
                 result_ttl_seconds=900, result_max_bytes=64 << 20, event_limit=4096,
                 transfer_tma_enabled=None):
        self.cache = cache
        self.scheduler = scheduler
        self.fingerprint = fingerprint
        self.restart_id = uuid.uuid4().hex
        self.epochs = {}
        self.operations = {}
        self.attempts = OrderedDict()
        self.receipt_limit = receipt_limit
        self.result_ttl_seconds = result_ttl_seconds
        self.result_max_bytes = result_max_bytes
        self.results = OrderedDict()
        self.result_bytes = 0
        self.events = deque(maxlen=event_limit)
        self.event_seq = 0
        self.sample_pool_enabled = os.environ.get("CONTEXTKV_SAMPLE_POOL") == "1"
        self.profiling_enabled = os.environ.get("CONTEXTKV_ENABLE_PROFILING") == "1"
        self.last_pool_sample = None
        cache.contextkv_runtime = self
        self.observed_seq = 0
        cc = cache.cache_controller
        allocator = getattr(cache, "token_to_kv_pool_allocator", None)
        config = {"page_size": cache.page_size, "backend": cache._tree_core_backend,
                  "hicache": cc is not None, "protocol": PROTOCOL,
                  "write_policy": getattr(cc, "write_policy", None),
                  "host_memory_mode": getattr(cache, "host_memory_mode", None),
                  "hicache_io_backend": getattr(cc, "io_backend", None),
                  "host_layout": getattr(getattr(cc, "mem_pool_host", None), "layout", None),
                  "tma_transfer_enabled": transfer_tma_enabled,
                  "gpu_capacity_tokens": getattr(allocator, "size", None),
                  "host_capacity_tokens": getattr(getattr(cc, "mem_pool_host", None), "size", None),
                  "load_back_threshold": getattr(cache, "load_back_threshold", None),
                  "sample_pool": self.sample_pool_enabled}
        self.cache_config = config
        self.cache_config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    def _restart(self, expected):
        if expected != self.restart_id:
            raise ContextKVError("STALE_RESTART")

    def _fingerprint(self, value):
        if value != self.fingerprint:
            raise ContextKVError("FINGERPRINT_MISMATCH")

    def dispatch(self, action, data):
        if action not in {"info", "quote", "prepare", "retire", "status", "publish_result", "metrics", "profile_prepare"}:
            raise ContextKVError("UNKNOWN_OPERATION")
        return getattr(self, action)(data)

    def info(self, data):
        return dict(engine_restart_id=self.restart_id, protocol_version=PROTOCOL,
                    engine_kind="sglang", host_quote_supported=True,
                    completion_status_supported=True, guarded_generate_supported=True,
                    scheduler_owned=True, cache_config_hash=self.cache_config_hash,
                    fingerprint=self.fingerprint, sglang_commit=COMMIT,
                    metrics_supported=True, result_replay_supported=True,
                    result_ttl_seconds=self.result_ttl_seconds,
                    profiling_enabled=self.profiling_enabled, streaming_generation_supported=True)

    def prepare(self, data):
        self._restart(data["expected_restart_id"])
        sid, op = data["engine_session_id"], data["operation_id"]
        if not isinstance(sid, str) or not sid or not isinstance(op, str) or not op:
            raise ContextKVError("INVALID_ID")
        epoch = self.epochs.get(sid)
        if epoch:
            if epoch["operation_id"] != op or epoch["state"] not in {"PREPARED", "COMMITTED"}:
                raise ContextKVError("EPOCH_CONFLICT")
        else:
            if op in self.operations:
                raise ContextKVError("OPERATION_CONFLICT")
            # Never wrap an existing native session or reuse an ID.
            if sid in self.cache.session_refs._session_generations:
                raise ContextKVError("EPOCH_CONFLICT")
            generation = self.cache.open_radix_session(sid)
            if generation is None:
                raise ContextKVError("SESSION_CACHE_DISABLED")
            epoch = self.epochs[sid] = dict(generation=generation, operation_id=op,
                                           state="PREPARED", inflight=set())
            self.operations[op] = ("prepare", sid, generation)
        return dict(engine_session_id=sid, session_generation=epoch["generation"],
                    engine_restart_id=self.restart_id, operation_id=op, state=epoch["state"])

    def admit(self, recv, req):
        """Executed before ensure_session_generation. Never reopens a retired ID."""
        sid = recv.session_id
        self._restart(recv.expected_restart_id)
        self._fingerprint(recv.contextkv_fingerprint)
        epoch = self.epochs.get(sid)
        generation = recv.expected_generation
        if not epoch or epoch["generation"] != generation:
            raise ContextKVError("STALE_GENERATION")
        if epoch["state"] not in {"PREPARED", "COMMITTED"}:
            raise ContextKVError("EPOCH_RETIRED")
        if self.cache.session_refs._session_generations.get(sid) != generation:
            raise ContextKVError("STALE_GENERATION")
        attempt = recv.contextkv_attempt_id
        if not attempt or attempt != recv.rid:
            raise ContextKVError("INVALID_ATTEMPT_ID")
        if (recv.session_params is not None or recv.lora_id is not None
                or recv.mm_inputs is not None or recv.input_embeds is not None
                or recv.return_logprob or recv.cache_salt is not None or recv.extra_key is not None):
            raise ContextKVError("UNSUPPORTED_GENERATION_MODE")
        if attempt in self.attempts:
            raise ContextKVError("ATTEMPT_EXISTS", "inspect status; generation is not replayed")
        if epoch["inflight"]:
            raise ContextKVError("EPOCH_BUSY")
        item = dict(state="INFLIGHT", engine_restart_id=self.restart_id,
                    engine_session_id=sid, session_generation=generation,
                    attempt_id=attempt, inflight=True, terminal=False,
                    result=None, receipt=None)
        self.attempts[attempt] = item
        epoch["inflight"].add(attempt)
        req.contextkv_runtime = self
        req.contextkv_attempt_id = attempt
        req.contextkv_input_hash = token_hash(recv.input_ids)
        req.contextkv_session_ref_registered = False
        req.contextkv_prefill_tokens = 0

    def record_rejection(self, recv, error):
        attempt = recv.contextkv_attempt_id
        if attempt and attempt == recv.rid and attempt not in self.attempts:
            self.attempts[attempt] = dict(state="REJECTED", engine_restart_id=self.restart_id,
                engine_session_id=recv.session_id, session_generation=recv.expected_generation,
                attempt_id=attempt, inflight=False, terminal=True, result=None, receipt=None)

    def finish(self, req):
        attempt = getattr(req, "contextkv_attempt_id", None)
        if not attempt or not req.finished():
            return
        item = self.attempts.get(attempt)
        if item is None or item["terminal"]:
            return
        reason = req.finished_reason.to_json()
        kind = reason.get("type", "unknown")
        complete = kind != "abort"
        device = int(req.cached_tokens_device)
        host = int(req.cached_tokens_host)
        if device == 0 and host == 0:
            device = int(req.cached_tokens)
        receipt = dict(rid=req.rid, attempt_id=attempt,
            engine_session_id=req.session_id, session_generation=req.session_generation,
            engine_restart_id=self.restart_id, input_hash=req.contextkv_input_hash,
            finish_reason=kind, generation_complete=complete,
            actual_device_hit_tokens=device, actual_host_loaded_tokens=host,
            actual_prefill_tokens=req.contextkv_prefill_tokens,
            inserted_cache_tokens=int(req.kv.cache_inserted_len),
            session_ref_registered=bool(getattr(req, "contextkv_session_ref_registered", False)))
        # Native HTTP output is published back to this owner before delivery.
        # A crash before that publication leaves a terminal receipt only.
        item.update(state="COMPLETED" if complete else "FAILED", terminal=True,
                    inflight=False, receipt=receipt)
        req.customized_info = dict(req.customized_info or {})
        req.customized_info["contextkv_receipt"] = [receipt]
        output_count = len(getattr(req, "output_ids", ()))
        if getattr(req, "finished_len", None) is not None:
            output_count = min(output_count, req.finished_len)
        timing = native_timing(getattr(req, "time_stats", None), output_count)
        req.customized_info["contextkv_timing"] = [timing]
        self._event("request_terminal", attempt_id=attempt, receipt=receipt, timing=timing)
        epoch = self.epochs[req.session_id]
        epoch["inflight"].discard(attempt)
        if epoch["state"] == "RETIRING" and not epoch["inflight"]:
            self._release(req.session_id, epoch)
        elif complete and epoch["state"] == "PREPARED":
            epoch["state"] = "COMMITTED"
        self._prune_results()
        for key in list(self.attempts):
            if len(self.attempts) <= self.receipt_limit:
                break
            if self.attempts[key]["terminal"]:
                del self.attempts[key]
                self._drop_result(key)

    def _release(self, sid, epoch):
        self.cache.release_radix_session(sid)
        epoch["state"] = "RETIRED"
        self._event("retire_ref", session_id=sid, freed_gpu_tokens=0, freed_host_tokens=0)

    def retire(self, data):
        self._restart(data["expected_restart_id"])
        sid, generation, op = (data["engine_session_id"], data["expected_generation"], data["operation_id"])
        identity = ("retire", sid, generation)
        if op in self.operations and self.operations[op] != identity:
            raise ContextKVError("OPERATION_CONFLICT")
        epoch = self.epochs.get(sid)
        if not epoch or epoch["generation"] != generation:
            raise ContextKVError("STALE_GENERATION")
        self.operations[op] = identity
        if epoch["state"] != "RETIRED":
            epoch["state"] = "RETIRING"
            if not epoch["inflight"]:
                self._release(sid, epoch)
        return dict(engine_session_id=sid, session_generation=generation,
            engine_restart_id=self.restart_id, operation_id=op, state=epoch["state"],
            acknowledged=epoch["state"] == "RETIRED", freed_gpu_tokens=0, freed_host_tokens=0)

    def status(self, data):
        if data.get("expected_restart_id") is not None:
            self._restart(data["expected_restart_id"])
        attempt = data.get("attempt_id")
        if attempt is not None:
            self._prune_results()
            item = dict(self.attempts.get(attempt, dict(state="UNKNOWN", engine_restart_id=self.restart_id,
                        attempt_id=attempt, inflight=False, terminal=False)))
            if attempt in self.results:
                item["result"] = self.results[attempt][2]
            return item
        sid = data.get("engine_session_id")
        if sid is None and data.get("operation_id") in self.operations:
            sid = self.operations[data["operation_id"]][1]
        epoch = self.epochs.get(sid)
        return dict(state=epoch["state"] if epoch else "UNKNOWN", engine_restart_id=self.restart_id,
                    engine_session_id=sid, session_generation=epoch["generation"] if epoch else None,
                    inflight=bool(epoch and epoch["inflight"]), terminal=bool(epoch and epoch["state"] == "RETIRED"))

    def quote(self, data):
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.managers.schedule_batch import Req
        self._fingerprint(data["fingerprint"])
        candidates = data["candidates"]
        if len({c["snapshot_version"] for c in candidates}) > 1:
            raise ContextKVError("MIXED_SNAPSHOT")
        quotes = []
        # No calls to loading_check/writing_check/maintenance or CUDA fences.
        free_gpu = int(self.cache.token_to_kv_pool_allocator.available_size())
        controller = self.cache.cache_controller
        free_host = int(controller.mem_pool_host.available_size()) if controller is not None else 0
        for c in candidates:
            ids = c["input_ids"]
            if c["input_hash"] != token_hash(ids):
                raise ContextKVError("INPUT_HASH_MISMATCH")
            # Reuse the native match limit method under the supported ordinary
            # dense generation contract; page alignment remains native too.
            from types import SimpleNamespace
            limit_req = SimpleNamespace(dllm_config=None, return_logprob=False, is_dllm=lambda: False)
            limit = Req._compute_max_prefix_len(limit_req, len(ids))
            limit = limit // self.cache.page_size * self.cache.page_size
            view = self.cache.tree_core.peek_full_prefix(RadixKey(array("q", ids[:limit])))
            host_load = view["host_only_load_tokens"]
            threshold = max(1, getattr(self.cache, "load_back_threshold", 10))
            native_load = host_load if host_load >= threshold else 0
            uncertain = (bool(view["inflight_h2d_tokens"] or view["inflight_d2h_tokens"])
                         or free_gpu < max(0, len(ids) - view["ready_gpu_prefix_tokens"]))
            quotes.append(dict(candidate_id=c["candidate_id"], input_hash=c["input_hash"],
                input_len=len(ids), match_limit=limit, **view,
                native_load_estimate=native_load, native_estimate_uncertain=uncertain))
        return dict(request_id=data["request_id"], observed_seq=self.observed_seq,
                    observed_time_ns=time.monotonic_ns(), engine_restart_id=self.restart_id,
                    cache_config_hash=self.cache_config_hash,
                    free_gpu_tokens=free_gpu, free_host_tokens=free_host,
                    quotes=quotes, fingerprint=self.fingerprint, host_quote_supported=True,
                    active_requests=len(self.scheduler.running_batch.reqs) if self.scheduler else 1,
                    queued_requests=len(self.scheduler.waiting_queue) if self.scheduler else 0)

    def _drop_result(self, attempt):
        item = self.results.pop(attempt, None)
        if item is not None:
            self.result_bytes -= item[1]

    def _prune_results(self):
        cutoff = time.monotonic() - self.result_ttl_seconds
        while self.results:
            attempt, (saved, _, _) = next(iter(self.results.items()))
            if saved > cutoff and self.result_bytes <= self.result_max_bytes:
                break
            self._drop_result(attempt)

    def publish_result(self, data):
        """Internal tokenizer→scheduler publication; no public HTTP route."""
        attempt = data["attempt_id"]
        item = self.attempts.get(attempt)
        if item is None or not item["terminal"] or item["receipt"] is None:
            raise ContextKVError("MISSING_RECEIPT")
        raw = data["response"]
        receipt = item["receipt"]
        meta = raw.get("meta_info", {})
        timing = meta.get("contextkv_timing", {})
        if isinstance(timing, list):
            timing = timing[0] if timing else {}
        result = dict(text=raw["text"], output_ids=raw["output_ids"],
                      finish_reason=receipt["finish_reason"], receipt=receipt,
                      generation_complete=receipt["generation_complete"], **timing)
        size = len(result["text"].encode("utf-8")) + 8*len(result["output_ids"])
        self._drop_result(attempt)
        if size <= self.result_max_bytes:
            self.results[attempt] = (time.monotonic(), size, result)
            self.result_bytes += size
        self._prune_results()
        return dict(stored=attempt in self.results)

    def _event(self, kind, **fields):
        self.event_seq += 1
        self.events.append(dict(seq=self.event_seq, event=kind,
                                observed_time_ns=time.monotonic_ns(), **fields))

    def record_transfer(self, direction, ack):
        """Runs after the native ACK completion fence; adds no synchronization."""
        duration = ack.start_event.elapsed_time(ack.finish_event) if ack.timing_enabled else None
        self._event("transfer", transfer_id=f"{self.restart_id}:{self.event_seq+1}",
                    status="completed", direction=direction, bytes=int(ack.num_bytes),
                    tokens=int(ack.num_tokens), duration_ms=duration,
                    clock="cuda_event" if ack.timing_enabled else None,
                    source="hardware", timing_available=bool(ack.timing_enabled))

    def pool_snapshot(self):
        allocator = self.cache.token_to_kv_pool_allocator
        free_gpu = int(allocator.available_size())
        gpu_capacity = int(allocator.size)
        cc = self.cache.cache_controller
        host_capacity = int(cc.mem_pool_host.size) if cc else 0
        free_host = int(cc.mem_pool_host.available_size()) if cc else 0
        return dict(gpu_capacity_tokens=gpu_capacity, host_capacity_tokens=host_capacity,
                    free_gpu_tokens=free_gpu, free_host_tokens=free_host,
                    unique_gpu_tokens=gpu_capacity-free_gpu,
                    unique_host_tokens=host_capacity-free_host,
                    page_size=self.cache.page_size)

    def observe_pool(self):
        """Optional O(1) scheduler-boundary samples, not an allocator event log."""
        if not self.sample_pool_enabled:
            return
        current = self.pool_snapshot()
        if current != self.last_pool_sample:
            self._event("pool_snapshot", **current)
            self.last_pool_sample = current

    def metrics(self, data):
        after = int(data.get("after_seq", 0))
        oldest = self.events[0]["seq"] if self.events else self.event_seq+1
        return dict(engine_restart_id=self.restart_id, observed_seq=self.observed_seq,
                    observed_time_ns=time.monotonic_ns(), clock="server_monotonic",
                    pool=self.pool_snapshot(), cache_config=self.cache_config,
                    active_requests=len(self.scheduler.running_batch.reqs) if self.scheduler else 0,
                    queued_requests=len(self.scheduler.waiting_queue) if self.scheduler else 0,
                    events=[e for e in self.events if e["seq"] > after],
                    next_seq=self.event_seq, dropped_before_seq=oldest-1,
                    events_truncated=after < oldest-1,
                    pool_sample_scope="scheduler_boundaries" if self.sample_pool_enabled else "on_demand")

    def profile_prepare(self, data):
        """Destructive cache fixtures for an explicitly isolated profiler only."""
        if not self.profiling_enabled or self.scheduler is None:
            raise ContextKVError("PROFILING_DISABLED")
        self._restart(data["expected_restart_id"])
        if not self.scheduler.is_fully_idle():
            raise ContextKVError("ENGINE_BUSY")
        mode = data["mode"]
        if mode == "cold":
            if not self.scheduler.flush_cache(empty_cache=False):
                raise ContextKVError("ENGINE_BUSY")
            # flush invalidates native generations; replace the protocol owner
            # and restart identity too so no stale admission can succeed.
            replacement = ContextKVRuntime(self.cache, self.fingerprint,
                self.receipt_limit, scheduler=self.scheduler,
                result_ttl_seconds=self.result_ttl_seconds, result_max_bytes=self.result_max_bytes,
                transfer_tma_enabled=self.cache_config["tma_transfer_enabled"])
            self.scheduler.contextkv_runtime = replacement
            return dict(mode=mode, info=replacement.info({}), pool=replacement.pool_snapshot())
        if mode == "settle":
            if self.cache.cache_controller is not None:
                self.cache.flush_pending_backups()
                self.cache.writing_check(write_back=True)
            return dict(mode=mode, info=self.info({}), pool=self.pool_snapshot())
        if mode == "pressure":
            from sglang.srt.mem_cache.base_prefix_cache import EvictParams
            tier, tokens = data["tier"], data["tokens"]
            if tier not in {"gpu", "host"} or type(tokens) is not int or tokens <= 0:
                raise ContextKVError("INVALID_PRESSURE")
            if tier == "host" and self.cache.cache_controller is None:
                raise ContextKVError("HICACHE_DISABLED")
            # Only settle already-submitted transfers. Session refs are retained;
            # native eviction still decides which referenced/unreferenced leaves go.
            if self.cache.cache_controller is not None:
                self.cache.flush_pending_backups()
                self.cache.writing_check(write_back=True)
                self.cache.loading_check()
            before = self.pool_snapshot()
            if tokens > before[f"{tier}_capacity_tokens"]:
                raise ContextKVError("INVALID_PRESSURE", "request exceeds the native pool capacity")
            if tier == "gpu":
                evicted = int(self.cache.evict(EvictParams(num_tokens=tokens)).num_tokens_evicted)
            else:
                evicted = int(self.cache.evict_host(tokens))
            after = self.pool_snapshot()
            result = dict(mode=mode, tier=tier, requested_tokens=tokens,
                native_evicted_tokens=evicted, pool_before=before, pool_after=after,
                gpu_tokens_freed=before["unique_gpu_tokens"]-after["unique_gpu_tokens"],
                host_tokens_freed=before["unique_host_tokens"]-after["unique_host_tokens"],
                refs_released=False)
            self._event("profiling_pressure", **result)
            return result
        if mode == "host_only":
            if self.cache.cache_controller is None:
                raise ContextKVError("HICACHE_DISABLED")
            from sglang.srt.mem_cache.base_prefix_cache import EvictParams
            # Profiling fixture only: settle native writes, release soft refs,
            # evict device copies through native policy, then settle write-back.
            self.cache.flush_pending_backups()
            self.cache.writing_check(write_back=True)
            for sid, epoch in self.epochs.items():
                if epoch["state"] != "RETIRED":
                    self._release(sid, epoch)
            self.cache.evict(EvictParams(num_tokens=self.cache.token_to_kv_pool_allocator.size))
            self.cache.writing_check(write_back=True)
            return dict(mode=mode, info=self.info({}), pool=self.pool_snapshot())
        raise ContextKVError("INVALID_CACHE_STATE")


def get_runtime(scheduler):
    runtime = getattr(scheduler, "contextkv_runtime", None)
    if runtime is not None:
        return runtime
    from sglang.srt.runtime_context import get_parallel, get_model
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    parallel = get_parallel()
    cache = scheduler.tree_cache
    if (parallel.tp_size != 1 or parallel.num_dp_ranks != 1 or parallel.pp_size != 1
            or not isinstance(cache, UnifiedRadixCache) or not cache.enable_session_radix_cache
            or cache._tree_core_backend != "python" or len(cache._components_tuple) != 1
            or cache.tree_core.is_eagle or cache.enable_storage or cache.linker is not None
            or cache.buffer_pipeline is not None
            or scheduler.dllm_config is not None
            or scheduler.rust_server is not None or not scheduler.spec_algorithm.is_none()
            or scheduler.enable_pdmux or scheduler.disaggregation_mode.value != "null"):

        raise ContextKVError("UNSUPPORTED_ENGINE", "requires Python, dense FULL, TP/DP/PP=1, session cache")
    try:
        fp = json.loads(os.environ["CONTEXTKV_FINGERPRINT_JSON"])
        if set(fp) != {"model_revision", "tokenizer_hash", "template_hash", "namespace_hash", "cache_salt", "extra_key"}:
            raise ValueError("incomplete fingerprint")
        if not all(isinstance(fp[k], str) and fp[k] for k in ("model_revision", "tokenizer_hash", "template_hash", "namespace_hash")):
            raise ValueError("empty fingerprint field")
        if fp["cache_salt"] is not None or fp["extra_key"] is not None:
            raise ValueError("v1 requires the default cache key")
    except (KeyError, ValueError, AssertionError) as exc:
        raise ContextKVError("INVALID_FINGERPRINT_CONFIG", "set complete CONTEXTKV_FINGERPRINT_JSON") from exc
    # Validate the running tokenizer and local frozen model config, not merely
    # echo a caller-provided fingerprint. This first patch supports Qwen2.5-7B.
    try:
        if get_model().revision != MODEL_REVISION or fp["model_revision"] != MODEL_REVISION:
            raise ValueError("wrong model revision")
        config = json.loads((Path(get_model().model_path) / "config.json").read_text())
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
        if hashlib.sha256(canonical.encode()).hexdigest() != MODEL_CONFIG_HASH:
            raise ValueError("wrong model config")
        tokenizer = scheduler.tokenizer
        actual_tokenizer = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
        template = json.dumps({"template": tokenizer.chat_template, "enable_thinking": False}, sort_keys=True)
        actual_template = hashlib.sha256(template.encode()).hexdigest()
        if actual_tokenizer != fp["tokenizer_hash"] or actual_template != fp["template_hash"]:
            raise ValueError("wrong tokenizer or template")
    except (AssertionError, AttributeError, OSError, ValueError) as exc:
        raise ContextKVError("FINGERPRINT_MISMATCH", "running model config/revision/tokenizer/template differs") from exc
    from sglang.srt.environ import envs
    scheduler.contextkv_runtime = ContextKVRuntime(cache, fp, scheduler=scheduler,
        transfer_tma_enabled=envs.SGLANG_HICACHE_TMA_TRANSFER.get(),
        result_ttl_seconds=float(os.environ.get("CONTEXTKV_RESULT_TTL_SECONDS", "900")),
        result_max_bytes=int(os.environ.get("CONTEXTKV_RESULT_MAX_BYTES", str(64 << 20))))
    return scheduler.contextkv_runtime
