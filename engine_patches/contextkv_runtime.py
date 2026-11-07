"""Scheduler-owned ContextKV protocol, for the pinned SGLang Python backend.

This module deliberately has no background thread or GPU synchronizations. It is
called only from scheduler dispatch and its terminal output callback. The FULL
GPU/host read view uses observation-only transfer markers; it never processes
ACKs, takes locks, splits nodes or treats queued transfers as ready GPU copies.
"""
from __future__ import annotations

from array import array
from collections import Counter, OrderedDict, deque
import hashlib
import json
import math
import os
import time
from types import SimpleNamespace
import uuid

COMMIT = "71de97b264b04dcd514cf904003028aefe9775c8"
PROTOCOL = "contextkv-v1"


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


class AdmissionDiagnostic:
    """Per guarded Req, bounded by native reason names, emitted only at terminal.

    Gate observations are not duration attribution. No cache contents or request
    text are copied, and the native request timestamps are never resampled.
    """
    def __init__(self):
        self.gates = {}
        self.outcomes = Counter()

    def observe(self, reason, snapshot):
        row = self.gates.get(reason)
        if row is None:
            self.gates[reason] = dict(count=1, first=snapshot, last=snapshot)
        else:
            row["count"] += 1
            row["last"] = snapshot

    def finish(self, req):
        stats = req.time_stats
        return dict(version="native_admission_diagnostic_v1", request_id=req.rid,
            clock="server_perf_counter_seconds", timestamps={
                name: getattr(stats, name) or None for name in (
                    "scheduler_recv_time", "forward_entry_time", "prefill_finished_time",
                    "completion_time", "wait_queue_entry_time")},
            queue_duration_s=stats.queue_duration_s,
            queue_duration_scope="native_accumulated_wait_including_cache_preparation_and_reentry",
            gates=self.gates, adder_outcomes=dict(self.outcomes),
            scope="visited_native_gates_only_not_wait_duration_attribution")


def record_admission_gate(req, adder, reason, **details):
    diagnostic = getattr(req, "contextkv_admission_diagnostic", None)
    if diagnostic is None:
        return
    # These are the live native pass budgets, including its temporary prefix
    # pin when called from _select_prefill_admission. No match/lock/ACK here.
    diagnostic.observe(reason, dict(observed_at=time.perf_counter(),
        prompt_tokens=len(req.origin_input_ids), full_fill_tokens=len(req.full_untruncated_fill_ids),
        prefix_tokens=len(req.prefix_indices),
        host_hit_tokens=req.host_hit_length, generated_tokens=len(req.output_ids),
        max_new_tokens=req.sampling_params.max_new_tokens,
        ignore_eos=req.sampling_params.ignore_eos,
        remaining_total=adder.rem_total_tokens, remaining_current=adder.cur_rem_tokens,
        total_offset=adder.rem_total_token_offset, current_offset=adder.cur_rem_token_offset,
        page_size=adder.page_size,
        remaining_input=adder.rem_input_tokens, remaining_chunk=adder.rem_chunk_tokens,
        new_token_ratio=adder.new_token_ratio, admitted_requests=len(adder.can_run_list),
        budget_type="PrefillAdder0518", **details))


def record_admission_outcome(req, adder, admitted_before, result):
    diagnostic = getattr(req, "contextkv_admission_diagnostic", None)
    if diagnostic is None:
        return
    accepted = len(adder.can_run_list) > admitted_before and adder.can_run_list[-1] is req
    # add_one_req's result can describe exhaustion AFTER committing this Req.
    # Membership, not NO_TOKEN/OTHER, determines whether it was admitted.
    outcome = ("accepted:" if accepted else "not_accepted:") + result.name
    diagnostic.outcomes[outcome] += 1
    record_admission_gate(req, adder, "adder_accepted" if accepted else "adder_not_accepted",
                          return_state=result.name, accepted=accepted)


def record_scheduler_gate(scheduler, running_batch, reason):
    # Only the head present at this early pass gate is sampled. Do not walk or
    # attribute the entire waiting queue to a cause that was never tested for it.
    if not scheduler.waiting_queue:
        return
    req = scheduler.waiting_queue[0]
    diagnostic = getattr(req, "contextkv_admission_diagnostic", None)
    if diagnostic is None:
        return
    diagnostic.observe(reason, dict(observed_at=time.perf_counter(),
        scope="scheduler_pass_head_observation", running_requests=len(running_batch.reqs),
        waiting_requests=len(scheduler.waiting_queue), batch_is_full=running_batch.batch_is_full,
        new_token_ratio=scheduler.new_token_ratio_tracker.current,
        budget=None))  # No PrefillAdder exists at these early gates.


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


def executed_prefix_hashes(ids, page_size):
    """Hash actual prompt page boundaries without retaining prompt token arrays."""
    digest = hashlib.sha256(b"[")
    boundaries = {}
    for position, token in enumerate(ids, 1):
        digest.update(("," if position > 1 else "").encode())
        digest.update(str(int(token)).encode())
        if position % page_size == 0:
            page_digest = digest.copy()
            page_digest.update(b"]")
            boundaries[position] = page_digest.hexdigest()
    return boundaries


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
                  evictable_gpu_prefix_tokens=0,
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
            if full.lock_ref == 0:
                counts["evictable_gpu_prefix_tokens"] += matched
        if recoverable_contiguous:
            counts["recoverable_prefix_tokens"] += matched
            if host_ready and full.value is None and not h2d_pending:
                counts["host_only_load_tokens"] += matched
        if matched < len(child.key):
            break
        node = child
        key = key[matched:]
    return counts


class HostAdmissionPolicy:
    """Bounded scalar state for optional request-triggered backup admission.

    Adapted from EfficientAgent b21b12a (Apache-2.0): pressure AND large write.
    Here a request spends its incremental budget only on native-admitted FULL
    BackupKV pages; failed allocations and already-backed specs spend nothing.
    8192 tokens starts from its 8 x 1024-token threshold; it is not tuned here.
    """
    def __init__(self, mode="off", threshold_tokens=8192, occupancy_threshold=.95,
                 window_seconds=60., prompt_window=256):
        if mode not in {"off", "fixed", "conditioned"}:
            raise ValueError("unknown host admission mode")
        if (threshold_tokens < 0 or not 0 <= occupancy_threshold <= 1
                or not math.isfinite(window_seconds) or window_seconds <= 0 or prompt_window < 1):
            raise ValueError("invalid host admission settings")
        self.mode, self.threshold_tokens = mode, threshold_tokens
        self.occupancy_threshold, self.window_seconds = occupancy_threshold, window_seconds
        self.prompt_window = prompt_window
        self.live = {}  # engine epoch -> business identity and last first-match time
        self.prompts = deque(maxlen=prompt_window)
        self.evictions = deque()
        self.counters = Counter(first_match_observations=0, allocation_evicted_host_tokens=0,
            considered_actions=0, allowed_actions=0, skipped_actions=0,
            allowed_planned_new_tokens=0, skipped_new_tokens=0,
            admitted_new_tokens=0, admitted_new_pages=0, unknown_business_actions_native=0)

    def config(self):
        return dict(version=1, mode=self.mode, threshold_tokens=self.threshold_tokens,
                    occupancy_threshold=self.occupancy_threshold,
                    window_seconds=self.window_seconds, prompt_window=self.prompt_window,
                    new_write_unit="native_FULL_BackupKV_pages",
                    admission_scope="incremental_request_backup_budget")

    def _prune(self, now):
        cutoff = now-self.window_seconds
        self.live = {sid: row for sid, row in self.live.items() if row[1] >= cutoff}
        while self.evictions and self.evictions[0][0] < cutoff:
            self.evictions.popleft()

    def observe(self, sid, business_id, prompt_tokens, now):
        self._prune(now)
        self.live[sid] = (business_id, now)
        self.prompts.append((sid, now, prompt_tokens))
        self.counters["first_match_observations"] += 1

    def retire(self, sid):
        self.live.pop(sid, None)
        self.prompts = deque((row for row in self.prompts if row[0] != sid), maxlen=self.prompt_window)

    def capacity_evicted(self, tokens, now):
        if tokens > 0:
            self._prune(now)
            self.evictions.append((now, tokens))
            self.counters["allocation_evicted_host_tokens"] += tokens

    def window(self, now):
        cutoff = now-self.window_seconds
        active = {sid: row for sid, row in self.live.items() if row[1] >= cutoff}
        businesses = {row[0] for row in active.values()}
        lengths = [n for sid, at, n in self.prompts if sid in active and at >= cutoff]
        mean_prompt = sum(lengths)/len(lengths) if lengths else 0.
        return dict(active_business_sessions=len(businesses), active_engine_epochs=len(active),
                    prompt_observations=len(lengths), mean_prompt_tokens=mean_prompt,
                    other_business_working_set_tokens=max(0, len(businesses)-1)*mean_prompt,
                    window_evicted_host_tokens=sum(n for at, n in self.evictions if at >= cutoff))

    def decide(self, new_tokens, page_size, host_capacity, host_used, now, *, admitted_pages=0):
        self._prune(now)
        window = self.window(now)
        new_pages = -(-new_tokens//page_size)
        threshold_pages = -(-self.threshold_tokens//page_size)
        above = window["other_business_working_set_tokens"] > host_capacity
        full_evicting = (host_capacity > 0 and host_used >= self.occupancy_threshold*host_capacity
                         and window["window_evicted_host_tokens"] > 0)
        pressure = self.mode == "fixed" or (above and full_evicting)
        projected_pages = admitted_pages + new_pages
        skip = self.mode != "off" and pressure and projected_pages > threshold_pages and new_pages > 0
        self.counters["considered_actions"] += 1
        self.counters["skipped_actions" if skip else "allowed_actions"] += 1
        self.counters["skipped_new_tokens" if skip else "allowed_planned_new_tokens"] += new_tokens
        reason = ("request_backup_budget_exceeded" if skip else "already_backed" if new_pages == 0 else
                  "within_request_page_budget" if projected_pages <= threshold_pages else "pressure_not_met")
        return dict(allowed=not skip, reason=reason, mode=self.mode, pressure=pressure,
                    working_set_above_capacity=above, host_full_and_evicting=full_evicting,
                    new_tokens=new_tokens, new_pages=new_pages, threshold_pages=threshold_pages,
                    request_admitted_pages=admitted_pages, projected_request_pages=projected_pages,
                    host_capacity_tokens=host_capacity, host_used_tokens=host_used, **window)

    def snapshot(self, now):
        return dict(config=self.config(), counters=dict(self.counters), **self.window(now),
                    eviction_scope="native_backup_capacity_reclaim",
                    skipped_tokens_meaning="planned_backup_omission_not_completed_DMA_savings")


class ContextKVRuntime:
    def __init__(self, cache, fingerprint, receipt_limit=1024, *, scheduler=None,
                 result_ttl_seconds=900, result_max_bytes=64 << 20, event_limit=4096,
                 transfer_tma_enabled=None, model_scope=None, decoder_settings=None):
        self.cache = cache
        self._native_generations = cache.session_refs._session_generations
        self.scheduler = scheduler
        self.fingerprint = fingerprint
        self.model_scope = model_scope
        self.decoder_settings = decoder_settings
        self.restart_id = uuid.uuid4().hex
        self.epochs = {}
        self.operations = {}
        self.retired_epochs = OrderedDict()
        self.operation_receipts = OrderedDict()
        self.attempts = OrderedDict()
        self.executed_prefixes = {}
        self.prefix_owners = {}
        self.receipt_limit = receipt_limit
        self.result_ttl_seconds = result_ttl_seconds
        self.result_max_bytes = result_max_bytes
        self.results = OrderedDict()
        self.result_bytes = 0
        self.events = deque(maxlen=event_limit)
        self.event_seq = 0
        self.sample_pool_enabled = os.environ.get("CONTEXTKV_SAMPLE_POOL") == "1"
        self.profiling_enabled = os.environ.get("CONTEXTKV_ENABLE_PROFILING") == "1"
        self.admission_diagnostics_enabled = (
            self.profiling_enabled and os.environ.get("CONTEXTKV_ADMISSION_DIAGNOSTICS") == "1")
        self.last_pool_sample = None
        cache.contextkv_runtime = self
        self.observed_seq = 0
        self.host_admission = HostAdmissionPolicy(
            mode=os.environ.get("CONTEXTKV_HOST_ADMISSION", "off"),
            threshold_tokens=int(os.environ.get("CONTEXTKV_HOST_ADMISSION_THRESHOLD_TOKENS", "8192")),
            occupancy_threshold=float(os.environ.get("CONTEXTKV_HOST_ADMISSION_OCCUPANCY_THRESHOLD", ".95")),
            window_seconds=float(os.environ.get("CONTEXTKV_HOST_ADMISSION_WINDOW_SECONDS", "60")),
            prompt_window=int(os.environ.get("CONTEXTKV_HOST_ADMISSION_PROMPT_WINDOW", "256")))
        cc = cache.cache_controller
        allocator = getattr(cache, "token_to_kv_pool_allocator", None)
        config = {"page_size": cache.page_size, "backend": type(cache.tree_core).__name__,
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
        if self.host_admission.mode != "off":
            config["host_admission"] = self.host_admission.config()
        if self.decoder_settings is not None:
            config["decoder"] = self.decoder_settings
        self.cache_config = config
        self.cache_config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        self.prefix_scope_hash = hashlib.sha256(json.dumps([self.fingerprint, self.cache_config_hash],
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _restart(self, expected):
        if expected != self.restart_id:
            raise ContextKVError("STALE_RESTART")

    def _replace_after_cache_reset(self, scheduler):
        """Native reset replaces its generation map and can reuse counter values."""
        replacement = ContextKVRuntime(self.cache, self.fingerprint,
            self.receipt_limit, scheduler=scheduler,
            result_ttl_seconds=self.result_ttl_seconds, result_max_bytes=self.result_max_bytes,
            event_limit=self.events.maxlen,
            transfer_tma_enabled=self.cache_config["tma_transfer_enabled"], model_scope=self.model_scope,
            decoder_settings=self.decoder_settings)
        scheduler.contextkv_runtime = replacement
        return replacement

    def _fingerprint(self, value):
        if value != self.fingerprint:
            raise ContextKVError("FINGERPRINT_MISMATCH")

    def dispatch(self, action, data):
        if action not in {"info", "quote", "prepare", "retire", "status", "publish_result", "metrics", "profile_prepare",
                          "retain_prefix", "release_prefix", "prefix_status"}:
            raise ContextKVError("UNKNOWN_OPERATION")
        return getattr(self, action)(data)

    def info(self, data):
        info = dict(engine_restart_id=self.restart_id, protocol_version=PROTOCOL,
                    engine_kind="sglang", host_quote_supported=True,
                    completion_status_supported=True, guarded_generate_supported=True,
                    scheduler_owned=True, cache_config_hash=self.cache_config_hash,
                    fingerprint=self.fingerprint, sglang_commit=COMMIT,
                    metrics_supported=True, result_replay_supported=True,
                    result_ttl_seconds=self.result_ttl_seconds,
                    profiling_enabled=self.profiling_enabled, streaming_generation_supported=True,
                    evidence_prefix_retention_supported=True)
        if self.host_admission.mode != "off":
            info["business_session_identity_supported"] = True
        if self.admission_diagnostics_enabled:
            info["admission_diagnostics_enabled"] = True
        if self.model_scope is not None:
            info["model_scope"] = self.model_scope
        return info

    def prepare(self, data):
        self._restart(data["expected_restart_id"])
        sid, op = data["engine_session_id"], data["operation_id"]
        if not isinstance(sid, str) or not sid or not isinstance(op, str) or not op:
            raise ContextKVError("INVALID_ID")
        business_id = data.get("business_session_id")
        if business_id is not None and (not isinstance(business_id, str) or not business_id):
            raise ContextKVError("INVALID_BUSINESS_SESSION_ID")
        epoch = self.epochs.get(sid)
        if epoch:
            if (business_id is not None and business_id != epoch.get("business_session_id")):
                raise ContextKVError("EPOCH_CONFLICT")
            if epoch["operation_id"] != op or epoch["state"] not in {"PREPARED", "COMMITTED"}:
                raise ContextKVError("EPOCH_CONFLICT")
        else:
            if self._operation(op) is not None:
                raise ContextKVError("OPERATION_CONFLICT")
            # Reuse the native bounded close guard. Once both histories expire,
            # an explicit prepare gets a new native generation, never the old one.
            if (sid in self.cache.session_refs._session_generations
                    or sid in self.cache.session_refs._closed_session_ids):
                raise ContextKVError("EPOCH_CONFLICT")
            generation = self.cache.open_radix_session(sid)
            if generation is None:
                raise ContextKVError("SESSION_CACHE_DISABLED")
            epoch = self.epochs[sid] = dict(generation=generation, operation_id=op,
                                           state="PREPARED", inflight=set(), business_session_id=business_id)
            self.operations[op] = ("prepare", sid, generation)
        ack = dict(engine_session_id=sid, session_generation=epoch["generation"],
                   engine_restart_id=self.restart_id, operation_id=op, state=epoch["state"])
        if epoch.get("business_session_id") is not None:
            ack["business_session_id"] = epoch["business_session_id"]
        return ack

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
        self.executed_prefixes[attempt] = executed_prefix_hashes(recv.input_ids, self.cache.page_size)
        epoch["inflight"].add(attempt)
        req.contextkv_runtime = self
        req.contextkv_attempt_id = attempt
        req.contextkv_input_hash = token_hash(recv.input_ids)
        req.contextkv_session_ref_registered = False
        req.contextkv_prefill_tokens = 0
        req.contextkv_inserted_cache_tokens = 0
        if self.admission_diagnostics_enabled:
            req.contextkv_admission_diagnostic = AdmissionDiagnostic()

    def record_rejection(self, recv, error):
        attempt = recv.contextkv_attempt_id
        if attempt and attempt == recv.rid and attempt not in self.attempts:
            self.attempts[attempt] = dict(state="REJECTED", engine_restart_id=self.restart_id,
                engine_session_id=recv.session_id, session_generation=recv.expected_generation,
                attempt_id=attempt, inflight=False, terminal=True, result=None, receipt=None)
            self._prune_attempts()

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
            inserted_cache_tokens=int(req.contextkv_inserted_cache_tokens),
            session_ref_registered=bool(getattr(req, "contextkv_session_ref_registered", False)))
        # Native HTTP output is published back to this owner before delivery.
        # A crash before that publication leaves a terminal receipt only.
        item.update(state="COMPLETED" if complete else "FAILED", terminal=True,
                    inflight=False, receipt=receipt)
        # Retention follows terminal time: a long-running request must still
        # publish its result after newer admission rejections have accumulated.
        self.attempts.move_to_end(attempt)
        req.customized_info = dict(req.customized_info or {})
        req.customized_info["contextkv_receipt"] = [receipt]
        output_count = len(getattr(req, "output_ids", ()))
        if getattr(req, "finished_len", None) is not None:
            output_count = min(output_count, req.finished_len)
        timing = native_timing(getattr(req, "time_stats", None), output_count)
        req.customized_info["contextkv_timing"] = [timing]
        diagnostic = getattr(req, "contextkv_admission_diagnostic", None)
        extra = {"admission_diagnostic": diagnostic.finish(req)} if diagnostic is not None else {}
        self._event("request_terminal", attempt_id=attempt, receipt=receipt, timing=timing, **extra)
        epoch = self.epochs[req.session_id]
        epoch["inflight"].discard(attempt)
        if epoch["state"] == "RETIRING" and not epoch["inflight"]:
            self._release(req.session_id, epoch)
        elif complete and epoch["state"] == "PREPARED":
            epoch["state"] = "COMMITTED"
        self._prune_results()
        self._prune_attempts()

    def _prune_attempts(self):
        # Both admission rejection and normal completion enter terminal state.
        # Active attempts are retained independently of the terminal budget.
        terminal = [key for key, item in self.attempts.items() if item["terminal"]]
        for key in terminal[:max(0, len(terminal)-self.receipt_limit)]:
            del self.attempts[key]
            self.executed_prefixes.pop(key, None)
            self._drop_result(key)

    def _executed_prefix(self, data):
        self._restart(data["expected_restart_id"])
        self._fingerprint(data["fingerprint"])
        attempt = data["attempt_id"]
        executed = self.attempts[attempt]
        if executed["state"] != "COMPLETED" or not executed["receipt"]["generation_complete"]:
            raise ContextKVError("PREFIX_REQUIRES_COMPLETED_EXECUTION")
        ids = data["input_ids"]
        boundary = len(ids) // self.cache.page_size * self.cache.page_size
        if boundary == 0:
            raise ContextKVError("PREFIX_BELOW_PAGE_SIZE")
        ids = array("q", ids[:boundary])
        path_hash = token_hash(ids)
        if self.executed_prefixes[attempt].get(boundary) != path_hash:
            raise ContextKVError("PREFIX_NOT_EXECUTED")
        return ids

    def retain_prefix(self, data):
        """Register an executed causal prefix as an independent native soft owner."""
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        owner_id = data["owner_id"]
        owner = self.prefix_owners.get(owner_id)
        if owner is None:
            ids = self._executed_prefix(data)
            attempt = data["attempt_id"]
        else:
            # The independent owner carries its original executed-path proof
            # beyond the bounded business-attempt receipt history.
            self._restart(data["expected_restart_id"])
            self._fingerprint(data["fingerprint"])
            boundary = len(data["input_ids"]) // self.cache.page_size * self.cache.page_size
            ids = array("q", data["input_ids"][:boundary])
            attempt = owner["source_attempt_id"]
        boundary, path_hash = len(ids), token_hash(ids)
        ttl = float(data["ttl_seconds"])
        if not math.isfinite(ttl) or ttl <= 0:
            raise ContextKVError("INVALID_PREFIX_TTL")
        if not isinstance(owner_id, str) or not owner_id:
            raise ContextKVError("INVALID_ID")
        sources = data["sources"]
        identity = path_hash, json.dumps(sources, sort_keys=True, separators=(",", ":"))
        if owner is not None:
            if owner["identity"] != identity:
                raise ContextKVError("PREFIX_OWNER_CONFLICT")
        # This is an intentional mutating match, unlike /cache/quote. It creates
        # a page-aligned native split and applies pending-transfer relocation.
        result = self.cache.tree_core.match_prefix(MatchPrefixParams(key=RadixKey(ids)))
        self.cache._apply_cache_actions(result.cache_actions)
        node = self.cache.tree_core.node_by_id(result.best_match_node)
        length, current = 0, node
        while current is not self.cache.tree_core.root_node:
            length += len(current.key)
            current = current.parent
        if length != boundary:
            raise ContextKVError("PREFIX_NOT_RESIDENT")
        sid = "contextkv:prefix:" + owner_id
        if owner is None and (sid in self.cache.session_refs._session_generations or sid in self.epochs):
            raise ContextKVError("PREFIX_OWNER_CONFLICT")
        generation = owner["generation"] if owner is not None else self.cache.open_radix_session(sid)
        for component in self.cache.session_refs.components:
            component.register_session_leaf(sid, node)
        owner = self.prefix_owners[owner_id] = dict(identity=identity, input_ids=ids,
            path_hash=path_hash, boundary_tokens=boundary, source_attempt_id=attempt,
            sources=sources, native_session_id=sid, generation=generation,
            expires_at=time.monotonic()+ttl)
        receipt = self._prefix_receipt(owner_id, owner)
        self._event("retain_prefix", **receipt)
        return receipt

    def _prefix_receipt(self, owner_id, owner):
        from sglang.srt.mem_cache.radix_cache import RadixKey
        tree = self.cache.tree_core
        ids = owner["input_ids"]
        key = RadixKey(ids)
        view = tree.peek_full_prefix(key)
        page_hashes = executed_prefix_hashes(ids, self.cache.page_size)
        gpu_pages, host_pages, referenced_pages = [], [], []
        current, position = tree.root_node, 0
        while len(key):
            child = current.children.get(key.child_key(self.cache.page_size))
            if child is None:
                break
            matched = child.key.match(key, page_size=self.cache.page_size)
            if matched == 0:
                break
            full = child.component_data[0]
            for end in range(position+self.cache.page_size, position+matched+1, self.cache.page_size):
                page_id = self.prefix_scope_hash + ":" + page_hashes[end]
                if full.value is not None and child.contextkv_h2d_pending_id is None:
                    gpu_pages.append(page_id)
                if full.host_value is not None and child.write_through_pending_id is None:
                    host_pages.append(page_id)
                if full.session_ref:
                    referenced_pages.append(page_id)
            position += matched
            if matched < len(child.key):
                break
            current, key = child, key[matched:]
        sid = owner["native_session_id"]
        reference_present = any(component._session_leaves.get(sid)
                                for component in self.cache.session_refs.components)
        return dict(owner_id=owner_id, native_session_id=sid,
            session_generation=owner["generation"], engine_restart_id=self.restart_id,
            cache_config_hash=self.cache_config_hash, fingerprint=self.fingerprint,
            cache_scope_hash=self.prefix_scope_hash,
            source_attempt_id=owner["source_attempt_id"], sources=owner["sources"],
            prefix_hash=owner["path_hash"], boundary_tokens=owner["boundary_tokens"],
            page_size=self.cache.page_size, expires_at=owner["expires_at"],
            clock="server_monotonic", state="RETAINED", reference_present=reference_present,
            gpu_page_ids=gpu_pages, host_page_ids=host_pages,
            referenced_page_ids=referenced_pages, residency=view,
            pool=self.pool_snapshot(), workload=self.workload_snapshot(),
            protection="native_session_soft_priority_both_tiers")

    def release_prefix(self, data):
        self._restart(data["expected_restart_id"])
        owner_id = data["owner_id"]
        owner = self.prefix_owners.pop(owner_id, None)
        if owner is not None:
            self.cache.release_radix_session(owner["native_session_id"])
        receipt = dict(owner_id=owner_id, engine_restart_id=self.restart_id,
            state="RELEASED", acknowledged=True, freed_gpu_tokens=0, freed_host_tokens=0)
        self._event("release_prefix", **receipt)
        return receipt

    def expire_prefixes(self):
        now = time.monotonic()
        for owner_id, owner in tuple(self.prefix_owners.items()):
            if now >= owner["expires_at"]:
                self.release_prefix(dict(owner_id=owner_id, expected_restart_id=self.restart_id))

    def prefix_status(self, data):
        self._restart(data["expected_restart_id"])
        if "input_ids" in data:
            ids = self._executed_prefix(data)
            receipt = self._prefix_receipt(None, dict(input_ids=ids, native_session_id=None,
                generation=None, source_attempt_id=data["attempt_id"], sources=data["sources"],
                path_hash=token_hash(ids), boundary_tokens=len(ids), expires_at=None))
            return {**receipt, "state": "OBSERVED", "protection": "none"}
        owner_id = data.get("owner_id")
        if owner_id is not None:
            owner = self.prefix_owners.get(owner_id)
            return self._prefix_receipt(owner_id, owner) if owner is not None else dict(
                owner_id=owner_id, state="RELEASED", engine_restart_id=self.restart_id)
        return dict(engine_restart_id=self.restart_id,
            prefixes=[self._prefix_receipt(key, owner) for key, owner in self.prefix_owners.items()])

    def _release(self, sid, epoch):
        self.cache.release_radix_session(sid)
        epoch["state"] = "RETIRED"
        self.host_admission.retire(sid)
        self._complete_operation(epoch["operation_id"])
        for op in epoch.pop("retire_operations", ()):
            self._complete_operation(op)
        self.retired_epochs[sid] = None
        while len(self.retired_epochs) > self.receipt_limit:
            old, _ = self.retired_epochs.popitem(last=False)
            del self.epochs[old]
        self._event("retire_ref", session_id=sid, freed_gpu_tokens=0, freed_host_tokens=0)

    def _operation(self, operation_id):
        return self.operations.get(operation_id) or self.operation_receipts.get(operation_id)

    def _complete_operation(self, operation_id):
        """Keep active operations until completion; bound only terminal receipts."""
        self.operation_receipts[operation_id] = self.operations.pop(operation_id)
        self.operation_receipts.move_to_end(operation_id)
        while len(self.operation_receipts) > self.receipt_limit:
            self.operation_receipts.popitem(last=False)

    def retire(self, data):
        self._restart(data["expected_restart_id"])
        sid, generation, op = (data["engine_session_id"], data["expected_generation"], data["operation_id"])
        identity = ("retire", sid, generation)
        previous = self._operation(op)
        if previous is not None and previous != identity:
            raise ContextKVError("OPERATION_CONFLICT")
        epoch = self.epochs.get(sid)
        self.operations[op] = identity
        if epoch is None or epoch["generation"] != generation:
            native_generation = self.cache.session_refs._session_generations.get(sid)
            # A lost retirement ACK can outlive the bounded metadata. Absence
            # or a newer native incarnation proves this old reference is gone;
            # acknowledge it without releasing or modifying the live incarnation.
            if native_generation is not None and native_generation <= generation:
                del self.operations[op]
                raise ContextKVError("STALE_GENERATION")
            self._complete_operation(op)
            state = "RETIRED"
        elif epoch["state"] != "RETIRED":
            epoch.setdefault("retire_operations", set()).add(op)
            epoch["state"] = "RETIRING"
            if not epoch["inflight"]:
                self._release(sid, epoch)
            state = epoch["state"]
        else:
            self._complete_operation(op)
            state = "RETIRED"
        return dict(engine_session_id=sid, session_generation=generation,
            engine_restart_id=self.restart_id, operation_id=op, state=state,
            acknowledged=state == "RETIRED", freed_gpu_tokens=0, freed_host_tokens=0)

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
        operation = self._operation(data.get("operation_id"))
        if sid is None and operation is not None:
            sid = operation[1]
        epoch = self.epochs.get(sid)
        return dict(state=epoch["state"] if epoch else "UNKNOWN", engine_restart_id=self.restart_id,
                    engine_session_id=sid, session_generation=epoch["generation"] if epoch else None,
                    inflight=bool(epoch and epoch["inflight"]), terminal=bool(epoch and epoch["state"] == "RETIRED"))

    def workload_snapshot(self):
        """Scheduler-visible logical work, never uniquely owned physical slots.

        Native in-flight collection handles overlap and repeated batch membership.
        Waiting wins over in-flight membership during retraction; a chunked
        request can be absent from both batches. Existing native IDs and scalar
        summaries leave the scheduler; token arrays and cache state do not change.
        """
        if self.scheduler is None:
            return dict(active_requests=None, queued_requests=None,
                        native_running_requests=None, native_waiting_requests=None,
                        native_new_token_ratio=None, native_max_new_tokens_clip=None,
                        native_running_budget_tokens=None, native_running_budget_requests=None)
        from sglang.srt.managers.schedule_policy import CLIP_MAX_NEW_TOKENS, PrefillAdder

        # Keep the exact current scheduler.running_batch membership. PrefillAdder
        # sums it without filtering finished/retracted requests. This is NOT the
        # broader overlap/last-batch/chunk union exported for service work below.
        budget_requests = self.scheduler.running_batch.reqs
        ratio = self.scheduler.new_token_ratio_tracker.current
        offset_owner = SimpleNamespace(new_token_ratio=ratio)
        running_budget = sum(PrefillAdder._get_running_request_total_token_offset(offset_owner, req)
                             for req in budget_requests)
        waiting = [req for req in self.scheduler.waiting_queue if not req.finished()]
        running = set(self.scheduler.running_batch.reqs)
        if self.scheduler.last_batch is not None:
            running.update(self.scheduler.last_batch.reqs)
        if self.scheduler.cur_batch_for_debug is not None:
            running.update(self.scheduler.cur_batch_for_debug.reqs)
        if self.scheduler.enable_overlap:
            for batch, _ in self.scheduler.result_queue:
                running.update(batch.reqs)
        if self.scheduler.chunked_req is not None:
            running.add(self.scheduler.chunked_req)
        running.difference_update(waiting)
        running = [req for req in running if not req.finished()]

        def summary(req):
            return dict(request_id=req.rid, input_tokens=len(req.origin_input_ids),
                        generated_tokens=len(req.output_ids),
                        max_new_tokens=req.sampling_params.max_new_tokens,
                        ignore_eos=req.sampling_params.ignore_eos,
                        kv_allocated_tokens=req.kv.kv_allocated_len if req.kv is not None else 0,
                        kv_committed_tokens=req.kv_committed_len,
                        cache_protected_tokens=req.cache_protected_len,
                        native_finished=req.finished(),
                        last_native_match=dict(req.contextkv_last_native_match)
                            if getattr(req, "contextkv_last_native_match", None) is not None else None)

        return dict(active_requests=len(running), queued_requests=len(waiting),
                    native_running_requests=[summary(req) for req in running],
                    native_waiting_requests=[summary(req) for req in waiting],
                    native_new_token_ratio=ratio, native_max_new_tokens_clip=CLIP_MAX_NEW_TOKENS,
                    native_running_budget_tokens=running_budget,
                    native_running_budget_requests=[summary(req) for req in budget_requests])

    def record_native_match(self, req, input_tokens):
        """Observe a completed native match; overwrite bounded per-request scalars.

        Called by match_kv_cache, never by quote. Subsequent locking, load-back,
        eviction or retraction can change residency: timestamp is observation age,
        not a promise that the saved tier layout remains valid.
        """
        req.contextkv_last_native_match = dict(observed_time_ns=time.monotonic_ns(),
            input_tokens=input_tokens, device_prefix_tokens=len(req.prefix_indices),
            host_hit_tokens=req.host_hit_length, matched_prefix_tokens=req.num_matched_prefix_tokens)
        if self.host_admission.mode != "off" and not getattr(req, "contextkv_host_admission_observed", False):
            req.contextkv_host_admission_observed = True
            epoch = self.epochs.get(req.session_id)
            if epoch and epoch.get("business_session_id") is not None and epoch["state"] != "RETIRED":
                self.host_admission.observe(req.session_id, epoch["business_session_id"],
                                            len(req.origin_input_ids), time.monotonic())

    def allow_host_backup(self, req, new_tokens):
        """Called before allocation/locking; no match, tree walk or DMA here."""
        policy = self.host_admission
        if policy.mode == "off":
            return True
        epoch = self.epochs.get(req.session_id)
        if epoch is None or epoch.get("business_session_id") is None:
            policy.counters["unknown_business_actions_native"] += 1
            return True
        host = self.cache.cache_controller.mem_pool_host
        decision = policy.decide(new_tokens, self.cache.page_size, int(host.size),
                                 int(host.size-host.available_size()), time.monotonic(),
                                 admitted_pages=getattr(req, "contextkv_host_backup_admitted_pages", 0))
        self._event("host_backup_admission", rid=req.rid, engine_session_id=req.session_id,
                    business_session_id=epoch["business_session_id"], **decision)
        return decision["allowed"]

    def record_host_backup_admitted(self, req, new_tokens):
        """Native host allocation/commit succeeded; DMA may still be queued."""
        epoch = self.epochs.get(req.session_id)
        if epoch is None or epoch.get("business_session_id") is None:
            return
        pages = -(-new_tokens//self.cache.page_size)
        req.contextkv_host_backup_admitted_pages = getattr(req, "contextkv_host_backup_admitted_pages", 0) + pages
        self.host_admission.counters["admitted_new_tokens"] += new_tokens
        self.host_admission.counters["admitted_new_pages"] += pages

    def record_host_capacity_eviction(self, tokens):
        if self.host_admission.mode != "off":
            self.host_admission.capacity_evicted(tokens, time.monotonic())

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
            reasons = []
            if free_gpu < max(0, len(ids) - view["ready_gpu_prefix_tokens"]):
                reasons.append("capacity_pressure")
            if view["inflight_h2d_tokens"] or view["inflight_d2h_tokens"]:
                reasons.append("inflight_transfer")
            quotes.append(dict(candidate_id=c["candidate_id"], input_hash=c["input_hash"],
                input_len=len(ids), match_limit=limit, **view,
                native_load_estimate=native_load, native_estimate_uncertain=bool(reasons),
                uncertainty_reasons=reasons))
        return dict(request_id=data["request_id"], observed_seq=self.observed_seq,
                    observed_time_ns=time.monotonic_ns(), engine_restart_id=self.restart_id,
                    cache_config_hash=self.cache_config_hash,
                    free_gpu_tokens=free_gpu, free_host_tokens=free_host,
                    gpu_capacity_tokens=int(self.cache.token_to_kv_pool_allocator.size),
                    evictable_gpu_tokens=int(self.cache.full_evictable_size()),
                    evictable_host_tokens=None,
                    quotes=quotes, fingerprint=self.fingerprint, host_quote_supported=True,
                    **self.workload_snapshot())

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
        """Expire independent refs, then optionally sample the scheduler pool."""
        self.expire_prefixes()
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
                    admission_diagnostics_enabled=self.admission_diagnostics_enabled,
                    control_metadata=dict(active_epochs=len(self.epochs)-len(self.retired_epochs),
                        retained_retired_epochs=len(self.retired_epochs),
                        active_operations=len(self.operations),
                        retained_operation_receipts=len(self.operation_receipts),
                        terminal_record_limit=self.receipt_limit),
                    host_admission=self.host_admission.snapshot(time.monotonic()),
                    **self.workload_snapshot(),
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
            replacement = self._replace_after_cache_reset(self.scheduler)
            return dict(mode=mode, info=replacement.info({}), pool=replacement.pool_snapshot())
        if mode == "settle":
            if self.cache.cache_controller is not None:
                self.cache.cache_controller.start_writing()
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
                self.cache.cache_controller.start_writing()
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
            self.cache.cache_controller.start_writing()
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
    cache = scheduler.tree_cache
    if runtime is not None and runtime.cache is cache:
        if runtime._native_generations is cache.session_refs._session_generations:
            return runtime
        # Native /flush_cache also resets generations. Rotate the existing
        # restart fence before any control operation or guarded admission.
        return runtime._replace_after_cache_reset(scheduler)
    from sglang.srt.runtime_context import get_parallel, get_model, get_server_args
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    parallel = get_parallel()
    if (parallel.tp_size != 1 or get_server_args().dp_size != 1 or parallel.pp_size != 1
            or not isinstance(cache, UnifiedRadixCache) or not cache.enable_session_radix_cache
            or not isinstance(cache.tree_core, UnifiedTreeCore) or len(cache._components_tuple) != 1
            or cache.tree_core.is_eagle or cache.enable_storage
            or scheduler.dllm_config is not None
            or scheduler.rust_server is not None
            or not (scheduler.spec_algorithm.is_none() or scheduler.spec_algorithm.is_ngram())
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
    # Actual engine admission requires an explicitly selected local Qwen3
    # snapshot. This one-time check has no legacy model/revision fallback.
    scope_path = os.environ.get("CONTEXTKV_MODEL_SCOPE_PATH")
    if not scope_path:
        raise ContextKVError("MODEL_SCOPE_REQUIRED", "set CONTEXTKV_MODEL_SCOPE_PATH to the local Qwen3 scope")
    try:
        from sglang.srt.managers.contextkv_model_scope import load_model_scope
        from sglang.srt.runtime_context import get_server_args
        scope = load_model_scope(scope_path)
        scope.verify_native_settings(get_server_args())
        scope.verify_files(get_model().model_path)
        scope.verify_tokenizer(scheduler.tokenizer)
        scope.verify_fingerprint(fp)
        model_scope = scope.manifest()
    except (AssertionError, AttributeError, OSError, ValueError) as exc:
        raise ContextKVError("FINGERPRINT_MISMATCH", "running model config/revision/tokenizer/template differs") from exc
    # Structured JSON and speculative decoding both change measured execution
    # and output lengths, so bind them to the calibration identity.
    from dataclasses import asdict
    resolved_args = asdict(get_server_args())
    decoder_settings = {name: value for name, value in resolved_args.items()
                        if name.startswith("constrained_json_") or name == "grammar_backend"}
    if not scheduler.spec_algorithm.is_none():
        decoder_settings.update({name: value for name, value in resolved_args.items()
            if name.startswith("speculative_ngram_") or name in {
                "speculative_algorithm", "speculative_num_draft_tokens",
                "speculative_accept_threshold_single", "speculative_accept_threshold_acc",
                "speculative_adaptive_config"}})
    scheduler.contextkv_runtime = ContextKVRuntime(cache, fp, scheduler=scheduler,
        model_scope=model_scope, decoder_settings=decoder_settings,
        transfer_tma_enabled=False,
        result_ttl_seconds=float(os.environ.get("CONTEXTKV_RESULT_TTL_SECONDS", "900")),
        result_max_bytes=int(os.environ.get("CONTEXTKV_RESULT_MAX_BYTES", str(64 << 20))))
    return scheduler.contextkv_runtime
