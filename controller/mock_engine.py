"""Deterministic CPU protocol simulator, never a model/performance substitute.

The trie models unique physical token slots separately from epoch soft refs.
Pending transfers do not become ready during quotes. All state changes share one
owner lock; generation waits outside that lock so retire can close admission.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
import uuid

from controller.engine_client import (EngineError, EngineInfo, EpochStatus, PrepareAck,
                                      RetireAck, UnknownOutcomeError, token_hash)
from controller.schema import (CacheQuote, CacheQuoteBatch, CompletionReceipt,
                               ConfigFingerprint, GenerationResult)


@dataclass
class _Slot:
    gpu: bool = False
    host: bool = False
    h2d: bool = False
    d2h: bool = False
    refs: set[str] = field(default_factory=set)
    touched: int = 0


@dataclass
class _Epoch:
    generation: int
    operation_id: str
    state: str = "PREPARED"
    inflight: set[str] = field(default_factory=set)


class MockEngine:
    """Executable fault-injection model. Timing is CPU wall time, not GPU cost."""
    def __init__(self, fingerprint=None, *, gpu_capacity_tokens=65536,
                 host_capacity_tokens=131072, page_size=1,
                 response_text="模拟回答：请依据当前证据核对配置与适用版本。",
                 generation_delay=0.0, receipt_limit=1024):
        self.fingerprint = fingerprint or ConfigFingerprint()
        self.gpu_capacity_tokens = gpu_capacity_tokens
        self.host_capacity_tokens = host_capacity_tokens
        if page_size < 1:
            raise ValueError("page_size must be positive")
        self.page_size = page_size
        self.response_text = response_text
        self.generation_delay = generation_delay
        self.receipt_limit = receipt_limit
        self.fail_next_generate = False
        self.drop_next_generate_response = False
        self.drop_next_retire_response = False
        self._lock = asyncio.Lock()
        self._reset()

    def _reset(self):
        self.engine_restart_id = uuid.uuid4().hex
        self._epochs: dict[str, _Epoch] = {}
        self._retire_operations: dict[str, tuple] = {}
        self._prepare_operations: dict[str, str] = {}
        self._attempts = OrderedDict()
        self._children: dict[tuple[int, int], int] = {}
        self._slots: dict[int, _Slot] = {}
        self._next_slot = 1
        self._generation = 0
        self._seq = 0
        self.events: list[dict] = []

    async def restart(self):
        async with self._lock:
            self._reset()

    def _check_restart(self, expected):
        if expected != self.engine_restart_id:
            raise EngineError("STALE_RESTART")

    def _check_fingerprint(self, fingerprint):
        if fingerprint != self.fingerprint:
            raise EngineError("FINGERPRINT_MISMATCH")
        if fingerprint.cache_salt is not None or fingerprint.extra_key is not None:
            raise EngineError("UNSUPPORTED_NAMESPACE", "v1 only supports default cache key")

    def _event(self, kind, **details):
        self._seq += 1
        self.events.append({"event": kind, "seq": self._seq,
                            "observed_time_ns": time.monotonic_ns(), "source": "mock", **details})

    def _usage(self):
        return sum(s.gpu or s.h2d for s in self._slots.values()), sum(s.host or s.d2h for s in self._slots.values())

    @property
    def unique_gpu_tokens(self):
        return self._usage()[0]

    @property
    def unique_host_tokens(self):
        return self._usage()[1]

    def state_digest(self):
        """A complete mutation-sensitive snapshot for purity assertions."""
        value = {"children": sorted((p, t, n) for (p, t), n in self._children.items()),
                 "slots": {i: {**asdict(s), "refs": sorted(s.refs)} for i, s in self._slots.items()},
                 "epochs": {k: {**asdict(v), "inflight": sorted(v.inflight)} for k, v in self._epochs.items()},
                 "seq": self._seq, "events": self.events, "next": self._next_slot}
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    async def info(self):
        return EngineInfo(engine_restart_id=self.engine_restart_id, fingerprint=self.fingerprint,
                          metrics_supported=True, result_replay_supported=True,
                          cache_config_hash=f"mock-trie-p{self.page_size}-{self.gpu_capacity_tokens}-{self.host_capacity_tokens}")

    def _walk(self, ids):
        parent = 0
        for token in ids:
            child = self._children.get((parent, token))
            if child is None:
                return
            yield child, self._slots[child]
            parent = child

    def _quote(self, candidate):
        limit = max(len(candidate.input_ids)-1, 0)
        limit = limit // self.page_size * self.page_size
        raw = gpu = recoverable = host_only = h2d = d2h = 0
        gpu_chain = recover_chain = True
        for _, slot in self._walk(candidate.input_ids[:limit]):
            raw += 1
            gpu_ready = slot.gpu and not slot.h2d
            host_ready = slot.host and not slot.d2h
            gpu_chain = gpu_chain and gpu_ready
            recover_chain = recover_chain and (gpu_ready or host_ready)
            gpu += int(gpu_chain)
            recoverable += int(recover_chain)
            host_only += int(recover_chain and host_ready and not slot.gpu and not slot.h2d)
            h2d += int(slot.h2d)
            d2h += int(slot.d2h)
        return CacheQuote(candidate_id=candidate.candidate_id, input_hash=candidate.input_hash,
                          input_len=len(candidate.input_ids), match_limit=limit,
                          raw_match_tokens=raw, ready_gpu_prefix_tokens=gpu,
                          recoverable_prefix_tokens=recoverable, host_only_load_tokens=host_only,
                          inflight_h2d_tokens=h2d, inflight_d2h_tokens=d2h,
                          native_load_estimate=host_only, native_estimate_uncertain=bool(h2d or d2h))

    async def quote_batch(self, candidates, fingerprint):
        async with self._lock:
            self._check_fingerprint(fingerprint)
            if len({c.snapshot_version for c in candidates}) > 1:
                raise EngineError("MIXED_SNAPSHOT")
            for c in candidates:
                if c.input_hash != token_hash(c.input_ids):
                    raise EngineError("INPUT_HASH_MISMATCH")
            gpu, host = self._usage()
            return CacheQuoteBatch(request_id=uuid.uuid4().hex, observed_seq=self._seq,
                observed_time_ns=time.monotonic_ns(), engine_restart_id=self.engine_restart_id,
                cache_config_hash=(await self.info()).cache_config_hash,
                free_gpu_tokens=self.gpu_capacity_tokens-gpu, free_host_tokens=self.host_capacity_tokens-host,
                quotes=tuple(self._quote(c) for c in candidates), fingerprint=fingerprint,
                evictable_gpu_tokens=sum(s.gpu and not s.refs and not s.h2d for s in self._slots.values()),
                evictable_host_tokens=sum(s.host and not s.refs and not s.d2h for s in self._slots.values()))

    async def prepare(self, engine_session_id, operation_id, expected_restart_id):
        async with self._lock:
            self._check_restart(expected_restart_id)
            epoch = self._epochs.get(engine_session_id)
            if operation_id in self._retire_operations or (operation_id in self._prepare_operations
                    and self._prepare_operations[operation_id] != engine_session_id):
                raise EngineError("OPERATION_CONFLICT")
            if epoch:
                if epoch.operation_id != operation_id or epoch.state not in {"PREPARED", "COMMITTED"}:
                    raise EngineError("EPOCH_CONFLICT")
            else:
                self._prepare_operations[operation_id] = engine_session_id
                self._generation += 1
                epoch = self._epochs[engine_session_id] = _Epoch(self._generation, operation_id)
                self._event("prepare", session_id=engine_session_id)
            return PrepareAck(engine_session_id, epoch.generation, self.engine_restart_id,
                              operation_id, epoch.state)

    def _guard(self, session_id, generation, restart_id):
        self._check_restart(restart_id)
        epoch = self._epochs.get(session_id)
        if epoch is None or epoch.generation != generation:
            raise EngineError("STALE_GENERATION")
        if epoch.state not in {"PREPARED", "COMMITTED"}:
            raise EngineError("EPOCH_RETIRED")
        return epoch

    def _store_ids(self, ids, *, gpu=False, host=False, h2d=False, d2h=False, session_id=None):
        parent = 0
        for token in ids:
            index = self._children.get((parent, token))
            if index is None:
                index = self._next_slot
                self._next_slot += 1
                self._children[(parent, token)] = index
                self._slots[index] = _Slot()
            slot = self._slots[index]
            old_gpu, old_host = slot.gpu or slot.h2d, slot.host or slot.d2h
            slot.gpu |= gpu
            slot.host |= host
            slot.h2d = (slot.h2d and not gpu) or h2d
            slot.d2h = (slot.d2h and not host) or d2h
            if session_id:
                slot.refs.add(session_id)
            slot.touched = self._seq
            if not old_gpu and (gpu or h2d):
                self._event("alloc", tier="gpu", slots=1)
            if not old_host and (host or d2h):
                self._event("alloc", tier="host", slots=1)
            parent = index

    async def seed_cache(self, input_ids, *, tier="gpu", pending=False):
        """Test fixture insertion; does not participate in production planning."""
        async with self._lock:
            self._store_ids(input_ids, gpu=tier == "gpu" and not pending,
                            host=tier == "host" and not pending,
                            h2d=tier == "gpu" and pending, d2h=tier == "host" and pending)

    async def complete_transfers(self):
        async with self._lock:
            for slot in self._slots.values():
                slot.gpu |= slot.h2d
                slot.host |= slot.d2h
                slot.h2d = slot.d2h = False
            self._event("transfer_complete")

    def _admit(self, ids):
        existing = {i for i, _ in self._walk(ids)}
        gpu, _ = self._usage()
        needed = len(ids) - sum(self._slots[i].gpu or self._slots[i].h2d for i in existing)
        shortage = gpu + needed - self.gpu_capacity_tokens
        if shortage > 0:
            # Soft references prefer retention; they never authorize hard locks.
            evictable = sorted(((bool(s.refs), s.touched, i) for i, s in self._slots.items()
                                if s.gpu and i not in existing and not s.h2d))
            if len(evictable) < shortage:
                raise EngineError("CAPACITY_EXCEEDED")
            for _, _, i in evictable[:shortage]:
                self._slots[i].gpu = False
                self._event("free", tier="gpu", slots=1)

    async def generate(self, *, input_ids, session_id, expected_generation, expected_restart_id,
                       attempt_id, fingerprint, max_new_tokens=128, ignore_eos=False):
        started = time.perf_counter()
        ids = tuple(input_ids)
        input_hash = token_hash(ids)
        identity = (session_id, expected_generation, input_hash)
        async with self._lock:
            self._check_fingerprint(fingerprint)
            epoch = self._guard(session_id, expected_generation, expected_restart_id)
            existing = self._attempts.get(attempt_id)
            if existing:
                if existing["identity"] != identity:
                    raise EngineError("ATTEMPT_CONFLICT")
                if existing["result"] is not None:
                    return existing["result"]
                raise EngineError("ATTEMPT_INFLIGHT" if not existing["terminal"] else "ATTEMPT_TERMINAL")
            if epoch.inflight:
                raise EngineError("EPOCH_BUSY")
            if not ids or max_new_tokens < 1:
                raise EngineError("INVALID_GENERATION_REQUEST")
            epoch.inflight.add(attempt_id)
            self._attempts[attempt_id] = {"identity": identity, "result": None, "terminal": False, "state": "INFLIGHT"}
            self._event("generate_start", attempt_id=attempt_id)
        try:
            if self.generation_delay:
                await asyncio.sleep(self.generation_delay)
            async with self._lock:
                self._check_restart(expected_restart_id)
                if self.fail_next_generate:
                    self.fail_next_generate = False
                    raise EngineError("GENERATION_FAILED")
                from types import SimpleNamespace
                quoted = self._quote(SimpleNamespace(input_ids=ids, candidate_id=attempt_id, input_hash=input_hash))
                output = tuple(self.response_text.encode("utf-8"))[:max_new_tokens]
                text = bytes(output).decode("utf-8", errors="replace")
                cached = (ids + output)[:-1]
                cached = cached[:len(cached)//self.page_size*self.page_size]
                self._admit(cached)
                # Retire has already blocked admission. An older admitted request
                # may finish, but must never recreate that epoch's soft refs.
                registered = epoch.state in {"PREPARED", "COMMITTED"} and bool(cached)
                self._store_ids(cached, gpu=True, session_id=session_id if registered else None)
                if registered:
                    epoch.state = "COMMITTED"
                finish = "length" if len(output) < len(self.response_text.encode("utf-8")) else "stop"
                receipt = CompletionReceipt(attempt_id, attempt_id, session_id, expected_generation,
                    self.engine_restart_id, input_hash, finish, True,
                    quoted.ready_gpu_prefix_tokens, quoted.host_only_load_tokens,
                    len(ids)-quoted.recoverable_prefix_tokens, len(cached), registered)
                result = GenerationResult(text, output, finish, receipt,
                                          elapsed_ms=(time.perf_counter()-started)*1000)
                self._attempts[attempt_id].update(result=result, terminal=True, state="COMPLETED")
                self._event("generate_complete", attempt_id=attempt_id)
        except BaseException:
            async with self._lock:
                if attempt_id in self._attempts:
                    self._attempts[attempt_id].update(terminal=True, state="FAILED")
            raise
        finally:
            async with self._lock:
                epoch.inflight.discard(attempt_id)
                if epoch.state == "RETIRING" and not epoch.inflight:
                    self._finish_retire(session_id, epoch)
                # Terminal receipts are bounded; absent receipts mean UNKNOWN,
                # never proof that a request did not execute.
                for key in list(self._attempts):
                    if len(self._attempts) <= self.receipt_limit:
                        break
                    if self._attempts[key]["terminal"]:
                        del self._attempts[key]
        if self.drop_next_generate_response:
            self.drop_next_generate_response = False
            raise UnknownOutcomeError("RESPONSE_LOST")
        return result

    def _finish_retire(self, session_id, epoch):
        for slot in self._slots.values():
            slot.refs.discard(session_id)
        epoch.state = "RETIRED"
        self._event("retire_ref", session_id=session_id, freed_gpu_tokens=0, freed_host_tokens=0)

    async def retire(self, engine_session_id, expected_generation, operation_id,
                     expected_restart_id, reason="replaced"):
        async with self._lock:
            self._check_restart(expected_restart_id)
            identity = (engine_session_id, expected_generation)
            if operation_id in self._prepare_operations or (operation_id in self._retire_operations
                    and self._retire_operations[operation_id] != identity):
                raise EngineError("OPERATION_CONFLICT")
            epoch = self._epochs.get(engine_session_id)
            if epoch is None or epoch.generation != expected_generation:
                raise EngineError("STALE_GENERATION")
            self._retire_operations[operation_id] = identity
            if epoch.state != "RETIRED":
                epoch.state = "RETIRING"
                if not epoch.inflight:
                    self._finish_retire(engine_session_id, epoch)
            ack = RetireAck(engine_session_id, expected_generation, self.engine_restart_id,
                            operation_id, state=epoch.state, acknowledged=epoch.state == "RETIRED")
        if self.drop_next_retire_response:
            self.drop_next_retire_response = False
            raise UnknownOutcomeError("RETIRE_RESPONSE_LOST")
        return ack

    async def status(self, engine_session_id=None, attempt_id=None, operation_id=None,
                     expected_restart_id=None):
        async with self._lock:
            if expected_restart_id is not None:
                self._check_restart(expected_restart_id)
            if attempt_id is not None:
                item = self._attempts.get(attempt_id)
                if item is None:
                    return EpochStatus("UNKNOWN", self.engine_restart_id, attempt_id=attempt_id)
                result = item["result"]
                return EpochStatus(item["state"], self.engine_restart_id,
                    item["identity"][0], item["identity"][1], attempt_id,
                    not item["terminal"], item["terminal"], result,
                    result.receipt if result else None)
            if engine_session_id is None and operation_id is not None:
                engine_session_id = next((sid for sid, e in self._epochs.items() if e.operation_id == operation_id), None)
                if engine_session_id is None and operation_id in self._retire_operations:
                    engine_session_id = self._retire_operations[operation_id][0]
            epoch = self._epochs.get(engine_session_id)
            if epoch is None:
                return EpochStatus("UNKNOWN", self.engine_restart_id, engine_session_id)
            return EpochStatus(epoch.state, self.engine_restart_id, engine_session_id,
                               epoch.generation, inflight=bool(epoch.inflight), terminal=epoch.state == "RETIRED")

    async def metrics(self, after_seq=0):
        """Exact simulated slot events, explicitly distinguished from hardware."""
        async with self._lock:
            gpu, host = self._usage()
            return dict(engine_restart_id=self.engine_restart_id, observed_seq=self._seq,
                observed_time_ns=time.monotonic_ns(), clock="server_monotonic", source="mock",
                pool=dict(gpu_capacity_tokens=self.gpu_capacity_tokens,
                    host_capacity_tokens=self.host_capacity_tokens,
                    free_gpu_tokens=self.gpu_capacity_tokens-gpu,
                    free_host_tokens=self.host_capacity_tokens-host,
                    unique_gpu_tokens=gpu, unique_host_tokens=host, page_size=self.page_size),
                active_requests=sum(len(epoch.inflight) for epoch in self._epochs.values()),
                queued_requests=0, events=[e for e in self.events if e["seq"] > after_seq],
                next_seq=self._seq, dropped_before_seq=0, events_truncated=False,
                pool_sample_scope="mock_exact_slots")
