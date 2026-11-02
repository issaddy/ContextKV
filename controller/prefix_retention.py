"""Independent evidence-prefix ownership and marginal native-cache value."""
from __future__ import annotations

import asyncio
from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from collections import defaultdict
import time
from pathlib import Path

from controller.schema import CompletionReceipt, ConfigFingerprint
from controller.serialization import to_jsonable


@dataclass(frozen=True)
class PrefixForecast:
    horizon_seconds: float
    reuse_probability: float
    native_reuse_cost_ms: float | None
    retained_reuse_cost_ms: float | None
    extra_gpu_seconds_per_page: float
    extra_host_seconds_per_page: float
    cache_scope_hash: str
    evidence_digest: str
    retention_control_ms: float = 0.0
    expected_collateral_saved_ms: float = 0.0
    native_window_cost_ms: float | None = None
    retained_window_cost_ms: float | None = None

    @property
    def saved_ms(self):
        benefit = (self.reuse_probability * (self.native_reuse_cost_ms-self.retained_reuse_cost_ms)
                   if self.reuse_probability else 0.0)
        return benefit+self.expected_collateral_saved_ms-self.retention_control_ms


@dataclass(frozen=True)
class RetentionRequest:
    owner_id: str
    forecast: PrefixForecast
    gpu_page_ids: frozenset[str]
    host_page_ids: frozenset[str]
    page_size: int

    @classmethod
    def from_receipt(cls, receipt, forecast):
        if receipt["cache_scope_hash"] != forecast.cache_scope_hash:
            raise ValueError("retention forecast differs from the executing reader/cache scope")
        # Fit normalizes future tier residency by the full eligible token path.
        # A currently Host-only page can occupy GPU again within this horizon.
        # These are logical protection capacities; the receipt retains the exact
        # presently ready replica sets for physical-residency observations.
        path = frozenset(receipt["gpu_page_ids"]) | frozenset(receipt["host_page_ids"])
        if len(path)*receipt["page_size"] != receipt["boundary_tokens"]:
            raise ValueError("retention needs the full recoverable causal page path")
        host = path if receipt["pool"]["host_capacity_tokens"] else frozenset()
        return cls(receipt["owner_id"], forecast, path, host, receipt["page_size"])


@dataclass(frozen=True)
class RetentionChoice:
    owner_id: str
    ttl_seconds: float
    expected_saved_ms: float
    extra_gpu_token_seconds: float
    extra_host_token_seconds: float
    value: float
    marginal_gpu_tokens: int
    marginal_host_tokens: int
    evidence_digest: str


def allocate_retention(requests, *, gpu_budget_tokens, host_budget_tokens,
                       lambda_gpu, lambda_host, protected_gpu_pages=frozenset(),
                       protected_host_pages=frozenset()):
    """Choose positive marginal value, counting a shared causal page only once.

    The forecasts come from paired native/retained traces. Occupancy penalties
    are their incremental residency, rather than the full current cache size.
    Capacity budgets bound the protected set; soft references allocate no KV.
    """
    requests = {request.owner_id: request for request in requests}
    gpu_pages, host_pages = set(protected_gpu_pages), set(protected_host_pages)
    choices = []
    while requests:
        candidates = []
        for request in requests.values():
            gpu = len(request.gpu_page_ids-gpu_pages)*request.page_size
            host = len(request.host_page_ids-host_pages)*request.page_size
            if not gpu and not host:
                continue  # Another chosen owner already gives this entire path soft protection.
            if ((len(gpu_pages | request.gpu_page_ids)*request.page_size > gpu_budget_tokens)
                    or (len(host_pages | request.host_page_ids)*request.page_size > host_budget_tokens)):
                continue
            forecast = request.forecast
            gpu_integral = gpu*forecast.extra_gpu_seconds_per_page
            host_integral = host*forecast.extra_host_seconds_per_page
            value = forecast.saved_ms-lambda_gpu*gpu_integral-lambda_host*host_integral
            if value > 0:
                choice = RetentionChoice(request.owner_id, forecast.horizon_seconds,
                    forecast.saved_ms, gpu_integral, host_integral, value, gpu, host,
                    forecast.evidence_digest)
                candidates.append(choice)
        if not candidates:
            break
        choice = max(candidates, key=lambda item: (item.value, item.owner_id))
        selected = requests.pop(choice.owner_id)
        gpu_pages.update(selected.gpu_page_ids)
        host_pages.update(selected.host_page_ids)
        choices.append(choice)
    return tuple(choices)


def retainable_prefix(receipt):
    residency = receipt["residency"]
    return (receipt["boundary_tokens"] > 0
        and residency["recoverable_prefix_tokens"] >= receipt["boundary_tokens"]
        and not residency["inflight_h2d_tokens"] and not residency["inflight_d2h_tokens"])


class EmpiricalRetentionModel:
    """Conditional estimates fitted to fixed-DAG/token paired observations.

    Each input row declares raw features, cache configuration, horizon and
    the actual cost/residency in its native and retained arm. A new cache recipe
    needs its own measurements; unobserved feature buckets have no prediction.
    """
    NUMERIC_BINS = {
        "prior_access_count": (0, 1, 2, 4, 8, 16),
        "last_access_interval_seconds": (1, 5, 30, 120, 600, 3600),
        "interval_seconds": (1, 5, 30, 120, 600, 3600),
        "source_turn_gap": (0, 1, 2, 4, 8, 16),
        "gpu_pressure": (.25, .5, .75, .9, .95),
        "host_pressure": (.25, .5, .75, .9, .95),
        "gpu_prefix_tokens": (64, 256, 1024, 4096, 16384),
        "host_load_tokens": (1, 64, 256, 1024, 4096, 16384),
    }

    def __init__(self, forecasts, feature_schema):
        self.forecasts = forecasts
        self.feature_schema = feature_schema
        self.feature_names = tuple(sorted(feature_schema))

    def feature_bucket(self, raw_features):
        bucket = {}
        for name, schema in self.feature_schema.items():
            value = raw_features[name]
            if schema["kind"] == "numeric_bins":
                bucket[name] = "missing" if value is None else bisect_right(schema["edges"], value)
            else:
                bucket[name] = value
        return bucket

    @classmethod
    def _schema(cls, rows, feature_names):
        schema = {}
        for name in feature_names:
            if name in cls.NUMERIC_BINS:
                schema[name] = {"kind": "numeric_bins", "edges": list(cls.NUMERIC_BINS[name]),
                                "origin": "registered_fixed_thresholds"}
            else:
                values = [row["features"][name] for row in rows if row["features"][name] is not None]
                if values and all(type(value) in (int, float) for value in values):
                    ordered = sorted(set(values))
                    edges = sorted({ordered[int((len(ordered)-1)*quantile)] for quantile in (.25, .5, .75)})
                    schema[name] = {"kind": "numeric_bins", "edges": edges,
                                    "origin": "fit_partition_quantiles"}
                else:
                    schema[name] = {"kind": "categorical"}
        return schema

    @staticmethod
    def _key(features, cache_scope_hash, horizon_seconds):
        return json.dumps([features, cache_scope_hash, float(horizon_seconds)],
                          sort_keys=True, separators=(",", ":"))

    @classmethod
    def fit(cls, rows, *, feature_names=None, feature_schema=None, reuse_rows=None):
        rows = tuple(rows)
        if not rows:
            raise ValueError("retention fit requires paired observations")
        names = tuple(sorted(rows[0]["features"])) if feature_names is None else tuple(feature_names)
        schema = cls._schema(rows, names) if feature_schema is None else feature_schema
        model = cls({}, schema)
        groups = defaultdict(list)
        reuse_groups = defaultdict(list)
        for row in rows:
            if row["native_input_hashes"] != row["retained_input_hashes"]:
                raise ValueError("C observations must use identical executed input token paths")
            bucket = model.feature_bucket(row["features"])
            groups[cls._key(bucket, row["cache_scope_hash"], row["horizon_seconds"])].append(row)
        for row in rows if reuse_rows is None else reuse_rows:
            bucket = model.feature_bucket(row["features"])
            reuse_groups[cls._key(bucket, row["cache_scope_hash"], row["horizon_seconds"])].append(row)
        forecasts = {}
        for key, cohort in groups.items():
            reused = [row for row in cohort if row["reused"]]
            frequency = reuse_groups[key]
            frequency_weight = sum(row.get("sampling_weight", 1.0) for row in frequency)
            probability = sum(row.get("sampling_weight", 1.0) for row in frequency if row["reused"])/frequency_weight
            if probability and not reused:
                continue
            absent = [row for row in cohort if not row["reused"]]
            if probability < 1 and not absent:
                continue
            def mean(values, field):
                weight = sum(row.get("sampling_weight", 1.0) for row in values)
                return sum(row.get("sampling_weight", 1.0)*row[field] for row in values)/weight
            native_window = ((probability*mean(reused, "native_window_cost_ms") if probability else 0)+
                ((1-probability)*mean(absent, "native_window_cost_ms") if probability < 1 else 0))
            retained_window = ((probability*mean(reused, "retained_window_cost_ms") if probability else 0)+
                ((1-probability)*mean(absent, "retained_window_cost_ms") if probability < 1 else 0))
            gpu_pages = sum(row.get("sampling_weight", 1.0)*row["gpu_page_count"] for row in cohort)
            host_pages = sum(row.get("sampling_weight", 1.0)*row["host_page_count"] for row in cohort)
            gpu_seconds = sum(row.get("sampling_weight", 1.0)*(row["retained_gpu_token_seconds"]-row["native_gpu_token_seconds"])
                              for row in cohort)
            host_seconds = sum(row.get("sampling_weight", 1.0)*(row["retained_host_token_seconds"]-row["native_host_token_seconds"])
                               for row in cohort)
            evidence = hashlib.sha256(json.dumps([cohort, frequency], sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()
            reused_weight = sum(row.get("sampling_weight", 1.0) for row in reused)
            control_weight = sum(row.get("sampling_weight", 1.0) for row in cohort)
            native_target = sum(row.get("sampling_weight", 1.0)*row["native_reuse_cost_ms"] for row in reused)/reused_weight if reused else None
            retained_target = sum(row.get("sampling_weight", 1.0)*row["retained_reuse_cost_ms"] for row in reused)/reused_weight if reused else None
            control = sum(row.get("sampling_weight", 1.0)*row["retention_control_ms"] for row in cohort)/control_weight
            target_gain = probability*(native_target-retained_target) if probability else 0
            forecasts[key] = PrefixForecast(cohort[0]["horizon_seconds"], probability,
                native_target, retained_target,
                gpu_seconds/(gpu_pages*cohort[0]["page_size"]) if gpu_pages else 0,
                host_seconds/(host_pages*cohort[0]["page_size"]) if host_pages else 0,
                cohort[0]["cache_scope_hash"], evidence,
                control, native_window-retained_window-target_gain+control, native_window, retained_window)
        return cls(forecasts, schema)

    def predict(self, features, cache_scope_hash, horizon_seconds):
        return self.forecasts[self._key(self.feature_bucket(features), cache_scope_hash, horizon_seconds)]

    def to_dict(self):
        return {"kind": "paired_native_retention_v4", "feature_schema": self.feature_schema,
                "value_basis": "full affected-response window Native-Retain; target, collateral and first control separated without double charging",
                "bin_convention": "left_closed_right_open; None is missing",
                "forecasts": {key: asdict(value) for key, value in self.forecasts.items()}}

    @classmethod
    def from_dict(cls, value):
        if value["kind"] != "paired_native_retention_v4":
            raise ValueError("unknown retention model format")
        return cls({key: PrefixForecast(**row) for key, row in value["forecasts"].items()}, value["feature_schema"])


class PrefixRuntimeClient:
    """Use the same guarded engine channel as generation and epoch retirement."""
    def __init__(self, engine):
        self.engine = engine

    async def retain(self, *, owner_id, receipt, prefix_input_ids, sources, ttl_seconds):
        return await self.engine._post("/context/prefix/retain", dict(owner_id=owner_id,
            attempt_id=receipt.attempt_id, input_ids=list(prefix_input_ids), sources=sources,
            ttl_seconds=ttl_seconds, expected_restart_id=receipt.engine_restart_id,
            fingerprint=asdict(self.engine.fingerprint)))

    async def release(self, owner_id, engine_restart_id):
        return await self.engine._post("/context/prefix/release", dict(owner_id=owner_id,
            expected_restart_id=engine_restart_id))

    async def status(self, engine_restart_id, owner_id=None):
        body = dict(expected_restart_id=engine_restart_id)
        if owner_id is not None:
            body["owner_id"] = owner_id
        return await self.engine._post("/context/prefix/status", body)

    async def inspect(self, *, receipt, prefix_input_ids, sources):
        return await self.engine._post("/context/prefix/status", dict(attempt_id=receipt.attempt_id,
            input_ids=list(prefix_input_ids), sources=sources,
            expected_restart_id=receipt.engine_restart_id, fingerprint=asdict(self.engine.fingerprint)))


@dataclass(frozen=True)
class PrefixOwnerRecord:
    owner_id: str
    session_id: str
    attempt_id: str
    step_id: str
    receipt: CompletionReceipt
    fingerprint: ConfigFingerprint
    prefix_input_ids: tuple[int, ...]
    sources: tuple[dict, ...]
    policy: str
    policy_choice: dict
    ttl_seconds: float
    stage: str = "RETAINING"
    runtime_receipt: dict | None = None
    last_status: dict | None = None
    created_ns: int = 0
    updated_ns: int = 0


class PrefixObserver:
    """Persist prefix intentions and reconcile them against native ownership.

    The ledger belongs to evidence prefixes. Business turn commit/retirement
    cannot discard it. Missing native ownership is recorded as such; a server
    restart never manufactures retained cache from old durable metadata.
    """
    def __init__(self, store, engine):
        self.store, self.engine = store, engine
        self.runtime = PrefixRuntimeClient(engine)
        self._lock = asyncio.Lock()

    async def observe(self, *, owner_id, session_id, attempt_id, step_id, receipt,
                      prefix_input_ids, sources, ttl_seconds, policy, policy_choice):
        async with self._lock:
            previous = self.store.prefix_owner(owner_id)
            now = time.monotonic_ns()
            record = PrefixOwnerRecord(owner_id, session_id, attempt_id, step_id, receipt,
                self.engine.fingerprint, tuple(prefix_input_ids), tuple(sources), policy,
                policy_choice, ttl_seconds, created_ns=previous.created_ns if previous else now,
                updated_ns=now)
            self.store.save_prefix_owner(record, "retain_requested")
            ack = await self.runtime.retain(owner_id=owner_id, receipt=receipt,
                prefix_input_ids=prefix_input_ids, sources=sources, ttl_seconds=ttl_seconds)
            record = replace(record, stage="RETAINED", runtime_receipt=ack,
                             last_status=ack, updated_ns=time.monotonic_ns())
            self.store.save_prefix_owner(record, "retain_acknowledged")
            return ack

    async def release(self, owner_id, *, reason):
        async with self._lock:
            record = self.store.prefix_owner(owner_id)
            info = await self.engine.info()
            if info.engine_restart_id != record.receipt.engine_restart_id:
                updated = replace(record, stage="LOST", last_status={"reason": "engine_restart",
                    "engine_restart_id": info.engine_restart_id}, updated_ns=time.monotonic_ns())
                self.store.save_prefix_owner(updated, "runtime_restarted")
                return updated.last_status
            record = replace(record, stage="RELEASING", updated_ns=time.monotonic_ns())
            self.store.save_prefix_owner(record, "release_requested", details={"reason": reason})
            ack = await self.runtime.release(owner_id, info.engine_restart_id)
            self.store.save_prefix_owner(replace(record, stage="RELEASED", last_status=ack,
                updated_ns=time.monotonic_ns()), "release_acknowledged")
            return ack

    async def reconcile(self, session_id=None):
        async with self._lock:
            info = await self.engine.info()
            records = self.store.prefix_owners(session_id, active_only=True)
            updated = []
            for record in records:
                if record.receipt.engine_restart_id != info.engine_restart_id:
                    current = replace(record, stage="LOST", last_status={"reason": "engine_restart",
                        "engine_restart_id": info.engine_restart_id}, updated_ns=time.monotonic_ns())
                    self.store.save_prefix_owner(current, "runtime_restarted")
                else:
                    status = await self.runtime.status(info.engine_restart_id, record.owner_id)
                    if status["state"] == "RETAINED" and record.stage == "RELEASING":
                        status = await self.runtime.release(record.owner_id, info.engine_restart_id)
                    stage = "RETAINED" if status["state"] == "RETAINED" else "RELEASED"
                    current = replace(record, stage=stage, last_status=status,
                        runtime_receipt=status if stage == "RETAINED" else record.runtime_receipt,
                        updated_ns=time.monotonic_ns())
                    self.store.save_prefix_owner(current, "native_ownership_observed")
                updated.append(current)
            return tuple(updated)


class PrefixReadObserver:
    """ReadingExecutor callback for source-first, actually executed READs."""
    def __init__(self, store, engine, template, settings):
        self.store, self.engine, self.template, self.settings = store, engine, template, settings
        self.owners = PrefixObserver(store, engine)
        self.runtime = self.owners.runtime
        if settings["mode"] not in {"fixed_ttl", "value"}:
            raise ValueError("prefix callback requires fixed_ttl or value mode")
        self.model = None
        if settings["mode"] == "value":
            raw = Path(settings["model_path"]).read_bytes()
            self.model = EmpiricalRetentionModel.from_dict(json.loads(raw))
            if not self.model.forecasts:
                raise ValueError("retention value needs paired native observations")
            self.model_digest = hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _features(plan, sources, accesses, now):
        state = plan.task_state
        last = accesses[-1] if accesses else None
        gap = state.current_turn_id-last["payload"]["turn_id"] if last else None
        keyed = [(source["document_family"], span["source_key"]) for source in sources for span in source["source_spans"]]
        current = [(unit.document_family or unit.source_spans[0].source_key.document_id, span.source_key)
                   for unit in plan.selected_set.selected_units for span in unit.source_spans]
        source_families = {(key["corpus_id"], family) for family, key in keyed}
        current_families = {(key.corpus_id, family) for family, key in current}
        versions = {(key["corpus_id"], family, key["version"]) for family, key in keyed}
        current_versions = {(key.corpus_id, family, key.version) for family, key in current}
        relation = "same_version" if versions & current_versions else (
            "other_version" if source_families & current_families else "other_source")
        return dict(prior_access_count=len(accesses),
            last_access_interval_seconds=(now-last["observed_ns"])/1e9 if last else None,
            source_turn_gap=gap, revisit=gap is not None and gap > 1,
            version_mode=state.version_intent.mode, source_task_relation=relation,
            need_operations=tuple(sorted({need.operation for need in state.needs})),
            condition_updates=tuple(sorted({condition.update for condition in state.conditions})))

    def _forecast(self, features, scope):
        bucket = self.model.feature_bucket(features)
        return self.model.forecasts.get(self.model._key(bucket, scope, self.settings["horizon_seconds"]))

    async def __call__(self, plan, task, units, rendered, record):
        from controller.reading import stable_prefix_input_ids, stable_source_payload
        session_id = record.engine_epoch.business_session_id
        info = await self.engine.info()
        if info.engine_restart_id != record.receipt.engine_restart_id:
            await self.owners.reconcile(session_id)
            return dict(state="LOST", reason="engine_restart", engine_restart_id=info.engine_restart_id,
                        source_attempt_id=record.receipt.attempt_id)
        prefix_ids = stable_prefix_input_ids(self.template, plan, task, units, rendered)
        source_payload, _ = stable_source_payload(units)
        sources = tuple({"source_unit": source["source_unit"], "source_spans": source["source_spans"],
                         "document_family": source["document_family"], "applicability": to_jsonable(source["applicability"])}
                        for source in source_payload)
        inspected = await self.runtime.inspect(receipt=record.receipt, prefix_input_ids=prefix_ids, sources=sources)
        owner_id = "evidence:"+hashlib.sha256(json.dumps([session_id, inspected["cache_scope_hash"],
            inspected["prefix_hash"]], separators=(",", ":")).encode()).hexdigest()
        inspected = {**inspected, "owner_id": owner_id}
        previous = self.store.prefix_accesses(owner_id)
        seen = any(row["native_attempt_id"] == record.receipt.attempt_id for row in previous)
        prior = tuple(row for row in previous if row["native_attempt_id"] != record.receipt.attempt_id)
        now = time.monotonic_ns()
        features = self._features(plan, sources, prior, now)
        pool = inspected["pool"]
        features.update(gpu_pressure=1-pool["free_gpu_tokens"]/pool["gpu_capacity_tokens"],
            host_pressure=1-pool["free_host_tokens"]/pool["host_capacity_tokens"] if pool["host_capacity_tokens"] else 0,
            gpu_prefix_tokens=inspected["residency"]["ready_gpu_prefix_tokens"],
            host_load_tokens=inspected["residency"]["host_only_load_tokens"])
        self.store.record_prefix_access(owner_id=owner_id, session_id=session_id,
            native_attempt_id=record.receipt.attempt_id, observed_ns=now,
            payload=dict(attempt_id=record.attempt_id, step_id=record.step_id,
                turn_id=plan.task_state.current_turn_id, sources=sources,
                receipt=asdict(record.receipt), inspection=inspected, features=features,
                clock="client_monotonic_nanoseconds"))
        if seen:
            updated = await self.owners.reconcile(session_id)
            owner = next((row for row in updated if row.owner_id == owner_id), self.store.prefix_owner(owner_id))
            return owner.last_status if owner is not None else {**inspected, "state": "UNRETAINED"}
        prefix_ids = prefix_ids[:inspected["boundary_tokens"]]
        arguments = dict(owner_id=owner_id, session_id=session_id, attempt_id=record.attempt_id,
            step_id=record.step_id, receipt=record.receipt, prefix_input_ids=prefix_ids, sources=sources)
        if self.settings["mode"] == "fixed_ttl":
            choice = dict(qualification_only=True, features=features)
            return await self.owners.observe(**arguments, ttl_seconds=self.settings["ttl_seconds"],
                policy="fixed_ttl_mechanism", policy_choice=choice)
        return await self._allocate_value(plan, arguments, inspected, features, now)

    async def _allocate_value(self, plan, current, inspected, features, now):
        scope = inspected["cache_scope_hash"]
        forecast = self._forecast(features, scope)
        requests, records = [], {}
        forecasts, feature_rows = {current["owner_id"]: forecast}, {current["owner_id"]: features}
        feasible = {current["owner_id"]: retainable_prefix(inspected)}
        if forecast is not None and feasible[current["owner_id"]]:
            requests.append(RetentionRequest.from_receipt(inspected, forecast))
        status = await self.runtime.status(inspected["engine_restart_id"])
        native = {row["owner_id"]: row for row in status["prefixes"]}
        for old in self.store.prefix_owners(active_only=True):
            if old.owner_id == current["owner_id"]:
                continue
            if old.receipt.engine_restart_id != inspected["engine_restart_id"]:
                self.store.save_prefix_owner(replace(old, stage="LOST",
                    last_status={"reason": "engine_restart"}, updated_ns=now), "runtime_restarted")
                continue
            if old.owner_id not in native:
                self.store.save_prefix_owner(replace(old, stage="RELEASED",
                    last_status={"state": "RELEASED"}, updated_ns=now), "native_ownership_observed")
                continue
            accesses = self.store.prefix_accesses(old.owner_id)
            old_features = self._features(plan, old.sources, accesses, now)
            old_features.update(gpu_pressure=features["gpu_pressure"], host_pressure=features["host_pressure"],
                gpu_prefix_tokens=native[old.owner_id]["residency"]["ready_gpu_prefix_tokens"],
                host_load_tokens=native[old.owner_id]["residency"]["host_only_load_tokens"])
            old_forecast = self._forecast(old_features, scope)
            forecasts[old.owner_id], feature_rows[old.owner_id] = old_forecast, old_features
            feasible[old.owner_id] = retainable_prefix(native[old.owner_id])
            if old_forecast is not None and feasible[old.owner_id]:
                requests.append(RetentionRequest.from_receipt(native[old.owner_id], old_forecast))
                records[old.owner_id] = old
        choices = allocate_retention(requests, gpu_budget_tokens=self.settings["gpu_budget_tokens"],
            host_budget_tokens=self.settings["host_budget_tokens"], lambda_gpu=self.settings["lambda_gpu"],
            lambda_host=self.settings["lambda_host"])
        selected = {choice.owner_id: choice for choice in choices}
        for owner_id in native.keys()-selected.keys():
            if self.store.prefix_owner(owner_id) is not None:
                await self.owners.release(owner_id, reason="nonpositive_or_uncalibrated_marginal_value")
        current_ack = {**inspected, "state": "UNRETAINED", "policy": "value",
            "policy_choice": {"calibrated": forecast is not None, "features": features,
                              "action_feasible": feasible[current["owner_id"]],
                              "feature_bucket": self.model.feature_bucket(features),
                              "model_digest": self.model_digest}}
        for owner_id, choice in selected.items():
            policy_choice = {**asdict(choice), "model_digest": self.model_digest,
                "capacity_basis": "full eligible causal page paths per tier; ready physical replicas remain in native receipt",
                "raw_features": feature_rows[owner_id],
                "feature_bucket": self.model.feature_bucket(feature_rows[owner_id]),
                "forecast": asdict(forecasts[owner_id])}
            if owner_id == current["owner_id"]:
                current_ack = await self.owners.observe(**current, ttl_seconds=choice.ttl_seconds,
                    policy="paired_native_value", policy_choice=policy_choice)
            else:
                old = records[owner_id]
                await self.owners.observe(owner_id=owner_id, session_id=old.session_id,
                    attempt_id=old.attempt_id, step_id=old.step_id, receipt=old.receipt,
                    prefix_input_ids=old.prefix_input_ids, sources=old.sources, ttl_seconds=choice.ttl_seconds,
                    policy="paired_native_value", policy_choice=policy_choice)
        return {**current_ack, "retention_choices": [asdict(choice) for choice in choices],
            "retention_candidates": [{"owner_id": owner_id,
                "action_feasible": feasible[owner_id],
                "raw_features": feature_rows[owner_id],
                "feature_bucket": self.model.feature_bucket(feature_rows[owner_id]),
                "forecast": asdict(prediction) if prediction is not None else None}
                for owner_id, prediction in forecasts.items()]}
