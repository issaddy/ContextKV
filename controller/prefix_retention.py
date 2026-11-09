"""Independent evidence-prefix ownership and marginal native-cache value."""
from __future__ import annotations

import asyncio
from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from collections import defaultdict, deque
from math import ceil
import time
from pathlib import Path

from controller.schema import CompletionReceipt, ConfigFingerprint
from controller.engine_client import EngineError, UnknownOutcomeError
from controller.serialization import to_jsonable
from controller.templates import token_hash
from engine_patches.contextkv_runtime import executed_prefix_hashes


RETENTION_FEATURE_NAMES = ("access_state", "native_residency", "pressure_concurrency")
TARGET_DEMAND_RULE = "retained_causal_body_pages_v3"
TARGET_COST_RULE = "native_source_tier_conditioned_server_ttft_v2"
MIN_RETENTION_GROUPS = 2
RETENTION_CONTROL_RULE = "owner_call_only_fixed_decision_once_v1"



def session_components(rows):
    """Windows sharing any session form one independent observation group."""
    components = []
    for row in rows:
        sessions = set(row["independent_session_ids"])
        if not sessions:
            raise ValueError("retention observations require actual session identities")
        overlapping = [group for group in components if group & sessions]
        for group in overlapping:
            sessions.update(group)
            components.remove(group)
        components.append(sessions)
    return tuple(tuple(sorted(group)) for group in components)


def session_group(row, components):
    sessions = set(row["independent_session_ids"])
    return next(group for group in components if sessions.intersection(group))


def balance_session_rows(rows, components):
    """Give each connected session group total weight one; arms are repeated measures."""
    rows = tuple(rows)
    totals = defaultdict(float)
    for row in rows:
        totals[session_group(row, components)] += row.get("sampling_weight", 1.0)
    return [{**row, "sampling_weight": row.get("sampling_weight", 1.0)/totals[session_group(row, components)]}
            for row in rows]

def body_page_offsets(boundary, ranges, page_size):
    return frozenset(page for first, last in ranges if first < last
        for page in range(first//page_size, min((last-1)//page_size+1, boundary//page_size)))


def source_demand_event_key(prefix_input_ids, evidence_token_ranges, page_size, scope):
    """Identity of the executed source-containing causal path."""
    path = executed_prefix_hashes(prefix_input_ids, page_size)
    body = body_page_offsets(len(prefix_input_ids), evidence_token_ranges, page_size)
    return json.dumps([scope, list(path.values()), evidence_token_ranges], separators=(",", ":")) if body else None


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
    source_token_range: tuple[int, int] | None = None

    @property
    def target_saved_ms(self):
        return (self.reuse_probability * (self.native_reuse_cost_ms-self.retained_reuse_cost_ms)
                if self.reuse_probability else 0.0)

    @property
    def saved_ms(self):
        """Next source consumer saving minus control and measured external-request penalty."""
        return (self.target_saved_ms-self.retention_control_ms
                +min(0.0, self.expected_collateral_saved_ms))

    @property
    def full_window_saved_ms(self):
        """Single-prefix paired response-window saving, including all affected turns."""
        if self.native_window_cost_ms is None or self.retained_window_cost_ms is None:
            return None
        return self.native_window_cost_ms-self.retained_window_cost_ms


@dataclass(frozen=True)
class RetentionRequest:
    owner_id: str
    forecast: PrefixForecast
    page_ids: frozenset[str]
    page_size: int
    reuse_event_key: str = ""
    # Per consumer, per causal page marginal savings. Distinct owners cannot
    # claim the same future request/page twice. Empty uses the legacy scalar.
    consumer_savings: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()
    value_unit: str = "milliseconds"

    @classmethod
    def from_receipt(cls, receipt, forecast, *, reuse_event_key=""):
        if receipt["cache_scope_hash"] != forecast.cache_scope_hash:
            raise ValueError("retention forecast differs from the executing reader/cache scope")
        # A soft owner protects one complete path in both native tiers. Native
        # receipts, rather than this logical capacity, describe physical replicas.
        path = owner_pages(receipt)
        if len(path)*receipt["page_size"] != receipt["boundary_tokens"]:
            raise ValueError("retention needs the full recoverable causal page path")
        return cls(receipt["owner_id"], forecast, path, receipt["page_size"], reuse_event_key)


@dataclass(frozen=True)
class RetentionChoice:
    owner_id: str
    ttl_seconds: float
    expected_saved_ms: float | None
    extra_gpu_token_seconds: float
    extra_host_token_seconds: float
    value: float
    marginal_protected_tokens: int
    evidence_digest: str
    reuse_event_key: str = ""
    value_unit: str = "milliseconds"


def allocate_retention(requests, *, protection_budget_tokens, lambda_gpu, lambda_host, baseline_pages=()):
    """Compare value-ordered and capacity-density constructions on the same objective."""
    requests = tuple(requests)
    if len({request.value_unit for request in requests}) > 1:
        raise ValueError("retention allocation cannot mix milliseconds and unpriced risk tokens")
    choices = tuple(_allocate_retention_greedy(requests, protection_budget_tokens=protection_budget_tokens,
        lambda_gpu=lambda_gpu, lambda_host=lambda_host, baseline_pages=baseline_pages, density=density) for density in (False, True))
    return max(choices, key=lambda selected: sum(choice.value for choice in selected))


def _allocate_retention_greedy(requests, *, protection_budget_tokens,
                              lambda_gpu, lambda_host, density, baseline_pages=()):
    """Choose positive marginal value, counting a shared causal page only once.

    Value uses the next source consumer saving, deduplicated against selected
    requests, minus this owner's control, measured external-request penalty and
    incremental residency prices.
    Single-prefix full-window residuals are not additive across protected sets.
    One logical capacity bounds protection; native tier placement is unchanged.
    """
    requests = {(request.owner_id, request.forecast.horizon_seconds): request for request in requests}
    protected_pages = set(baseline_pages)
    gpu_leases, host_leases = {}, {}
    covered_events, covered_consumer_pages = {}, {}

    def cover(request):
        forecast = request.forecast
        event = request.reuse_event_key or request.owner_id
        covered_events[event] = max(covered_events.get(event, 0), forecast.target_saved_ms)
        for consumer, pages in request.consumer_savings:
            for page, saving in pages:
                key = consumer, page
                covered_consumer_pages[key] = max(covered_consumer_pages.get(key, 0), saving)
        for leases, seconds in ((gpu_leases, forecast.extra_gpu_seconds_per_page),
                                (host_leases, forecast.extra_host_seconds_per_page)):
            for page in request.page_ids:
                previous = leases.get(page, (0, 0))
                leases[page] = max(forecast.horizon_seconds, previous[0]), max(seconds, previous[1])
        protected_pages.update(request.page_ids)

    def added_residency(pages, leases, forecast, seconds, size):
        return size*sum(max(0, seconds-leases.get(page, (0, 0))[1]) for page in pages)

    choices = []
    while requests:
        candidates = []
        for request in requests.values():
            added_tokens = len(request.page_ids-protected_pages)*request.page_size
            forecast = request.forecast
            if len(protected_pages | request.page_ids)*request.page_size > protection_budget_tokens:
                continue
            gpu_integral = added_residency(request.page_ids, gpu_leases, forecast,
                                           forecast.extra_gpu_seconds_per_page, request.page_size)
            host_integral = added_residency(request.page_ids, host_leases, forecast,
                                            forecast.extra_host_seconds_per_page, request.page_size)
            event = request.reuse_event_key or request.owner_id
            # Credit each target consumer once. Include the paired intervention's
            # external-request penalty; positive residual benefits may overlap
            # another target and are not an additional reward.
            if request.consumer_savings:
                target = sum(max(0, saving-covered_consumer_pages.get((consumer, page), 0))
                    for consumer, pages in request.consumer_savings for page, saving in pages)
            else:
                target = forecast.target_saved_ms
                target -= min(max(target, 0), covered_events.get(event, 0))
            saved = target-forecast.retention_control_ms+min(0.0, forecast.expected_collateral_saved_ms)
            value = saved-lambda_gpu*gpu_integral-lambda_host*host_integral
            if value > 0:
                choice = RetentionChoice(request.owner_id, forecast.horizon_seconds,
                    saved if request.value_unit == "milliseconds" else None, gpu_integral, host_integral, value, added_tokens,
                    forecast.evidence_digest, event, request.value_unit)
                candidates.append(choice)
        if not candidates:
            break
        def priority(item):
            if not density:
                return (False, item.value, item.value, -item.ttl_seconds, item.owner_id)
            capacity = item.marginal_protected_tokens
            # Renewing an already covered page can add value without adding capacity.
            return (capacity == 0, item.value/capacity if capacity else item.value,
                    item.value, -item.ttl_seconds, item.owner_id)
        choice = max(candidates, key=priority)
        selected = requests[choice.owner_id, choice.ttl_seconds]
        requests = {key: request for key, request in requests.items() if request.owner_id != choice.owner_id}
        cover(selected)
        choices.append(choice)
    return tuple(choices)


def owner_pages(receipt):
    return frozenset(receipt["gpu_page_ids"]) | frozenset(receipt["host_page_ids"])


def referenced_gpu_pages(receipts):
    return {page for row in receipts if row["reference_present"]
            for page in set(row["gpu_page_ids"]).intersection(row["referenced_page_ids"])}


def shared_body_page(left, right, boundary, page_size):
    for first, last in left:
        for other_first, other_last in right:
            start = max(first, other_first)
            page_end = ((start+page_size-1)//page_size+1)*page_size
            if page_end <= min(last, other_last, boundary):
                return True
    return False


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
        "host_prefix_tokens": (64, 256, 1024, 4096, 16384),
        "host_load_tokens": (1, 64, 256, 1024, 4096, 16384),
        "active_requests": (1, 2, 4, 8, 16, 32),
        "queued_requests": (1, 2, 4, 8, 16, 32),
    }

    def __init__(self, forecasts, feature_schema, *, independent_samples=0):
        self.independent_samples = independent_samples
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
        reuse_rows = rows if reuse_rows is None else tuple(reuse_rows)
        if not rows:
            raise ValueError("retention fit requires paired observations")
        if any(row.get("target_demand_rule") != TARGET_DEMAND_RULE for row in (*rows, *reuse_rows)):
            raise ValueError("retention fit requires measured causal source-page demand labels")
        if any(row.get("control_cost_rule") != RETENTION_CONTROL_RULE for row in rows):
            raise ValueError("retention fit requires owner-call control labels separated from allocation cost")
        if any(row.get("target_cost_rule") != TARGET_COST_RULE for row in rows):
            raise ValueError("retention fit requires cache-work-conditioned acquisition labels")
        names = RETENTION_FEATURE_NAMES if feature_names is None else tuple(feature_names)
        schema = cls._schema(rows, names) if feature_schema is None else feature_schema
        model = cls({}, schema)
        groups = defaultdict(list)
        reuse_groups = defaultdict(list)
        components = session_components((*rows, *reuse_rows))
        for row in rows:
            if row["native_input_hashes"] != row["retained_input_hashes"]:
                raise ValueError("C observations must use identical executed input token paths")
            bucket = model.feature_bucket(row["features"])
            groups[cls._key(bucket, row["cache_scope_hash"], row["horizon_seconds"])].append(row)
        for row in reuse_rows:
            bucket = model.feature_bucket(row["features"])
            reuse_groups[cls._key(bucket, row["cache_scope_hash"], row["horizon_seconds"])].append(row)
        forecasts = {}
        for key, cohort in groups.items():
            if len({session_group(row, components) for row in cohort}) < MIN_RETENTION_GROUPS:
                continue
            cohort = balance_session_rows(cohort, components)
            reused = [row for row in cohort if row["reused"]]
            frequency = balance_session_rows(reuse_groups[key], components)
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
            target_step_delta = probability*mean(reused, "target_revisit_step_wall_difference_ms") if probability else 0
            source_tokens = [row["gpu_page_count"]*row["page_size"] for row in cohort]
            forecasts[key] = PrefixForecast(cohort[0]["horizon_seconds"], probability,
                native_target, retained_target,
                gpu_seconds/(gpu_pages*cohort[0]["page_size"]) if gpu_pages else 0,
                host_seconds/(host_pages*cohort[0]["page_size"]) if host_pages else 0,
                cohort[0]["cache_scope_hash"], evidence,
                control, native_window-retained_window-target_step_delta+control, native_window, retained_window,
                (min(source_tokens), max(source_tokens)))
        if not forecasts:
            raise ValueError("retention fit has no supported feature/scope/TTL bucket: each needs at least "
                             f"{MIN_RETENTION_GROUPS} independent session groups and observed reuse outcomes")
        model = cls(forecasts, schema)
        model.independent_samples = len(components)
        return model

    def predict(self, features, cache_scope_hash, horizon_seconds):
        return self.forecasts[self._key(self.feature_bucket(features), cache_scope_hash, horizon_seconds)]

    def to_dict(self):
        return {"kind": "paired_native_retention_v7", "feature_schema": self.feature_schema,
                "target_demand_rule": TARGET_DEMAND_RULE,
                "target_cost_rule": TARGET_COST_RULE,
                "control_cost_rule": RETENTION_CONTROL_RULE,
                "minimum_independent_groups_per_bucket": MIN_RETENTION_GROUPS,
                "value_basis": "next READ or ANSWER acquisition of the retained causal source pages, minus control and paired external-request penalty",
                "bin_convention": "left_closed_right_open; None is missing",
                "independent_samples": self.independent_samples,
                "sample_unit": "connected session groups; TTL arms and overlapping windows are repeated measures",
                "set_intervention_required": "same fixed workload Native versus protected owner set; record all response intervals, controls and physical residency",
                "forecasts": {key: asdict(value) for key, value in self.forecasts.items()}}

    @classmethod
    def from_dict(cls, value):
        if (value["kind"] != "paired_native_retention_v7"
                or value.get("target_demand_rule") != TARGET_DEMAND_RULE
                or value.get("target_cost_rule") != TARGET_COST_RULE
                or value.get("control_cost_rule") != RETENTION_CONTROL_RULE):
            raise ValueError("unknown retention model format")
        return cls({key: PrefixForecast(**row) for key, row in value["forecasts"].items()}, value["feature_schema"],
                   independent_samples=value["independent_samples"])


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
    lease_started_ns: int = 0


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
            prior_status = ((previous.last_status or previous.runtime_receipt or {})
                            if previous else {})
            policy_choice = {**policy_choice,
                "previous_native_expires_at": prior_status.get("expires_at")}
            record = PrefixOwnerRecord(owner_id, session_id, attempt_id, step_id, receipt,
                self.engine.fingerprint, tuple(prefix_input_ids), tuple(sources), policy,
                policy_choice, ttl_seconds, created_ns=previous.created_ns if previous else now,
                updated_ns=now, lease_started_ns=now)
            self.store.save_prefix_owner(record, "retain_requested")
            try:
                ack = await self.runtime.retain(owner_id=owner_id, receipt=receipt,
                    prefix_input_ids=prefix_input_ids, sources=sources, ttl_seconds=ttl_seconds)
            except EngineError as error:
                if error.code != "PREFIX_NOT_RESIDENT":
                    raise
                ack = dict(owner_id=owner_id, state="UNRETAINED", reason="prefix_not_resident",
                    engine_restart_id=receipt.engine_restart_id)
                self.store.save_prefix_owner(replace(record, stage="RELEASED", last_status=ack,
                    updated_ns=time.monotonic_ns()), "retention_opportunity_lost")
                return ack
            record = replace(record, stage="RETAINED", runtime_receipt=ack,
                             last_status=ack, updated_ns=time.monotonic_ns())
            self.store.save_prefix_owner(record, "retain_acknowledged")
            return ack

    async def release(self, owner_id, *, reason):
        async with self._lock:
            record = self.store.prefix_owner(owner_id)
            if record.stage != "RELEASING":
                record = replace(record, stage="RELEASING", updated_ns=time.monotonic_ns())
                self.store.save_prefix_owner(record, "release_requested", details={"reason": reason})
            try:
                ack = await self.runtime.release(owner_id, record.receipt.engine_restart_id)
            except EngineError as exc:
                if exc.code != "STALE_RESTART":
                    raise
                status = {"state": "LOST", "reason": "engine_restart"}
                self.store.save_prefix_owner(replace(record, stage="LOST", last_status=status,
                    updated_ns=time.monotonic_ns()), "runtime_restarted")
                return status
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
                    if record.stage == "RELEASING":
                        status = await self.runtime.release(record.owner_id, info.engine_restart_id)
                    else:
                        status = await self.runtime.status(info.engine_restart_id, record.owner_id)
                    stage = "RETAINED" if status["state"] == "RETAINED" else "RELEASED"
                    current = replace(record, stage=stage, last_status=status,
                        runtime_receipt=status if stage == "RETAINED" else record.runtime_receipt,
                        updated_ns=time.monotonic_ns())
                    self.store.save_prefix_owner(current, "native_ownership_observed")
                updated.append(current)
            return tuple(updated)


class PrefixReadObserver:
    """ReadingExecutor callback for source-first, actually executed READs."""
    def __init__(self, store, engine, settings):
        self.store, self.engine, self.settings = store, engine, settings
        self.owners = PrefixObserver(store, engine)
        self.runtime = self.owners.runtime
        self._allocation_lock = asyncio.Lock()
        self._control_observations = deque(maxlen=32)
        if settings["mode"] not in {"fixed_ttl", "value", "demand"}:
            raise ValueError("prefix callback requires fixed_ttl, value or demand mode")
        if settings.get("unpriced_policy", "native") not in {"native", "risk_tokens"}:
            raise ValueError("unpriced demand policy must be native or the explicit risk_tokens comparator")
        self.model = None
        model_path = settings["model_path"] if settings["mode"] == "value" else settings.get("pricing_asset")
        if model_path:
            raw = Path(model_path).read_bytes()
            self.model = EmpiricalRetentionModel.from_dict(json.loads(raw))
            if not self.model.forecasts:
                raise ValueError("retention value needs paired native observations")
            self.model_digest = hashlib.sha256(raw).hexdigest()
        self.observe_reads = settings["mode"] != "demand" or self.model is not None

    @property
    def control_observation_count(self):
        return len(self._control_observations)

    def predicted_control_ms(self):
        """One future READ callback, estimated only from actual complete callbacks."""
        if not self._control_observations:
            return None
        return sum(self._control_observations)/len(self._control_observations)

    def observe_control(self, elapsed_ms, receipt):
        """Decompose the executor's existing prefix_control_ms; charge nothing again."""
        self._control_observations.append(elapsed_ms)
        owner = receipt["prefix_owner_control"]
        owner_ms = owner["retain_ms"]+owner["release_ms"]
        receipt["control_observation"] = dict(total_ms=elapsed_ms,
            allocation_fixed_ms=elapsed_ms-owner_ms, owner_mutations_ms=owner_ms,
            source="actual_prefix_control_ms", sample_count=self.control_observation_count,
            accounting="decomposition of the already charged callback, not an additional cost")
        if "predicted_set_value_before_control_ms" in receipt:
            receipt["predicted_set_net_value_ms"] = receipt["predicted_set_value_before_control_ms"]-elapsed_ms
            receipt["set_value_scope"] = "deduplicated expected savings and residency prices minus this complete observed callback once; not a measured set-intervention benefit"

    async def demand_options(self, plans, state, session_id, batch, *, page_size, candidate_arrival_ms=None):
        """Allocate short leases for exact, already compiled continuations."""
        from .admission import quoted_native_work
        from .demand import DemandLease, JointOption, joint_options
        from .prefix_view import native_cache_scope_hash
        from .templates import longest_common_prefix

        own = {row.owner_id: row for row in self.store.prefix_owners(session_id, active_only=True)
               if row.policy == "joint_demand"}
        unpriced_policy = self.settings.get("unpriced_policy", "native")
        if self.model is None and unpriced_policy == "native" and not own:
            # Without a net-value calibration, supplementary protection has no
            # justified gain over Native. Avoid even the unneeded status call.
            return joint_options(plans, state, allocate=lambda demand, *_:
                JointOption(demand, retention_evidence="native_unpriced_net_value"))
        sources = {record.step_id: record for record in state.steps if record.task.kind == "READ"
                   and record.receipt is not None and record.receipt.engine_restart_id == batch.engine_restart_id
                   and record.metrics["cache_config_hash"] == batch.cache_config_hash
                   and record.metrics["source_view"]["source_prefix_tokens"] >= page_size}
        if not own and not sources:
            return joint_options(plans, state)
        statuses = await self.runtime.status(batch.engine_restart_id)
        active = {row["owner_id"]: row for row in statuses["prefixes"]}
        own = {key: record for key, record in own.items() if key in active}
        if not own and not sources:
            return joint_options(plans, state)
        other_owners = tuple(row for key, row in active.items() if key not in own)
        shared = set().union(*(owner_pages(row) for row in other_owners))
        shared_gpu = referenced_gpu_pages(other_owners)
        quotes = {quote.candidate_id: quote for quote in batch.quotes}
        limit = self.settings["protection_budget_tokens"]
        scope = native_cache_scope_hash(batch.fingerprint, batch.cache_config_hash)
        queued_growth = sum(max(0, request.input_tokens+request.max_new_tokens-request.kv_allocated_tokens)
                            for request in batch.native_waiting_requests or ())
        concurrent_growth = ceil(max(0, batch.native_running_budget_tokens or 0))+queued_growth

        def work(candidate):
            return sum(quoted_native_work(quotes[candidate.candidate_id]))+candidate.output_budget

        matched_paths = {}

        def matching_sources(compiled):
            key = compiled.candidate_id
            if key not in matched_paths:
                quote = quotes[key]
                upper = min(quote.ready_gpu_prefix_tokens, quote.match_limit, compiled.source_prefix_tokens, limit)
                matches = {}
                for record in sources.values():
                    view = record.metrics["source_view"]
                    observed_end = min(upper, view["source_prefix_tokens"])
                    common = longest_common_prefix(compiled.input_ids, record.input_ids[:observed_end])
                    boundary = common//page_size*page_size
                    if shared_body_page(view["body_token_ranges"],
                            compiled.source_body_token_ranges or (), boundary, page_size):
                        matches.setdefault(boundary, (record.step_id, record.attempt_id))
                hashes = executed_prefix_hashes(compiled.input_ids[:max(matches, default=0)], page_size)
                matched_paths[key] = tuple(("demand:"+hashlib.sha256(json.dumps(
                    [session_id, attempt_id, scope, hashes[boundary]], separators=(",", ":")).encode()).hexdigest(),
                    step_id, boundary, tuple(scope+":"+value for end, value in hashes.items() if end <= boundary))
                    for boundary, (step_id, attempt_id) in matches.items())
            return matched_paths[key]

        now = time.monotonic_ns()
        horizons = tuple(sorted(set(self.settings.get("horizon_seconds_choices", (
            self.settings["ttl_seconds"],)))))
        elapsed = defaultdict(list)
        task_kinds = {task.step_id: task.kind for task in state.plan.steps}
        for result in state.results:
            if result.elapsed_ms > 0 and result.step_id in task_kinds:
                elapsed[task_kinds[result.step_id]].append(result.elapsed_ms)
        means = {kind: sum(values)/len(values) for kind, values in elapsed.items()}

        def remaining(owner):
            old = own[owner]
            return max(0, old.ttl_seconds-(now-old.lease_started_ns)/1e9)

        def allocate(demand, task, candidate, actions):
            current = quotes[candidate.candidate_id] if candidate is not None else None
            # Do not drop a lease backing the selected quote before Native has
            # locked that request. Its existing expiry is never extended here.
            immediate = set()
            if current is not None:
                for key, record in own.items():
                    if not active[key]["reference_present"] or remaining(key) <= 0:
                        continue
                    boundary = longest_common_prefix(candidate.input_ids[:current.recoverable_prefix_tokens],
                        record.prefix_input_ids)//page_size*page_size
                    if (boundary == len(record.prefix_input_ids) or
                            record.step_id in sources and shared_body_page(
                                sources[record.step_id].metrics["source_view"]["body_token_ranges"],
                                candidate.source_body_token_ranges or (), boundary, page_size)):
                        immediate.add(key)
            kept = tuple(active[key] for key in immediate)
            base_pages = shared.union(*(owner_pages(row) for row in kept))
            base_gpu = shared_gpu | referenced_gpu_pages(kept)
            current_growth = (current.input_len-current.ready_gpu_prefix_tokens+candidate.output_budget
                              if current is not None else 0)
            pressure = max(0, current_growth+concurrent_growth-batch.free_gpu_tokens)
            if task.kind == "ANSWER" or not pressure or not sources:
                return JointOption(demand, release=tuple(sorted(own.keys()-immediate)),
                    protected_tokens=len(base_pages-shared)*page_size, keep=tuple(sorted(immediate)))
            current_prefix = candidate.input_ids[:current.ready_gpu_prefix_tokens] if current is not None else ()
            targets = {}
            unmet = {need.need_id: need.priority for need in demand.needs}
            # Consumer arrival is just the work before that consumer plus queue
            # and observed progress. It is never the task's complete-response P95.
            observed = means.get(task.kind)
            arrival = (observed*(1+batch.queued_requests) if observed is not None else None)
            seen = {task.step_id}
            arrivals = {}
            for target, compiled in actions:
                if target.step_id == task.step_id:
                    continue
                if target.step_id not in arrivals:
                    arrivals[target.step_id] = arrival
                if target.step_id not in seen:
                    duration = means.get(target.kind)
                    arrival = arrival+duration if arrival is not None and duration is not None else None
                    seen.add(target.step_id)
                if compiled is None:
                    continue
                priority = sum(unmet.get(need, 0) for need in target.need_ids
                               if target.step_id in demand.remaining_read_ids)
                if target.kind == "ANSWER" and compiled.source_body_token_ranges:
                    # Original material remains a consumer even after facts
                    # have closed the semantic gaps. Facts-only inputs have no
                    # source body and therefore create no source-prefix demand.
                    priority = max(1, sum(unmet.get(need, 1) for need in target.need_ids))
                predicted_arrival = ((candidate_arrival_ms or {}).get(compiled.candidate_id, arrivals[target.step_id]))
                if priority and (target.step_id not in targets or work(compiled) < work(targets[target.step_id][0])):
                    targets[target.step_id] = compiled, priority, predicted_arrival

            paths, demands, consumer_paths = {}, {}, {}
            own_gpu = referenced_gpu_pages(tuple(active[key] for key in own))
            for target, (compiled, priority, arrival_ms) in targets.items():
                quote = quotes[compiled.candidate_id]
                matches = matching_sources(compiled)
                if not matches:
                    continue
                evictable = quote.evictable_gpu_prefix_tokens or 0
                locked_pages = (quote.ready_gpu_prefix_tokens-evictable)//page_size
                if not concurrent_growth:
                    locked_pages = max(locked_pages, longest_common_prefix(compiled.input_ids, current_prefix)//page_size)
                _, _, _, path = max(matches, key=lambda match: match[2])
                # Existing supplementary leases mask eviction in the quote.
                # Evaluate their release against Native, along with new leases.
                evictable_pages = (frozenset(path[locked_pages:]) | (set(path) & own_gpu))-base_gpu
                risk = min(pressure, len(evictable_pages)*page_size)
                if risk:
                    demands[target] = (risk, priority, evictable_pages, arrival_ms)
                    consumer_paths[target] = frozenset(path), body_page_offsets(len(path)*page_size,
                        compiled.source_body_token_ranges or (), page_size)
                    paths.update((owner, (step, end, frozenset(pages))) for owner, step, end, pages in matches)
            # Include old leases even when their original path is not the longest
            # observed match. Their remaining TTL competes with renew/new choices.
            for owner, old in own.items():
                if owner in immediate:
                    continue
                pages = owner_pages(active[owner])
                if any(pages & row[2] for row in demands.values()):
                    paths[owner] = old.step_id, len(old.prefix_input_ids), pages

            requests, actions_by_option = [], {}
            curves = {}
            for owner, (step, boundary, _) in paths.items():
                if self.model is None or step not in sources or owner in immediate:
                    continue
                inspection = sources[step].metrics.get("evidence_prefix", {})
                if inspection.get("demand_observation") and inspection["boundary_tokens"] == boundary:
                    curves[owner] = tuple(forecast for horizon in horizons
                        if (forecast := self._demand_forecast(inspection, batch, horizon)) is not None)
            priced = any(curves.values())
            if not priced and unpriced_policy == "native":
                return JointOption(demand, release=tuple(sorted(own.keys()-immediate)),
                    protected_tokens=len(base_pages-shared)*page_size,
                    uncovered_risk_tokens=sum(row[0] for row in demands.values()),
                    retention_evidence="native_unpriced_net_value", keep=tuple(sorted(immediate)))
            for owner, (step, boundary, pages) in paths.items():
                if owner in immediate:
                    continue
                curve = curves.get(owner, ())
                if priced and not curve:
                    continue
                if priced:
                    source_body = body_page_offsets(boundary,
                        sources[step].metrics["source_view"]["body_token_ranges"], page_size)
                    # The paired effect prices the first consumer of the owner.
                    # Later requests can already hit pages loaded by that consumer.
                    first_consumer = next((target for target, (path, body) in consumer_paths.items()
                        if pages <= path and source_body and source_body <= body), None)
                    if first_consumer is None:
                        continue
                leases = [(horizon, "renew" if owner in own else "new") for horizon in horizons]
                if owner in own and remaining(owner) > 0:
                    leases.append((remaining(owner), "keep"))
                for horizon, action in leases:
                    events, target_ids, forecasts = [], [], []
                    for target, (risk, priority, evictable_pages, arrival_ms) in demands.items():
                        if arrival_ms is not None and arrival_ms >= horizon*1000:
                            continue
                        covered = pages & evictable_pages
                        if not covered:
                            continue
                        forecast = None
                        if priced:
                            if target != first_consumer:
                                continue
                            forecast = (self._remaining_forecast(curve, horizon) if action == "keep" else
                                next((point for point in curve if point.horizon_seconds == horizon), None))
                            if forecast is None:
                                continue
                            per_page = max(0, forecast.target_saved_ms)*min(risk, boundary)/boundary/len(evictable_pages)
                            forecasts.append(forecast)
                        else:
                            per_page = risk*priority/len(evictable_pages)
                        events.append((target, tuple((page, per_page) for page in sorted(covered))))
                        target_ids.append(target)
                    if not events:
                        continue
                    # Control belongs to the owner call once. Shared residence
                    # is unioned by the existing allocator. Positive collateral
                    # full-window effects are deliberately never added.
                    forecast = PrefixForecast(horizon, 0, None, None,
                        max((point.extra_gpu_seconds_per_page for point in forecasts), default=0),
                        max((point.extra_host_seconds_per_page for point in forecasts), default=0),
                        scope, "consumer_paired_marginal" if priced else "native_at_risk_token_rule",
                        retention_control_ms=0 if action == "keep" else max((point.retention_control_ms for point in forecasts), default=0),
                        expected_collateral_saved_ms=min((point.expected_collateral_saved_ms for point in forecasts), default=0))
                    request = RetentionRequest(owner, forecast, pages, page_size,
                        consumer_savings=tuple(events), value_unit="milliseconds" if priced else "risk_tokens")
                    requests.append(request)
                    target = max(target_ids, key=lambda key: demands[key][0]*demands[key][1])
                    actions_by_option[owner, horizon] = DemandLease(owner, step, target, boundary,
                        min(demands[target][0], len(pages & demands[target][2])*page_size), page_size, scope,
                        horizon, action, demands[target][3])
            choices = allocate_retention(requests, protection_budget_tokens=limit,
                lambda_gpu=self.settings.get("lambda_gpu", 0) if priced else 0,
                lambda_host=self.settings.get("lambda_host", 0) if priced else 0, baseline_pages=base_pages)
            selected = {choice.owner_id: actions_by_option[choice.owner_id, choice.ttl_seconds] for choice in choices}
            protected = base_pages.union(*(paths[owner][2] for owner in selected))
            kept_ids = immediate | {owner for owner, lease in selected.items() if lease.action == "keep"}
            uncovered = sum(max(0, risk-min(risk, len(path & protected)*page_size))
                for risk, _, path, _ in demands.values())
            # B fits actual Native branch cost, including future cache behavior.
            # These consumer marginals are not conditioned on that branch's
            # realized continuation and this retention intervention. Allocation
            # value also folds in control, collateral and residency prices, so
            # retain it as a lease diagnostic rather than a branch latency delta.
            allocation_value = sum(choice.value for choice in choices) if priced else None
            chosen_forecasts = {(request.owner_id, request.forecast.horizon_seconds): request.forecast
                                for request in requests}
            mutations = tuple(lease for lease in selected.values() if lease.action != "keep")
            owner_control_ms = (sum(chosen_forecasts[lease.owner_id, lease.ttl_seconds].retention_control_ms
                for lease in mutations) if priced else None) if mutations else 0.0
            return JointOption(demand, mutations,
                tuple(sorted(own.keys()-immediate-selected.keys())), len(protected-shared)*page_size,
                uncovered, "consumer_paired_marginal" if priced else "native_at_risk_token_rule",
                tuple(sorted(kept_ids)), "candidate_arrival_ms_or_observed_step_means_plus_queue; unknown_when_unobserved",
                allocation_value, owner_control_ms)

        return joint_options(plans, state, allocate=allocate)

    def _demand_forecast(self, inspected, batch, horizon):
        features = inspected.get("retention_features")
        if features is None:
            return None
        current = {**inspected, "pool": {**inspected["pool"], "free_gpu_tokens": batch.free_gpu_tokens,
                   "free_host_tokens": batch.free_host_tokens},
                   "residency": {"ready_gpu_prefix_tokens": inspected["boundary_tokens"],
                                 "host_only_load_tokens": 0},
                   "workload": {"active_requests": batch.active_requests, "queued_requests": batch.queued_requests}}
        bucket = self.model.feature_bucket({**features, **self._native_features(current)})
        forecast = self.model.forecasts.get(self.model._key(bucket, inspected["cache_scope_hash"],
                                                            horizon))
        if forecast is not None and forecast.source_token_range is not None:
            low, high = forecast.source_token_range
            if low <= inspected["boundary_tokens"] <= high:
                return forecast

    async def apply_demand(self, option, state, session_id, *, batch):
        started = time.monotonic()
        receipts = {}
        async with self._allocation_lock:
            for owner in option.release:
                receipts[owner] = await self.owners.release(owner, reason="joint_remaining_demand")
            if option.retain:
                steps = {record.step_id: record for record in state.steps}
                status = await self.runtime.status(batch.engine_restart_id)
                protected = set().union(*(owner_pages(row) for row in status["prefixes"]))
                protected_gpu = referenced_gpu_pages(status["prefixes"])
            for lease in option.retain:
                sources = (self.store.prefix_owner(lease.owner_id).sources
                           if lease.action == "renew" else None)
                record = steps[lease.source_step_id]
                prefix = record.input_ids[:lease.boundary_tokens]
                page_size, scope = lease.page_size, lease.cache_scope_hash
                pages = {scope+":"+value for value in executed_prefix_hashes(prefix, page_size).values()}
                if lease.action != "renew" and pages <= protected_gpu:
                    continue
                next_pages = protected | pages
                if len(next_pages)*page_size > self.settings["protection_budget_tokens"]:
                    receipts[lease.owner_id] = {"state": "UNRETAINED", "reason": "shared_protection_budget"}
                    continue
                receipts[lease.owner_id] = await self.owners.observe(owner_id=lease.owner_id,
                    session_id=session_id, attempt_id=record.attempt_id, step_id=record.step_id,
                    receipt=record.receipt, prefix_input_ids=prefix,
                    sources=(sources if lease.action == "renew" else
                        ({"kind": "executed_source_body_prefix", "prefix_hash": token_hash(prefix)},)),
                    ttl_seconds=lease.ttl_seconds,
                    policy="joint_demand", policy_choice={**asdict(lease), "observed_quote_seq": batch.observed_seq,
                        "protected_tokens": option.protected_tokens})
                if receipts[lease.owner_id]["state"] == "RETAINED":
                    protected.update(owner_pages(receipts[lease.owner_id]))
                    protected_gpu.update(referenced_gpu_pages((receipts[lease.owner_id],)))
        return {"elapsed_ms": (time.monotonic()-started)*1000, "kept_owners": option.keep,
                "owner_status": {owner: {key: ack[key] for key in ("state", "reason", "boundary_tokens", "expires_at")
                                         if key in ack} for owner, ack in receipts.items()}}

    async def finish_demand(self, outcome, session_id, attempt_id):
        from .demand import DemandState, JointOption

        release = tuple(owner.owner_id for owner in self.store.prefix_owners(session_id, active_only=True)
                        if owner.policy == "joint_demand" and owner.attempt_id == attempt_id)
        if release:
            return await self.apply_demand(JointOption(DemandState((), ()), release=release),
                                           outcome, session_id, batch=None)

    async def flush_demand_releases(self, session_id, *, events):
        async with self._allocation_lock:
            for owner in self.store.prefix_owners(session_id, stage="RELEASING"):
                if owner.policy != "joint_demand":
                    continue
                try:
                    await self.owners.release(owner.owner_id, reason="terminal_demand")
                except UnknownOutcomeError as exc:
                    events.emit("joint_prefix_release_retry", owner_id=owner.owner_id, error=str(exc))

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
        return dict(access_state=("first" if not accesses else "recent" if (now-last["observed_ns"])/1e9 <= 60 else "revisit"),
            prior_access_count=len(accesses),
            last_access_interval_seconds=(now-last["observed_ns"])/1e9 if last else None,
            source_turn_gap=gap, revisit=gap is not None and gap > 1,
            version_mode=state.version_intent.mode, source_task_relation=relation,
            need_operations=tuple(sorted({need.operation for need in state.needs})),
            condition_updates=tuple(sorted({condition.update for condition in state.conditions})))

    @staticmethod
    def _native_features(inspected):
        pool, residency, workload = inspected["pool"], inspected["residency"], inspected["workload"]
        ready = residency["ready_gpu_prefix_tokens"]
        residency_state = ("device" if ready >= inspected["boundary_tokens"] else
                           "host" if residency["recoverable_prefix_tokens"] >= inspected["boundary_tokens"] else "missing")
        pressure = max(1-pool["free_gpu_tokens"]/pool["gpu_capacity_tokens"],
            1-pool["free_host_tokens"]/pool["host_capacity_tokens"] if pool["host_capacity_tokens"] else 0)
        load = "contended" if workload["active_requests"] > 1 or workload["queued_requests"] else "single"
        return dict(native_residency=residency_state,
            pressure_concurrency=("high" if pressure >= .9 else "normal")+":"+load,
            gpu_pressure=1-pool["free_gpu_tokens"]/pool["gpu_capacity_tokens"],
            host_pressure=1-pool["free_host_tokens"]/pool["host_capacity_tokens"] if pool["host_capacity_tokens"] else 0,
            gpu_prefix_tokens=residency["ready_gpu_prefix_tokens"],
            host_prefix_tokens=len(inspected["host_page_ids"])*inspected["page_size"],
            host_load_tokens=residency["host_only_load_tokens"],
            active_requests=workload["active_requests"], queued_requests=workload["queued_requests"])

    def _forecasts(self, features, scope):
        bucket = self.model.feature_bucket(features)
        horizons = (self.settings["horizon_seconds_choices"] if "horizon_seconds_choices" in self.settings
                    else (self.settings["horizon_seconds"],))
        return tuple(self.model.forecasts[key] for horizon in sorted(horizons)
            if (key := self.model._key(bucket, scope, horizon)) in self.model.forecasts)

    @staticmethod
    def _accepted_policy_call(owner, status):
        """Distinguish this retain from an unchanged lease predating its intent."""
        prior_expiry = owner.policy_choice.get("previous_native_expires_at")
        if status.get("state") != "RETAINED":
            return False
        if prior_expiry is not None:
            return status.get("expires_at") != prior_expiry
        return status.get("source_attempt_id") == owner.receipt.attempt_id

    @staticmethod
    def _remaining_forecast(curve, remaining):
        """Interpolate measured horizon curves for the lease still in force.

        Reuse is forecast from the current access history. Retain control was
        already paid; keeping the existing lease issues no new native call.
        """
        right = next((point for point in curve if point.horizon_seconds >= remaining), curve[-1])
        left = next((point for point in reversed(curve) if point.horizon_seconds < remaining), None)
        lower = left.horizon_seconds if left else 0
        upper = right.horizon_seconds
        fraction = (min(remaining, upper)-lower)/(upper-lower) if upper > lower else 1

        def interpolate(field):
            before = getattr(left, field) if left else 0
            return before+(getattr(right, field)-before)*fraction

        def expected(point, field):
            return point.reuse_probability*getattr(point, field) if point and point.reuse_probability else 0

        probability = interpolate("reuse_probability")
        def cost(field):
            before = expected(left, field)
            value = before+(expected(right, field)-before)*fraction
            return value/probability if probability else None

        return PrefixForecast(remaining, probability, cost("native_reuse_cost_ms"),
            cost("retained_reuse_cost_ms"), interpolate("extra_gpu_seconds_per_page"),
            interpolate("extra_host_seconds_per_page"), right.cache_scope_hash, right.evidence_digest,
            expected_collateral_saved_ms=interpolate("expected_collateral_saved_ms"))

    async def __call__(self, plan, task, units, record):
        # Observation and the resulting owner changes must see one allocation
        # order, including callbacks restoring an already completed READ.
        async with self._allocation_lock:
            control = dict(retain_ms=0.0, release_ms=0.0, retain_calls=0, release_calls=0)
            result = await self._observe_and_allocate(plan, task, units, record, control)
            return {**result, "prefix_owner_control": control}

    async def _observe_and_allocate(self, plan, task, units, record, control):
        from controller.source_prompt import stable_source_payload
        session_id = record.engine_epoch.business_session_id
        info = await self.engine.info()
        if info.engine_restart_id != record.receipt.engine_restart_id:
            await self.owners.reconcile(session_id)
            return dict(state="LOST", reason="engine_restart", engine_restart_id=info.engine_restart_id,
                        source_attempt_id=record.receipt.attempt_id)
        source_view = record.metrics["source_view"]
        prefix_ids = record.input_ids[:source_view["source_prefix_tokens"]]
        source_payload, _ = stable_source_payload(units)
        sources = tuple({"source_unit": source["source_unit"], "source_spans": source["source_spans"],
                         "document_family": source["document_family"], "applicability": to_jsonable(source["applicability"])}
                        for source in source_payload)
        inspected = await self.runtime.inspect(receipt=record.receipt, prefix_input_ids=prefix_ids, sources=sources)
        owner_id = "evidence:"+hashlib.sha256(json.dumps([session_id, inspected["cache_scope_hash"],
            inspected["prefix_hash"]], separators=(",", ":")).encode()).hexdigest()
        inspected = {**inspected, "owner_id": owner_id}
        previous = self.store.prefix_accesses(owner_id)
        access = next((row for row in previous
                       if row["native_attempt_id"] == record.receipt.attempt_id), None)
        prior = tuple(row for row in previous if row["native_attempt_id"] != record.receipt.attempt_id)
        now = time.monotonic_ns()
        features = self._features(plan, sources, prior, now)
        features.update(self._native_features(inspected))
        self.store.record_prefix_access(owner_id=owner_id, session_id=session_id,
            native_attempt_id=record.receipt.attempt_id, observed_ns=now,
            payload=dict(attempt_id=record.attempt_id, step_id=record.step_id,
                turn_id=plan.task_state.current_turn_id, sources=sources,
                receipt=asdict(record.receipt), inspection=inspected, features=features,
                clock="client_monotonic_nanoseconds"))
        if access is not None:
            completed = access["payload"].get("policy_result")
            if completed is not None and self.settings["mode"] == "demand":
                return completed
            updated = await self.owners.reconcile(session_id)
            owner = next((row for row in updated if row.owner_id == owner_id), self.store.prefix_owner(owner_id))
            if completed is not None:
                return owner.last_status if owner is not None else completed
            accepted = (self.settings["mode"] == "fixed_ttl"
                and owner is not None and owner.stage == "RETAINED"
                and owner.receipt.attempt_id == record.receipt.attempt_id
                and self._accepted_policy_call(owner, owner.last_status))
            if accepted:
                self.store.complete_prefix_access(owner_id=owner_id,
                    native_attempt_id=record.receipt.attempt_id, result=owner.last_status)
                return owner.last_status
        prefix_ids = prefix_ids[:inspected["boundary_tokens"]]
        ranges = source_view["body_token_ranges"]
        event = source_demand_event_key(prefix_ids, ranges, inspected["page_size"], inspected["cache_scope_hash"])
        if event is None:
            result = {**inspected, "state": "UNRETAINED", "reason": "no_complete_source_body_page"}
            self.store.complete_prefix_access(owner_id=owner_id,
                native_attempt_id=record.receipt.attempt_id, result=result)
            return result
        arguments = dict(owner_id=owner_id, session_id=session_id, attempt_id=record.attempt_id,
            step_id=record.step_id, receipt=record.receipt, prefix_input_ids=prefix_ids, sources=sources)
        if self.settings["mode"] == "demand":
            result = {**inspected, "demand_observation": retainable_prefix(inspected), "retention_features": features}
        elif self.settings["mode"] == "fixed_ttl":
            choice = dict(qualification_only=True, features=features, reuse_event_key=event)
            started = time.monotonic()
            result = await self.owners.observe(**arguments, ttl_seconds=self.settings["ttl_seconds"],
                policy="fixed_ttl_mechanism", policy_choice=choice)
            control["retain_ms"] += (time.monotonic()-started)*1000
            control["retain_calls"] += 1
        else:
            result = await self._allocate_value(plan, arguments, inspected, features, time.monotonic_ns(), event,
                                                control=control)
        self.store.complete_prefix_access(owner_id=owner_id,
            native_attempt_id=record.receipt.attempt_id, result=result)
        return result

    async def _allocate_value(self, plan, current, inspected, features, now, event_key, *, control=None):
        control = control if control is not None else dict(retain_ms=0.0, release_ms=0.0, retain_calls=0, release_calls=0)
        scope, current_id = inspected["cache_scope_hash"], current["owner_id"]
        curves = {current_id: self._forecasts(features, scope)}
        feature_rows, events = {current_id: features}, {current_id: event_key}
        feasible = {current_id: retainable_prefix(inspected)}
        receipts, records, active = {current_id: inspected}, {}, []
        trigger = current["receipt"].attempt_id
        applied = set()
        status = await self.runtime.status(inspected["engine_restart_id"])
        native = {row["owner_id"]: row for row in status["prefixes"]}
        for old in self.store.prefix_owners(active_only=True):
            if old.receipt.engine_restart_id != inspected["engine_restart_id"]:
                self.store.save_prefix_owner(replace(old, stage="LOST",
                    last_status={"reason": "engine_restart"}, updated_ns=now), "runtime_restarted")
                continue
            if old.owner_id not in native:
                self.store.save_prefix_owner(replace(old, stage="RELEASED",
                    last_status={"state": "RELEASED"}, updated_ns=now), "native_ownership_observed")
                continue
            records[old.owner_id] = old
            receipts[old.owner_id] = native[old.owner_id]
            if (old.policy_choice.get("trigger_native_attempt_id") == trigger
                    and self._accepted_policy_call(old, native[old.owner_id])):
                applied.add(old.owner_id)
            if old.owner_id != current_id:
                accesses = self.store.prefix_accesses(old.owner_id)
                old_features = self._features(plan, old.sources, accesses, now)
                old_features.update(self._native_features(native[old.owner_id]))
                curves[old.owner_id] = self._forecasts(old_features, scope)
                feature_rows[old.owner_id] = old_features
                events[old.owner_id] = old.policy_choice["reuse_event_key"]
                feasible[old.owner_id] = retainable_prefix(native[old.owner_id])
            remaining = max(0, old.ttl_seconds-(now-old.lease_started_ns)/1e9)
            if remaining and curves[old.owner_id] and feasible[old.owner_id]:
                forecast = self._remaining_forecast(curves[old.owner_id], remaining)
                active.append(RetentionRequest.from_receipt(native[old.owner_id], forecast,
                    reuse_event_key=events[old.owner_id]))

        budgets = dict(protection_budget_tokens=self.settings["protection_budget_tokens"], lambda_gpu=self.settings["lambda_gpu"],
            lambda_host=self.settings["lambda_host"])
        requests = [RetentionRequest.from_receipt(receipts[owner_id], forecast,
                reuse_event_key=events[owner_id])
            for owner_id, curve in curves.items() if feasible[owner_id] and owner_id not in applied
            for forecast in curve]
        # Keep, renew and new leases compete for the same future capacity.
        # A remaining lease has no new control call; at an identical horizon
        # it replaces the renewal option. Unselected old leases are released.
        keep_options = {(request.owner_id, request.forecast.horizon_seconds) for request in active}
        allocated = allocate_retention((*requests, *active), **budgets)
        kept_ids = {choice.owner_id for choice in allocated
                    if (choice.owner_id, choice.ttl_seconds) in keep_options}
        choices = tuple(choice for choice in allocated if choice.owner_id not in kept_ids)
        selected = {choice.owner_id: choice for choice in choices}
        final_ids = kept_ids | selected.keys()
        for owner_id in native.keys()-final_ids:
            if self.store.prefix_owner(owner_id) is not None:
                started = time.monotonic()
                await self.owners.release(owner_id, reason="nonpositive_or_uncalibrated_marginal_value")
                control["release_ms"] += (time.monotonic()-started)*1000
                control["release_calls"] += 1
        current_ack = (native[current_id] if current_id in kept_ids else
            {**inspected, "state": "UNRETAINED", "policy": "value",
             "policy_choice": {"calibrated": bool(curves[current_id]), "features": features,
                "action_feasible": feasible[current_id], "feature_bucket": self.model.feature_bucket(features),
                "model_digest": self.model_digest}})
        for owner_id, choice in selected.items():
            forecast = next(point for point in curves[owner_id] if point.horizon_seconds == choice.ttl_seconds)
            policy_choice = {**asdict(choice), "model_digest": self.model_digest,
                "trigger_native_attempt_id": trigger,
                "capacity_basis": "one logical capacity of unique full causal pages; native GPU/Host placement unchanged",
                "raw_features": feature_rows[owner_id],
                "feature_bucket": self.model.feature_bucket(feature_rows[owner_id]),
                "forecast": asdict(forecast)}
            started = time.monotonic()
            if owner_id == current_id:
                current_ack = await self.owners.observe(**current, ttl_seconds=choice.ttl_seconds,
                    policy="paired_native_value", policy_choice=policy_choice)
            else:
                old = records[owner_id]
                await self.owners.observe(owner_id=owner_id, session_id=old.session_id,
                    attempt_id=old.attempt_id, step_id=old.step_id, receipt=old.receipt,
                    prefix_input_ids=old.prefix_input_ids, sources=old.sources, ttl_seconds=choice.ttl_seconds,
                    policy="paired_native_value", policy_choice=policy_choice)
            control["retain_ms"] += (time.monotonic()-started)*1000
            control["retain_calls"] += 1
        owner_prediction = sum(next(point.retention_control_ms for point in curves[choice.owner_id]
            if point.horizon_seconds == choice.ttl_seconds) for choice in choices)
        return {**current_ack, "retention_choices": [asdict(choice) for choice in choices],
            "predicted_set_value_before_control_ms": sum(choice.value for choice in allocated)+owner_prediction,
            "predicted_owner_control_ms": owner_prediction,
            "prefix_owner_control": control,
            "existing_retention_owners": sorted(kept_ids-selected.keys()),
            "retention_candidates": [{"owner_id": owner_id,
                "reuse_event_key": events[owner_id], "action_feasible": feasible[owner_id],
                "raw_features": feature_rows[owner_id],
                "feature_bucket": self.model.feature_bucket(feature_rows[owner_id]),
                "forecasts": [asdict(prediction) for prediction in curve]}
                for owner_id, curve in curves.items()]}
