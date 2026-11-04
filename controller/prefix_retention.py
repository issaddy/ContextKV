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
from controller.engine_client import EngineError
from controller.serialization import to_jsonable


RETENTION_FEATURE_NAMES = ("access_state", "native_residency", "pressure_concurrency")
TARGET_DEMAND_RULE = "exact_complete_causal_prefix_and_body_ranges_v1"



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

def source_token_ranges(template, rendered, messages, units):
    """Locate encoded original-source text, excluding shared instructions/catalogue."""
    from controller.templates import ByteTokenizer, longest_common_prefix
    if type(template.tokenizer) is ByteTokenizer:
        offsets = [(index, index+1) for index, char in enumerate(rendered.text) for _ in char.encode("utf-8")]
    else:
        encoded = template.tokenizer(rendered.text, add_special_tokens=False, return_offsets_mapping=True)
        if tuple(encoded["input_ids"]) != tuple(rendered.input_ids):
            raise ValueError("source-body offset tokenizer differs from actual rendered input")
        offsets = encoded["offset_mapping"]
    content = messages[1].content
    message_start = rendered.text.index(content)
    ranges, cursor = [], 0
    for unit in units:
        literal = json.dumps(unit.text, ensure_ascii=False)
        marker = '"text":'+literal
        found = content.index(marker, cursor)
        first = message_start+found+len('"text":')+1
        last = first+len(literal)-2
        tokens = [index for index, (start, end) in enumerate(offsets) if end > first and start < last and end > start]
        if tokens:
            ranges.append([tokens[0], tokens[-1]+1])
        cursor = found+len(marker)
    system = template.render((messages[0],)).input_ids
    return ranges, longest_common_prefix(system, rendered.input_ids)



def source_demand_event_key(prefix_input_ids, evidence_token_ranges, page_size, scope):
    """The next READ demanding this known source-containing causal prefix."""
    from engine_patches.contextkv_runtime import executed_prefix_hashes
    path = executed_prefix_hashes(prefix_input_ids, page_size)
    body = [page for end, page in path.items()
            if any(end > first and end-page_size < last for first, last in evidence_token_ranges)]
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

    @property
    def target_saved_ms(self):
        return (self.reuse_probability * (self.native_reuse_cost_ms-self.retained_reuse_cost_ms)
                if self.reuse_probability else 0.0)

    @property
    def saved_ms(self):
        """Next READ saving minus control and measured external-request penalty."""
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
    gpu_page_ids: frozenset[str]
    host_page_ids: frozenset[str]
    page_size: int
    reuse_event_key: str = ""

    @classmethod
    def from_receipt(cls, receipt, forecast, *, reuse_event_key=""):
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
        return cls(receipt["owner_id"], forecast, path, host, receipt["page_size"], reuse_event_key)


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
    reuse_event_key: str = ""


def allocate_retention(requests, *, gpu_budget_tokens, host_budget_tokens,
                       lambda_gpu, lambda_host, protected_gpu_pages=frozenset(),
                       protected_host_pages=frozenset(), protected_requests=()):
    """Compare value-ordered and capacity-density constructions on the same objective."""
    requests, protected_requests = tuple(requests), tuple(protected_requests)
    choices = tuple(_allocate_retention_greedy(requests, gpu_budget_tokens=gpu_budget_tokens,
        host_budget_tokens=host_budget_tokens, lambda_gpu=lambda_gpu, lambda_host=lambda_host,
        protected_gpu_pages=protected_gpu_pages, protected_host_pages=protected_host_pages,
        protected_requests=protected_requests, density=density) for density in (False, True))
    return max(choices, key=lambda selected: sum(choice.value for choice in selected))


def _allocate_retention_greedy(requests, *, gpu_budget_tokens, host_budget_tokens,
                              lambda_gpu, lambda_host, protected_gpu_pages,
                              protected_host_pages, protected_requests, density):
    """Choose positive marginal value, counting a shared causal page only once.

    Value uses the next READ target saving, deduplicated against protected
    requests, minus this call's control, measured external-request penalty and
    incremental residency prices.
    Single-prefix full-window residuals are not additive across protected sets.
    Capacity budgets bound the protected set; soft references allocate no KV.
    """
    requests = {(request.owner_id, request.forecast.horizon_seconds): request for request in requests}
    gpu_pages, host_pages = set(protected_gpu_pages), set(protected_host_pages)
    gpu_leases = {page: (float("inf"), 0) for page in gpu_pages}
    host_leases = {page: (float("inf"), 0) for page in host_pages}
    covered_events = {}

    def cover(request):
        forecast = request.forecast
        event = request.reuse_event_key or request.owner_id
        covered_events[event] = max(covered_events.get(event, 0), forecast.target_saved_ms)
        for pages, leases, seconds in ((request.gpu_page_ids, gpu_leases, forecast.extra_gpu_seconds_per_page),
                                      (request.host_page_ids, host_leases, forecast.extra_host_seconds_per_page)):
            for page in pages:
                if page not in leases or forecast.horizon_seconds > leases[page][0]:
                    leases[page] = forecast.horizon_seconds, seconds
        gpu_pages.update(request.gpu_page_ids)
        host_pages.update(request.host_page_ids)

    for request in protected_requests:
        cover(request)

    def added_residency(pages, leases, forecast, seconds, size):
        return size*sum(seconds if page not in leases else seconds-leases[page][1]
            for page in pages if page not in leases or forecast.horizon_seconds > leases[page][0])

    choices = []
    while requests:
        candidates = []
        for request in requests.values():
            gpu = len(request.gpu_page_ids-gpu_pages)*request.page_size
            host = len(request.host_page_ids-host_pages)*request.page_size
            forecast = request.forecast
            if not any(page not in leases or forecast.horizon_seconds > leases[page][0]
                       for pages, leases in ((request.gpu_page_ids, gpu_leases), (request.host_page_ids, host_leases))
                       for page in pages):
                continue
            if ((len(gpu_pages | request.gpu_page_ids)*request.page_size > gpu_budget_tokens)
                    or (len(host_pages | request.host_page_ids)*request.page_size > host_budget_tokens)):
                continue
            gpu_integral = added_residency(request.gpu_page_ids, gpu_leases, forecast,
                                           forecast.extra_gpu_seconds_per_page, request.page_size)
            host_integral = added_residency(request.host_page_ids, host_leases, forecast,
                                            forecast.extra_host_seconds_per_page, request.page_size)
            event = request.reuse_event_key or request.owner_id
            # Credit each target READ once. Include the paired intervention's
            # external-request penalty; positive residual benefits may overlap
            # another target and are not an additional reward.
            target = forecast.target_saved_ms
            target -= min(max(target, 0), covered_events.get(event, 0))
            saved = target-forecast.retention_control_ms+min(0.0, forecast.expected_collateral_saved_ms)
            value = saved-lambda_gpu*gpu_integral-lambda_host*host_integral
            if value > 0:
                choice = RetentionChoice(request.owner_id, forecast.horizon_seconds,
                    saved, gpu_integral, host_integral, value, gpu, host,
                    forecast.evidence_digest, event)
                candidates.append(choice)
        if not candidates:
            break
        def priority(item):
            if not density:
                return (False, item.value, item.value, item.owner_id)
            capacity = ((item.marginal_gpu_tokens/gpu_budget_tokens if gpu_budget_tokens else 0)
                        +(item.marginal_host_tokens/host_budget_tokens if host_budget_tokens else 0))
            # Renewing an already covered page can add value without adding capacity.
            return (capacity == 0, item.value/capacity if capacity else item.value,
                    item.value, item.owner_id)
        choice = max(candidates, key=priority)
        selected = requests[choice.owner_id, choice.ttl_seconds]
        requests = {key: request for key, request in requests.items() if request.owner_id != choice.owner_id}
        cover(selected)
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
            raise ValueError("retention fit requires exact complete READ demand labels")
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
            if len({session_group(row, components) for row in cohort}) < 2:
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
            target_gain = probability*(native_target-retained_target) if probability else 0
            forecasts[key] = PrefixForecast(cohort[0]["horizon_seconds"], probability,
                native_target, retained_target,
                gpu_seconds/(gpu_pages*cohort[0]["page_size"]) if gpu_pages else 0,
                host_seconds/(host_pages*cohort[0]["page_size"]) if host_pages else 0,
                cohort[0]["cache_scope_hash"], evidence,
                control, native_window-retained_window-target_gain+control, native_window, retained_window)
        model = cls(forecasts, schema)
        model.independent_samples = len(components)
        return model

    def predict(self, features, cache_scope_hash, horizon_seconds):
        return self.forecasts[self._key(self.feature_bucket(features), cache_scope_hash, horizon_seconds)]

    def to_dict(self):
        return {"kind": "paired_native_retention_v5", "feature_schema": self.feature_schema,
                "target_demand_rule": TARGET_DEMAND_RULE,
                "value_basis": "marginal exact source-demand READ saving minus control and paired external-request penalty; full protected-set net measured by set intervention",
                "bin_convention": "left_closed_right_open; None is missing",
                "independent_samples": self.independent_samples,
                "sample_unit": "connected session groups; TTL arms and overlapping windows are repeated measures",
                "set_intervention_required": "same fixed workload Native versus protected owner set; record all response intervals, controls and physical residency",
                "forecasts": {key: asdict(value) for key, value in self.forecasts.items()}}

    @classmethod
    def from_dict(cls, value):
        if (value["kind"] != "paired_native_retention_v5"
                or value.get("target_demand_rule") != TARGET_DEMAND_RULE):
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
        self._allocation_lock = asyncio.Lock()
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

    async def __call__(self, plan, task, units, rendered, record):
        # Observation and the resulting owner changes must see one allocation
        # order, including callbacks restoring an already completed READ.
        async with self._allocation_lock:
            return await self._observe_and_allocate(plan, task, units, rendered, record)

    async def _observe_and_allocate(self, plan, task, units, rendered, record):
        from controller.reading import stable_prefix_input_ids, stable_prefix_messages, stable_source_payload
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
            updated = await self.owners.reconcile(session_id)
            owner = next((row for row in updated if row.owner_id == owner_id), self.store.prefix_owner(owner_id))
            completed = access["payload"].get("policy_result")
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
        ranges, _ = source_token_ranges(self.template, rendered, stable_prefix_messages(plan, task, units), units)
        event = source_demand_event_key(prefix_ids, ranges, inspected["page_size"], inspected["cache_scope_hash"])
        if event is None:
            result = {**inspected, "state": "UNRETAINED", "reason": "no_complete_source_body_page"}
            self.store.complete_prefix_access(owner_id=owner_id,
                native_attempt_id=record.receipt.attempt_id, result=result)
            return result
        arguments = dict(owner_id=owner_id, session_id=session_id, attempt_id=record.attempt_id,
            step_id=record.step_id, receipt=record.receipt, prefix_input_ids=prefix_ids, sources=sources)
        if self.settings["mode"] == "fixed_ttl":
            choice = dict(qualification_only=True, features=features)
            result = await self.owners.observe(**arguments, ttl_seconds=self.settings["ttl_seconds"],
                policy="fixed_ttl_mechanism", policy_choice=choice)
        else:
            result = await self._allocate_value(plan, arguments, inspected, features, time.monotonic_ns(), event)
        self.store.complete_prefix_access(owner_id=owner_id,
            native_attempt_id=record.receipt.attempt_id, result=result)
        return result

    async def _allocate_value(self, plan, current, inspected, features, now, event_key):
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

        budgets = dict(gpu_budget_tokens=self.settings["gpu_budget_tokens"],
            host_budget_tokens=self.settings["host_budget_tokens"], lambda_gpu=self.settings["lambda_gpu"],
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
                await self.owners.release(owner_id, reason="nonpositive_or_uncalibrated_marginal_value")
        current_ack = (native[current_id] if current_id in kept_ids else
            {**inspected, "state": "UNRETAINED", "policy": "value",
             "policy_choice": {"calibrated": bool(curves[current_id]), "features": features,
                "action_feasible": feasible[current_id], "feature_bucket": self.model.feature_bucket(features),
                "model_digest": self.model_digest}})
        for owner_id, choice in selected.items():
            forecast = next(point for point in curves[owner_id] if point.horizon_seconds == choice.ttl_seconds)
            policy_choice = {**asdict(choice), "model_digest": self.model_digest,
                "trigger_native_attempt_id": trigger,
                "capacity_basis": "unique full causal pages; incremental lease residency beyond active protection",
                "raw_features": feature_rows[owner_id],
                "feature_bucket": self.model.feature_bucket(feature_rows[owner_id]),
                "forecast": asdict(forecast)}
            if owner_id == current_id:
                current_ack = await self.owners.observe(**current, ttl_seconds=choice.ttl_seconds,
                    policy="paired_native_value", policy_choice=policy_choice)
            else:
                old = records[owner_id]
                await self.owners.observe(owner_id=owner_id, session_id=old.session_id,
                    attempt_id=old.attempt_id, step_id=old.step_id, receipt=old.receipt,
                    prefix_input_ids=old.prefix_input_ids, sources=old.sources, ttl_seconds=choice.ttl_seconds,
                    policy="paired_native_value", policy_choice=policy_choice)
        return {**current_ack, "retention_choices": [asdict(choice) for choice in choices],
            "existing_retention_owners": sorted(kept_ids-selected.keys()),
            "retention_candidates": [{"owner_id": owner_id,
                "reuse_event_key": events[owner_id], "action_feasible": feasible[owner_id],
                "raw_features": feature_rows[owner_id],
                "feature_bucket": self.model.feature_bucket(feature_rows[owner_id]),
                "forecasts": [asdict(prediction) for prediction in curve]}
                for owner_id, curve in curves.items()]}
