"""CPU regressions for measured demand value and scheduler-owned soft quotas.

The controller, allocator, Native execution-proof checks and retain/release
entrypoints are real. Only transport and the Radix tree's storage are fakes;
these tests make no model-quality, physical-memory or latency-speedup claims.
"""
import asyncio
from collections import deque
from dataclasses import asdict, replace
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from controller.context_schema import (ContextPlan, TaskState, Need, EvidenceSet,
    ReadTask, ReadingOutcome, ReadResult, StepRecord, CompletionReceipt)
from controller.demand import DemandState, JointOption
from controller.engine_client import EngineError
from controller.plan_policy import PlanCandidate, PlanStep
from controller.prefix_retention import (EmpiricalRetentionModel, PrefixForecast,
    PrefixReadObserver, RetentionRequest, allocate_retention)
from controller.prefix_view import native_cache_scope_hash
from controller.schema import PromptCandidate, CacheQuote, CacheQuoteBatch, ConfigFingerprint
from controller.templates import token_hash
from engine_patches.contextkv_runtime import ContextKVError, ContextKVRuntime, executed_prefix_hashes


class MemoryStore:
    def __init__(self):
        self.records = {}

    def prefix_owners(self, session_id=None, active_only=False, stage=None):
        return tuple(row for row in self.records.values()
            if (session_id is None or row.session_id == session_id)
            and (stage is None or row.stage == stage)
            and (not active_only or row.stage in {"RETAINING", "RETAINED", "RELEASING"}))

    def prefix_owner(self, owner_id):
        return self.records.get(owner_id)

    def save_prefix_owner(self, record, *args, **kwargs):
        self.records[record.owner_id] = record


class FakeRadixKey(tuple):
    def __getitem__(self, index):
        value = super().__getitem__(index)
        return type(self)(value) if isinstance(index, slice) else value

    def child_key(self, page_size):
        return tuple(self[:page_size])

    def match(self, other, *, page_size):
        common = 0
        for left, right in zip(self, other):
            if left != right:
                break
            common += 1
        return common//page_size*page_size


class FakeTree:
    def __init__(self):
        self.nodes = {}
        self.root_node = self._node((), None)

    def _node(self, ids, parent):
        node = SimpleNamespace(id=len(self.nodes), key=FakeRadixKey(ids), parent=parent,
            children={}, component_data=[SimpleNamespace(value=True, host_value=True, session_ref=0)],
            contextkv_h2d_pending_id=None, write_through_pending_id=None)
        self.nodes[node.id] = node
        return node

    def add(self, ids):
        node = self.root_node
        for offset in range(0, len(ids)//64*64, 64):
            page = tuple(ids[offset:offset+64])
            if page not in node.children:
                node.children[page] = self._node(page, node)
            node = node.children[page]

    def match_prefix(self, params):
        node, key = self.root_node, params.key
        while len(key) >= 64 and key.child_key(64) in node.children:
            node = node.children[key.child_key(64)]
            key = key[64:]
        return SimpleNamespace(best_match_node=node.id, cache_actions=())

    def node_by_id(self, identity):
        return self.nodes[identity]

    def peek_full_prefix(self, key):
        node = self.node_by_id(self.match_prefix(SimpleNamespace(key=key)).best_match_node)
        size = 0
        while node is not self.root_node:
            size += len(node.key)
            node = node.parent
        return dict(ready_gpu_prefix_tokens=size, recoverable_prefix_tokens=size,
            host_only_load_tokens=0, inflight_h2d_tokens=0, inflight_d2h_tokens=0)


class FakeComponent:
    def __init__(self, tree):
        self.tree, self._session_leaves = tree, {}

    def register_session_leaf(self, sid, node):
        self._session_leaves[sid] = [node]
        self.recount()

    def recount(self):
        for node in self.tree.nodes.values():
            node.component_data[0].session_ref = 0
        for leaves in self._session_leaves.values():
            for node in leaves:
                while node is not self.tree.root_node:
                    node.component_data[0].session_ref += 1
                    node = node.parent


class FakeCache:
    page_size = 64

    def __init__(self):
        self.tree_core = FakeTree()
        self.session_refs = SimpleNamespace(_session_generations={},
            components=[FakeComponent(self.tree_core)])

    def _apply_cache_actions(self, actions):
        assert not actions

    def open_radix_session(self, sid):
        self.session_refs._session_generations[sid] = 1
        return 1

    def release_radix_session(self, sid):
        self.session_refs._session_generations.pop(sid, None)
        for component in self.session_refs.components:
            component._session_leaves.pop(sid, None)
            component.recount()


class NativeTransport:
    fingerprint = ConfigFingerprint()

    def __init__(self):
        runtime = self.runtime = ContextKVRuntime.__new__(ContextKVRuntime)
        runtime.cache = FakeCache()
        runtime.fingerprint = asdict(self.fingerprint)
        runtime.restart_id, runtime.cache_config_hash = "restart", "cache"
        runtime.prefix_scope_hash = native_cache_scope_hash(self.fingerprint, "cache")
        runtime.prefix_owners, runtime.attempts, runtime.executed_prefixes, runtime.epochs = {}, {}, {}, {}
        runtime.events, runtime.event_seq = deque(), 0
        runtime.pool_snapshot = lambda: dict(gpu_capacity_tokens=1024, host_capacity_tokens=1024,
            free_gpu_tokens=0, free_host_tokens=1024, page_size=64)
        runtime.workload_snapshot = lambda: dict(active_requests=1, queued_requests=0)
        self.calls = []

    def completed(self, attempt_id, ids):
        self.runtime.cache.tree_core.add(ids)
        self.runtime.attempts[attempt_id] = dict(state="COMPLETED", receipt={"generation_complete": True})
        self.runtime.executed_prefixes[attempt_id] = executed_prefix_hashes(ids, 64)

    def call(self, action, body):
        modules = {
            "sglang.srt.mem_cache.base_prefix_cache": SimpleNamespace(MatchPrefixParams=SimpleNamespace),
            "sglang.srt.mem_cache.radix_cache": SimpleNamespace(RadixKey=FakeRadixKey),
        }
        with patch.dict(sys.modules, modules):
            return self.runtime.dispatch(action, body)

    async def _post(self, route, body):
        self.calls.append((route, body))
        await asyncio.sleep(0)
        action = {"retain": "retain_prefix", "release": "release_prefix", "status": "prefix_status"}[route.rsplit("/", 1)[1]]
        try:
            return self.call(action, body)
        except ContextKVError as exc:
            raise EngineError(exc.code) from exc


def case(offset=0):
    fp = ConfigFingerprint()
    scope = native_cache_scope_hash(fp, "cache")
    source_ids = tuple(range(offset, offset+129))
    other_ids = tuple(range(offset+1000, offset+1129))

    def candidate(name, ids):
        return PromptCandidate(name, "REBUILD", 0, (), ids, token_hash(ids), frozenset(),
            frozenset(), output_budget=1, fingerprint=fp,
            source_prefix_tokens=128, source_body_token_ranges=((64, 128),))

    source, compete, answer = candidate("source", source_ids), candidate("compete", other_ids), candidate("answer", source_ids)
    read = ReadTask("read-source", "READ", ("form",))
    next_read, ans = ReadTask("compete", "READ", ("other",)), ReadTask("answer", "ANSWER", ("form",))
    plan = ContextPlan("context", TaskState("task", "Question", needs=(Need("form"), Need("other"))),
        EvidenceSet("sources", ()), steps=(read, next_read, ans))
    receipt = CompletionReceipt("rid", "native-source-"+str(offset), "epoch", 1, "restart",
        source.input_hash, "stop", True)
    inspection = dict(demand_observation=True, cache_scope_hash=scope, page_size=64,
        retention_features={"access_state": "first"},
        pool=dict(gpu_capacity_tokens=1024, host_capacity_tokens=10000))
    record = StepRecord("attempt-"+str(offset), "read-source", read, input_ids=source_ids,
        receipt=receipt, metrics={"cache_config_hash": "cache", "evidence_prefix": inspection,
            "source_view": {"body_token_ranges": ((64, 128),), "source_prefix_tokens": 128}})
    state = ReadingOutcome(plan, results=(ReadResult("read-source", elapsed_ms=1),), steps=(record,))
    route = PlanCandidate("continue", "read_once", plan, (PlanStep(next_read, (compete,)),),
        "fixed-qwen", route_steps=(PlanStep(next_read, (compete,)), PlanStep(ans, (answer,))))
    q_compete = CacheQuote("compete", compete.input_hash, 129, 128)
    q_answer = CacheQuote("answer", answer.input_hash, 129, 128, raw_match_tokens=128,
        ready_gpu_prefix_tokens=128, recoverable_prefix_tokens=128,
        gpu_token_ranges=((0, 128),), host_token_ranges=((0, 128),),
        evictable_gpu_token_ranges=((0, 128),), evictable_host_token_ranges=((0, 128),))
    batch = CacheQuoteBatch("quote", 1, 0, "restart", "cache", 0, 10000,
        (q_compete, q_answer), fingerprint=fp, queued_requests=0, active_requests=1,
        host_write_policy="write_back")
    return route, state, batch


def settings(budget=8192):
    return {"mode": "demand", "allocation": "next_use", "ttl_seconds": 15,
        "horizon_seconds_choices": (15, 60, 180), "protection_budget_tokens": budget}


def model(**overrides):
    scope = native_cache_scope_hash(ConfigFingerprint(), "cache")
    features = {"access_state": "first", "native_residency": "device", "pressure_concurrency": "high:single"}
    forecast = PrefixForecast(15, 1, 600, 100, 0, 0, scope, "test-paired-observations",
        retention_control_ms=1, native_window_cost_ms=1000, retained_window_cost_ms=501,
        source_token_range=(128, 128))
    forecast = replace(forecast, **overrides)
    return EmpiricalRetentionModel({EmpiricalRetentionModel._key(features, scope, 15): forecast},
        {key: {"kind": "categorical"} for key in features}, independent_samples=2)


async def option(observer, bundle, session="session"):
    route, state, batch = bundle
    return (await observer.demand_options((route,), state, session, batch, page_size=64,
        candidate_arrival_ms={"answer": 1}))["compete"]


class DemandValueTests(unittest.IsolatedAsyncioTestCase):
    async def test_unpriced_next_use_does_not_call_native_or_allocate_risk_tokens(self):
        for observed_control in (0, 10000):
            engine = NativeTransport()
            observer = PrefixReadObserver(MemoryStore(), engine, settings())
            observer._owner_control_observations.append(observed_control)
            selected = await option(observer, case())
            self.assertFalse(selected.retain)
            self.assertIsNone(selected.retention_allocation_value_ms)
            self.assertEqual(selected.retention_evidence, "native_unpriced_net_value")
            self.assertEqual(engine.calls, [])

    async def test_measured_owner_control_changes_the_decision(self):
        outcomes = []
        for observed_control in (0, 10000):
            observer = PrefixReadObserver(MemoryStore(), NativeTransport(), settings())
            observer.model = model()
            observer._owner_control_observations.append(observed_control)
            outcomes.append(await option(observer, case()))
        self.assertEqual(len(outcomes[0].retain), 1)
        self.assertGreater(outcomes[0].retention_allocation_value_ms, 0)
        self.assertEqual(outcomes[1].retain, ())
        self.assertLessEqual(outcomes[1].retention_allocation_value_ms, 0)

    async def test_fixed_control_is_charged_once_and_can_veto_lease_set(self):
        observer = PrefixReadObserver(MemoryStore(), NativeTransport(), settings())
        observer.model = model()
        observer._control_observations.append(10000)
        selected = await option(observer, case())
        self.assertEqual(selected.retain, ())
        self.assertEqual(selected.protected_tokens, 0)
        self.assertEqual(selected.uncovered_risk_tokens, 128)
        self.assertEqual(selected.retention_control_ms, 0)
        self.assertGreaterEqual(selected.retention_fixed_control_ms, 10000)
        self.assertEqual(selected.retention_evidence, "native_nonpositive_net_value")

    async def test_collateral_and_residency_costs_are_not_discarded(self):
        for forecast, resource_prices in (
            (model(expected_collateral_saved_ms=-600), {}),
            (model(extra_gpu_seconds_per_page=1), {"lambda_gpu": 10}),
            (model(extra_host_seconds_per_page=1), {"lambda_host": 10}),
        ):
            observer = PrefixReadObserver(MemoryStore(), NativeTransport(), {**settings(), **resource_prices})
            observer.model = forecast
            self.assertEqual((await option(observer, case())).retain, ())

    async def test_missing_feature_bucket_and_unmeasured_lengths_use_native(self):
        for forecast in (model(source_token_range=(256, 512)), EmpiricalRetentionModel({}, {})):
            observer = PrefixReadObserver(MemoryStore(), NativeTransport(), settings())
            observer.model = forecast
            selected = await option(observer, case())
            self.assertFalse(selected.retain)
            self.assertEqual(selected.retention_evidence, "native_unpriced_net_value")

    def test_next_use_loads_the_same_supported_pricing_asset(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            asset = Path(directory)/"pricing.json"
            asset.write_text(json.dumps(model().to_dict()))
            observer = PrefixReadObserver(MemoryStore(), NativeTransport(),
                {**settings(), "pricing_asset": str(asset)})
            self.assertIsNotNone(observer.model)
            self.assertTrue(observer.observe_reads)

    def test_shared_pages_and_consumer_savings_are_counted_once(self):
        forecast = PrefixForecast(15, 1, 10, 0, 1, 1, "scope", "measured", retention_control_ms=1)
        requests = [RetentionRequest(owner, forecast, frozenset({"a", "b"}), 64,
            consumer_savings=(("answer", (("a", 5), ("b", 5))),)) for owner in ("one", "two")]
        selected = allocate_retention(requests, protection_budget_tokens=128, lambda_gpu=0, lambda_host=0)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].marginal_protected_tokens, 128)
        self.assertEqual(selected[0].value, 9)


class NativeQuotaTests(unittest.IsolatedAsyncioTestCase):
    async def test_stale_options_from_two_observers_cannot_exceed_shared_quota(self):
        engine = NativeTransport()
        observers = [PrefixReadObserver(MemoryStore(), engine, settings(128)) for _ in range(2)]
        bundles = [case(0), case(10000)]
        for observer, bundle in zip(observers, bundles):
            observer.model = model()
            record = bundle[1].steps[0]
            engine.completed(record.receipt.attempt_id, record.input_ids)
        selected = await asyncio.gather(*(option(observer, bundle, session)
            for observer, bundle, session in zip(observers, bundles, ("A", "B"))))
        self.assertEqual([len(row.retain) for row in selected], [1, 1])
        applied = await asyncio.gather(*(observer.apply_demand(choice, bundle[1], session, batch=bundle[2])
            for observer, choice, bundle, session in zip(observers, selected, bundles, ("A", "B"))))
        states = sorted(ack["state"] for result in applied for ack in result["owner_status"].values())
        self.assertEqual(states, ["RETAINED", "UNRETAINED"])
        self.assertEqual(len(engine.runtime.prefix_owners), 1)
        union = set().union(*(set(executed_prefix_hashes(owner["input_ids"], 64).values())
            for owner in engine.runtime.prefix_owners.values()))
        self.assertEqual(len(union)*64, 128)
        self.assertEqual(sorted(row.stage for observer in observers for row in observer.store.records.values()),
            ["RELEASED", "RETAINED"])
        for observer, bundle, session in zip(observers, bundles, ("A", "B")):
            await observer.finish_demand(bundle[1], session, bundle[1].steps[0].attempt_id)
        self.assertEqual(engine.runtime.prefix_owners, {})

    def request(self, engine, owner="A", offset=0, budget=128):
        ids = tuple(range(offset, offset+128))
        attempt = "executed-"+str(offset)
        engine.completed(attempt, ids)
        return dict(owner_id=owner, attempt_id=attempt, input_ids=ids, sources=[],
            ttl_seconds=15, expected_restart_id="restart", fingerprint=asdict(engine.fingerprint),
            protection_budget_tokens=budget)

    def test_identical_shared_paths_renew_and_release_idempotently(self):
        engine = NativeTransport()
        first = self.request(engine)
        engine.call("retain_prefix", first)
        second = {**first, "owner_id": "B"}
        engine.call("retain_prefix", second)
        renewed = engine.call("retain_prefix", {**first, "protection_budget_tokens": 64})
        self.assertEqual(renewed["state"], "RETAINED")
        self.assertEqual(len(engine.runtime.prefix_owners), 2)
        with self.assertRaises(ContextKVError) as caught:
            engine.call("retain_prefix", self.request(engine, "C", 10000, 64))
        self.assertEqual(caught.exception.code, "PREFIX_PROTECTION_BUDGET_EXCEEDED")
        for _ in range(2):
            engine.call("release_prefix", dict(owner_id="A", expected_restart_id="restart"))
        remaining = engine.call("prefix_status", dict(owner_id="B", expected_restart_id="restart"))
        self.assertTrue(remaining["reference_present"])
        self.assertEqual(len(remaining["referenced_page_ids"])*64, 128)

    def test_quota_does_not_bypass_execution_fingerprint_or_restart_checks(self):
        engine = NativeTransport()
        request = self.request(engine)
        for changed, code in (
            ({"input_ids": tuple(range(1000, 1128))}, "PREFIX_NOT_EXECUTED"),
            ({"expected_restart_id": "old"}, "STALE_RESTART"),
            ({"fingerprint": {}}, "FINGERPRINT_MISMATCH"),
            ({"protection_budget_tokens": -1}, "INVALID_PREFIX_PROTECTION_BUDGET"),
            ({"protection_budget_tokens": True}, "INVALID_PREFIX_PROTECTION_BUDGET"),
        ):
            with self.assertRaises(ContextKVError) as caught:
                engine.call("retain_prefix", {**request, **changed})
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(engine.runtime.prefix_owners, {})

    def test_expired_owner_releases_budget_through_native_expiry(self):
        engine = NativeTransport()
        engine.call("retain_prefix", self.request(engine))
        engine.runtime.prefix_owners["A"]["expires_at"] = time.monotonic()-1
        engine.runtime.expire_prefixes()
        receipt = engine.call("retain_prefix", self.request(engine, "B", 10000))
        self.assertEqual(receipt["state"], "RETAINED")
        self.assertEqual(set(engine.runtime.prefix_owners), {"B"})

    async def test_nonresident_failure_leaves_no_active_owner(self):
        engine = NativeTransport()
        observer = PrefixReadObserver(MemoryStore(), engine, settings(128))
        observer.model = model()
        bundle = case()
        record = bundle[1].steps[0]
        engine.completed(record.receipt.attempt_id, record.input_ids)
        selected = await option(observer, bundle)
        engine.runtime.cache.tree_core.root_node.children.clear()
        result = await observer.apply_demand(selected, bundle[1], "session", batch=bundle[2])
        ack, = result["owner_status"].values()
        self.assertEqual(ack["state"], "UNRETAINED")
        self.assertEqual(ack["reason"], "prefix_not_resident")
        self.assertEqual(observer.store.prefix_owners(active_only=True), ())
        self.assertEqual(engine.runtime.prefix_owners, {})

    async def test_repeated_terminal_cleanup_is_safe(self):
        engine = NativeTransport()
        observer = PrefixReadObserver(MemoryStore(), engine, settings(128))
        observer.model = model()
        bundle = case()
        record = bundle[1].steps[0]
        engine.completed(record.receipt.attempt_id, record.input_ids)
        selected = await option(observer, bundle)
        await observer.apply_demand(selected, bundle[1], "session", batch=bundle[2])
        release = JointOption(DemandState((), ()), release=(selected.retain[0].owner_id,))
        await observer.apply_demand(release, bundle[1], "session", batch=None)
        await observer.apply_demand(release, bundle[1], "session", batch=None)
        self.assertEqual(engine.runtime.prefix_owners, {})
        self.assertEqual(observer.store.prefix_owners(active_only=True), ())


class IntegrationArtifactTests(unittest.TestCase):
    def test_deployment_patch_contains_current_runtime_and_hash(self):
        root = Path(__file__).resolve().parents[1]
        integration = root/"engine_patches/integrations/sglang-0.5.18"
        manifest = json.loads((integration/"manifest.json").read_text())
        patch_bytes = (integration/manifest["patch"]).read_bytes()
        self.assertEqual(hashlib.sha256(patch_bytes).hexdigest(), manifest["patch_sha256"])
        target = "python/sglang/srt/managers/contextkv_runtime.py"
        section = patch_bytes.decode().split(f"diff --git a/{target} b/{target}\n", 1)[1]
        section = section.split("\ndiff --git ", 1)[0]
        actual = "\n".join(line[1:] for line in section.splitlines()
                           if line.startswith("+") and not line.startswith("+++"))+"\n"
        self.assertEqual(actual, (root/"engine_patches/contextkv_runtime.py").read_text())
        compile(actual, target, "exec")


if __name__ == "__main__":
    unittest.main()
