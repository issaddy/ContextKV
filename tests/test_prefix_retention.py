"""Protocol/allocator checks; actual native eviction is exercised by the GPU probe."""
from collections import OrderedDict
from dataclasses import replace
from types import ModuleType, SimpleNamespace
import sys

import pytest

from controller.prefix_retention import (EmpiricalRetentionModel, PrefixForecast,
    RetentionRequest, allocate_retention, retainable_prefix)
from engine_patches.contextkv_runtime import (ContextKVError, ContextKVRuntime,
    executed_prefix_hashes, token_hash)


class Key:
    def __init__(self, ids):
        self.ids = tuple(ids)
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, item):
        return Key(self.ids[item])
    def child_key(self, page_size):
        return self.ids[:page_size]
    def page_aligned(self, page_size):
        return self[:len(self)//page_size*page_size]
    def match(self, key, page_size):
        count = 0
        for left, right in zip(self.ids, key.ids):
            if left != right:
                break
            count += 1
        return count//page_size*page_size


class Node:
    def __init__(self, ids, parent=None):
        self.key, self.parent, self.children = Key(ids), parent, {}
        self.component_data = [SimpleNamespace(value=list(ids), host_value=list(ids),
            session_ref=0, lock_ref=0)]
        self.contextkv_h2d_pending_id = self.write_through_pending_id = None


class Component:
    def __init__(self, root):
        self.root, self._session_leaves = root, {}
    def register_session_leaf(self, sid, node):
        if node in self._session_leaves.get(sid, ()):
            return
        self._session_leaves.setdefault(sid, set()).add(node)
        while node is not self.root:
            node.component_data[0].session_ref += 1
            node = node.parent
    def release_session(self, sid):
        for node in self._session_leaves.pop(sid, ()):
            while node is not self.root:
                node.component_data[0].session_ref -= 1
                node = node.parent


@pytest.fixture
def owner(monkeypatch):
    radix = ModuleType("sglang.srt.mem_cache.radix_cache")
    radix.RadixKey = Key
    base = ModuleType("sglang.srt.mem_cache.base_prefix_cache")
    base.MatchPrefixParams = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, radix.__name__, radix)
    monkeypatch.setitem(sys.modules, base.__name__, base)
    root = Node(())
    prefix = Node((1, 2), root)
    tail = Node((3, 4), prefix)
    root.children[(1, 2)], prefix.children[(3, 4)] = prefix, tail
    component = Component(root)
    actions, matches = [], []
    def match(params):
        matches.append(params.key.ids)
        return SimpleNamespace(best_match_node=1 if len(params.key) == 2 else 2,
                               cache_actions=["native_split_action"])
    tree = SimpleNamespace(root_node=root, page_size=2, match_prefix=match,
                           node_by_id=lambda node_id: {1: prefix, 2: tail}[node_id])
    from engine_patches.contextkv_runtime import peek_full_prefix
    tree.peek_full_prefix = lambda key: peek_full_prefix(tree, key)
    tracker = SimpleNamespace(_session_generations={"business": 1},
        _closed_session_ids=OrderedDict(), components=(component,))
    cache = SimpleNamespace(page_size=2, _tree_core_backend="python", cache_controller=None,
        tree_core=tree, session_refs=tracker, _apply_cache_actions=actions.extend,
        token_to_kv_pool_allocator=SimpleNamespace(size=128, available_size=lambda: 124))
    def open_session(sid):
        tracker._session_generations[sid] = max(tracker._session_generations.values(), default=0)+1
        return tracker._session_generations[sid]
    def release_session(sid):
        component.release_session(sid)
        tracker._session_generations.pop(sid, None)
        tracker._closed_session_ids[sid] = None
    cache.open_radix_session, cache.release_radix_session = open_session, release_session
    runtime = ContextKVRuntime(cache, {"reader": "R0"})
    component.register_session_leaf("business", tail)
    runtime.attempts["read"] = dict(state="COMPLETED", terminal=True,
        receipt=dict(generation_complete=True))
    runtime.executed_prefixes["read"] = executed_prefix_hashes((1, 2, 3, 4, 5), 2)
    return runtime, cache, component, prefix, tail, actions, matches


def retain_data(runtime, ids=(1, 2, 3)):
    return dict(expected_restart_id=runtime.restart_id, fingerprint=runtime.fingerprint,
        attempt_id="read", owner_id="source-v1", input_ids=ids,
        sources=[{"source_id": "policy", "version": "v1"}], ttl_seconds=30)


def test_prefix_hashes_are_exact_existing_wire_hashes_at_actual_pages():
    ids = (21, 88, 0, 103, -4)
    assert executed_prefix_hashes(ids, 2) == {2: token_hash(ids[:2]), 4: token_hash(ids[:4])}


def test_independent_owner_uses_page_endpoint_and_survives_business_release(owner):
    runtime, cache, component, prefix, tail, actions, matches = owner
    ack = runtime.retain_prefix(retain_data(runtime))
    assert ack["boundary_tokens"] == 2 and ack["prefix_hash"] == token_hash((1, 2))
    assert matches == [(1, 2)] and actions == ["native_split_action"]
    sid = ack["native_session_id"]
    assert component._session_leaves[sid] == {prefix}
    assert tail.component_data[0].session_ref == 1  # Independent owner stops before this suffix.
    cache.release_radix_session("business")
    assert tail.component_data[0].session_ref == 0
    status = runtime.prefix_status(dict(expected_restart_id=runtime.restart_id, owner_id="source-v1"))
    assert status["reference_present"] and prefix.component_data[0].session_ref == 1
    assert status["residency"]["ready_gpu_prefix_tokens"] == 2
    assert len(status["gpu_page_ids"]) == len(status["host_page_ids"]) == 1
    runtime.release_prefix(dict(expected_restart_id=runtime.restart_id, owner_id="source-v1"))
    assert prefix.component_data[0].session_ref == 0 and prefix.component_data[0].value == [1, 2]


def test_unexecuted_path_and_incomplete_request_cannot_register_owner(owner):
    runtime, cache, component, *_ = owner
    before = dict(cache.session_refs._session_generations)
    with pytest.raises(ContextKVError, match="PREFIX_NOT_EXECUTED"):
        runtime.retain_prefix(retain_data(runtime, (1, 99)))
    runtime.attempts["read"]["state"] = "INFLIGHT"
    with pytest.raises(ContextKVError, match="PREFIX_REQUIRES_COMPLETED_EXECUTION"):
        runtime.retain_prefix(retain_data(runtime))
    assert cache.session_refs._session_generations == before
    assert set(component._session_leaves) == {"business"}


def test_inspection_is_read_only_and_returns_actual_two_tier_pages_before_ownership(owner):
    runtime, cache, component, prefix, tail, actions, matches = owner
    before = dict(cache.session_refs._session_generations), prefix.component_data[0].session_ref
    inspected = runtime.prefix_status(retain_data(runtime))
    assert inspected["state"] == "OBSERVED" and inspected["protection"] == "none"
    assert inspected["boundary_tokens"] == 2 and not inspected["reference_present"]
    assert len(inspected["gpu_page_ids"]) == len(inspected["host_page_ids"]) == 1
    assert not runtime.prefix_owners and not actions and not matches
    assert before == (cache.session_refs._session_generations, prefix.component_data[0].session_ref)
    with pytest.raises(ContextKVError, match="PREFIX_NOT_EXECUTED"):
        runtime.prefix_status(retain_data(runtime, (1, 99)))


def test_owner_renewal_is_idempotent_and_ttl_releases_only_its_reference(owner, monkeypatch):
    runtime, _, component, prefix, *_ = owner
    monkeypatch.setattr("engine_patches.contextkv_runtime.time.monotonic", lambda: 100)
    first = runtime.retain_prefix(retain_data(runtime))
    second = runtime.retain_prefix({**retain_data(runtime), "ttl_seconds": 60})
    assert first["session_generation"] == second["session_generation"]
    assert prefix.component_data[0].session_ref == 2 and second["expires_at"] == 160
    with pytest.raises(ContextKVError, match="PREFIX_OWNER_CONFLICT"):
        runtime.retain_prefix({**retain_data(runtime), "sources": [{"version": "v2"}]})
    monkeypatch.setattr("engine_patches.contextkv_runtime.time.monotonic", lambda: 161)
    runtime.expire_prefixes()
    assert prefix.component_data[0].session_ref == 1
    assert set(component._session_leaves) == {"business"}


def test_new_completed_read_reestablishes_an_evicted_owner_at_its_exact_endpoint(owner):
    runtime, _, component, prefix, *_ = owner
    ack = runtime.retain_prefix(retain_data(runtime))
    component.release_session(ack["native_session_id"])
    assert not runtime.prefix_status(dict(expected_restart_id=runtime.restart_id,
        owner_id="source-v1"))["reference_present"]
    renewed = runtime.retain_prefix(retain_data(runtime))
    assert renewed["session_generation"] == ack["session_generation"]
    assert renewed["reference_present"] and prefix.component_data[0].session_ref == 2


def test_independent_owner_can_renew_after_business_receipt_history_expires(owner):
    runtime, _, component, prefix, *_ = owner
    first = runtime.retain_prefix(retain_data(runtime))
    runtime.attempts.clear()
    runtime.executed_prefixes.clear()
    renewed = runtime.retain_prefix(retain_data(runtime))
    assert renewed["source_attempt_id"] == "read"
    assert renewed["session_generation"] == first["session_generation"]
    assert prefix.component_data[0].session_ref == 2


def test_shared_pages_are_charged_once_and_native_hits_have_no_added_value():
    forecast = PrefixForecast(10, .8, 100, 10, 5, 5, "cache", "paired-observation-digest")
    a = RetentionRequest("a", forecast, frozenset({"root", "a"}), frozenset(), 64)
    b = RetentionRequest("b", forecast, frozenset({"root", "b"}), frozenset(), 64)
    natural = RetentionRequest("native-hit", replace(forecast, native_reuse_cost_ms=10),
        frozenset({"native"}), frozenset(), 64)
    chosen = allocate_retention((a, b, natural), gpu_budget_tokens=192, host_budget_tokens=0,
        lambda_gpu=.01, lambda_host=0)
    assert {row.owner_id for row in chosen} == {"a", "b"}
    assert sum(row.marginal_gpu_tokens for row in chosen) == 192
    assert [row.marginal_gpu_tokens for row in chosen] == [128, 64]
    assert all(row.expected_saved_ms == 72 for row in chosen)
    assert all(row.evidence_digest == forecast.evidence_digest for row in chosen)


def test_host_only_path_keeps_future_gpu_opportunity_cost_and_shared_protection_once():
    forecast = PrefixForecast(10, 1, 100, 20, 2, 1, "scope", "real-paired-proof")
    observed = dict(owner_id="host-only", cache_scope_hash="scope", boundary_tokens=128,
        page_size=64, gpu_page_ids=[], host_page_ids=["root", "source"],
        pool={"host_capacity_tokens": 1024}, residency={"recoverable_prefix_tokens": 128,
            "inflight_h2d_tokens": 0, "inflight_d2h_tokens": 0})
    request = RetentionRequest.from_receipt(observed, forecast)
    assert request.gpu_page_ids == request.host_page_ids == frozenset({"root", "source"})
    assert retainable_prefix(observed)
    assert not retainable_prefix({**observed, "residency": {**observed["residency"], "recoverable_prefix_tokens": 64}})
    chosen = allocate_retention((request, replace(request, owner_id="same-path")),
        gpu_budget_tokens=128, host_budget_tokens=128, lambda_gpu=.1, lambda_host=.1)
    assert len(chosen) == 1 and chosen[0].extra_gpu_token_seconds == 256
    assert chosen[0].extra_host_token_seconds == 128 and chosen[0].value == pytest.approx(41.6)
    gpu_only = RetentionRequest.from_receipt({**observed, "pool": {"host_capacity_tokens": 0},
        "gpu_page_ids": observed["host_page_ids"], "host_page_ids": []}, forecast)
    assert not gpu_only.host_page_ids and gpu_only.gpu_page_ids == request.gpu_page_ids


def test_empirical_value_requires_fixed_paths_and_same_cache_recipe():
    rows = [dict(features={"revisit": True}, cache_scope_hash="cache", horizon_seconds=10,
        reused=True, native_input_hashes=["same"], retained_input_hashes=["same"],
        native_reuse_cost_ms=100, retained_reuse_cost_ms=20,
        native_window_cost_ms=100, retained_window_cost_ms=20,
        retention_control_ms=0,
        gpu_page_count=2, host_page_count=0, page_size=64,
        native_gpu_token_seconds=128, retained_gpu_token_seconds=256,
        native_host_token_seconds=0, retained_host_token_seconds=0)]
    model = EmpiricalRetentionModel.fit(rows)
    forecast = model.predict({"revisit": True}, "cache", 10)
    assert forecast.saved_ms == 80 and forecast.extra_gpu_seconds_per_page == 1
    assert EmpiricalRetentionModel.from_dict(model.to_dict()).predict({"revisit": True}, "cache", 10) == forecast
    with pytest.raises(KeyError):
        model.predict({"revisit": True}, "new-cache", 10)
    with pytest.raises(ValueError, match="identical executed input"):
        EmpiricalRetentionModel.fit([{**rows[0], "retained_input_hashes": ["changed"]}])


def test_fit_and_predict_use_frozen_buckets_for_continuous_access_and_pressure_features():
    raw = dict(prior_access_count=2, last_access_interval_seconds=5.1,
        source_turn_gap=5, gpu_pressure=.81, host_pressure=.51)
    row = dict(features=raw, cache_scope_hash="cache", horizon_seconds=30, reused=True,
        native_input_hashes=["same"], retained_input_hashes=["same"],
        native_reuse_cost_ms=100, retained_reuse_cost_ms=20,
        native_window_cost_ms=100, retained_window_cost_ms=20,
        retention_control_ms=0,
        gpu_page_count=1, host_page_count=1, page_size=64,
        native_gpu_token_seconds=0, retained_gpu_token_seconds=0,
        native_host_token_seconds=0, retained_host_token_seconds=0)
    model = EmpiricalRetentionModel.fit([row])
    nearby = dict(prior_access_count=3, last_access_interval_seconds=5.9,
        source_turn_gap=7, gpu_pressure=.84, host_pressure=.56)
    assert model.predict(nearby, "cache", 30).saved_ms == 80
    saved = model.to_dict()
    assert saved["feature_schema"]["last_access_interval_seconds"]["origin"] == "registered_fixed_thresholds"
    assert EmpiricalRetentionModel.from_dict(saved).feature_bucket(nearby) == model.feature_bucket(raw)
    assert raw["last_access_interval_seconds"] == 5.1
    with pytest.raises(KeyError):
        model.predict({**nearby, "gpu_pressure": .96}, "cache", 30)
