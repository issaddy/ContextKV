"""Protocol/allocator checks; actual native eviction is exercised by the GPU probe."""
from collections import OrderedDict
from dataclasses import replace
from types import ModuleType, SimpleNamespace
import sys

import pytest

from controller.prefix_retention import (PrefixForecast,
    RetentionRequest, allocate_retention)
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


def test_shared_pages_are_charged_once_and_native_hits_have_no_added_value():
    forecast = PrefixForecast(10, .8, 100, 10, 5, 5, "cache", "paired-observation-digest")
    a = RetentionRequest("a", forecast, frozenset({"root", "a"}), 64)
    b = RetentionRequest("b", forecast, frozenset({"root", "b"}), 64)
    natural = RetentionRequest("native-hit", replace(forecast, native_reuse_cost_ms=10),
        frozenset({"native"}), 64)
    chosen = allocate_retention((a, b, natural), protection_budget_tokens=192,
        lambda_gpu=.01, lambda_host=0)
    assert {row.owner_id for row in chosen} == {"a", "b"}
    assert sum(row.marginal_protected_tokens for row in chosen) == 192
    assert [row.marginal_protected_tokens for row in chosen] == [128, 64]
    assert all(row.expected_saved_ms == 72 for row in chosen)
    assert all(row.evidence_digest == forecast.evidence_digest for row in chosen)


def test_complete_source_demand_identity_and_single_owner_collateral_are_distinct():
    from controller.prefix_retention import source_demand_event_key
    # Shared first body page does not make two complete READ paths one demand.
    first = source_demand_event_key((1, 2, 3, 4), [[0, 4]], 2, "scope")
    second = source_demand_event_key((1, 2, 5, 6), [[0, 4]], 2, "scope")
    assert first != second
    assert first == source_demand_event_key((1, 2, 3, 4), [[0, 4]], 2, "scope")
    forecast = PrefixForecast(60, 1, 100, 0, 0, 0, "scope", "paired",
        retention_control_ms=5, expected_collateral_saved_ms=-120,
        native_window_cost_ms=200, retained_window_cost_ms=225)
    assert forecast.saved_ms == -25
    assert forecast.full_window_saved_ms == -25
    request = RetentionRequest("losing-owner", forecast, frozenset({"page"}), 64, "read")
    assert allocate_retention((request,), protection_budget_tokens=64,
                              lambda_gpu=0, lambda_host=0) == ()


@pytest.mark.asyncio
async def test_demand_retains_partial_causal_prefix_only_when_it_contains_shared_body():
    from controller.context_schema import ContextPlan, EvidenceSet, Need, ReadResult, ReadTask, ReadingOutcome, StepRecord, TaskState
    from controller.plan_policy import PlanCandidate, PlanStep
    from controller.prefix_retention import PrefixReadObserver
    from controller.schema import CacheQuote, CacheQuoteBatch, PromptCandidate

    old, later = (ReadTask(name, "READ", ("n",)) for name in ("old", "later"))
    plan = ContextPlan("plan", TaskState("task", "question", needs=(Need("n"),)),
                       EvidenceSet("evidence", ()), steps=(old, later))
    record = StepRecord("attempt", old.step_id, old, input_ids=(1, 2, 3, 4), metrics={
        "source_view": {"body_token_ranges": ((0, 4),)},
        "evidence_prefix": {"demand_observation": True, "engine_restart_id": "restart",
            "cache_config_hash": "cache", "owner_id": "old", "page_size": 2,
            "boundary_tokens": 4, "cache_scope_hash": "scope"}})
    state = ReadingOutcome(plan, results=(ReadResult(old.step_id),), steps=(record,))
    ids = (1, 2, 5, 6)
    candidate = PromptCandidate("later", "REBUILD", 0, (), ids, token_hash(ids), frozenset(), frozenset(),
        output_budget=2, source_body_token_ranges=((0, 4),), source_prefix_tokens=4)
    quote = CacheQuote(candidate.candidate_id, candidate.input_hash, 4, 4,
                       ready_gpu_prefix_tokens=2, recoverable_prefix_tokens=2, evictable_gpu_prefix_tokens=2)
    batch = CacheQuoteBatch("quote", 1, 1, "restart", "cache", 0, 0, (quote,))
    gap = PlanCandidate("gap", "gap_retrieval", plan, (PlanStep(ReadTask("gap", "GAP")),), "reader")
    observer = PrefixReadObserver(SimpleNamespace(prefix_owners=lambda *args, **kwargs: ()), None,
                                  {"mode": "demand", "protection_budget_tokens": 4})

    async def after_gap(target):
        read = PlanCandidate("next", "read_once", plan, (PlanStep(later, (target,)),), "reader")
        return (await observer.demand_options((gap, read), state, "session", batch,
            prefix_status={"prefixes": []}))["gap"]

    shared = await after_gap(candidate)
    lease, = shared.retain
    assert (lease.source_step_id, lease.target_step_id, lease.boundary_tokens) == ("old", "later", 2)
    assert shared.protected_tokens == 2
    assert (await after_gap(replace(candidate, source_body_token_ranges=((2, 4),)))).retain == ()
