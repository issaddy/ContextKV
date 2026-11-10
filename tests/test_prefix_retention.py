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
    from unittest.mock import AsyncMock

    from controller.context_schema import (CompletionReceipt, ContextPlan, EvidenceSet, Need,
        ReadResult, ReadTask, ReadingOutcome, StepRecord, TaskState)
    from controller.plan_policy import PlanCandidate, PlanStep
    from controller.prefix_retention import PrefixReadObserver
    from controller.schema import CacheQuote, CacheQuoteBatch, PromptCandidate
    from controller.templates import ByteTemplate

    old, later = (ReadTask(name, "READ", ("n",)) for name in ("old", "later"))
    tail = ReadTask("tail", "READ", ("other",))
    plan = ContextPlan("plan", TaskState("task", "question", needs=(Need("n", operation="count"), Need("other"))),
                       EvidenceSet("evidence", ()), steps=(old, later, tail))
    original = (1, 2, 3, 4)
    receipt = CompletionReceipt("old", "old", "epoch", 1, "restart", token_hash(original), "stop", True)
    record = StepRecord("attempt", old.step_id, old, input_ids=original, receipt=receipt, metrics={
        "cache_config_hash": "cache",
        "source_view": {"body_token_ranges": ((0, 4),), "source_prefix_tokens": 4}})
    state = ReadingOutcome(plan, results=(ReadResult(old.step_id),), steps=(record,))
    ids = (1, 2, 5, 6)
    candidate = PromptCandidate("later", "REBUILD", 0, (), ids, token_hash(ids), frozenset(), frozenset(),
        output_budget=2, source_body_token_ranges=((0, 4),), source_prefix_tokens=4)
    quote = CacheQuote(candidate.candidate_id, candidate.input_hash, 4, 4,
                       ready_gpu_prefix_tokens=2, recoverable_prefix_tokens=2, evictable_gpu_prefix_tokens=2)
    batch = CacheQuoteBatch("quote", 1, 1, "restart", "cache", 0, 0, (quote,),
                            fingerprint=ByteTemplate().fingerprint, native_running_budget_tokens=4)
    gap = PlanCandidate("gap", "gap_retrieval", plan, (PlanStep(ReadTask("gap", "GAP")),), "reader")
    observer = PrefixReadObserver(SimpleNamespace(prefix_owners=lambda *args, **kwargs: ()), None,
                                  {"mode": "demand", "ttl_seconds": 15, "protection_budget_tokens": 4})
    observer.runtime = SimpleNamespace(status=AsyncMock())

    async def after_gap(target, active=()):
        read = PlanCandidate("next", "read_once", plan, (PlanStep(later, (target,)),), "reader")
        observer.runtime.status.return_value = {"prefixes": active}
        planned_gap = replace(gap, route_steps=(PlanStep(later, (target,)),))
        return (await observer.demand_options((planned_gap, read), state, "session", batch, page_size=2))["gap"]

    shared = await after_gap(candidate)
    observer.allocation = "paired_value"
    observer.runtime.status.reset_mock()
    native = await after_gap(candidate)
    assert native.retain == () and native.retention_evidence == "native_unpriced_net_value"
    observer.runtime.status.assert_not_awaited()
    observer.allocation = "next_use"
    lease, = shared.retain
    assert (lease.source_step_id, lease.target_step_id, lease.boundary_tokens) == ("old", "later", 2)
    assert shared.protected_tokens == 2
    page = lease.cache_scope_hash + ":" + token_hash(ids[:2])
    existing = {"owner_id": "another-session", "gpu_page_ids": [page], "host_page_ids": [],
                "referenced_page_ids": [page], "reference_present": True}
    reused = await after_gap(candidate, (existing,))
    assert reused.retain == ()
    assert reused.protected_tokens == reused.uncovered_risk_tokens == 0
    assert (await after_gap(replace(candidate, source_body_token_ranges=((2, 4),)))).retain == ()

    owner = SimpleNamespace(owner_id=lease.owner_id, policy="joint_demand", prefix_input_ids=ids[:2],
        step_id="old", ttl_seconds=15, lease_started_ns=__import__("time").monotonic_ns())
    observer.store = SimpleNamespace(prefix_owners=lambda *args, **kwargs: (owner,))
    read = PlanCandidate("next", "read_once", plan, (PlanStep(later, (candidate,)),), "reader")
    observer.runtime.status.return_value = {"prefixes": ({**existing, "owner_id": owner.owner_id},)}
    options = await observer.demand_options((gap, read), state, "session", replace(batch, free_gpu_tokens=16),
                                            page_size=2)
    assert options[candidate.candidate_id].release == ()
    assert options[candidate.candidate_id].protected_tokens == 2
    assert options["gap"].release == (owner.owner_id,)
    # The selected ANSWER reuses only the first body page of a longer owner.
    # Keep that owner through execution instead of unpinning the quoted page.
    owner.prefix_input_ids = original
    full_pages = [lease.cache_scope_hash+":"+value for value in executed_prefix_hashes(original, 2).values()]
    observer.runtime.status.return_value = {"prefixes": ({**existing, "owner_id": owner.owner_id,
        "gpu_page_ids": full_pages, "referenced_page_ids": full_pages},)}
    answer = ReadTask("answer", "ANSWER", ("n",))
    answer_plan = PlanCandidate("answer", "direct_answer", plan, (PlanStep(answer, (candidate,)),), "reader")
    final = (await observer.demand_options((answer_plan,), state, "session", batch,
                                           page_size=2))[candidate.candidate_id]
    assert final.keep == (owner.owner_id,) and final.release == () and final.retain == ()
    assert final.protected_tokens == 4
    # A common non-source header alone does not prolong the original-body lease.
    header = replace(candidate, source_body_token_ranges=((2, 4),))
    answer_plan = replace(answer_plan, steps=(PlanStep(answer, (header,)),))
    final = (await observer.demand_options((answer_plan,), state, "session", batch,
                                           page_size=2))[candidate.candidate_id]
    assert final.keep == () and final.release == (owner.owner_id,)

    long_ids = (1, 2, 3, 4, 7)
    long_candidate = replace(candidate, candidate_id="tail", input_ids=long_ids, input_hash=token_hash(long_ids))
    long_quote = CacheQuote("tail", long_candidate.input_hash, 5, 4,
        ready_gpu_prefix_tokens=4, recoverable_prefix_tokens=4, evictable_gpu_prefix_tokens=4)
    reads = (PlanCandidate(step.step_id, "read_once", plan, (PlanStep(step, (compiled,)),), "reader")
             for step, compiled in ((later, candidate), (tail, long_candidate)))
    observer.runtime.status.return_value = {"prefixes": ()}
    gap = replace(gap, route_steps=(PlanStep(later, (candidate,)), PlanStep(tail, (long_candidate,))))
    joint = (await observer.demand_options((gap, *reads), state, "session",
        replace(batch, quotes=(quote, long_quote)), page_size=2))["gap"]
    assert [lease.boundary_tokens for lease in joint.retain] == [4]
    assert joint.protected_tokens == 4 and joint.uncovered_risk_tokens == 0


@pytest.mark.asyncio
async def test_future_hybrid_original_body_remains_a_consumer_after_facts_complete():
    from unittest.mock import AsyncMock
    from controller.context_schema import (AnswerPlan, AnswerPoint, ContextPlan, EvidenceClaim,
        EvidenceSet, EvidenceUnit, Need, ReadResult, ReadTask, ReadingOutcome, SourceSpan, StepRecord, TaskState)
    from controller.demand import DemandState
    from controller.evidence import make_evidence
    from controller.plan_policy import PlanCandidate, PlanStep
    from controller.prefix_retention import PrefixReadObserver
    from controller.schema import CacheQuote, CacheQuoteBatch, CompletionReceipt, PromptCandidate
    from controller.templates import ByteTemplate

    block = make_evidence("ABCD", "source")
    span = SourceSpan("span", block.key, 0, 4)
    unit = EvidenceUnit("u", (span,), block.text, need_ids=("n",))
    fact = EvidenceClaim("fact", "n", "extract", "ABCD", source_spans=(span,))
    source = ReadTask("source", "READ", ("n",), input_unit_ids=("u",))
    reduce = ReadTask("reduce", "REDUCE", ("n",), input_claim_ids=("fact",), operation="summarize")
    answer = ReadTask("answer", "ANSWER", ("n",))
    plan = ContextPlan("plan", TaskState("task", "question", needs=(Need("n"),)),
        EvidenceSet("evidence", (unit,)), steps=(source, reduce),
        answer_plan=AnswerPlan((AnswerPoint("n", ("fact",)),)))
    ids = (1, 2, 3, 4)
    receipt = CompletionReceipt("source", "source", "epoch", 1, "restart", token_hash(ids), "stop", True)
    record = StepRecord("attempt", "source", source, input_ids=ids, receipt=receipt, metrics={
        "cache_config_hash": "cache", "source_view": {"body_token_ranges": ((0, 4),), "source_prefix_tokens": 4}})
    state = ReadingOutcome(plan, results=(ReadResult("source", (fact,), ("u",), ("n",)),),
                           steps=(record,), claims=(fact,))
    assert DemandState.from_outcome(state).needs == ()
    candidate = PromptCandidate("hybrid", "REBUILD", 0, (), ids, token_hash(ids), frozenset(), frozenset(),
        output_budget=2, representation="hybrid", source_body_token_ranges=((0, 4),), source_prefix_tokens=4)
    quote = CacheQuote("hybrid", candidate.input_hash, 4, 4, ready_gpu_prefix_tokens=4,
                       recoverable_prefix_tokens=4, evictable_gpu_prefix_tokens=4)
    batch = CacheQuoteBatch("quote", 1, 1, "restart", "cache", 0, 0, (quote,),
        fingerprint=ByteTemplate().fingerprint, native_running_budget_tokens=4)
    proposal = PlanCandidate("reduce", "reduce_once", plan, (PlanStep(reduce),), "reader",
        route_steps=(PlanStep(answer, (candidate,)),))
    observer = PrefixReadObserver(SimpleNamespace(prefix_owners=lambda *args, **kwargs: ()), None,
        {"mode": "demand", "ttl_seconds": 15, "protection_budget_tokens": 4})
    observer.runtime = SimpleNamespace(status=AsyncMock(return_value={"prefixes": ()}))
    option = (await observer.demand_options((proposal,), state, "session", batch, page_size=2))["reduce"]
    lease, = option.retain
    assert lease.target_step_id == "answer" and lease.boundary_tokens == 4
    assert option.retention_allocation_value_ms is None
    facts = replace(candidate, representation="facts", source_body_token_ranges=(), source_prefix_tokens=0)
    proposal = replace(proposal, route_steps=(PlanStep(answer, (facts,)),))
    assert (await observer.demand_options((proposal,), state, "session", batch, page_size=2))["reduce"].retain == ()


def test_consumer_page_savings_are_unioned_across_owners_and_residence_is_nonnegative():
    base = PrefixForecast(15, 0, None, None, 5, 0, "scope", "paired")
    # Each owner can prevent part of the same future request's recomputation.
    first = RetentionRequest("a", base, frozenset({"root", "a"}), 1,
        consumer_savings=(("read", (("root", 20), ("a", 20))),))
    second = RetentionRequest("b", replace(base, horizon_seconds=5, extra_gpu_seconds_per_page=10),
        frozenset({"root", "b"}), 1,
        consumer_savings=(("read", (("root", 20), ("b", 20))),))
    choices = allocate_retention((first, second), protection_budget_tokens=3, lambda_gpu=.1, lambda_host=0)
    assert len(choices) == 2
    assert sum(choice.expected_saved_ms for choice in choices) == 60
    # Shared root uses max(5, 10), despite the shorter lease's higher forecast.
    assert sum(choice.extra_gpu_token_seconds for choice in choices) == 25
    assert all(choice.extra_gpu_token_seconds >= 0 for choice in choices)


def test_unpriced_risk_is_never_reported_as_milliseconds_and_prefers_shorter_equal_lease():
    forecast = PrefixForecast(15, 0, None, None, 0, 0, "scope", "risk_rule")
    renew = RetentionRequest("owner", forecast, frozenset({"page"}), 2,
        consumer_savings=(("read", (("page", 2),)),), value_unit="risk_tokens")
    keep = replace(renew, forecast=replace(forecast, horizon_seconds=7))
    choice, = allocate_retention((renew, keep), protection_budget_tokens=2, lambda_gpu=0, lambda_host=0)
    assert choice.ttl_seconds == 7
    assert choice.value == 2 and choice.expected_saved_ms is None
    assert choice.value_unit == "risk_tokens"


@pytest.mark.asyncio
async def test_demand_keep_avoids_retain_and_renew_uses_same_owner_despite_resident_pages():
    from unittest.mock import AsyncMock
    from controller.context_schema import ReadingOutcome, ContextPlan, TaskState, EvidenceSet
    from controller.demand import DemandState, DemandLease, JointOption
    from controller.prefix_retention import PrefixReadObserver

    prefix = (1, 2)
    owner = SimpleNamespace(sources=({"original": "source"},))
    record = SimpleNamespace(step_id="source", attempt_id="attempt", input_ids=prefix,
                             receipt=SimpleNamespace(engine_restart_id="restart"))
    state = ReadingOutcome(ContextPlan("p", TaskState("t", "q"), EvidenceSet("e", ())), steps=(record,))
    observer = PrefixReadObserver(SimpleNamespace(prefix_owner=lambda key: owner), None,
        {"mode": "demand", "ttl_seconds": 15, "protection_budget_tokens": 2})
    page = "scope:" + token_hash(prefix)
    native = {"owner_id": "lease", "gpu_page_ids": [page], "host_page_ids": [],
              "reference_present": True, "referenced_page_ids": [page], "state": "RETAINED"}
    observer.runtime = SimpleNamespace(status=AsyncMock(return_value={"prefixes": (native,)}))
    observer.owners = SimpleNamespace(observe=AsyncMock(return_value=native), release=AsyncMock())
    await observer.apply_demand(JointOption(DemandState((), ()), keep=("lease",)), state, "session", batch=None)
    observer.runtime.status.assert_not_awaited()
    observer.owners.observe.assert_not_awaited()
    lease = DemandLease("lease", "source", "future", 2, 2, 2, "scope", 60, "renew")
    await observer.apply_demand(JointOption(DemandState((), ()), retain=(lease,)), state,
                                "session", batch=SimpleNamespace(engine_restart_id="restart", observed_seq=1))
    args = observer.owners.observe.await_args.kwargs
    assert args["owner_id"] == "lease" and args["ttl_seconds"] == 60
    assert args["sources"] == owner.sources


@pytest.mark.asyncio
async def test_demand_remaining_lease_competes_with_renewal_at_consumer_arrival():
    import time
    from unittest.mock import AsyncMock
    from controller.context_schema import (CompletionReceipt, ContextPlan, EvidenceSet, Need,
        ReadResult, ReadTask, ReadingOutcome, StepRecord, TaskState)
    from controller.plan_policy import PlanCandidate, PlanStep
    from controller.prefix_retention import PrefixReadObserver
    from controller.schema import CacheQuote, CacheQuoteBatch, PromptCandidate
    from controller.templates import ByteTemplate

    source, later = (ReadTask(name, "READ", ("n",)) for name in ("source", "later"))
    plan = ContextPlan("p", TaskState("t", "q", needs=(Need("n", operation="count"),)),
                       EvidenceSet("e", ()), steps=(source, later))
    ids = (1, 2, 3, 4)
    receipt = CompletionReceipt("source", "source", "epoch", 1, "restart", token_hash(ids), "stop", True)
    record = StepRecord("attempt", source.step_id, source, input_ids=ids, receipt=receipt, metrics={
        "cache_config_hash": "cache", "source_view": {"body_token_ranges": ((0, 4),), "source_prefix_tokens": 4}})
    state = ReadingOutcome(plan, results=(ReadResult(source.step_id),), steps=(record,))
    candidate = PromptCandidate("later", "REBUILD", 0, (), ids, token_hash(ids), frozenset(), frozenset(),
        output_budget=2, source_body_token_ranges=((0, 4),), source_prefix_tokens=4)
    quote = CacheQuote("later", candidate.input_hash, 4, 4, ready_gpu_prefix_tokens=4,
                       recoverable_prefix_tokens=4, evictable_gpu_prefix_tokens=4)
    batch = CacheQuoteBatch("quote", 1, 1, "restart", "cache", 0, 0, (quote,),
        fingerprint=ByteTemplate().fingerprint, native_running_budget_tokens=4)
    gap = PlanCandidate("gap", "gap_retrieval", plan, (PlanStep(ReadTask("gap", "GAP")),), "reader",
                        route_steps=(PlanStep(later, (candidate,)),))
    observer = PrefixReadObserver(SimpleNamespace(prefix_owners=lambda *args, **kwargs: ()), None,
        {"mode": "demand", "ttl_seconds": 15, "horizon_seconds_choices": (15, 60, 180),
         "protection_budget_tokens": 4})
    observer.runtime = SimpleNamespace(status=AsyncMock(return_value={"prefixes": ()}))
    first = (await observer.demand_options((gap,), state, "session", batch, page_size=2))["gap"]
    lease, = first.retain
    pages = [lease.cache_scope_hash+":"+page for page in executed_prefix_hashes(ids, 2).values()]
    owner = SimpleNamespace(owner_id=lease.owner_id, policy="joint_demand", prefix_input_ids=ids,
        step_id="source", ttl_seconds=15, lease_started_ns=time.monotonic_ns()-8_000_000_000)
    observer.store = SimpleNamespace(prefix_owners=lambda *args, **kwargs: (owner,))
    observer.runtime.status.return_value = {"prefixes": ({"owner_id": owner.owner_id,
        "gpu_page_ids": pages, "host_page_ids": [], "referenced_page_ids": pages, "reference_present": True},)}
    keep = (await observer.demand_options((gap,), state, "session", batch, page_size=2,
                                         candidate_arrival_ms={"later": 6000}))["gap"]
    assert keep.keep == (owner.owner_id,) and keep.retain == () and keep.release == ()
    renew = (await observer.demand_options((gap,), state, "session", batch, page_size=2,
                                          candidate_arrival_ms={"later": 20000}))["gap"]
    renewal, = renew.retain
    assert renewal.owner_id == owner.owner_id and renewal.action == "renew"
    assert renewal.ttl_seconds == 60 and renewal.consumer_arrival_ms == 20000




@pytest.mark.asyncio
async def test_demand_priced_multiowner_counts_consumers_not_owner_window_sums(monkeypatch):
    from unittest.mock import AsyncMock
    from controller.context_schema import (CompletionReceipt, ContextPlan, EvidenceSet, Need,
        ReadResult, ReadTask, ReadingOutcome, StepRecord, TaskState)
    from controller.plan_policy import PlanCandidate, PlanStep
    import controller.prefix_retention as retention
    from controller.schema import CacheQuote, CacheQuoteBatch, PromptCandidate
    from controller.templates import ByteTemplate

    old = tuple(ReadTask("old"+str(i), "READ", ("n",)) for i in range(2))
    future = tuple(ReadTask("future"+str(i), "READ", ("n",)) for i in range(2))
    plan = ContextPlan("p", TaskState("t", "q", needs=(Need("n", operation="count"),)),
        EvidenceSet("e", ()), steps=(*old, *future))
    records, candidates, quotes = [], [], []
    for previous, target, ids in zip(old, future, ((1, 2, 3, 4), (1, 2, 5, 6)), strict=True):
        receipt = CompletionReceipt(previous.step_id, previous.step_id, "epoch", 1, "restart",
                                    token_hash(ids), "stop", True)
        records.append(StepRecord("attempt", previous.step_id, previous, input_ids=ids, receipt=receipt, metrics={
            "cache_config_hash": "cache", "source_view": {"body_token_ranges": ((0, 4),), "source_prefix_tokens": 4},
            "evidence_prefix": {"boundary_tokens": 4, "demand_observation": True}}))
        candidates.append(PromptCandidate(target.step_id, "REBUILD", 0, (), ids, token_hash(ids), frozenset(), frozenset(),
            output_budget=2, source_body_token_ranges=((0, 4),), source_prefix_tokens=4))
        quotes.append(CacheQuote(target.step_id, token_hash(ids), 4, 4, ready_gpu_prefix_tokens=4,
                                 recoverable_prefix_tokens=4, evictable_gpu_prefix_tokens=4))
    state = ReadingOutcome(plan, results=tuple(ReadResult(task.step_id) for task in old), steps=tuple(records))
    batch = CacheQuoteBatch("quote", 1, 1, "restart", "cache", 0, 0, tuple(quotes),
        fingerprint=ByteTemplate().fingerprint, native_running_budget_tokens=4)
    gap = PlanCandidate("gap", "gap_retrieval", plan, (PlanStep(ReadTask("gap", "GAP")),), "reader",
        route_steps=tuple(PlanStep(task, (candidate,)) for task, candidate in zip(future, candidates, strict=True)))
    observer = retention.PrefixReadObserver(SimpleNamespace(prefix_owners=lambda *args, **kwargs: ()), None,
        {"mode": "demand", "ttl_seconds": 15, "protection_budget_tokens": 6})
    observer.runtime = SimpleNamespace(status=AsyncMock(return_value={"prefixes": ()}))
    observer.model = object()
    observer._demand_forecast = lambda inspected, batch, horizon, boundary: PrefixForecast(horizon, 1, 100, 0,
        0, 0, "scope", "paired", retention_control_ms=5,
        native_window_cost_ms=20000, retained_window_cost_ms=0)
    chosen = []
    original_allocate = retention.allocate_retention
    def capture(requests, **kwargs):
        result = original_allocate(requests, **kwargs)
        chosen.extend(result)
        return result
    monkeypatch.setattr(retention, "allocate_retention", capture)
    option = (await observer.demand_options((gap,), state, "session", batch, page_size=2))["gap"]
    assert len(option.retain) == 2 and option.protected_tokens == 6
    assert sum(choice.expected_saved_ms for choice in chosen) == 190
    assert option.retention_allocation_value_ms == 190
    assert option.retention_control_ms == 10  # Two owners; not one call per consumer.
    # A longer ANSWER can consume the whole calibrated READ path.
    answer = ReadTask("answer", "ANSWER", ("n",))
    answer_ids = (*candidates[0].input_ids, 7, 8)
    answer_input = replace(candidates[0], candidate_id="answer", representation="hybrid", generation_kind="ANSWER",
        input_ids=answer_ids, input_hash=token_hash(answer_ids), source_body_token_ranges=((0, 6),), source_prefix_tokens=6)
    answer_quote = replace(quotes[0], candidate_id="answer", input_hash=answer_input.input_hash, input_len=6, match_limit=6)
    gap = replace(gap, route_steps=(PlanStep(answer, (answer_input,)),))
    chosen.clear()
    option = (await observer.demand_options((gap,), state, "session",
        replace(batch, quotes=(*batch.quotes, answer_quote)), page_size=2))["gap"]
    assert len(option.retain) == 1 and option.protected_tokens == 4
    assert option.retention_allocation_value_ms == 95 and option.retention_control_ms == 5
