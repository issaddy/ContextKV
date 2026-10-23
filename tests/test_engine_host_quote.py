"""Offline CPU checks of pure quotes and patched native source fragments.

The small fixture contains exact original SGLang definitions, their pinned
commit/file hashes and original line numbers. Native-method tests apply this
repository's actual patch hunks before AST extraction. No network, /tmp checkout
or CUDA is needed; these tests are not hardware validation.
"""
from __future__ import annotations

from array import array
import ast
import copy
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import textwrap
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runtime():
    spec = importlib.util.spec_from_file_location(
        "contextkv_runtime_host_quote_test", ROOT / "engine_patches/contextkv_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StrictKey:
    """Small test double that preserves native array and page-match contracts."""
    def __init__(self, ids, limit=None):
        self.token_ids = ids
        self.limit = limit

    def __len__(self):
        return min(len(self.token_ids),self.limit) if self.limit is not None else len(self.token_ids)

    def __getitem__(self, index):
        return StrictKey(self.token_ids[:len(self)][index])

    def page_aligned(self, size):
        return self[:len(self)//size*size]

    def child_key(self, page_size=1):
        return self.child_key_at(0,page_size)

    def child_key_at(self, offset, page_size=1):
        assert offset+page_size <= len(self)
        return self.token_ids[offset] if page_size == 1 else tuple(self.token_ids[offset:offset+page_size])

    def match(self, other, page_size=1):
        return self.match_at(other,0,page_size)

    def match_at(self, other, offset, page_size=1):
        assert type(self.token_ids) is type(other.token_ids)
        count = 0
        for a,b in zip(self.token_ids[:len(self)],other.token_ids[offset:len(other)]):
            if a != b:
                break
            count += 1
        return count//page_size*page_size


def node(ids=(), *, gpu=True, host=False, h2d=None, d2h=None, node_id=1, key_cls=StrictKey):
    return SimpleNamespace(
        id=node_id,key=key_cls(array("q",ids)),children={},parent=None,
        component_data=[SimpleNamespace(value=tuple(ids) if gpu else None,
                                         host_value=tuple(ids) if host else None)],
        contextkv_h2d_pending_id=h2d,write_through_pending_id=d2h,
        load_back_pending_id=None,last_access_time=123,lock_ref=0,session_ref=0)


def tree_chain(*specs, page_size=1, key_cls=StrictKey):
    root = node((),node_id=0,key_cls=key_cls)
    current = root
    nodes = []
    all_ids = []
    for i,spec in enumerate(specs,1):
        ids,kwargs = spec
        child = node(ids,node_id=i,key_cls=key_cls,**kwargs)
        child.parent = current
        current.children[child.key.child_key(page_size)] = child
        nodes.append(child)
        current = child
        all_ids.extend(ids)
    tree = SimpleNamespace(root_node=root,page_size=page_size,is_eagle=False)
    return tree,nodes,key_cls(array("q",all_ids))


def compact_state(tree):
    """Avoid parent cycles while checking every field quote may accidentally touch."""
    state = []
    pending = [tree.root_node]
    while pending:
        current = pending.pop()
        state.append((current.id,tuple(current.key.token_ids),tuple(current.children),
            current.parent.id if current.parent else None,
            copy.deepcopy(current.component_data[0].__dict__),
            current.contextkv_h2d_pending_id,current.write_through_pending_id,
            current.last_access_time,current.lock_ref,current.session_ref))
        pending.extend(current.children.values())
    return tuple(state)


def test_contiguous_gpu_then_host_prefix_has_no_double_count(runtime):
    tree,_,key = tree_chain(((1,2,3,4),{}),((5,6,7,8),{"gpu":False,"host":True}))
    quote = runtime.peek_full_prefix(tree,key)
    assert quote["raw_match_tokens"] == 8
    assert quote["ready_gpu_prefix_tokens"] == 4
    assert quote["recoverable_prefix_tokens"] == 8
    assert quote["host_only_load_tokens"] == 4
    assert quote["inflight_h2d_tokens"] == quote["inflight_d2h_tokens"] == 0


def test_pending_d2h_is_not_host_ready_but_existing_gpu_is_ready(runtime):
    tree,_,key = tree_chain(((1,2,3,4),{"gpu":True,"host":True,"d2h":7}))
    quote = runtime.peek_full_prefix(tree,key)
    assert quote["ready_gpu_prefix_tokens"] == quote["recoverable_prefix_tokens"] == 4
    assert quote["host_only_load_tokens"] == 0
    assert quote["inflight_d2h_tokens"] == 4


def test_recoverable_prefix_cannot_jump_over_pending_d2h_hole(runtime):
    tree,_,key = tree_chain(((1,2,3,4),{}),
        ((5,6,7,8),{"gpu":False,"host":True,"d2h":7}),
        ((9,10,11,12),{"gpu":False,"host":True}))
    quote = runtime.peek_full_prefix(tree,key)
    assert quote["raw_match_tokens"] == 12
    assert quote["ready_gpu_prefix_tokens"] == quote["recoverable_prefix_tokens"] == 4
    assert quote["host_only_load_tokens"] == 0
    assert quote["inflight_d2h_tokens"] == 4


def test_pending_h2d_is_counted_once_and_not_new_h2d(runtime):
    tree,_,key = tree_chain(((1,2,3,4),{}),
        ((5,6,7,8),{"gpu":True,"host":True,"h2d":7}),
        ((9,10,11,12),{"gpu":False,"host":True}))
    quote = runtime.peek_full_prefix(tree,key)
    assert quote["ready_gpu_prefix_tokens"] == 4
    assert quote["recoverable_prefix_tokens"] == 12
    assert quote["inflight_h2d_tokens"] == 4
    assert quote["host_only_load_tokens"] == 4


def test_pending_device_allocation_without_ready_host_is_not_recoverable(runtime):
    tree,_,key = tree_chain(((1,2,3,4),{}),
        ((5,6,7,8),{"gpu":True,"host":False,"h2d":7}))
    quote = runtime.peek_full_prefix(tree,key)
    assert quote["ready_gpu_prefix_tokens"] == quote["recoverable_prefix_tokens"] == 4
    assert quote["inflight_h2d_tokens"] == 4


def test_non_page_aligned_input_matches_partial_node_without_split(runtime):
    tree,nodes,_ = tree_chain((tuple(range(12)),{"host":True}),page_size=4)
    before = compact_state(tree)
    quote = runtime.peek_full_prefix(tree,StrictKey(array("q",range(10))))
    assert quote["raw_match_tokens"] == quote["ready_gpu_prefix_tokens"] == 8
    assert len(nodes[0].key) == 12
    assert compact_state(tree) == before


def test_partial_match_counts_only_matched_pending_tokens(runtime):
    tree,_,_ = tree_chain((tuple(range(12)),{"host":True,"h2d":1}),page_size=4)
    quote = runtime.peek_full_prefix(tree,StrictKey(array("q",range(6))))
    assert quote["raw_match_tokens"] == 4
    assert quote["inflight_h2d_tokens"] == 4
    assert quote["recoverable_prefix_tokens"] == 4
    assert quote["ready_gpu_prefix_tokens"] == 0


def test_unrelated_pending_branch_does_not_zero_ready_prefix(runtime):
    tree,_,key = tree_chain(((1,2,3,4),{}))
    other = node((9,10,11,12),host=True,h2d=99,node_id=99)
    other.parent = tree.root_node
    tree.root_node.children[9] = other
    before = compact_state(tree)
    assert runtime.peek_full_prefix(tree,key)["ready_gpu_prefix_tokens"] == 4
    assert compact_state(tree) == before


def test_quote_order_does_not_mutate_tree_or_keys(runtime):
    tree,_,a = tree_chain(((1,2,3,4),{}),((5,6,7,8),{"gpu":False,"host":True}))
    b = StrictKey(array("q",(1,2,3,4,9)))
    before = compact_state(tree)
    a_key,b_key = copy.deepcopy(a.__dict__),copy.deepcopy(b.__dict__)
    ab = [runtime.peek_full_prefix(tree,k) for k in (a,b)]
    ba = [runtime.peek_full_prefix(tree,k) for k in (b,a)]
    assert ab == ba[::-1]
    assert compact_state(tree) == before
    assert a.__dict__ == a_key and b.__dict__ == b_key


def native_source(name, *, patched=False):
    fixture = json.loads((ROOT / "tests/fixtures/sglang_quote_source.json").read_text())
    manifest = json.loads((ROOT / "engine_patches/source_manifest.json").read_text())
    assert fixture["commit"] == manifest["commit"]
    snippet = fixture["snippets"][name]
    assert snippet["original_file_sha256"] == manifest["original_sha256"][snippet["path"]]
    assert hashlib.sha256(snippet["source"].encode()).hexdigest() == snippet["source_sha256"]
    if not patched:
        return snippet["source"]
    first,last = snippet["start_line"],snippet["end_line"]
    lines = snippet["source"].splitlines(keepends=True)
    additions,deletions = {},set()
    active,old_line = False,None
    for line in (ROOT / "engine_patches/0001-contextkv-protocol.patch").read_text().splitlines(keepends=True):
        if line.startswith("diff --git "):
            active = line.rstrip().endswith(" b/"+snippet["path"])
            old_line = None
            continue
        if not active:
            continue
        if line.startswith("@@ "):
            old_line = int(re.match(r"@@ -(\d+)",line).group(1))
            continue
        if old_line is None:
            continue
        if line.startswith("+"):
            if first <= old_line <= last:
                additions.setdefault(old_line,[]).append(line[1:])
        elif line.startswith((" ","-")):
            if first <= old_line <= last:
                assert lines[old_line-first] == line[1:], "patch context differs from pinned fixture"
                if line.startswith("-"):
                    deletions.add(old_line)
            old_line += 1
    result = []
    for number,line in enumerate(lines,first):
        result.extend(additions.get(number,()))
        if number not in deletions:
            result.append(line)
    return "".join(result)


def extract_native_class(source, name, namespace=None):
    module = ast.parse(source)
    definition = next(node for node in module.body if isinstance(node,ast.ClassDef) and node.name == name)
    selected = ast.Module(body=[ast.ImportFrom(module="__future__",
                names=[ast.alias(name="annotations")],level=0),definition],type_ignores=[])
    ast.fix_missing_locations(selected)
    namespace = dict(namespace or {})
    exec(compile(selected,f"pinned-source:{name}","exec"),namespace)
    return namespace[name]


def extract_native_method(source, method_name, namespace=None):
    module = ast.parse(textwrap.dedent(source))
    method = next(node for node in module.body if isinstance(node,ast.FunctionDef) and node.name == method_name)
    method.decorator_list = []
    selected = ast.Module(body=[ast.ImportFrom(module="__future__",
                names=[ast.alias(name="annotations")],level=0),method],type_ignores=[])
    ast.fix_missing_locations(selected)
    namespace = dict(namespace or {})
    exec(compile(selected,f"pinned-source:{method_name}","exec"),namespace)
    return namespace[method_name]


def test_native_radix_key_array_contract_and_partial_page_quote(runtime):
    native_key = extract_native_class(native_source("RadixKey"),"RadixKey",{"array":array})
    native = native_key(array("q",range(12)))
    with pytest.raises(AssertionError):
        native.match(native_key(list(range(12))))
    tree,_,_ = tree_chain((tuple(range(12)),{"gpu":False,"host":True}),
                          page_size=4,key_cls=native_key)
    quote = runtime.peek_full_prefix(tree,native_key(array("q",range(10))))
    assert quote["recoverable_prefix_tokens"] == quote["host_only_load_tokens"] == 8


@pytest.mark.parametrize("write_back",[False,True])
def test_native_ack_clears_observation_markers_in_every_write_policy(write_back):
    finish = extract_native_method(native_source("finish_load_back",patched=True),"finish_load_back")
    tree,nodes,_ = tree_chain(((1,2),{}),((3,4),{"host":True,"h2d":2}))
    # Split fragments carry the same transfer owner, while an older ready
    # ancestor remains untouched by that owner's ACK.
    nodes[0].contextkv_h2d_pending_id = 2
    for item in nodes:
        item.load_back_pending_id = 2 if write_back else None
    tree.is_write_back = write_back
    tree.node_by_id = lambda _:nodes[-1]
    tree._update_duplicate_tracking = lambda _:None
    finish(tree,2)
    assert all(item.contextkv_h2d_pending_id is None for item in nodes)


@pytest.mark.parametrize("write_back",[False,True])
def test_native_load_publishes_observation_before_device_value_for_every_policy(write_back):
    commit = extract_native_method(native_source("commit_load_back",patched=True),"commit_load_back",
        {"BASE_COMPONENT_TYPE":0,"CacheTransferPhase":SimpleNamespace(LOAD_BACK="LOAD_BACK"),
         "StorageMedium":SimpleNamespace(GPU="GPU")})
    tree,nodes,_ = tree_chain(((1,2),{"gpu":False,"host":True}),
                              ((3,4),{"gpu":False,"host":True}))
    tree.is_write_back = write_back
    tree.node_by_id = lambda node_id:nodes[node_id-1]
    tree._update_evictable_leaf_sets = lambda _:None
    tree.kv_events = SimpleNamespace(record_store=lambda *args,**kwargs:None)
    def publish(*args,**kwargs):
        assert all(n.contextkv_h2d_pending_id == 2 for n in nodes)
        for n in nodes:
            n.component_data[0].value = n.component_data[0].host_value
    tree.components_by_type = {0:SimpleNamespace(commit_hicache_transfer=publish)}
    transfer = SimpleNamespace(nodes_to_load=[1,2],device_indices=None)
    commit(tree,2,(1,2,3,4),transfer,{})
    assert all(n.contextkv_h2d_pending_id == 2 for n in nodes)
    assert all(n.load_back_pending_id == (2 if write_back else None) for n in nodes)


def test_native_ack_does_not_clear_another_transfer_owner():
    finish = extract_native_method(native_source("finish_load_back",patched=True),"finish_load_back")
    tree,nodes,_ = tree_chain(((1,2),{"host":True,"h2d":99}),
                              ((3,4),{"host":True,"h2d":2}))
    tree.is_write_back = False
    tree.node_by_id = lambda _:nodes[-1]
    tree._update_duplicate_tracking = lambda _:None
    finish(tree,2)
    assert nodes[0].contextkv_h2d_pending_id == 99
    assert nodes[1].contextkv_h2d_pending_id is None


def test_native_split_inherits_observation_marker():
    module = ast.parse(textwrap.dedent(native_source("_split_node",patched=True)))
    split = next(n for n in ast.walk(module) if isinstance(n,ast.FunctionDef) and n.name == "_split_node")
    assignments = [n for n in ast.walk(split) if isinstance(n,ast.Assign) and
        any(isinstance(target,ast.Attribute) and target.attr == "contextkv_h2d_pending_id"
            for target in n.targets)]
    assert assignments, "split must inherit the all-policy H2D observation marker"
    new_node,child = SimpleNamespace(),SimpleNamespace(contextkv_h2d_pending_id=73)
    selected = ast.Module(body=assignments,type_ignores=[])
    exec(compile(ast.fix_missing_locations(selected),"pinned-source:_split_node","exec"),
         {"new_node":new_node,"child":child})
    assert new_node.contextkv_h2d_pending_id == 73


@pytest.mark.parametrize("threshold,expected",[(4,8),(9,0)])
@pytest.mark.parametrize("free_gpu", [0, 100])
@pytest.mark.parametrize("pending", [None, 7])
def test_runtime_quote_uses_native_array_limit_and_host_threshold(runtime,monkeypatch,threshold,expected,free_gpu,pending):
    native_key = extract_native_class(native_source("RadixKey"),"RadixKey",{"array":array})
    max_prefix = extract_native_method(native_source("_compute_max_prefix_len"),"_compute_max_prefix_len")
    for name in ("sglang","sglang.srt","sglang.srt.mem_cache","sglang.srt.managers"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules,name,module)
    radix_module,schedule_module = ModuleType("radix_cache"),ModuleType("schedule_batch")
    radix_module.RadixKey = native_key
    schedule_module.Req = type("Req",(),{"_compute_max_prefix_len":max_prefix})
    monkeypatch.setitem(sys.modules,"sglang.srt.mem_cache.radix_cache",radix_module)
    monkeypatch.setitem(sys.modules,"sglang.srt.managers.schedule_batch",schedule_module)
    tree,_,_ = tree_chain((tuple(range(8)),{"gpu":False,"host":True,"d2h":pending}),
                          page_size=4,key_cls=native_key)
    tree.peek_full_prefix = lambda key:runtime.peek_full_prefix(tree,key)
    cache = SimpleNamespace(tree_core=tree,page_size=4,_tree_core_backend="python",
        cache_controller=SimpleNamespace(mem_pool_host=SimpleNamespace(available_size=lambda:100)),
        token_to_kv_pool_allocator=SimpleNamespace(available_size=lambda:free_gpu,size=256),
        full_evictable_size=lambda:128,load_back_threshold=threshold)
    fingerprint = {"model_revision":"test"}
    bridge = runtime.ContextKVRuntime(cache,fingerprint)
    ids = list(range(10))
    before = compact_state(tree)
    quote = bridge.quote({"request_id":"q","fingerprint":fingerprint,"candidates":[
        {"candidate_id":"a","snapshot_version":0,"input_ids":ids,"input_hash":runtime.token_hash(ids)}]})
    row = quote["quotes"][0]
    assert row["match_limit"] == 8
    assert row["native_load_estimate"] == (expected if pending is None else 0)
    assert row["host_only_load_tokens"] == (8 if pending is None else 0)
    assert quote["free_host_tokens"] == 100 and quote["host_quote_supported"]
    assert quote["gpu_capacity_tokens"] == 256 and quote["evictable_gpu_tokens"] == 128
    assert quote["evictable_host_tokens"] is None
    assert row["uncertainty_reasons"] == ((["capacity_pressure"] if free_gpu == 0 else [])
                                        + (["inflight_transfer"] if pending else []))
    assert row["native_estimate_uncertain"] == (free_gpu == 0 or pending is not None)
    assert quote["native_running_requests"] is quote["native_waiting_requests"] is None
    from controller.schema import CacheQuoteBatch
    from controller.serialization import dumps, loads
    decoded = loads(CacheQuoteBatch, dumps(quote))
    assert decoded.gpu_capacity_tokens == 256 and decoded.evictable_host_tokens is None
    assert decoded.quotes[0].uncertainty_reasons == tuple(row["uncertainty_reasons"])
    assert loads(CacheQuoteBatch, dumps(decoded)) == decoded
    assert compact_state(tree) == before


def test_workload_snapshot_reuses_native_overlap_collection_without_mutation(runtime):
    native_kv = extract_native_class(native_source("ReqKvInfo"), "ReqKvInfo",
                                    {"dataclasses":dataclasses})
    native_finished = extract_native_method(native_source("finished"), "finished")
    class Request:
        def __init__(self, n, generated, maximum, *, done=False):
            self.origin_input_ids = array("q",range(n))
            self.output_ids = array("q",range(generated))
            self.sampling_params = SimpleNamespace(max_new_tokens=maximum,ignore_eos=False)
            self.kv = native_kv(kv_allocated_len=n+generated+4,
                               kv_committed_len=n+generated-1,cache_protected_len=n//2)
            self.finished_reason = "length" if done else None
        finished = native_finished
    # running + last + overlap results contain aliases, including a retracted
    # request already back in waiting. An unfinished chunk may be in neither.
    active, retracted, queued, chunked, done = (
        Request(20,3,10), Request(30,4,4), Request(40,0,12),
        Request(50,0,8), Request(60,10,10,done=True))
    collect_batches = extract_native_method(native_source("_collect_inflight_batches"),
        "_collect_inflight_batches", {"get_parallel":lambda:SimpleNamespace(pp_size=1)})
    collect = extract_native_method(native_source("collect_inflight_reqs"),"collect_inflight_reqs")
    class Scheduler:
        _collect_inflight_batches = collect_batches
        collect_inflight_reqs = collect
    scheduler = Scheduler()
    scheduler.running_batch = SimpleNamespace(reqs=[active,retracted,done])
    scheduler.last_batch = SimpleNamespace(reqs=[active])
    scheduler.enable_continuous_input_polling = True
    scheduler.result_queue = [(SimpleNamespace(reqs=[active,retracted]),object())]
    scheduler.chunked_req = chunked
    scheduler.waiting_queue = [queued,retracted,done]
    bridge = object.__new__(runtime.ContextKVRuntime)
    bridge.scheduler = scheduler
    state_before = copy.deepcopy([r.__dict__ for r in (active,retracted,queued,chunked,done)])
    result = bridge.workload_snapshot()
    assert result["active_requests"] == result["queued_requests"] == 2
    assert [r["input_tokens"] for r in result["native_waiting_requests"]] == [40,30]
    by_input = {r["input_tokens"]:r for r in result["native_running_requests"]}
    assert set(by_input) == {20,50}
    assert by_input[20] == dict(input_tokens=20,generated_tokens=3,max_new_tokens=10,
        ignore_eos=False,kv_allocated_tokens=27,kv_committed_tokens=22,cache_protected_tokens=10)
    from controller.schema import CacheQuoteBatch, NativeRequestSummary
    from controller.serialization import dumps, loads
    wire = dict(request_id="q",observed_seq=0,observed_time_ns=1,engine_restart_id="r",
                cache_config_hash="c",free_gpu_tokens=0,free_host_tokens=10,quotes=[],
                gpu_capacity_tokens=16384,evictable_gpu_tokens=128,evictable_host_tokens=None,
                **result)
    decoded = loads(CacheQuoteBatch,dumps(wire))
    assert isinstance(decoded.native_running_requests,tuple)
    assert all(isinstance(row,NativeRequestSummary) for row in decoded.native_running_requests)
    assert decoded.native_waiting_requests[1].generated_tokens == 4
    assert loads(CacheQuoteBatch,dumps(decoded)) == decoded
    assert [r.__dict__ for r in (active,retracted,queued,chunked,done)] == state_before
    assert scheduler.waiting_queue == [queued,retracted,done]
    assert scheduler.running_batch.reqs == [active,retracted,done]
    scheduler.waiting_queue = []
    scheduler.running_batch.reqs = []
    scheduler.last_batch = None
    scheduler.result_queue = []
    scheduler.chunked_req = done
    assert bridge.workload_snapshot() == dict(active_requests=0,queued_requests=0,
        native_running_requests=[],native_waiting_requests=[])


def test_native_gpu_evictable_counter_is_constant_time_and_not_host_capacity():
    evictable = extract_native_method(native_source("evictable_size"), "evictable_size",
                                      {"BASE_COMPONENT_TYPE":0})
    full = extract_native_method(native_source("full_evictable_size"), "full_evictable_size")
    class Tree:
        component_evictable_size_ = {0:128}
        evictable_size = evictable
        full_evictable_size = full
        @property
        def root_node(self):
            raise AssertionError("counter access must not walk the tree")
    assert Tree().full_evictable_size() == 128
