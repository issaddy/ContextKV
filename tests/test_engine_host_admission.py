"""CPU-only host-admission lifecycle, native action-order and wire contracts."""
import asyncio
import ast
from dataclasses import asdict, replace
import hashlib
import json
from types import SimpleNamespace
import textwrap

import httpx
import pytest

from controller.engine_client import SGLangEngineClient
from bench.mock_engine import MockEngine
from controller.planner import PlannerKnobs
from controller.schema import ConfigFingerprint
from controller.service import Controller
from controller.session_store import SessionStore
from engine_patches.contextkv_runtime import ContextKVRuntime, ContextKVError, HostAdmissionPolicy
from test_engine import FakeNativeCache
from test_engine_host_quote import extract_native_method, native_source


def runtime(monkeypatch, mode="conditioned"):
    monkeypatch.setenv("CONTEXTKV_HOST_ADMISSION", mode)
    monkeypatch.delenv("CONTEXTKV_SAMPLE_POOL",raising=False)
    cache=FakeNativeCache()
    cache.page_size=4
    cache.cache_controller=SimpleNamespace(write_policy="write_through",io_backend="kernel",
        mem_pool_host=SimpleNamespace(size=100,available_size=lambda:2,layout="layer_first"))
    cache.host_memory_mode="cache"
    cache.token_to_kv_pool_allocator=SimpleNamespace(size=100,available_size=lambda:100)
    owner=ContextKVRuntime(cache,asdict(ConfigFingerprint()))
    return owner


def prepare(owner,sid,business=None):
    value=dict(engine_session_id=sid,operation_id='p-'+sid,expected_restart_id=owner.restart_id)
    if business is not None:
        value['business_session_id']=business
    return owner.prepare(value)


def request(sid,rid,n=80):
    return SimpleNamespace(session_id=sid,rid=rid,origin_input_ids=[1]*n,
        prefix_len=0,host_hit_length=0,num_matched_prefix_tokens=0)


def test_predicate_pressure_requires_working_set_occupancy_and_eviction():
    policy=HostAdmissionPolicy('conditioned',threshold_tokens=8,window_seconds=60,prompt_window=2)
    policy.observe('a','business-a',200,0)
    policy.observe('b','business-b',200,0)
    assert policy.decide(12,4,100,99,1)['allowed']  # no completed allocation eviction observed
    policy.capacity_evicted(4,1)
    assert policy.decide(8,4,100,99,2)['allowed']  # exact threshold retained
    assert policy.decide(12,4,100,50,2)['allowed']
    assert not policy.decide(12,4,100,99,2)['allowed']
    assert policy.decide(12,4,100,99,62)['allowed']  # activity/eviction window expired
    assert policy.snapshot(62)['active_business_sessions']==0


def test_epoch_dedup_retirement_and_actual_occupancy_are_separate():
    policy=HostAdmissionPolicy('conditioned',threshold_tokens=8)
    policy.observe('old','business-a',200,0)
    policy.observe('new','business-a',200,1)
    policy.capacity_evicted(4,1)
    assert policy.window(2)['active_engine_epochs']==2
    assert policy.window(2)['active_business_sessions']==1
    assert policy.decide(12,4,100,99,2)['allowed']
    policy.observe('other','business-b',200,2)
    assert not policy.decide(12,4,100,99,2)['allowed']
    policy.retire('old')
    assert policy.window(2)['active_engine_epochs']==2
    policy.retire('new')
    decision=policy.decide(12,4,100,99,3)
    assert decision['host_used_tokens']==99 and decision['active_business_sessions']==1
    assert decision['allowed']


def test_fixed_predicate_and_bounded_prompt_window():
    policy=HostAdmissionPolicy('fixed',threshold_tokens=8192,prompt_window=2)
    for index in range(3):
        policy.observe(str(index),str(index),100+index,index)
    assert len(policy.prompts)==2
    assert policy.decide(8192,64,1,0,3)['allowed']
    assert not policy.decide(8256,64,1,0,3)['allowed']
    assert HostAdmissionPolicy().decide(9999,64,1,1,3)['allowed']


def test_runtime_business_binding_is_optional_immutable_and_first_match_once(monkeypatch):
    owner=runtime(monkeypatch)
    a=prepare(owner,'a','business')
    assert a['business_session_id']=='business'
    assert prepare(owner,'a','business')==a
    assert prepare(owner,'a')==a  # old wire never clears an existing identity
    with pytest.raises(ContextKVError,match='EPOCH_CONFLICT'):
        prepare(owner,'a','other-business')
    old=prepare(owner,'unknown')
    assert 'business_session_id' not in old
    assert owner.allow_host_backup(request('unknown','u'),99999)
    req=request('a','r')
    owner.record_native_match(req,80)
    owner.record_native_match(req,80)  # retraction/requeue refreshes match, not the prompt sample
    assert owner.host_admission.counters['first_match_observations']==1
    owner.retire(dict(engine_session_id='a',expected_generation=a['session_generation'],
                      operation_id='retire',expected_restart_id=owner.restart_id))
    assert owner.host_admission.window(0)['active_business_sessions']==0
    assert owner.cache.cache_controller.mem_pool_host.available_size()==2


def test_default_off_cache_identity_and_wire_remain_legacy(monkeypatch):
    owner=runtime(monkeypatch,'off')
    assert owner.cache_config == dict(page_size=4,backend='python',hicache=True,protocol='contextkv-v1',
        write_policy='write_through',host_memory_mode='cache',hicache_io_backend='kernel',host_layout='layer_first',
        tma_transfer_enabled=None,gpu_capacity_tokens=100,host_capacity_tokens=100,load_back_threshold=None,sample_pool=False)
    expected=hashlib.sha256(json.dumps(owner.cache_config,sort_keys=True).encode()).hexdigest()
    assert owner.cache_config_hash==expected
    assert 'business_session_identity_supported' not in owner.info({})
    assert owner.host_admission.snapshot(0)['config']['mode']=='off'
    assert 'business_session_id' not in prepare(owner,'legacy')
    active=runtime(monkeypatch,'fixed')
    assert active.cache_config['host_admission']['threshold_tokens']==8192
    assert active.cache_config_hash!=owner.cache_config_hash


class Values(tuple):
    def numel(self):
        return len(self)


class Backup:
    def __init__(self,ids):
        self.node_ids=ids


class ComponentAction:
    pass


class Split(ComponentAction):
    pass


class Free:
    def __init__(self,values):
        self.indices=[values]


class FullFree(Free):
    pass


def native_cache(owner, *, policy='write_through', values=None):
    calls=[]
    backed=set()
    specs=values or {1:Values((1,2,3,4)),2:Values((5,6,7,8))}
    def build(node):
        calls.append(('spec',node))
        return Values(()) if node in backed else specs[node],{}
    def commit(node,host,components):
        calls.append(('commit',node))
        backed.add(node)
    cache=SimpleNamespace(buffer_pipeline=None,linker=None,contextkv_runtime=owner,
        cache_controller=SimpleNamespace(write_policy=policy),tree_core=SimpleNamespace(build_backup_spec=build,commit_backup=commit),
        _build_backup_sidecar=lambda *a:[],_backup_publish_node_ids=lambda node,xfers:[node],
        _track_write_through_node=lambda node,*a,**k:calls.append(('pending',node)),
        inc_lock_ref=lambda node:SimpleNamespace(to_dec_params=lambda:None),
        token_to_kv_pool_allocator=SimpleNamespace(free_segment=lambda indices,start_pos:calls.append(('free',tuple(indices)))))
    def execute(node,*args):
        calls.append(('write',node))
        return specs[node]
    cache._execute_kv_backup=execute
    namespace={'BackupKV':Backup,'ComponentAction':ComponentAction,'UpdatePendingBackupOnNodeSplit':Split,
               'FreeDeviceKV':Free,'FreeDeviceKVFullOnly':FullFree}
    for name in ('_execute_and_commit_kv_backup','_apply_cache_actions','_apply_cache_action'):
        fn=extract_native_method(native_source(name,patched=True),name,namespace)
        setattr(cache,name,fn.__get__(cache))
    return cache,calls,backed


def test_native_whole_action_skip_preserves_other_actions_then_allows_shared_ancestors(monkeypatch):
    owner=runtime(monkeypatch,'fixed')
    owner.host_admission.threshold_tokens=4
    prepare(owner,'s','business')
    cache,calls,backed=native_cache(owner)
    req=request('s','r')
    actions=[Free((10,)),Backup([1,2]),Free((11,))]
    cache._apply_cache_actions(actions,contextkv_request=req)
    assert not actions and calls==[('free',(10,)),('spec',1),('spec',2),('free',(11,))]
    assert not backed
    assert owner.host_admission.counters['skipped_new_tokens']==8
    # No node is permanently blacklisted; a later allowed action backs the
    # same complete ancestor chain with the native commit/ACK ordering.
    owner.host_admission.threshold_tokens=8
    calls.clear()
    cache._apply_cache_actions([Backup([1,2])],contextkv_request=req)
    assert backed=={1,2}
    assert calls==[('spec',1),('spec',2),('write',1),('commit',1),('pending',1),('write',2),('commit',2),('pending',2)]
    calls.clear()
    cache._apply_cache_actions([Backup([1,2])],contextkv_request=req)
    assert calls==[('spec',1),('spec',2)]  # native existing-copy dedup preserved


@pytest.mark.parametrize('mode,write_back,identity', [('off',False,True),('fixed',True,True),('fixed',False,False)])
def test_native_default_write_back_and_unknown_identity_keep_original_writes(monkeypatch,mode,write_back,identity):
    owner=runtime(monkeypatch,mode)
    owner.host_admission.threshold_tokens=0
    prepare(owner,'s','business' if identity else None)
    cache,calls,backed=native_cache(owner,policy='write_back' if write_back else 'write_through')
    cache._execute_and_commit_kv_backup(Backup([1,2]),write_back=write_back,contextkv_request=request('s','r'))
    assert backed=={1,2}
    assert [x for x in calls if x[0]=='write']==[('write',1),('write',2)]
    assert owner.host_admission.counters['skipped_actions']==0


def test_source_checkpoint_threads_identity_through_both_inserts_and_finally():
    source=textwrap.dedent(native_source('checkpoint_into_tree',patched=True))
    calls=[n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='insert']
    assert len(calls)==2 and all(any(k.arg=='contextkv_request' and isinstance(k.value,ast.Name) and k.value.id=='req' for k in c.keywords) for c in calls)
    insert=textwrap.dedent(native_source('insert',patched=True))
    calls=[n for n in ast.walk(ast.parse(insert)) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='_apply_cache_actions']
    assert len(calls)==2 and all(any(k.arg=='contextkv_request' for k in c.keywords) for c in calls)


def test_http_prepare_business_identity_and_old_wire(monkeypatch):
    async def scenario():
        owner=runtime(monkeypatch,'fixed')
        def handler(request):
            if request.url.path.endswith('/info'):
                return httpx.Response(200,json=owner.info({}))
            body=json.loads(request.content)
            return httpx.Response(200,json=owner.prepare(body))
        client=SGLangEngineClient('http://test',ConfigFingerprint(),transport=httpx.MockTransport(handler))
        ack=await client.prepare('s','op',owner.restart_id,business_session_id='business')
        assert ack.business_session_id=='business'
        old=await client.prepare('old','old-op',owner.restart_id)
        assert old.business_session_id is None
        await client.aclose()
    asyncio.run(scenario())


@pytest.mark.asyncio
async def test_controller_rebuild_epochs_share_explicit_business_identity(tmp_path):
    engine=MockEngine()
    service=Controller(SessionStore(tmp_path/'sessions.sqlite3'),engine,knobs=PlannerKnobs(strategy='always_rebuild'))
    await service.serve_turn('business','first',())
    await service.serve_turn('business','second',())
    assert len(engine._epochs)==2
    assert {epoch.business_session_id for epoch in engine._epochs.values()}=={'business'}
    await service.aclose()


@pytest.mark.asyncio
async def test_controller_old_engine_without_identity_capability_keeps_legacy_prepare(tmp_path):
    class OldWire(MockEngine):
        async def info(self):
            return replace(await super().info(),business_session_identity_supported=False)
        async def prepare(self,sid,op,restart):
            return await super().prepare(sid,op,restart)
    engine=OldWire()
    service=Controller(SessionStore(tmp_path/'sessions.sqlite3'),engine)
    await service.serve_turn('business','query',())
    assert all(epoch.business_session_id is None for epoch in engine._epochs.values())
    await service.aclose()



def test_allocation_eviction_observation_preserves_native_failure_and_write_order(monkeypatch):
    owner=runtime(monkeypatch,'conditioned')
    calls=[]
    host=SimpleNamespace(anchor_entry=SimpleNamespace(host_pool=SimpleNamespace(shared_allocation_domain=None),host_evict_fn=None),
                         available_size=lambda:0)
    cache=SimpleNamespace(contextkv_runtime=owner,cache_controller=SimpleNamespace(mem_pool_host=host,
        write=lambda *a,**k:(calls.append(('write',k)) or Values((1,2,3,4)))),
        evict_host=lambda n:(calls.append(('evict',n)) or 2))
    function=extract_native_method(native_source('_execute_kv_backup',patched=True),'_execute_kv_backup')
    assert function(cache,1,Values((1,2,3,4)),{},[]) is None
    assert calls==[('evict',4)]
    assert owner.host_admission.counters['allocation_evicted_host_tokens']==2
    cache.evict_host=lambda n:(calls.append(('evict',n)) or 4)
    assert len(function(cache,1,Values((1,2,3,4)),{},[]))==4
    assert calls[-1]==('write',{'node_id':1,'extra_pools':None,'flush':False})
    assert owner.host_admission.counters['allocation_evicted_host_tokens']==6


def test_retiring_inflight_failed_request_keeps_work_until_native_terminal(monkeypatch):
    owner=runtime(monkeypatch,'fixed')
    ack=prepare(owner,'s','business')
    req=request('s','r')
    receive=SimpleNamespace(session_id='s',expected_restart_id=owner.restart_id,
        expected_generation=ack['session_generation'],contextkv_fingerprint=owner.fingerprint,
        contextkv_attempt_id='r',rid='r',session_params=None,lora_id=None,mm_inputs=None,input_embeds=None,
        return_logprob=False,cache_salt=None,extra_key=None,input_ids=req.origin_input_ids)
    owner.admit(receive,req)
    owner.record_native_match(req,80)
    result=owner.retire(dict(engine_session_id='s',expected_generation=ack['session_generation'],
                            operation_id='retire',expected_restart_id=owner.restart_id))
    assert not result['acknowledged'] and owner.host_admission.window(0)['active_business_sessions']==1
    req.finished=lambda:True
    req.finished_reason=SimpleNamespace(to_json=lambda:{'type':'abort'})
    req.session_generation=ack['session_generation']
    req.cached_tokens_device=req.cached_tokens_host=req.cached_tokens=0
    req.kv=SimpleNamespace(cache_inserted_len=0)
    req.customized_info=None
    req.output_ids=[]
    owner.finish(req)
    assert owner.epochs['s']['state']=='RETIRED'
    assert owner.host_admission.window(0)['active_business_sessions']==0
    assert owner.status({'attempt_id':'r'})['state']=='FAILED'
    assert owner.cache.cache_controller.mem_pool_host.available_size()==2



def test_incremental_budget_handles_native_chunks_requeue_and_other_request(monkeypatch):
    owner=runtime(monkeypatch,'fixed')
    owner.cache.page_size=64
    prepare(owner,'s','business')
    cache,calls,backed=native_cache(owner,values={1:Values(range(8192)),2:Values(range(4096))})
    req=request('s','r')
    assert cache._execute_and_commit_kv_backup(Backup([1]),contextkv_request=req)==8192
    assert req.contextkv_host_backup_admitted_pages==128
    assert cache._execute_and_commit_kv_backup(Backup([1,2]),contextkv_request=req)==0
    assert backed=={1} and req.contextkv_host_backup_admitted_pages==128
    # Requeue/repeated checkpoint exposes zero new spec pages; no second charge.
    assert cache._execute_and_commit_kv_backup(Backup([1]),contextkv_request=req)==0
    assert req.contextkv_host_backup_admitted_pages==128
    other=request('s','another-native-request')
    assert cache._execute_and_commit_kv_backup(Backup([1,2]),contextkv_request=other)==4096
    assert backed=={1,2} and other.contextkv_host_backup_admitted_pages==64
    assert owner.host_admission.counters['admitted_new_pages']==192
    assert owner.host_admission.counters['skipped_new_tokens']==4096


def test_native_allocation_failure_does_not_spend_incremental_budget(monkeypatch):
    owner=runtime(monkeypatch,'fixed')
    prepare(owner,'s','business')
    cache,_,backed=native_cache(owner)
    cache._execute_kv_backup=lambda *args:None
    req=request('s','r')
    assert cache._execute_and_commit_kv_backup(Backup([1,2]),contextkv_request=req)==0
    assert not backed and getattr(req,'contextkv_host_backup_admitted_pages',0)==0
    assert owner.host_admission.counters['admitted_new_tokens']==0
    assert owner.host_admission.counters['allowed_planned_new_tokens']==8


def test_partial_chain_allocation_failure_spends_only_admitted_ancestor_pages(monkeypatch):
    owner=runtime(monkeypatch,'fixed')
    prepare(owner,'s','business')
    cache,_,backed=native_cache(owner)
    execute=cache._execute_kv_backup
    cache._execute_kv_backup=lambda node,*args:execute(node,*args) if node==1 else None
    req=request('s','r')
    assert cache._execute_and_commit_kv_backup(Backup([1,2]),contextkv_request=req)==0
    assert backed=={1} and req.contextkv_host_backup_admitted_pages==1
    assert owner.host_admission.counters['admitted_new_tokens']==4


def test_no_pressure_does_not_limit_a_request_already_over_budget():
    policy=HostAdmissionPolicy('conditioned')
    policy.observe('one','business',12000,0)
    result=policy.decide(4096,64,1000,999,1,admitted_pages=200)
    assert result['allowed'] and not result['pressure']
    assert result['projected_request_pages']==264 and result['reason']=='pressure_not_met'
