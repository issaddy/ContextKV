"""No-GPU contracts for the bounded three-request observation probe."""
import asyncio
from dataclasses import replace

import pytest

from controller.engine_client import EngineError, EngineInfo, EpochStatus, PrepareAck, RetireAck
from controller.schema import CacheQuoteBatch, CompletionReceipt, ConfigFingerprint, GenerationResult, NativePrefixMatch, NativeRequestSummary
from profiling.observation_probe import ObservationProbe, verify_snapshot


def snapshot(*,waiting=(),running=()):
    return CacheQuoteBatch("q",0,10000,"r","c",0,0,(),native_new_token_ratio=.5,
        native_max_new_tokens_clip=4096,native_running_requests=running,native_waiting_requests=waiting,
        native_running_budget_requests=running,native_running_budget_tokens=sum(
            min(r.max_new_tokens-r.generated_tokens,4096)*(1 if r.ignore_eos else .5) for r in running))


def test_probe_rejects_bad_formula_clock_and_missing_contract():
    request=NativeRequestSummary(100,1,10,101,100,32,last_native_match=NativePrefixMatch(9000,100,32,0,32))
    batch=snapshot(running=(request,))
    assert verify_snapshot(batch)["formula_matches"]
    assert verify_snapshot(batch)["maximum_match_age_ms"]==.001
    with pytest.raises(ValueError,match="offset"):
        verify_snapshot(replace(batch,native_running_budget_tokens=5))
    with pytest.raises(ValueError,match="timestamps"):
        verify_snapshot(replace(batch,observed_time_ns=8000))
    with pytest.raises(ValueError,match="lacks"):
        verify_snapshot(replace(batch,native_new_token_ratio=None))


class Engine:
    fingerprint=ConfigFingerprint()
    def __init__(self,unknown_waiting):
        self.unknown_waiting=unknown_waiting
        self.epochs={}
        self.jobs=[]
        self.rejected=set()
        self.quote_seq=0
    async def info(self):
        return EngineInfo("r",engine_kind="sglang",profiling_enabled=True)
    async def metrics(self):
        return {"pool":{"gpu_capacity_tokens":16384,"host_capacity_tokens":279040,"page_size":64}}
    async def prepare(self,sid,operation,restart):
        self.epochs[sid]="PREPARED"
        return PrepareAck(sid,1,restart,operation)
    async def generate(self,**kwargs):
        sid=kwargs["session_id"]
        if self.epochs[sid]=="RETIRED":
            self.rejected.add(kwargs["attempt_id"])
            raise EngineError(409,"EPOCH_RETIRED")
        event=asyncio.Event()
        self.jobs.append((sid,event,kwargs))
        await event.wait()
        receipt=CompletionReceipt("rid",kwargs["attempt_id"],sid,1,"r","input","length",True,
                                  session_ref_registered=True)
        return GenerationResult("fake",tuple(range(kwargs["max_new_tokens"])),"length",receipt)
    async def retire(self,sid,generation,operation,restart,reason):
        self.epochs[sid]="RETIRED"
        return RetireAck(sid,generation,restart,operation)
    async def status(self,*,engine_session_id=None,attempt_id=None,expected_restart_id=None):
        if attempt_id in self.rejected:
            return EpochStatus("REJECTED","r",terminal=True)
        return EpochStatus(self.epochs[engine_session_id],"r",engine_session_id=engine_session_id,terminal=True)
    async def quote_batch(self,candidates,fingerprint):
        assert candidates==[]  # polling must not send synthetic input IDs again
        self.quote_seq+=1
        active=[j for j in self.jobs if self.epochs[j[0]]!="RETIRED"]
        match=NativePrefixMatch(9000,9216,1024,0,1024)
        summary=NativeRequestSummary(9216,1,2,9217,9216,1024,ignore_eos=True,last_native_match=match)
        running=(summary,) if active else ()
        waiting=tuple(replace(summary,generated_tokens=0,last_native_match=None if self.unknown_waiting and i==1 else match)
                      for i in range(max(0,len(active)-1)))
        if len(active)==3:
            for _,event,_ in active:
                event.set()
        return replace(snapshot(running=running,waiting=waiting),observed_seq=self.quote_seq,
                       observed_time_ns=10000+self.quote_seq)


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown_waiting",[True,False])
async def test_probe_preserves_lifecycle_and_separates_missing_coverage(unknown_waiting):
    engine=Engine(unknown_waiting)
    probe=ObservationProbe(engine,(1,2,3,4),poll_seconds=0)
    async def prepare(case):
        probe.profiler._inputs=tuple((leader,)*case.prompt_tokens for leader in (1,2,3))
    probe.profiler.prepare=prepare
    async def settle(mode):
        assert mode=="settle"
    probe.profiler._fixture=settle
    result=await probe.run(output_tokens=2)
    assert result["status"]=="passed" and result["contract_checks_passed"]
    assert result["coverage_status"]==("complete" if unknown_waiting else "partial")
    assert result["coverage"]["one_running_two_waiting"] and result["coverage"]["known_waiting_match"]
    assert result["stale_epoch_rejected"] and len(result["results"])==3
    assert all(r["retired"] for r in result["results"])
    assert set(engine.epochs.values())=={"RETIRED"}
