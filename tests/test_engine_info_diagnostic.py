"""Real runtime wire through the HTTP client, including the opt-in capability."""
import asyncio
from dataclasses import asdict

import httpx
import pytest

from controller.engine_client import EngineInfo, SGLangEngineClient
from controller.schema import ConfigFingerprint
from controller.serialization import dumps, loads
from engine_patches.contextkv_runtime import ContextKVRuntime
from test_engine import FakeNativeCache


@pytest.mark.parametrize("profiling,diagnostics", [(False,False),(False,True),(True,False),(True,True)])
def test_real_info_wire_preserves_legacy_omission_and_decodes_enabled_capability(
        monkeypatch,profiling,diagnostics):
    monkeypatch.setenv("CONTEXTKV_ENABLE_PROFILING",str(int(profiling)))
    monkeypatch.setenv("CONTEXTKV_ADMISSION_DIAGNOSTICS",str(int(diagnostics)))
    fp=ConfigFingerprint()
    owner=ContextKVRuntime(FakeNativeCache(),asdict(fp))
    wire=owner.info({})
    enabled=profiling and diagnostics
    assert ("admission_diagnostics_enabled" in wire) is enabled
    calls=[]
    def transport(request):
        calls.append(request.url.path)
        assert request.url.path=="/context/engine/info"
        return httpx.Response(200,json=wire)
    async def scenario():
        client=SGLangEngineClient("http://test",fp,transport=httpx.MockTransport(transport))
        try:
            info=await client.info()
            assert info.admission_diagnostics_enabled is enabled
            assert info.fingerprint==fp and info.engine_restart_id==owner.restart_id
            assert loads(EngineInfo,dumps(info))==info
        finally:
            await client.aclose()
    asyncio.run(scenario())
    assert calls==["/context/engine/info"]
