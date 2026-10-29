"""Local snapshot/artifact identity and client/bridge startup contracts, without CUDA."""
from __future__ import annotations

import asyncio
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from controller.config import CostSettings, EngineConfig, ModelConfig, RunConfig, create_controller, create_model_template, load_config
from controller.cost_model import CostModel
from controller.engine_client import EngineError, EngineInfo, SGLangEngineClient
from controller.schema import ConfigFingerprint
from controller.serialization import dumps, loads
from engine_patches import contextkv_model_scope as scopes
from engine_patches import contextkv_runtime as native
from test_engine import FakeNativeCache

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def local_scope(tmp_path):
    record = json.loads((ROOT/"configs/model_scopes/qwen3-32b.local.json").read_text())
    directory = tmp_path/"local model"
    directory.mkdir()
    record.update(model_path=str(directory), revision="cpu-qwen-fixture")
    tokenizer = SimpleNamespace(chat_template="CPU template", split_special_tokens=False,
                                backend_tokenizer=SimpleNamespace(to_str=lambda: '{"fixture":"CPU tokenizer"}'))
    record["tokenizer_hash"] = hashlib.sha256(b'{"fixture":"CPU tokenizer"}').hexdigest()
    record["template_hash"] = hashlib.sha256(json.dumps({"template": tokenizer.chat_template,
                                                        "enable_thinking": False}, sort_keys=True).encode()).hexdigest()
    (directory/"config.json").write_text(json.dumps(record["config"]))
    (directory/"model.safetensors").write_bytes(b"fake")
    (directory/"model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": 4}, "weight_map": {"weight": "model.safetensors"}}))
    path = tmp_path/"scope.json"
    path.write_text(json.dumps(record))
    scope = scopes.load_model_scope(path)
    fingerprint = ConfigFingerprint(model_revision=scope.identity, tokenizer_hash=record["tokenizer_hash"],
                                    template_hash=record["template_hash"])
    return SimpleNamespace(path=path, scope=scope, record=record, model_path=directory,
                           tokenizer=tokenizer, fingerprint=fingerprint)


def test_actual_qwen3_32b_record_has_explicit_head_dimension():
    large = scopes.load_model_scope(ROOT/"configs/model_scopes/qwen3-32b.local.json")
    record = large.manifest()["record"]
    assert large.identity.startswith("local-snapshot:") and record["hf_revision"] is None
    assert record["config"]["head_dim"] == 128
    assert large.context_limit == 40960
    config = large.manifest()["record"]["config"]
    assert config["hidden_size"] // config["num_attention_heads"] == 80
    assert large.identity == "local-snapshot:7442623e99dd0b456628ddc48805a4ea11249ee2ee66484b5d1ab44d2940beb8"


def test_cpu_default_has_no_model_and_generic_runtime_cache_identity_stays_unchanged(local_scope, monkeypatch):
    monkeypatch.delenv("CONTEXTKV_HOST_ADMISSION", raising=False)
    model = ModelConfig()
    assert model.name is None and model.revision is None
    assert model.local_files_only is True and model.scope is None
    old = native.ContextKVRuntime(FakeNativeCache(), asdict(ConfigFingerprint()))
    selected = native.ContextKVRuntime(FakeNativeCache(), asdict(local_scope.fingerprint), model_scope=local_scope.scope.manifest())
    assert "model_scope" not in old.info({})
    assert old.cache_config == selected.cache_config and old.cache_config_hash == selected.cache_config_hash


def test_missing_scope_never_loads_a_real_model(monkeypatch):
    monkeypatch.setattr("controller.config.HuggingFaceTemplate", lambda *a, **kw: pytest.fail("model loading"))
    with pytest.raises(ValueError, match="scope_path"):
        RunConfig(engine=EngineConfig(kind="sglang"))
    with pytest.raises(ValueError, match="scope_path"):
        create_model_template(ModelConfig(name="Qwen/Qwen2.5-7B-Instruct", revision="a" * 40))


def test_scope_config_resolves_once_without_faking_hf_revision(local_scope, monkeypatch):
    calls = []
    original = scopes.load_model_scope
    monkeypatch.setattr(scopes, "load_model_scope", lambda path: (calls.append(path) or original(path)))
    model = ModelConfig(scope_path=str(local_scope.path))
    assert len(calls) == 1 and model.scope is model.scope
    assert model.name == str(local_scope.model_path) and model.revision == local_scope.scope.identity
    assert model.local_files_only is True
    config = RunConfig(engine=EngineConfig(kind="sglang"), model=model)
    assert config.model.revision == local_scope.scope.identity
    assert "_model_scope" not in asdict(config)["model"]
    assert loads(RunConfig, dumps(config)).model.revision == model.revision
    with pytest.raises(ValueError, match="context budget"):
        replace(config, context=replace(config.context, max_total_model_tokens=50000))


@pytest.mark.parametrize("extra", [{"name": "/other/model"}, {"revision": "a"*40}, {"local_files_only": False}])
def test_scope_conflicting_config_is_not_silently_overridden(local_scope, extra):
    with pytest.raises(ValueError, match="conflict"):
        ModelConfig(scope_path=str(local_scope.path), **extra)


def test_real_hf_template_adapter_checks_scope_without_a_second_tokenizer_serialization(local_scope, monkeypatch):
    from transformers import AutoTokenizer
    calls, serializations = [], []
    local_scope.tokenizer.backend_tokenizer.to_str = lambda: (serializations.append(1) or '{"fixture":"CPU tokenizer"}')
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda path, **kwargs:
                        (calls.append((path, kwargs)) or local_scope.tokenizer))
    model = ModelConfig(scope_path=str(local_scope.path))
    template = create_model_template(model)
    assert template.fingerprint == local_scope.fingerprint
    assert serializations == [1]
    assert calls == [(str(local_scope.model_path), {"local_files_only": True})]
    with pytest.raises(ValueError, match="path differs"):
        create_model_template(model, model_path=str(local_scope.path.parent))


def test_model_configuration_and_tokenizer_match_the_declared_revision(local_scope):
    local_scope.scope.verify_files()
    local_scope.scope.verify_tokenizer(local_scope.tokenizer)
    local_scope.tokenizer.chat_template += " changed"
    with pytest.raises(ValueError, match="tokenizer/nonthinking"):
        local_scope.scope.verify_tokenizer(local_scope.tokenizer)
    (local_scope.model_path/"config.json").write_text('{}')
    with pytest.raises(ValueError, match="config differs"):
        local_scope.scope.verify_files()


def install_native_environment(monkeypatch, fixture):
    class Unified(FakeNativeCache):
        enable_session_radix_cache = True
        _components_tuple = (object(),)
        tree_core = SimpleNamespace(is_eagle=False)
        enable_storage = False
        linker = None
        buffer_pipeline = None
    cache_module = ModuleType("sglang.srt.mem_cache.unified_radix_cache")
    cache_module.UnifiedRadixCache = Unified
    context = ModuleType("sglang.srt.runtime_context")
    settings = SimpleNamespace(model_path=str(fixture.model_path), tokenizer_path=str(fixture.model_path), revision=None,
        dtype="bfloat16", kv_cache_dtype="bfloat16", quantization=None, load_format="auto",
        json_model_override_args="{}", context_length=32768)
    context.get_model = lambda: settings
    context.get_server_args = lambda: settings
    context.get_parallel = lambda: SimpleNamespace(tp_size=1, num_dp_ranks=1, pp_size=1)
    environment = ModuleType("sglang.srt.environ")
    environment.envs = SimpleNamespace(SGLANG_HICACHE_TMA_TRANSFER=SimpleNamespace(get=lambda: False))
    for module in (cache_module, context, environment):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.contextkv_model_scope", scopes)
    monkeypatch.setenv("CONTEXTKV_MODEL_SCOPE_PATH", str(fixture.path))
    monkeypatch.setenv("CONTEXTKV_FINGERPRINT_JSON", json.dumps(asdict(fixture.fingerprint)))
    scheduler = SimpleNamespace(tree_cache=Unified(), tokenizer=fixture.tokenizer,
        dllm_config=None, rust_server=None, spec_algorithm=SimpleNamespace(is_none=lambda: True),
        enable_pdmux=False, disaggregation_mode=SimpleNamespace(value="null"))
    return scheduler, settings


def test_native_scope_guard_real_http_info_and_initialization_once(local_scope, monkeypatch):
    scheduler, _ = install_native_environment(monkeypatch, local_scope)
    owner = native.get_runtime(scheduler)
    assert owner.info({})["model_scope"] == local_scope.scope.manifest()
    # Changes after initialization do not cause per-request file reads.
    monkeypatch.setattr(scopes, "load_model_scope", lambda _: (_ for _ in ()).throw(AssertionError("reloaded")))
    assert native.get_runtime(scheduler) is owner
    async def scenario():
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=owner.info({})))
        client = SGLangEngineClient("http://test", local_scope.fingerprint, transport=transport,
                                   model_scope=local_scope.scope.manifest())
        try:
            info = await client.info()
            assert info.model_scope["identity"] == local_scope.scope.identity
            assert loads(EngineInfo, dumps(info)) == info
        finally:
            await client.aclose()
    asyncio.run(scenario())


@pytest.mark.parametrize("other_path", [False, True])
def test_http_info_compares_canonical_model_path_not_scope_file_location(local_scope, other_path):
    expected = local_scope.scope.manifest()
    actual = local_scope.scope.manifest()
    actual["source_path"] = "/different/process/scope-record.json"
    actual["record"]["source"] = {"note": "same model, different inspection annotation"}
    if other_path:
        actual["record"]["model_path"] += "-another-model-directory"
    owner = native.ContextKVRuntime(FakeNativeCache(), asdict(local_scope.fingerprint), model_scope=actual)
    async def scenario():
        client = SGLangEngineClient("http://test", local_scope.fingerprint, model_scope=expected,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=owner.info({}))))
        try:
            if other_path:
                with pytest.raises(EngineError) as caught:
                    await client.info()
                assert caught.value.code == "MODEL_SCOPE_MISMATCH"
            else:
                assert (await client.info()).model_scope == actual
        finally:
            await client.aclose()
    asyncio.run(scenario())


def test_scope_model_path_symlink_is_canonical(local_scope, tmp_path):
    alias = tmp_path/"alias"
    alias.symlink_to(local_scope.model_path, target_is_directory=True)
    record = {**local_scope.record, "model_path": str(alias)}
    path = tmp_path/"aliased-scope.json"
    path.write_text(json.dumps(record))
    other = scopes.load_model_scope(path)
    assert other.model_path == local_scope.scope.model_path
    assert other.identity == local_scope.scope.identity
    assert other.manifest()["record"]["model_path"] == str(local_scope.model_path)


@pytest.mark.parametrize("binding", ["missing", "old_7b", "matching", "raw_rows", "missing_cache"])
def test_scoped_factory_requires_matching_bound_cost_but_old_default_remains_compatible(
        local_scope, tmp_path, monkeypatch, binding):
    monkeypatch.setattr("controller.config.create_model_template", lambda *a, **kw:
                        SimpleNamespace(fingerprint=local_scope.fingerprint))
    artifact = tmp_path/"cost.json"
    fingerprint = asdict(local_scope.fingerprint if binding in ("matching", "missing_cache") else ConfigFingerprint())
    metadata = {"profile_manifest": {"fingerprint": fingerprint}} if binding in ("old_7b", "matching", "missing_cache") else {}
    if binding in ("old_7b", "matching"):
        metadata["profile_manifest"]["engine"] = {"cache_config_hash": "fixture-cache"}
    CostModel().save(artifact, metadata=metadata)
    observed = []
    monkeypatch.setattr("controller.config.SGLangEngineClient", lambda *a, **kw:
                        (observed.append(kw) or SimpleNamespace()))
    raw = tmp_path/"measurements.jsonl"
    raw.write_text("")
    config = RunConfig(engine=EngineConfig(kind="sglang"), model=ModelConfig(scope_path=str(local_scope.path)),
        state_dir=str(tmp_path/"state"),
        cost_model=CostSettings(artifact=str(artifact) if binding != "raw_rows" else None),
        calibration_path=str(raw) if binding == "raw_rows" else None)
    if binding == "matching":
        service = create_controller(config)
        assert service.template.fingerprint == local_scope.fingerprint
        assert observed[0]["model_scope"] == local_scope.scope.manifest()
    else:
        with pytest.raises(ValueError, match="bound to its actual fingerprint"):
            create_controller(config)
        assert not observed
    # Scope-free legacy startup remains explicitly compatible with its old path.
    create_controller(RunConfig(state_dir=str(tmp_path/"legacy"), cost_model=CostSettings(artifact=str(artifact))))


@pytest.mark.parametrize("field,value", [("revision", "a"*40), ("kv_cache_dtype", "fp8_e4m3"),
    ("dtype", "float16"), ("json_model_override_args", '{"head_dim":80}'), ("context_length", 131072),
    ("load_format", "dummy")])
def test_native_scope_rejects_wrong_actual_settings_before_protocol_owner(local_scope, monkeypatch, field, value):
    scheduler, settings = install_native_environment(monkeypatch, local_scope)
    setattr(settings, field, value)
    with pytest.raises(native.ContextKVError) as caught:
        native.get_runtime(scheduler)
    assert caught.value.code == "FINGERPRINT_MISMATCH"
    assert not hasattr(scheduler, "contextkv_runtime")


def test_actual_shell_launcher_uses_scope_paths_and_no_fake_revision(local_scope, tmp_path):
    checkout = tmp_path/"stock"
    package = checkout/"python/sglang"
    package.mkdir(parents=True)
    (package/"__init__.py").write_text("")
    (package/"launch_server.py").write_text('import json,os,sys\nprint(json.dumps({"argv":sys.argv[1:],"scope":os.environ["CONTEXTKV_MODEL_SCOPE_PATH"],"tma":os.environ["SGLANG_HICACHE_TMA_TRANSFER"]}))\n')
    env = {**os.environ, "SGLANG_CHECKOUT": str(checkout), "CONTEXTKV_MODEL_SCOPE_PATH": str(local_scope.path),
           "CONTEXTKV_FINGERPRINT_JSON": json.dumps(asdict(local_scope.fingerprint)),
           "PATH": str(Path(sys.executable).parent)+os.pathsep+os.environ["PATH"]}
    env.pop("CONTEXTKV_MODEL_PATH", None)
    env.pop("CONTEXTKV_HICACHE_LAYOUT", None)
    env.pop("SGLANG_HICACHE_TMA_TRANSFER", None)
    completed = subprocess.run(["bash", str(ROOT/"scripts/launch_sglang.sh"), "--port", "30001"],
                               env=env, text=True, capture_output=True, check=True)
    result = json.loads(completed.stdout)
    argv = result["argv"]
    assert argv[argv.index("--model-path")+1] == str(local_scope.model_path)
    assert argv[argv.index("--tokenizer-path")+1] == str(local_scope.model_path)
    assert "--revision" not in argv and "--enable-session-radix-cache" in argv
    assert argv[argv.index("--hicache-mem-layout")+1] == "layer_first"
    assert result["tma"] == "0"
    assert result["scope"] == str(local_scope.path)
    conflict = subprocess.run(["bash", str(ROOT/"scripts/launch_sglang.sh"), "--revision", "a"*40],
                              env=env, text=True, capture_output=True)
    assert conflict.returncode != 0 and "supplied only by model scope" in conflict.stderr


def test_probe_manifest_binds_explicit_scope_without_changing_measurement_schema(local_scope, tmp_path, monkeypatch):
    import profiling.sglang_probe as probe_module
    from test_engine_profiling import ProtocolDouble
    tokenizer = SimpleNamespace(all_special_ids=(), get_vocab=lambda: {"a": 1, "b": 2})
    monkeypatch.setattr("controller.config.HuggingFaceTemplate", lambda *a, **kw:
                        SimpleNamespace(tokenizer=tokenizer, fingerprint=local_scope.fingerprint))
    engine = ProtocolDouble(fingerprint=local_scope.fingerprint)
    async def metrics(after_seq=0):
        return {"pool": {"page_size": 1}, "cache_config": {}}
    async def close():
        pass
    async def collect(*a, **kw):
        return 0
    engine.metrics, engine.aclose = metrics, close
    monkeypatch.setattr(probe_module, "SGLangEngineClient", lambda *a, **kw: engine)
    monkeypatch.setattr(probe_module, "collect_case", collect)
    config = tmp_path/"config.yaml"
    config.write_text(f"engine:\n  kind: sglang\nmodel:\n  scope_path: {local_scope.path}\n")
    assert load_config(config).model.revision == local_scope.scope.identity
    output = tmp_path/"probe.jsonl"
    args = SimpleNamespace(config=str(config), runtime_manifest=None, model_path=None, base_url=None,
        output=str(output), seed=42, lengths=(8,), prefix_fractions=(0,), outputs=(4,), concurrency=(1,),
        cache_states=("cold",), device_prefix_tokens=None, repetitions=3, warmup=1)
    asyncio.run(probe_module._run(args))
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert manifest["model_scope"] == local_scope.scope.manifest()
    assert manifest["fingerprint"]["model_revision"] == local_scope.scope.identity
    assert manifest["cases"] == [{"prompt_tokens": 8, "prefix_tokens": 0, "output_tokens": 4,
                                   "concurrency": 1, "cache_state": "cold"}]
