"""Explicit local models through the formal native path; no GPU or test_mode."""
from dataclasses import asdict, replace
from copy import deepcopy
from contextlib import nullcontext
import json
from pathlib import Path

import httpx
import pytest

from adapters.mtrag import OnlineSession, load_rows
from bench.native_baseline import NativeBaseline, NativeProtocolError, NativeSGLangClient, validate_engine_manifest
from controller.config import ModelConfig, RunConfig
from controller.sampling import SamplingRecipe, request_sampling
from controller.templates import ByteTemplate
from profiling.common import ENGINE_COMMIT
import test_model_scope as scope_fixtures

local_scope = scope_fixtures.local_scope

RECIPE = SamplingRecipe(.7, .8, 20, 0)
FIXTURE = Path(__file__).parent / "fixtures/mtrag_small.jsonl"


def contract(scope, baseline="B0", recipe=RECIPE):
    template = ByteTemplate()
    template.fingerprint = scope.fingerprint
    enabled = baseline == "B1"
    manifest = {"engine": {"base_commit": ENGINE_COMMIT, "research_patch_applied": False, "tree_backend": "python"},
        "model": {"name": scope.scope.model_path, "revision": None,
                  "identity_kind": scope.scope.manifest()["record"]["identity_kind"],
                  "identity": scope.scope.identity, "dtype": "bfloat16", "kv_dtype": "bfloat16",
                  "tokenizer_hash": scope.fingerprint.tokenizer_hash, "chat_template_hash": scope.fingerprint.template_hash},
        "model_scope": scope.scope.manifest(), "sampling_recipe": request_sampling(recipe),
        "server": {"enable_session_radix_cache": enabled, "enable_hierarchical_cache": enabled,
                   "disable_radix_cache": False, "tp_size": 1, "dp_size": 1, "pp_size": 1,
                   "page_size": 64, "context_length": 32768},
        "runtime": {"gpu_model": "CPU contract fixture", "gpu_memory_bytes": 1,
                    "driver_cuda_torch_versions": {"driver": "fixture", "cuda": "fixture", "torch": "fixture"},
                    "actual_kv_pool_tokens": 16384, "actual_host_pool_bytes": 1024 if enabled else 0}}
    actual = {**manifest["server"], "model_path": scope.scope.model_path, "tokenizer_path": scope.scope.model_path,
              "revision": None, "dtype": "bfloat16", "kv_cache_dtype": "bfloat16", "quantization": None,
              "load_format": "safetensors", "json_model_override_args": "{}"}
    return template, manifest, {"server_args": actual, "max_total_num_tokens": 16384}


@pytest.mark.parametrize("baseline", ["B0", "B1"])
@pytest.mark.asyncio
async def test_formal_scoped_native_preserves_definition_payload_full_history(local_scope, baseline):
    template, manifest, info = contract(local_scope, baseline)
    calls = []
    def handler(request):
        if request.url.path == "/server_info":
            return httpx.Response(200, json=info)
        if request.url.path == "/close_session":
            return httpx.Response(200)
        body = json.loads(request.content)
        calls.append(body)
        text = f"Actual answer {len(calls)}"
        return httpx.Response(200, json={"text": text, "output_ids": list(text.encode()),
            "prompt_token_ids": body["input_ids"], "meta_info": {"finish_reason": {"type": "stop"}, "completion_tokens": len(text)}})
    async with httpx.AsyncClient(base_url="http://fake", transport=httpx.MockTransport(handler)) as http:
        client = NativeSGLangClient("http://fake", http_client=http, audit_prompt_ids=True, sampling_params=RECIPE.record())
        service = NativeBaseline(baseline, client, template, corpus_snapshot_id="fixture", engine_manifest=manifest,
                                 model_scope=local_scope.scope, sampling=RECIPE)
        assert not service.test_mode
        await service.verify_server()
        session = OnlineSession(load_rows(FIXTURE)[:3])
        first = await service(session.current())
        session.commit(session.current().task_id, first.assistant_text)
        second = await service(session.current())
        assert first.assistant_text in bytes(calls[1]["input_ids"]).decode()
        from controller.templates import citation_regex
        for p, result in zip(calls, (first, second)):
            assert p["sampling_params"] == {**RECIPE.record(), "max_new_tokens": 512,
                                            "regex": citation_regex(result.citation_scope)}
        assert all(("session_id" in p) is (baseline == "B1") for p in calls)
        assert not any(any("contextkv" in key for key in p) for p in calls)
        await service.close_all()


@pytest.mark.parametrize("field,value", [
    ("model_path", "/wrong-model"), ("tokenizer_path", "/wrong-tokenizer"),
    ("load_format", "dummy"), ("revision", "f" * 40), ("quantization", "fp8"),
    ("json_model_override_args", '{"head_dim":80}'), ("context_length", 8192),
    ("pp_size", 2), ("dtype", "float16"),
])
@pytest.mark.asyncio
async def test_scoped_actual_native_settings_rejected_before_generation(local_scope, field, value):
    template, manifest, info = contract(local_scope)
    info["server_args"][field] = value
    calls = []
    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json=info)
    async with httpx.AsyncClient(base_url="http://fake", transport=httpx.MockTransport(handler)) as http:
        client = NativeSGLangClient("http://fake", http_client=http, sampling_params=RECIPE.record())
        service = NativeBaseline("B0", client, template, corpus_snapshot_id="fixture", engine_manifest=manifest,
                                 model_scope=local_scope.scope, sampling=RECIPE)
        with pytest.raises(NativeProtocolError):
            await service.verify_server()
        assert service.observed_server_info == info
    assert calls == ["/server_info"]


@pytest.mark.parametrize("corruption", ["identity", "missing_recipe", "wrong_recipe", "client_recipe", "legacy_validator"])
def test_scope_and_recipe_cannot_be_relabelled(local_scope, corruption):
    template, manifest, _ = contract(local_scope)
    recipe = RECIPE
    scope = local_scope.scope
    if corruption == "identity":
        manifest["model"]["identity"] = "local-snapshot:other"
    elif corruption == "missing_recipe":
        manifest.pop("sampling_recipe")
    elif corruption == "wrong_recipe":
        manifest["sampling_recipe"] = {"temperature": 0}
    elif corruption == "legacy_validator":
        scope = None
    settings = None if corruption == "client_recipe" else recipe.record()
    # No request is sent; supplying an external http client avoids owning resources.
    client = NativeSGLangClient("http://fake", http_client=object(), sampling_params=settings)
    with pytest.raises(ValueError):
        NativeBaseline("B0", client, template, corpus_snapshot_id="fixture", engine_manifest=manifest,
                       model_scope=scope, sampling=recipe)


def test_explicit_qwen3_32b_head_dim_and_cross_checkpoint_identity(local_scope):
    from engine_patches.contextkv_model_scope import load_model_scope
    template, manifest, _ = contract(local_scope)
    assert local_scope.scope.manifest()["record"]["config"]["head_dim"] == 128
    assert not validate_engine_manifest(manifest, "B0", template, model_scope=local_scope.scope, sampling=RECIPE)
    other = load_model_scope(Path(__file__).parents[1] / "configs/model_scopes/qwen3-32b.local.json")
    assert other.identity != local_scope.scope.identity
    assert validate_engine_manifest(manifest, "B0", template, model_scope=other, sampling=RECIPE)


@pytest.mark.parametrize("change", ["missing", "gpu_capacity"])
@pytest.mark.asyncio
async def test_scoped_missing_actual_settings_and_wrong_pool_are_not_attested(local_scope, change):
    template, manifest, info = contract(local_scope)
    if change == "missing":
        info["server_args"].pop("enable_session_radix_cache")
    else:
        info["max_total_num_tokens"] = 8192
    async with httpx.AsyncClient(base_url="http://fake", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=info))) as http:
        service = NativeBaseline("B0", NativeSGLangClient("http://fake", http_client=http, sampling_params=RECIPE.record()),
            template, corpus_snapshot_id="fixture", engine_manifest=manifest, model_scope=local_scope.scope, sampling=RECIPE)
        with pytest.raises(NativeProtocolError):
            await service.verify_server()


def test_matrix_shared_scope_recipe_and_method_override_rejection(tmp_path, local_scope):
    from bench.experiments import matrix_jobs
    from test_experiments import spec
    from controller.serialization import to_jsonable
    matrix = spec(tmp_path)
    config = RunConfig(model=ModelConfig(scope_path=str(local_scope.path)), sampling=RECIPE)
    path = tmp_path / "shared.json"
    path.write_text(json.dumps(to_jsonable(config)))
    matrix["base_config"] = str(path)
    matrix["methods"].append({"id": "Native", "kind": "native", "native": {"baseline": "B0"}})
    assert len(matrix_jobs(matrix)) == 4
    for override in ({"sampling": {"temperature": 0}}, {"model": {"namespace_hash": "different"}}):
        bad = deepcopy(matrix)
        bad["methods"][0]["candidates"][0].update(override)
        with pytest.raises(ValueError, match="differs from the shared"):
            matrix_jobs(bad)
    for field, value in (("model_path", "/wrong"), ("sampling", RECIPE.record()), ("run_config", str(path))):
        bad = deepcopy(matrix)
        bad["methods"][-1]["native"][field] = value
        with pytest.raises(ValueError):
            matrix_jobs(bad)


@pytest.mark.parametrize("baseline", ["B0", "B1"])
@pytest.mark.parametrize("export_failure", ["none", "first", "all"])
def test_run_config_cli_uses_real_scope_factory_and_recipe_without_test_mode(
        tmp_path, local_scope, monkeypatch, baseline, export_failure):
    import asyncio
    import bench.native_baseline as module
    import controller.config as configuration
    from controller.candidates import CandidatePolicy
    template, manifest, info = contract(local_scope, baseline)
    manifest_path, config_path = tmp_path / "engine.json", tmp_path / "shared.json"
    manifest_path.write_text(json.dumps(manifest))
    config = RunConfig(model=ModelConfig(scope_path=str(local_scope.path)), sampling=RECIPE,
                       context=replace(CandidatePolicy(), max_new_tokens=64))
    config_path.write_text(json.dumps(asdict(config)))
    loaded, requests, closes, exports = [], [], [], []
    def template_factory(path, revision, **options):
        loaded.append((path, revision, options))
        return template
    monkeypatch.setattr(configuration, "HuggingFaceTemplate", template_factory)
    def handler(request):
        if request.url.path == "/server_info":
            return httpx.Response(200, json=info)
        body = json.loads(request.content)
        if request.url.path == "/close_session":
            closes.append(body["session_id"])
            return httpx.Response(200)
        requests.append(body)
        text = f"Complete native answer {len(requests)}"
        return httpx.Response(200, json={"text": text, "output_ids": list(text.encode()), "prompt_token_ids": body["input_ids"],
            "meta_info": {"finish_reason": {"type": "stop"}, "completion_tokens": len(text)}})
    http = httpx.AsyncClient(base_url="http://fake", transport=httpx.MockTransport(handler))
    original_client = module.NativeSGLangClient
    monkeypatch.setattr(module, "NativeSGLangClient", lambda *a, **k: original_client(*a, **k, http_client=http))
    original_export = module.export_evaluation

    def export(turn, answer, **kwargs):
        exports.append(turn.task_id)
        if export_failure == "all" or (export_failure == "first" and len(exports) == 1):
            raise OSError("controlled native export failure")
        return original_export(turn, answer, **kwargs)

    monkeypatch.setattr(module, "export_evaluation", export)
    output = tmp_path / "run"
    try:
        expected = (nullcontext() if export_failure == "none" else
                    pytest.raises(RuntimeError, match="evaluation export failed for"))
        with expected:
            assert module.main([str(FIXTURE), "--baseline", baseline, "--base-url", "http://fake", "--domain", "all",
                                "--run-config", str(config_path), "--engine-manifest", str(manifest_path),
                                "--output", str(output), "--concurrency", "2", "--audit-prompt-ids", "--no-stream"]) == 0
    finally:
        asyncio.run(http.aclose())
    assert loaded == [(local_scope.scope.model_path, None, {"namespace_hash": "default", "local_files_only": True,
                                                            "model_identity": local_scope.scope.identity})]
    assert len(requests) == len(exports) == 6
    assert all({k:v for k,v in r["sampling_params"].items() if k != "regex"}
               == {**RECIPE.record(), "max_new_tokens": 64} for r in requests)
    assert all(isinstance(r["sampling_params"]["regex"], str) for r in requests)
    payload = json.loads(next(output.glob("native-*.json")).read_text())
    assert payload["shared_configuration"]["sampling_recipe"] == RECIPE.record()
    assert payload["template_fingerprint"]["model_revision"] == local_scope.scope.identity
    assert payload["engine_manifest"]["model"]["revision"] is None
    turns = payload["report"]["turns"]
    assert len(turns) == 6 and all(t["status"] == "success" for t in turns)
    assert payload["report"]["unfinished_turns"] == 0
    assert payload["close_error"] is None and payload["undispatched_native_sessions"] == []
    assert len(closes) == (2 if baseline == "B1" else 0)
    failures = payload["evaluation_export"]["failures"]
    assert len(failures) == {"none": 0, "first": 1, "all": 6}[export_failure]
    assert payload["evaluation_export"]["status"] == ("completed" if export_failure == "none" else "failed")
    if failures:
        assert failures[0]["task_id"] == exports[0]
        assert failures[0]["error"] == "OSError: controlled native export failure"
    for turn in turns:
        if turn["turn"] > 1:
            previous = next(t for t in turns if t["conversation_id"] == turn["conversation_id"]
                            and t["turn"] == turn["turn"] - 1)
            assert turn["online_turn"]["history"][-1]["text"] == previous["assistant_text"]


@pytest.mark.asyncio
async def test_matrix_native_executor_receives_the_shared_typed_scope_and_recipe(tmp_path, local_scope, monkeypatch):
    import bench.native_baseline as module
    from bench.experiments import execute_job, matrix_jobs
    from test_experiments import spec
    matrix = spec(tmp_path)
    shared = RunConfig(model=ModelConfig(scope_path=str(local_scope.path)), sampling=RECIPE)
    config_path, manifest_path = tmp_path / "shared.json", tmp_path / "engine.json"
    config_path.write_text(json.dumps(asdict(shared)))
    manifest_path.write_text(json.dumps(contract(local_scope)[1]))
    matrix["base_config"] = str(config_path)
    matrix["methods"] = [{"id": "Native", "kind": "native", "native": {
        "baseline": "B0", "base_url": "http://fake", "engine_manifest": str(manifest_path)}}]
    class ReachedNative(Exception):
        pass
    async def capture(args):
        assert args.model_config == shared.model and args.sampling == RECIPE
        assert args.context_policy == shared.context
        raise ReachedNative
    monkeypatch.setattr(module, "_run", capture)
    with pytest.raises(ReachedNative):
        await execute_job(matrix, matrix_jobs(matrix)[0], tmp_path / "job")


@pytest.mark.parametrize("use_scope", [False, True])
def test_owned_scope_recipe_job_declaration_keeps_verified_executor_requirement(tmp_path, local_scope, use_scope):
    from bench.experiments import matrix_jobs
    from test_experiments import spec
    matrix = spec(tmp_path)
    model = ModelConfig(scope_path=str(local_scope.path)) if use_scope else ModelConfig()
    path = tmp_path / "shared.json"
    path.write_text(json.dumps(asdict(RunConfig(model=model, sampling=None if use_scope else RECIPE))))
    matrix["base_config"] = str(path)
    matrix["server_profiles"] = {"stock": {}}
    matrix["methods"].append({"id": "Native", "kind": "native", "server_profile": "stock", "native": {"baseline": "B0"}})
    if use_scope:
        jobs = matrix_jobs(matrix)
        assert next(job for job in jobs if job["method"] == "Native")["server_profile"] == "stock"
    else:
        with pytest.raises(ValueError, match="scope_path"):
            matrix_jobs(matrix)


def test_legacy_manifest_is_read_only_and_cannot_authorize_generation(local_scope):
    from profiling.common import LEGACY_MODEL_REVISION
    template, manifest, _ = contract(local_scope)
    manifest.pop("model_scope")
    manifest.pop("sampling_recipe")
    manifest["model"] = {"name": "Qwen/Qwen2.5-7B-Instruct", "revision": LEGACY_MODEL_REVISION,
        "tokenizer_hash": template.fingerprint.tokenizer_hash, "chat_template_hash": template.fingerprint.template_hash,
        "dtype": "bfloat16", "kv_dtype": "bfloat16"}
    template.fingerprint = replace(template.fingerprint, model_revision=LEGACY_MODEL_REVISION)
    original = deepcopy(manifest)
    client = NativeSGLangClient("http://fake", http_client=object(), sampling_params=SamplingRecipe().record())
    assert not validate_engine_manifest(manifest,"B0",template,sampling=SamplingRecipe())
    with pytest.raises(ValueError, match="explicit Qwen3"):
        NativeBaseline("B0", client, template, corpus_snapshot_id="fixture", engine_manifest=manifest,
                       sampling=SamplingRecipe())
    assert manifest == original
