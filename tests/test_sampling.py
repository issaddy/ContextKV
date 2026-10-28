"""Request recipe is opt-in; timing and output-length bindings stay separate."""
import asyncio
from dataclasses import FrozenInstanceError, replace
import json
from types import SimpleNamespace

import httpx
import pytest
from test_model_scope import local_scope as local_scope

from controller.config import RunConfig, create_controller, load_config
from controller.cost_model import CostModel
from controller.engine_client import SGLangEngineClient
from controller.output_length import OutputLengthModel
from controller.sampling import SamplingRecipe, bound_recipe, request_sampling
from controller.schema import ConfigFingerprint
from profiling.decision_pairs import _model_and_knobs
from profiling.fit import measurement_provenance
from test_cost_fit import candidate_and_quote, row
from test_engine_streaming import frame, info
from test_optional_cost_assets import assets
from test_stage_fit import examples, make_base


RECIPE = SamplingRecipe(.7, .8, 20, 0)


@pytest.mark.parametrize("values", [
    {"temperature": -1}, {"temperature": float("nan")}, {"temperature": True},
    {"top_p": 0}, {"top_p": 1.1}, {"top_p": float("inf")},
    {"min_p": -1}, {"min_p": 1.1}, {"top_k": 0}, {"top_k": -2},
    {"top_k": 2.0}, {"top_k": True},
])
def test_recipe_rejects_invalid_values_at_construction(values):
    with pytest.raises(ValueError):
        SamplingRecipe(**values)


@pytest.mark.parametrize("field", ["sampling_seed", "max_new_tokens", "ignore_eos", "unknown"])
def test_recipe_cannot_override_output_budget_or_other_protocol(tmp_path, field):
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps({"sampling": {field: 2}}))
    with pytest.raises(ValueError, match="Unknown fields"):
        load_config(path)


def test_typed_config_and_only_standard_greedy_equivalence(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps({"sampling": RECIPE.record()}))
    assert load_config(path).sampling == RECIPE
    assert load_config().sampling is None
    assert bound_recipe(None) is bound_recipe({}) is None  # unknown is not a source claim of greedy
    with pytest.raises(ValueError, match="complete recipe"):
        bound_recipe({"top_p": .8})
    assert bound_recipe({"temperature": 0}) == SamplingRecipe()
    assert bound_recipe({**SamplingRecipe().record(), "top_k": 1}) != SamplingRecipe()
    assert request_sampling(None) == {"temperature": 0}
    assert request_sampling(SamplingRecipe()) == SamplingRecipe().record()
    with pytest.raises(FrozenInstanceError):
        RECIPE.temperature = .5


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("recipe", [None, SamplingRecipe(), RECIPE])
def test_real_http_client_default_and_explicit_wire_share_one_budget(stream, recipe):
    fp, requests = ConfigFingerprint(), []
    def handler(request):
        if request.url.path == "/context/engine/info":
            return httpx.Response(200, json=info(fp))
        body = json.loads(request.content)
        requests.append(body)
        terminal = frame([7], "answer", terminal=True)
        if stream:
            return httpx.Response(200, text="data: " + json.dumps(terminal) + "\n\ndata: [DONE]\n\n",
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=terminal)
    async def scenario():
        engine = SGLangEngineClient("http://test", fp, sampling=recipe, stream=stream,
                                    transport=httpx.MockTransport(handler))
        try:
            for cap, ignore, citations in ((2, False, None), (3, True, None), (4, False, ("E_aa",))):
                result = await engine.generate(input_ids=[1, 2], session_id="s", expected_generation=1,
                    expected_restart_id="r", attempt_id="a", fingerprint=fp, max_new_tokens=cap, ignore_eos=ignore,
                    citation_ids=citations)
                assert result.output_ids == (7,)
        finally:
            await engine.aclose()
    asyncio.run(scenario())
    from controller.templates import citation_regex
    for body, cap, ignore, citations in zip(requests, (2, 3, 4), (False, True, False), (None, None, ("E_aa",))):
        assert body["sampling_params"] == {**request_sampling(recipe), "max_new_tokens": cap,
                                           **({"ignore_eos": True} if ignore else {}),
                                           **({"regex": citation_regex(citations)} if citations is not None else {})}
        assert body["stream"] is stream


def test_recipe_mismatch_keeps_time_points_but_not_eligibility_or_old_prior(tmp_path):
    model = CostModel.from_measurements([row(), row(phase="decode", elapsed=6)],
                                       default_output_tokens=4, p90_output_tokens=4)
    model.stage_model = {"development_output_lengths": {"lengths": [20, 30]}}
    # Configure without stage prediction, then inspect the distribution policy.
    points = model.prefill_points, model.decode_points
    report = model.configure_sampling(RECIPE, fallback_mean=2, fallback_p90=2)
    assert report["timing_compatible"] is False
    assert report["output_prior_status"] == "configured_prior_fallback"
    assert model._development_lengths == ()
    model.stage_model = {}
    candidate, quote = candidate_and_quote()
    predicted = model.predict_native(candidate, quote)
    assert not predicted.calibrated and predicted.predicted_output_tokens == 2
    assert predicted.restore_prefill_ms == 10 and predicted.decode_ms == 2
    assert (model.prefill_points, model.decode_points) == points
    # Rechecking an engine/restart cannot erase the independent recipe mismatch.
    scope = model.check_calibration_scope({})
    assert scope["sampling"]["timing_compatible"] is False
    assert not model.predict_native(candidate, quote).calibrated
    model.save(tmp_path / "unchanged.json")
    loaded = CostModel.load(tmp_path / "unchanged.json")
    assert loaded.default_output_tokens == 4  # runtime fallback did not relabel fitted priors
    loaded.configure_sampling(SamplingRecipe(), fallback_mean=2, fallback_p90=2)
    assert loaded.predict_native(candidate, quote).calibrated
    assert loaded.predict_native(candidate, quote).predicted_output_tokens == 4


def test_timing_and_development_lengths_require_their_own_matching_bindings():
    model = CostModel.from_measurements([row(), row(phase="decode", elapsed=6)],
                                       default_output_tokens=4, p90_output_tokens=4)
    model.calibration_metadata = {"profile_manifest": {"sampling_recipe": RECIPE.record()}}
    model.stage_model = {"development_output_lengths": {"lengths": [3, 4]}}
    result = model.configure_sampling(RECIPE, fallback_mean=2, fallback_p90=2)
    assert result["timing_compatible"] and model._development_lengths == ()
    assert result["output_prior_status"] == "configured_prior_fallback"
    model.calibration_metadata["output_prior_sampling"] = RECIPE.record()
    model.configure_sampling(RECIPE, fallback_mean=2, fallback_p90=2)
    assert model._development_lengths == [3, 4] and model._sampling_prior_override is None
    model.configure_sampling(None, fallback_mean=2, fallback_p90=2)
    assert not model.sampling_scope_compatible and model._development_lengths == ()


def test_new_answer_protocol_reuses_timings_without_old_length_distribution():
    model = CostModel.from_measurements([row(), row(phase="decode", elapsed=6)],
                                       default_output_tokens=900, p90_output_tokens=1024)
    model.calibration_metadata = {"profile_manifest": {"sampling_recipe": RECIPE.record()},
                                  "output_prior_sampling": RECIPE.record(),
                                  "output_prior_system_prompt": "old verbose policy"}
    model.stage_model = {"development_output_lengths": {"lengths": [800, 1000]}}
    points = model.prefill_points, model.decode_points
    report = model.configure_sampling(RECIPE, fallback_mean=128, fallback_p90=256,
                                      system_prompt="current concise policy")
    assert report["timing_compatible"]
    assert model.effective_output_prior == (128, 256)
    assert model._development_lengths == ()
    assert report["development_lengths_status"] == "disabled_prompt_mismatch"
    assert (model.prefill_points, model.decode_points) == points
    model.observe_output(80)
    model.observe_output(0)
    model.observe_output(1024)  # capped completion is still a served length
    assert list(model._observed_output_lengths) == [80, 1024]
    model.configure_sampling(RECIPE, fallback_mean=128, fallback_p90=256,
                             system_prompt="old verbose policy")
    assert model.effective_output_prior == (900, 1024)
    assert model._development_lengths == [800, 1000]
    assert not model._observed_output_lengths


def test_serving_and_pair_replay_apply_same_recipe_and_length_boundary(tmp_path):
    from controller.generation import GenerationProtocol
    config, cost, length = assets(tmp_path)
    config = replace(config, sampling=RECIPE)
    with pytest.raises(ValueError, match="output-length artifact sampling"):
        create_controller(config)
    content = json.loads(length.read_text())
    content["target"]["sampling_recipe"] = RECIPE.record()
    content["target"]["generation_protocol"] = GenerationProtocol.from_config(config).record()
    content["metadata"] = {"global_priors": [128, 256]}
    length.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="sampling"):
        OutputLengthModel.load(length)  # matching is required in both directions
    controller = create_controller(config)
    candidate, _ = candidate_and_quote()
    offline, _ = _model_and_knobs(config, [SimpleNamespace(max_new_tokens=10, candidates=(candidate,))], str(cost))
    assert controller.cost_model.sampling_scope_report == offline.sampling_scope_report
    assert controller.cost_model.output_length_model is not None
    asyncio.run(controller.aclose())
    controller.store.close()


def test_profile_merge_refuses_different_recipe_and_binds_length_report_separately(tmp_path):
    from test_calibration_provenance import manifest
    paths = [tmp_path / "p1.json", tmp_path / "p2.json"]
    first, second = manifest(), {**manifest(), "run_id": "profile-2", "sampling_recipe": RECIPE.record()}
    rows = [{**row(), "run_id": value} for value in ("profile-1", "profile-2")]
    measurements = tmp_path / "rows.jsonl"
    measurements.write_text("\n".join(json.dumps(value) for value in rows))
    for path, value in zip(paths, (first, second)):
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="same sampling recipe"):
        measurement_provenance(rows, measurements, profile_manifest_path=paths)
    first["sampling_recipe"] = RECIPE.record()
    paths[0].write_text(json.dumps(first))
    report = tmp_path / "dev.json"
    report.write_text(json.dumps({"sampling_recipe": {"temperature": 0}, "report": {"turns": []}}))
    metadata = measurement_provenance(rows, measurements, profile_manifest_path=paths, development_report_path=report)
    assert metadata["output_prior_sampling"] == {"temperature": 0}
    assert metadata["profile_manifest"]["sampling_recipe"] == RECIPE.record()


def test_stage_refinement_cannot_relabel_copied_decode_recipe(tmp_path):
    from profiling.stage_fit import fit_artifact
    base, provenance = make_base(tmp_path)
    provenance["profile_manifest"]["sampling_recipe"] = RECIPE.record()
    with pytest.raises(ValueError, match="same sampling recipe"):
        fit_artifact(examples(), provenance, base)


def test_unknown_profiles_never_gain_an_explicit_recipe_through_merge(tmp_path):
    from test_calibration_provenance import manifest
    paths = [tmp_path / "p1.json", tmp_path / "p2.json"]
    rows = [{**row(), "run_id": value} for value in ("profile-1", "profile-2")]
    measurements = tmp_path / "rows.jsonl"
    measurements.write_text("\n".join(json.dumps(value) for value in rows))
    values = [manifest(), {**manifest(), "run_id": "profile-2"}]
    for path, value in zip(paths, values):
        path.write_text(json.dumps(value))
    legacy = measurement_provenance(rows, measurements, profile_manifest_path=paths)
    assert legacy["measurement_provenance"]["sampling_recipe_binding"] == "unknown_legacy_unbound"
    assert all("sampling_recipe" not in value for value in legacy["profile_manifests"])
    values[1]["sampling_recipe"] = SamplingRecipe().record()
    paths[1].write_text(json.dumps(values[1]))
    with pytest.raises(ValueError, match="same sampling recipe"):
        measurement_provenance(rows, measurements, profile_manifest_path=paths)
    values[0]["sampling_recipe"] = {"temperature": 0}
    paths[0].write_text(json.dumps(values[0]))
    explicit = measurement_provenance(rows, measurements, profile_manifest_path=paths)
    assert explicit["measurement_provenance"]["sampling_recipe_binding"] == "explicit"


def test_profiler_cli_records_the_recipe_actually_given_to_its_client(tmp_path, monkeypatch, local_scope):
    from argparse import Namespace
    import profiling.sglang_probe as module
    from test_engine_profiling import ProtocolDouble
    engine, captured = ProtocolDouble(), []
    async def metrics(after_seq=0):
        return {"pool": {"page_size": 1}, "cache_config": {}}
    async def aclose():
        pass
    engine.metrics, engine.aclose = metrics, aclose
    def client(*args, **kwargs):
        captured.append(kwargs["sampling"])
        return engine
    monkeypatch.setattr(module, "SGLangEngineClient", client)
    tokenizer = SimpleNamespace(all_special_ids=(), get_vocab=lambda: {"a": 1, "b": 2})
    monkeypatch.setattr("controller.config.create_model_template", lambda *a, **k:
                        SimpleNamespace(tokenizer=tokenizer, fingerprint=ConfigFingerprint()))
    async def no_generations(*args, **kwargs):
        return 0
    monkeypatch.setattr(module, "collect_case", no_generations)
    config, output = tmp_path / "config.yaml", tmp_path / "probe.jsonl"
    config.write_text(json.dumps({"engine": {"kind": "sglang"}, "model": {"scope_path": str(local_scope.path)},
                                 "sampling": RECIPE.record()}))
    args = Namespace(config=str(config), runtime_manifest=None, model_path=None, base_url=None, output=str(output),
        seed=42, lengths=(8,), prefix_fractions=(0,), outputs=(4,), concurrency=(1,),
        cache_states=("cold",), device_prefix_tokens=None, repetitions=3, warmup=1)
    asyncio.run(module._run(args))
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert captured == [RECIPE] and manifest["sampling_recipe"] == RECIPE.record()
    assert manifest["generation_protocol"] == {"ignore_eos": True, "output_budget": "per_case",
                                               "seed_scope": "input_fixture_only_not_request_sampler"}


@pytest.mark.parametrize("location", ["base", "controller_candidate"])
def test_native_matrix_accepts_shared_recipe_but_refuses_method_sampling_difference(tmp_path, location, local_scope):
    from bench.experiments import matrix_jobs
    from test_experiments import spec
    matrix = spec(tmp_path)
    matrix["methods"].append({"id": "Native", "kind": "native", "native": {"baseline": "B1"}})
    path = tmp_path / "base.json"
    base = {"model": {"scope_path": str(local_scope.path)}}
    if location == "base":
        base["sampling"] = RECIPE.record()
    else:
        matrix["methods"][0]["candidates"][0]["sampling"] = RECIPE.record()
    path.write_text(json.dumps(base))
    matrix["base_config"] = str(path)
    if location == "base":
        assert len(matrix_jobs(matrix)) == 4
    else:
        with pytest.raises(ValueError, match="method sampling differs"):
            matrix_jobs(matrix)


def test_controller_only_matrix_allows_recipe_and_native_requires_shared_scope(tmp_path, local_scope):
    from bench.experiments import matrix_jobs
    from test_experiments import spec
    matrix, path = spec(tmp_path), tmp_path / "base.json"
    path.write_text(json.dumps({"sampling": RECIPE.record()}))
    matrix["base_config"] = str(path)
    assert len(matrix_jobs(matrix)) == 3
    path.write_text(json.dumps({"model": {"scope_path": str(local_scope.path)},
                                "sampling": SamplingRecipe().record()}))
    matrix["methods"].append({"id": "Native", "kind": "native", "native": {"baseline": "B1"}})
    assert len(matrix_jobs(matrix)) == 4


def test_owned_cost_preflight_rejects_cross_recipe_before_generation(tmp_path):
    from bench import b6_joint_sweep
    from test_controller_capacity_sweep import cost_artifact, runtime
    path = cost_artifact(tmp_path)
    config = replace(RunConfig(), sampling=RECIPE)
    with pytest.raises(ValueError, match="actual engine configuration"):
        b6_joint_sweep.fitted_scope(config, {"path": str(path)}, runtime())


@pytest.mark.parametrize("source_config,requested", [
    ({}, RECIPE), (None, RECIPE), ({"sampling": RECIPE.record()}, None),
])
def test_offline_rescore_cannot_relabel_outcomes_with_new_recipe(tmp_path, monkeypatch, source_config, requested):
    from argparse import Namespace
    from controller.generation import GenerationProtocol
    import profiling.decision_pairs as module
    frozen = {"cases_sha256": "same-frozen-inputs"}
    monkeypatch.setattr(module, "load_frozen", lambda _: ([], frozen))
    source, config, output = tmp_path / "pairs.jsonl", tmp_path / "config.json", tmp_path / "out.jsonl"
    source.with_suffix(".manifest.json").write_text(json.dumps({
        "status": "complete", "split": "dev", "frozen": frozen,
        "generation_protocol": GenerationProtocol.from_config(source_config).record() if source_config is not None else None}))
    config.write_text(json.dumps({"sampling": requested.record() if requested else None}))
    def forbidden(*args, **kwargs):
        pytest.fail("recipe mismatch must stop before model construction or any generation")
    monkeypatch.setattr(module, "_model_and_knobs", forbidden)
    with pytest.raises(ValueError, match="cannot change the measured generation protocol"):
        module.rescore_file(Namespace(cases="unused", input=str(source), output=str(output),
                                     config=str(config), cost_artifact="unused"))
    assert not output.exists()


def test_length_training_and_serving_use_same_recipe_fallback_center():
    from profiling.length_fit import training_priors
    from controller.templates import SYSTEM_PROMPT
    baseline = {"settings": {"default_output_tokens": 433, "p90_output_tokens": 512}, "metadata": {},
                "prefill_points": [], "decode_points": []}
    config = {"sampling": RECIPE.record(), "cost_model": {"default_output_tokens": 128, "p90_output_tokens": 256}}
    fitted, binding = training_priors(baseline, config)
    from controller.config import assemble_cost_model
    serving = assemble_cost_model(config, artifact=baseline)
    assert fitted == serving.effective_output_prior == (128, 256)
    assert binding["output_prior_status"] == "configured_prior_fallback"
    # A matching empirical source keeps its center; missing source config does
    # not manufacture a new recipe binding for a legacy offline run.
    baseline["metadata"]["output_prior_sampling"] = RECIPE.record()
    assert training_priors(baseline, config)[0] == (128, 256)  # Missing prompt binding still falls back.
    baseline["metadata"]["output_prior_system_prompt"] = SYSTEM_PROMPT
    assert training_priors(baseline, config)[0] == (433, 512)


@pytest.mark.parametrize("recipe", [None, RECIPE])
def test_learned_residual_center_is_checked_once_when_loading(tmp_path, recipe):
    _, _, path = assets(tmp_path)
    content = json.loads(path.read_text())
    content["target"]["sampling_recipe"] = request_sampling(recipe)
    content["metadata"] = {"global_priors": [128, 256]}
    path.write_text(json.dumps(content))
    OutputLengthModel.load(path, sampling=recipe, expected_priors=(128, 256))
    with pytest.raises(ValueError, match="prior center differs"):
        OutputLengthModel.load(path, sampling=recipe, expected_priors=(433, 512))
    content["metadata"].pop("global_priors")
    path.write_text(json.dumps(content))
    if recipe is None:
        OutputLengthModel.load(path, expected_priors=(128, 256))  # legacy compatibility, not fitted proof
    else:
        with pytest.raises(ValueError, match="bound shared prior center"):
            OutputLengthModel.load(path, sampling=recipe, expected_priors=(128, 256))


def test_length_rescore_rejects_legacy_unconstrained_measurements_before_model_load(tmp_path, monkeypatch):
    from argparse import Namespace
    import profiling.decision_pairs as module
    frozen = {"cases_sha256": "same-frozen-inputs"}
    monkeypatch.setattr(module, "load_frozen", lambda _: ([], frozen))
    source, config, output = tmp_path / "pairs.jsonl", tmp_path / "config.json", tmp_path / "out.jsonl"
    source.with_suffix(".manifest.json").write_text(json.dumps({
        "status": "complete", "split": "dev", "frozen": frozen, "config": {}}))
    config.write_text(json.dumps({"cost_model": {"output_length_artifact": "current-length-model.json"}}))
    def forbidden(*args, **kwargs):
        pytest.fail("unbound measured protocol must stop before loading the current length model")
    monkeypatch.setattr(module, "_model_and_knobs", forbidden)
    with pytest.raises(ValueError, match="cannot change the measured generation protocol"):
        module.rescore_file(Namespace(cases="unused", input=str(source), output=str(output),
                                     config=str(config), cost_artifact="unused"))
    assert not output.exists()
