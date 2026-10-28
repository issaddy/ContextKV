"""Optional timing/length artifacts bind consistently at real assembly boundaries."""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from types import SimpleNamespace

import pytest

from controller.cli import _manifest
from controller.config import CostSettings, RunConfig, create_controller, load_config
from controller.cost_model import CostModel, DecodePoint, LookupPoint
from controller.generation import GenerationProtocol
from controller.output_length import OutputLengthModel
from controller.schema import CacheQuote, ConfigFingerprint, CostEstimate, PromptCandidate
from controller.serialization import dumps, loads
from controller.templates import SYSTEM_PROMPT
from profiling.decision_pairs import _model_and_knobs


def assets(tmp_path):
    fingerprint = ConfigFingerprint()
    length = {"schema_version": 1, "kind": "contextkv_output_length_ridge_v1", "model_kind": "action_only",
        "target": {"fingerprint": asdict(fingerprint), "max_new_tokens": 10,
                   "generation_protocol": GenerationProtocol.from_config(RunConfig()).record()},
        "feature_schema": {"version": 1, "names": ["is_rebuild"]},
        "regression": {"target_scale": "linear", "centers": [0], "scales": [1], "coefficients": [-2], "intercept": 0},
        "training_range": {"input_tokens": [50, 150]},
        "uncertainty": {"absolute_error_p90_tokens": 30, "positive_residual_p90_tokens": 2},
        "metadata": {"global_priors": [5, 10]}}
    length_path = tmp_path/"length.json"
    OutputLengthModel(length).save(length_path)
    positive = {"coefficients": [2., 0., 0., 0., 0., 0., 0., 0.], "fit_sources": ["hardware"],
        "uncertainty_sufficient": True, "error_p99_ms": 100.,
        "conditional_uncertainty": {"minimum_feature_cells": 20, "strata": [{"concurrency": 1,
            "host_load": None, "host_load_states": [False], "feature_cells": 20, "error_p99_ms": 2.}]}}
    cost_path = tmp_path/"cost.json"
    CostModel([LookupPoint(0, 100, 1, 2.)],
        [DecodePoint(110, 1, 1., context_start=101, context_end=120)],
        default_output_tokens=5, p90_output_tokens=10, max_new_tokens=10,
        prefill_estimator="positive_basis", positive_prefill=positive, uncertainty_mode="conditional").save(
            cost_path, metadata={"output_prior_system_prompt": SYSTEM_PROMPT})
    config = RunConfig(cost_model=CostSettings(artifact=str(cost_path), output_length_artifact=str(length_path)),
                       state_dir=str(tmp_path/"state"))
    config = replace(config, context=replace(config.context, max_new_tokens=10))
    path = tmp_path/"config.yaml"
    path.write_text(json.dumps(asdict(config)))
    return load_config(path), cost_path, length_path


@pytest.mark.asyncio
async def test_serving_and_offline_assembly_share_assets_and_schema_roundtrip(tmp_path):
    config, cost, length = assets(tmp_path)
    controller = create_controller(config)
    try:
        candidate = PromptCandidate("candidate", "REBUILD", 0, (), (1,)*100, "unused",
                                    frozenset(), frozenset(), fingerprint=ConfigFingerprint())
        case = SimpleNamespace(max_new_tokens=10, candidates=(candidate,))
        offline, _ = _model_and_knobs(config, (case,), str(cost))
        quote = CacheQuote("candidate", "unused", 100, 99)
        online_value = controller.cost_model.predict_native(candidate, quote)
        assert offline.predict_native(candidate, quote) == online_value
        assert controller.cost_model.uncertainty_mode == offline.uncertainty_mode == "conditional"
        assert online_value.predicted_output_tokens == 3 and online_value.output_length_model == "action_only"
        assert online_value.output_length_error_tokens == 30 and online_value.error_ms == 2.
        assert loads(CostEstimate, dumps(online_value)) == online_value
        legacy = asdict(online_value)
        for key in ("output_length_error_tokens", "output_length_model", "output_length_in_range"):
            legacy.pop(key)
        old_value = loads(CostEstimate, json.dumps(legacy))
        assert old_value.output_length_model == "shared" and old_value.output_length_error_tokens is None
        manifest = json.loads(_manifest(config, controller, await controller.engine.info(), "P2").read_text())
        assert manifest["output_length_artifact_sha256"] == sha256(length.read_bytes()).hexdigest()
        assert manifest["cost_model_artifact_sha256"] == sha256(cost.read_bytes()).hexdigest()
    finally:
        await controller.aclose()
        controller.store.close()


@pytest.mark.asyncio
async def test_explicit_global_override_does_not_disable_optional_length_model(tmp_path):
    config, _, _ = assets(tmp_path)
    config = replace(config, cost_model=replace(config.cost_model, uncertainty_mode="global"))
    controller = create_controller(config)
    try:
        assert controller.cost_model._positive_error(1, 0) == 100.
        assert controller.cost_model.output_length_model.model_kind == "action_only"
    finally:
        await controller.aclose()
        controller.store.close()


@pytest.mark.parametrize("field", ["fingerprint", "budget"])
def test_optional_asset_mismatch_fails_before_serving_state_is_created(tmp_path, field):
    config, _, length = assets(tmp_path)
    content = json.loads(length.read_text())
    if field == "fingerprint":
        content["target"]["fingerprint"]["namespace_hash"] = "other-namespace"
    else:
        content["target"]["max_new_tokens"] = 11
    length.write_text(json.dumps(content))
    with pytest.raises(ValueError, match=field):
        create_controller(config)
    assert not (tmp_path/"state").exists()


@pytest.mark.parametrize("missing", ["generation_protocol", "global_priors"])
def test_unbound_legacy_length_artifact_cannot_enter_serving_or_pair_replay(tmp_path, missing):
    config, cost, length = assets(tmp_path)
    content = json.loads(length.read_text())
    content["target" if missing == "generation_protocol" else "metadata"].pop(missing)
    # A null source prompt cannot prove what an older default prompt contained.
    content["metadata"]["source_configuration"] = asdict(config)
    length.write_text(json.dumps(content))
    OutputLengthModel.load(length)  # Historical diagnostics remain inspectable.
    candidate = SimpleNamespace(fingerprint=ConfigFingerprint())
    case = SimpleNamespace(max_new_tokens=10, candidates=(candidate,))
    for assemble in (lambda: create_controller(config), lambda: _model_and_knobs(config, (case,), str(cost))):
        with pytest.raises(ValueError, match="generation protocol|bound shared prior center"):
            assemble()
    assert not (tmp_path/"state").exists()
