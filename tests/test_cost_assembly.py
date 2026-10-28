from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from controller.config import RunConfig
from controller.config import assemble_cost_model, load_cost_model
from controller.cost_model import CostModel
from controller.schema import ConfigFingerprint
from profiling.length_fit import training_priors


def test_effective_assembly_is_pure_and_identical_for_files_and_length_fitting(tmp_path):
    path = tmp_path / "cost.json"
    CostModel(default_output_tokens=580, p90_output_tokens=1024, paired_error_ms=7).save(path,
        metadata={"output_prior_system_prompt": "retired verbose prompt"})
    config = RunConfig(state_dir=str(tmp_path / "unused-state"))
    config = replace(config, cost_model=replace(config.cost_model, artifact=str(path)),
                     ablation=replace(config.ablation, host_cost=False))
    artifact = json.loads(path.read_text())
    original = deepcopy(artifact)
    pure = assemble_cost_model(asdict(config), artifact=artifact)
    loaded = load_cost_model(config, fingerprint=ConfigFingerprint())
    assert artifact == original and not (tmp_path / "unused-state").exists()
    assert pure.effective_output_prior == loaded.effective_output_prior == training_priors(artifact, asdict(config))[0] == (128, 256)
    assert pure.paired_error_ms == loaded.paired_error_ms == 7
    assert pure.include_host_cost is loaded.include_host_cost is False
    assert pure.sampling_scope_report == loaded.sampling_scope_report


@pytest.mark.parametrize("changed_prompt", [False, True])
def test_owned_preflight_and_serving_share_optional_length_protocol(tmp_path, changed_prompt):
    from bench.b6_joint_sweep import fitted_scope
    from controller.cost_model import LookupPoint, DecodePoint
    from controller.generation import GenerationProtocol
    from test_controller_capacity_sweep import runtime
    from test_output_length import specification

    cost_path, length_path = tmp_path / "cost.json", tmp_path / "length.json"
    CostModel([LookupPoint(0, 512, 1, 10)], [DecodePoint(512, 1, 1)]).save(cost_path,
        metadata={"profile_manifest": {"engine": runtime()["engine"]}})
    config = RunConfig()
    config = replace(config, context=replace(config.context, max_new_tokens=1024),
        cost_model=replace(config.cost_model, artifact=str(cost_path), output_length_artifact=str(length_path)))
    spec = specification()
    spec["metadata"]["global_priors"] = [128, 256]
    spec["target"]["generation_protocol"] = GenerationProtocol.from_config(config).record()
    length_path.write_text(json.dumps(spec))
    if changed_prompt:
        config = replace(config, context=replace(config.context, system_prompt="Changed answering protocol"))
        with pytest.raises(ValueError, match="generation protocol"):
            load_cost_model(config, fingerprint=ConfigFingerprint())
        with pytest.raises(ValueError, match="generation protocol"):
            fitted_scope(config, {"path": str(cost_path)}, runtime())
    else:
        assert load_cost_model(config, fingerprint=ConfigFingerprint()).effective_output_prior == (128, 256)
        assert fitted_scope(config, {"path": str(cost_path)}, runtime())["status"] == "matching_scope"
