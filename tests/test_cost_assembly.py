from copy import deepcopy
from dataclasses import asdict, replace
import json

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
