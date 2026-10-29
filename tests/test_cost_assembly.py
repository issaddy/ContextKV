from copy import deepcopy
from dataclasses import asdict, replace
import json


from controller.config import RunConfig
from controller.config import assemble_cost_model, load_cost_model
from controller.cost_model import CostModel


def test_effective_assembly_is_pure_and_identical_for_files_and_serving(tmp_path):
    path = tmp_path / "cost.json"
    CostModel(default_output_tokens=580, p90_output_tokens=1024).save(path,
        metadata={"output_prior_system_prompt": "retired verbose prompt"})
    config = RunConfig(state_dir=str(tmp_path / "unused-state"))
    config = replace(config, cost_model=replace(config.cost_model, artifact=str(path)),
                     ablation=replace(config.ablation, host_cost=False))
    artifact = json.loads(path.read_text())
    original = deepcopy(artifact)
    pure = assemble_cost_model(asdict(config), artifact=artifact)
    loaded = load_cost_model(config)
    assert artifact == original and not (tmp_path / "unused-state").exists()
    assert pure.effective_output_prior == loaded.effective_output_prior == (128, 256)
    assert pure.include_host_cost is loaded.include_host_cost is False
    assert pure.sampling_scope_report == loaded.sampling_scope_report
