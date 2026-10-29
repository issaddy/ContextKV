from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json

import pytest

from controller.cost_model import CostModel
from controller.schema import ConfigFingerprint
from profiling.stage_fit import (features, feature_cell, fit_artifact, fit_cells,
                                 grouped_folds, validate)


def examples():
    rows=[]
    for n in (1024,2048,4096,8192):
        for budget in (128,512):
            for repetition in (1,2):
                batch=f"{n}-{budget}-{repetition}"
                for index in range(2):
                    forward=n*.01+index*.2
                    wait=budget*.1*index
                    rows.append(dict(run_id="run",batch_id=batch,input_hash=f"{batch}:{index}",
                        request_index=index,prefix_tokens=0,suffix_tokens=n,host_load_tokens=0,
                        concurrency=2,requested_output_budget=budget,gpu_capacity_tokens=16384,
                        forward_ms=forward,pre_forward_ms=wait,total_ttft_ms=forward+wait))
    return rows


def test_validation_keeps_cells_budgets_and_every_batch_member_together():
    rows=examples()
    for train,test in grouped_folds(rows,3,42):
        assert not {feature_cell(r) for r in train}&{feature_cell(r) for r in test}
        assert not {r["batch_id"] for r in train}&{r["batch_id"] for r in test}
        for cell in {feature_cell(r) for r in test}:
            assert {r["requested_output_budget"] for r in test if feature_cell(r)==cell}=={128,512}


def test_no_ttft_rank_index_or_realized_output_feature():
    row=examples()[0]
    changed={**row,"request_index":999,"TTFT_rank":0,"actual_output_tokens":1,"actual_host_loaded_tokens":10000,"total_ttft_ms":99999}
    assert features(row,budget=True)==features(changed,budget=True)


def test_outer_heldout_labels_cannot_change_own_predictions():
    rows=examples()
    before=validate(rows,folds=3)
    held={tuple(cell) for cell in before["folds"][0]["heldout_cells"]}
    changed=deepcopy(rows)
    for row in changed:
        if feature_cell(row) in held:
            for target in ("forward_ms","pre_forward_ms","total_ttft_ms"):
                row[target]+=10000
    after=validate(changed,folds=3)
    for target,values in before["predictions"].items():
        a=[row["predicted_ms"] for row in values if row["fold"]==0]
        b=[row["predicted_ms"] for row in after["predictions"][target] if row["fold"]==0]
        assert a==b


def test_homogeneous_waiting_spread_retained_without_predicting_rank():
    cells=fit_cells(examples(),"pre_forward_ms",budget=True)
    for cell in cells:
        assert cell["p95_ms"]==cell["max_ms"]>cell["mean_ms"]
        assert cell["batches"]==2 and cell["requests"]==4


def make_base(tmp_path,report=None):
    manifest={"run_id":"run","engine":{"fingerprint":asdict(ConfigFingerprint()),"cache_config_hash":"cache"}}
    metadata={"profile_manifest":manifest}
    if report:
        metadata["measurement_provenance"]={"files":{"development_report":{"sha256":sha256(report.read_bytes()).hexdigest()}}}
    base=tmp_path/"base.json"
    CostModel(default_output_tokens=33,p90_output_tokens=55).save(base,metadata=metadata)
    return base,{"profile_manifest":manifest}


def test_artifact_keeps_decode_prior_and_uses_actual_oof_errors_without_cell_gate(tmp_path):
    base,provenance=make_base(tmp_path)
    original=base.read_bytes()
    artifact,validation=fit_artifact(examples(),provenance,base,folds=3)
    assert base.read_bytes()==original
    assert artifact["decode_points"]==json.loads(original)["decode_points"]
    assert artifact["settings"]["default_output_tokens"]==33
    assert {p["scope"] for p in artifact["prefill_points"]}=={"forward_first"}
    stage=artifact["stage_model"]
    assert stage["waiting_calibration_scope"]=="homogeneous_burst_only"
    assert not stage["pre_forward"]["online_heterogeneous_queue_calibrated"]
    assert all(p["error_ms"]==artifact["metadata"]["stage_fit"]["forward_error_p99_ms"] for p in artifact["prefill_points"])
    assert artifact["metadata"]["validation"]["metrics"]==validation["metrics"]


def test_development_length_distribution_must_match_bound_completed_report(tmp_path):
    report=tmp_path/"dev.json"
    report.write_text(json.dumps({"report":{"turns":[{"status":"success","conversation_id":"s", "result":{"output_tokens":37,"truncated":True}},
                                                        {"status":"failed","conversation_id":"s","result":{"output_tokens":999}}]}}))
    base,provenance=make_base(tmp_path,report)
    artifact,_=fit_artifact(examples(),provenance,base,development_report=report)
    distribution=artifact["stage_model"]["development_output_lengths"]
    assert distribution["lengths"]==[37] and distribution["truncated_outputs"]==1
    report.write_text('{}')
    with pytest.raises(ValueError,match="already bound"):
        fit_artifact(examples(),provenance,base,development_report=report)


@pytest.mark.parametrize("supply_report", [False, True])
def test_refitting_stage_base_preserves_frozen_empirical_prior(tmp_path, supply_report):
    report = tmp_path/"dev.json"
    report.write_text(json.dumps({"turns": [{"status": "success", "conversation_id": "dev-session",
        "result": {"output_tokens": 37, "truncated": True}}]}))
    base, provenance = make_base(tmp_path, report)
    # Stage fitting replaces measurement provenance with the new profile files.
    provenance["measurement_provenance"] = {"files": {"profile": {"sha256": "fixture-profile"}}}
    first, _ = fit_artifact(examples(), provenance, base, development_report=report)
    stage_base = tmp_path/"stage-base.json"
    stage_base.write_text(json.dumps(first))
    original = stage_base.read_bytes()
    refined, _ = fit_artifact(examples(), provenance, stage_base,
                              development_report=report if supply_report else None)
    assert stage_base.read_bytes() == original
    assert refined["stage_model"]["development_output_lengths"] == first["stage_model"]["development_output_lengths"]
    assert refined["decode_points"] == first["decode_points"]
    assert refined["settings"] == first["settings"]
    assert refined["stage_model"]["development_output_lengths"] is not first["stage_model"]["development_output_lengths"]
    report.write_text('{}')
    with pytest.raises(ValueError, match="already bound"):
        fit_artifact(examples(), provenance, stage_base, development_report=report)
