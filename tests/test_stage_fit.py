from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json

import pytest

from controller.cost_model import CostModel
from controller.schema import ConfigFingerprint
from profiling.common import Measurement
from profiling.stage_fit import (features, feature_cell, fit_artifact, fit_cells,
                                 grouped_folds, load_bound_rows, validate)


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


def inputs(tmp_path):
    case=dict(prompt_tokens=1024,prefix_tokens=512,output_tokens=128,concurrency=2,cache_state="gpu_ready")
    fingerprint=asdict(ConfigFingerprint())
    manifest=dict(run_id="run",status="complete",source="hardware",cases=[case],
                  engine=dict(fingerprint=fingerprint,cache_config_hash="cache"),
                  pool=dict(gpu_capacity_tokens=16384))
    manifest_path=tmp_path/"profile.json"
    manifest_path.write_text(json.dumps(manifest))
    compact=[]
    measurements=[]
    for repetition,warmup in ((1,True),(2,False),(3,False)):
        for index in range(2):
            # A prepared GPU-ready prefix may actually restore after admission;
            # this realized event must not become an online input feature.
            host=512 if index else 0
            ttft=20.0+index+repetition
            compact.append(dict(group="group",case=case,index=index,warmup=warmup,repetition=None,
                server_ttft_ms=ttft,forward_to_first_token_ms=10.0,output_tokens=128,
                actual_device_hit_tokens=512-host,actual_host_loaded_tokens=host,
                actual_prefill_tokens=512,input_hash=f"{repetition}-{index}"))
            if not warmup:
                measurements.append(asdict(Measurement("first_token",1024,512,128,2,0,ttft,"hardware",
                    host_load_tokens=host,run_id="run",repetition=repetition,cache_state="gpu_ready",timing_scope="server_request")))
    cp,mp=tmp_path/"compact.jsonl",tmp_path/"measurements.jsonl"
    cp.write_text("".join(json.dumps(row)+"\n" for row in compact))
    mp.write_text("".join(json.dumps(row)+"\n" for row in measurements))
    return cp,mp,{"group":manifest_path},compact


def test_original_batch_binding_excludes_warmup_and_future_restore_feature(tmp_path):
    cp,mp,manifests,_=inputs(tmp_path)
    rows,provenance=load_bound_rows(cp,mp,manifests)
    assert len(rows)==4 and {r["batch_id"] for r in rows}=={"run:2","run:3"}
    assert all(r["host_load_tokens"]==0 for r in rows)
    assert sum(r["actual_host_loaded_tokens"]>0 for r in rows)==2
    assert provenance["stage_compact"]["warmup_requests_excluded"]==2
    assert provenance["stage_compact"]["prepared_vs_actual_host_load_changed_requests"]==2


@pytest.mark.parametrize("change", ["order","negative_stage","case_budget"])
def test_binding_refuses_inconsistent_compact_source(tmp_path,change):
    cp,mp,manifests,compact=inputs(tmp_path)
    if change=="order":
        compact[2],compact[3]=compact[3],compact[2]
    elif change=="negative_stage":
        compact[2]["forward_to_first_token_ms"]=compact[2]["server_ttft_ms"]+1
    else:
        compact[2]["case"]={**compact[2]["case"],"output_tokens":1024}
    cp.write_text("".join(json.dumps(row)+"\n" for row in compact))
    with pytest.raises(ValueError):
        load_bound_rows(cp,mp,manifests)


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
    assert stage["forward"]["independent_feature_cells"]<8
    assert all(p["error_ms"]==stage["forward"]["empirical_error_p99_ms"] for p in artifact["prefill_points"])
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



def mixed_inputs(tmp_path, *, drift=False):
    cp, mp, manifests, compact = inputs(tmp_path)
    manifest = json.loads(manifests["group"].read_text())
    case = dict(manifest["cases"][0],cache_state="mixed",device_prefix_tokens=256)
    manifest["cases"] = [case]
    manifests["group"].write_text(json.dumps(manifest))
    for row in compact:
        row["case"] = case
        row["actual_device_hit_tokens"] = 0 if drift else 256
        row["actual_host_loaded_tokens"] = 512 if drift else 256
    cp.write_text("".join(json.dumps(row)+"\n" for row in compact))
    measurements = [json.loads(line) for line in mp.read_text().splitlines()]
    for row in measurements:
        row.update(cache_state="mixed",host_load_tokens=512 if drift else 256)
    mp.write_text("".join(json.dumps(row)+"\n" for row in measurements))
    return cp, mp, manifests


def test_mixed_stage_features_use_the_declared_split_with_exact_receipts(tmp_path):
    cp, mp, manifests = mixed_inputs(tmp_path)
    rows, provenance = load_bound_rows(cp,mp,manifests)
    assert len(rows)==4
    assert all(row["host_load_tokens"]==row["prepared_device_prefix_tokens"]==256 for row in rows)
    assert provenance["stage_compact"]["groups"]["group"] == {
        "prefix_tier_binding":["prepared_and_execution_exact"],"formal_requests":4,
        "prepared_vs_actual_host_load_changed_requests":0}


def test_mixed_stage_refuses_tier_drift_even_if_total_prefix_and_measurements_agree(tmp_path):
    cp, mp, manifests = mixed_inputs(tmp_path,drift=True)
    with pytest.raises(ValueError,match="mixed device/host split changed"):
        load_bound_rows(cp,mp,manifests)
