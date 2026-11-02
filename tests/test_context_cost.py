from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from controller.context_cost import quote_read_costs
from controller.cost_model import CostModel, DecodePoint, LookupPoint
from controller.reading import build_read_plan
from controller.schema import CacheQuote, CacheQuoteBatch, SessionSnapshot
from controller.templates import ByteTemplate
from test_context_planning import context_plan
from profiling.reading_fit import fit_reading_priors, fit_step_models


@pytest.mark.asyncio
async def test_resume_quote_skips_completed_reads_and_renders_durable_parent_claims():
    from controller.context_schema import EvidenceClaim, ReadResult, StepRecord
    from controller.reading import step_messages

    template, snapshot = ByteTemplate(), SessionSnapshot("s")
    original, _ = context_plan()
    plan = build_read_plan(original.task_state, original.selected_set, "selected_sources",
                           original.budget, template=template, snapshot=snapshot)
    first = plan.steps[0]
    leaf = EvidenceClaim(first.step_id+":claim:0", "need", "extract", text="parent fact",
                         source_spans=plan.selected_set.selected_units[0].source_spans)
    complete = StepRecord("attempt", first.step_id, first, result=ReadResult(first.step_id, claims=(leaf,)))
    hop = replace(first, step_id="hop:1:read:0", input_claim_ids=(leaf.claim_id,))
    plan = replace(plan, steps=(first, hop), reading_mode="coverage_only")
    expected = template.render(step_messages(plan, hop, plan.selected_set.selected_units,
                                            (leaf,), snapshot=snapshot))
    class Engine:
        async def quote_batch(self, candidates, fingerprint):
            assert len(candidates) == 1
            candidate = candidates[0]
            assert candidate.input_ids == expected.input_ids
            assert candidate.generation_kind == "READ_COVERAGE"
            return CacheQuoteBatch("batch", 1, 1, "r", "c", 100000, 100000,
                (CacheQuote(candidate.candidate_id, candidate.input_hash, len(candidate.input_ids), 0),),
                fingerprint=fingerprint)
    _, trace = await quote_read_costs((plan,), snapshot, template, Engine(), CostModel(),
                                    completed_steps=(complete,))
    assert len(trace["steps"]) == 1 and trace["unquoted"] == []


@pytest.mark.asyncio
async def test_unknown_future_claim_input_is_not_quoted_as_a_known_read():
    template, snapshot = ByteTemplate(), SessionSnapshot("s")
    original, _ = context_plan()
    plan = build_read_plan(original.task_state, original.selected_set, "selected_sources",
                           original.budget, template=template, snapshot=snapshot)
    plan = replace(plan, steps=(replace(plan.steps[0], input_claim_ids=("future:claim",)),))
    class Engine:
        async def quote_batch(self, *args):
            raise AssertionError("unknown future input must not receive a native quote")
    updated, trace = await quote_read_costs((plan,), snapshot, template, Engine(), CostModel())
    assert not updated[0].cost_prediction_eligible and not trace["steps"]
    assert trace["unquoted"][0]["unknown_parent_claim_ids"] == ("future:claim",)


def test_read_output_length_uses_its_own_development_profile():
    plan, _ = context_plan()
    template = ByteTemplate()
    from controller.candidates import build_plan_candidates
    candidate = build_plan_candidates(SessionSnapshot("s"), "How?", plan, template)[0]
    candidate = replace(candidate, generation_kind="READ", output_budget=2048)
    quote = CacheQuote(candidate.candidate_id, candidate.input_hash, len(candidate.input_ids), len(candidate.input_ids)-1)
    model = CostModel([LookupPoint(0, len(candidate.input_ids), 1, 10)],
        [DecodePoint(len(candidate.input_ids)+16, 1, 1)],
        step_output_priors={"READ:2048": {"mean_tokens": 20, "p90_tokens": 30}})
    prediction = model.predict_native(candidate, quote)
    assert prediction.predicted_output_tokens == 20
    assert prediction.p90_output_tokens == 30
    model.step_output_priors = {}
    assert not model.predict_native(candidate, quote).calibrated
    assert model.predict_native(candidate, quote).predicted_output_tokens == 2048


@pytest.mark.asyncio
async def test_all_known_read_inputs_from_all_sets_receive_independent_native_quotes():
    template, snapshot = ByteTemplate(), SessionSnapshot("s")
    plan, _ = context_plan()
    plans = tuple(build_read_plan(replace(plan.task_state, task_id=name),
        replace(plan.selected_set, set_id=name), "selected_sources", plan.budget,
        template=template, snapshot=snapshot) for name in ("one", "two"))

    class Engine:
        async def quote_batch(self, candidates, fingerprint):
            assert len(candidates) == 2
            assert len({c.candidate_id for c in candidates}) == 2
            assert all(c.generation_kind == "READ" for c in candidates)
            return CacheQuoteBatch("batch", 1, 1, "r", "c", 100000, 100000,
                tuple(CacheQuote(c.candidate_id, c.input_hash, len(c.input_ids), len(c.input_ids)-1)
                      for c in candidates), fingerprint=fingerprint)

    updated, trace = await quote_read_costs(plans, snapshot, template, Engine(), CostModel())
    assert len(trace["steps"]) == 2
    assert all(not p.cost_prediction_eligible for p in updated)


def test_step_length_fitting_uses_only_recorded_development_generations():
    rows = [{"split": "dev", "task_id": "t", "reading_outcome": {"steps": [
        {"task": {"kind": "READ", "output_budget": 2048}, "input_ids": [1], "output_ids": [1]*20},
        {"task": {"kind": "REDUCE", "output_budget": 1024}, "input_ids": [1], "output_ids": [1]*10}]}}]
    assert fit_reading_priors(rows)["READ:2048"]["mean_tokens"] == 20
    with pytest.raises(ValueError):
        fit_reading_priors([{**rows[0], "split": "test"}])


def test_independent_step_model_survives_serialization_and_keeps_step_budget_prior():
    from controller.candidates import build_plan_candidates

    plan, _ = context_plan()
    candidate = build_plan_candidates(SessionSnapshot("s"), "How?", plan, ByteTemplate())[0]
    n = len(candidate.input_ids)
    candidate = replace(candidate, generation_kind="READ", output_budget=2048)
    quote = CacheQuote(candidate.candidate_id, candidate.input_hash, n, n-1)
    parent = CostModel(default_output_tokens=900)
    parent.step_models["READ"] = CostModel([LookupPoint(0, n, 1, 10)],
        [DecodePoint(n+15, 1, 2, context_start=n, context_end=n+2048)], step_output_priors={"READ:2048": {
            "mean_tokens": 20, "p90_tokens": 30, "mean_control_ms": 7}})
    parent.step_models["READ"].calibration_metadata = {"generation_kind": "READ"}
    restored = CostModel.from_spec(parent.to_spec())
    prediction = restored.predict_native(candidate, quote)
    assert prediction.predicted_output_tokens == 20
    assert prediction.control_ms == 7
    assert prediction.decode_ms == 38
    assert prediction.calibrated
    del restored.step_models["READ"]
    restored.step_output_priors = {"READ:2048": {"mean_tokens": 20, "p90_tokens": 30}}
    assert not restored.predict_native(candidate, quote).calibrated


def observed_step_profile(tmp_path, *, reading_mode="coverage_only", operation="extract"):
    from controller.reading import COVERAGE_READ_SYSTEM, READ_SYSTEM, SUMMARY_READ_SYSTEM, step_generation_kind
    from controller.schema import ConfigFingerprint
    fingerprint = asdict(ConfigFingerprint())
    kind = step_generation_kind("READ", operation, reading_mode)
    prompt = {"READ_COVERAGE": COVERAGE_READ_SYSTEM, "READ": READ_SYSTEM,
              "READ_SUMMARY": SUMMARY_READ_SYSTEM}[kind]
    manifest = {"run_id": "operator-profile-name", "fingerprint": fingerprint,
                "engine": {"cache_config_hash": "cache-observed"}, "sampling_recipe": {"temperature": 0}}
    path = tmp_path/"reading-profile.json"
    path.write_text(json.dumps(manifest))
    row = {"task_id": "dev-read", "split": "dev", "reading_outcome": {
        "plan": {"reading_mode": reading_mode}, "steps": [{"step_id": "read:0",
            "task": {"kind": "READ", "operation": operation, "output_budget": 128},
            "input_messages": [{"role": "system", "content": prompt}, {"role": "user", "content": "original source"}],
            "input_ids": [1]*16, "output_ids": [2]*4,
            "receipt": {"actual_device_hit_tokens": 5, "actual_host_loaded_tokens": 3},
            "metrics": {"generation_kind": kind, "cache_quote": {"active_requests": 2, "queued_requests": 7,
                "fingerprint": fingerprint, "cache_config_hash": "cache-observed", "request_id": "actual-quote-id",
                "observed_seq": 19, "engine_restart_id": "observed-engine"},
                "ttft_ms": 12, "last_token_ms": 18, "step_wall_ms": 25, "elapsed_ms": 20}}]}}
    return row, path


def test_read_coverage_prior_uses_generation_kind_and_control_phase_scope(tmp_path):
    row, _ = observed_step_profile(tmp_path)
    prior = fit_reading_priors([row])
    assert set(prior) == {"READ_COVERAGE:128"}
    assert prior["READ_COVERAGE:128"]["mean_control_ms"] == 7
    assert prior["READ_COVERAGE:128"]["control_steps"] == 1
    del row["reading_outcome"]["steps"][0]["metrics"]["last_token_ms"]
    unmeasured = fit_reading_priors([row])["READ_COVERAGE:128"]
    assert "mean_control_ms" not in unmeasured
    assert unmeasured["control_steps"] == 0


def test_reading_calibration_binds_observed_quote_and_prompt_without_inventing_run_id(tmp_path):
    from controller.reading import COVERAGE_READ_SYSTEM
    row, manifest_path = observed_step_profile(tmp_path)
    records = tmp_path/"actual-steps.jsonl"
    records.write_text(json.dumps(row)+"\n")
    models, report = fit_step_models([row], manifest_path, records_path=records)
    child = CostModel.from_spec(models["READ_COVERAGE"])
    metadata = child.calibration_metadata
    binding = metadata["measurement_provenance"]
    assert binding["binding_status"] == "observed_step_configuration_and_prompt_matched"
    assert binding["run_ids"] == []
    assert binding["run_id_binding"] == "operator_declared_unverified"
    assert binding["observed_step_bindings"][0]["quote_request_id"] == "actual-quote-id"
    assert binding["files"]["measurements"]["path"] == str(records.resolve())
    assert metadata["output_prior_system_prompt"] == COVERAGE_READ_SYSTEM
    assert set(child.step_output_priors) == {"READ_COVERAGE:128"}
    assert child.step_output_priors["READ_COVERAGE:128"]["mean_control_ms"] == 7
    assert all(step["run_id"] == "" for step in report["step_ledger"])


@pytest.mark.parametrize("missing", ("cache_quote", "fingerprint", "cache_config_hash", "input_messages"))
def test_step_calibration_cannot_bind_missing_native_identity_or_prompt(tmp_path, missing):
    row, manifest_path = observed_step_profile(tmp_path)
    step = row["reading_outcome"]["steps"][0]
    if missing == "input_messages":
        del step["input_messages"]
    elif missing == "cache_quote":
        del step["metrics"]["cache_quote"]
    else:
        del step["metrics"]["cache_quote"][missing]
    with pytest.raises(ValueError, match="lacks observed|actual system prompt"):
        fit_step_models([row], manifest_path)


@pytest.mark.parametrize("conflict", ("cache", "fingerprint", "prompt", "generation_kind"))
def test_step_calibration_rejects_mixed_engine_or_prompt_bindings(tmp_path, conflict):
    row, manifest_path = observed_step_profile(tmp_path)
    changed = deepcopy(row)
    changed["task_id"] = "different-step-source"
    step = changed["reading_outcome"]["steps"][0]
    if conflict == "cache":
        step["metrics"]["cache_quote"]["cache_config_hash"] = "other-cache"
    elif conflict == "fingerprint":
        step["metrics"]["cache_quote"]["fingerprint"]["model_revision"] = "other-model"
    elif conflict == "prompt":
        step["input_messages"][0]["content"] = "Different system instructions"
    else:
        step["metrics"]["generation_kind"] = "READ"
    with pytest.raises(ValueError, match="differs"):
        fit_step_models([row, changed], manifest_path)


def test_step_profiles_keep_summary_and_coverage_observations_separate(tmp_path):
    coverage, manifest_path = observed_step_profile(tmp_path)
    summary, _ = observed_step_profile(tmp_path, operation="summarize")
    summary["task_id"] = "dev-summary"
    models, _ = fit_step_models([coverage, summary], manifest_path)
    assert set(models) == {"READ_COVERAGE", "READ_SUMMARY"}
    for kind, spec in models.items():
        child = CostModel.from_spec(spec)
        assert set(child.step_output_priors) == {kind+":128"}


def test_step_calibration_requires_observed_native_timing(tmp_path):
    row, manifest_path = observed_step_profile(tmp_path)
    del row["reading_outcome"]["steps"][0]["metrics"]["last_token_ms"]
    with pytest.raises(ValueError, match="no observed native step timing"):
        fit_step_models([row], manifest_path)


def test_step_calibration_requires_manifest_engine_identity(tmp_path):
    row, manifest_path = observed_step_profile(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    del manifest["engine"]["cache_config_hash"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest requires fingerprint and cache_config_hash"):
        fit_step_models([row], manifest_path)
