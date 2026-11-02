"""Token/control decomposition and fitted-asset binding, using unit fixtures."""
from dataclasses import asdict
import json

import pytest

from controller.cost_model import CostModel
from controller.sampling import SamplingRecipe
from controller.schema import CacheQuote, ConfigFingerprint, Message, PromptCandidate
from controller.templates import token_hash
from profiling.plan_native_fit import fit_native_assets, timing_phases


def observed(kind="ANSWER"):
    return dict(observation_id="fixture-phase", generation_kind=kind,
        generation_result={"ttft_ms": 40, "last_token_ms": 90}, latency_ms=100,
        receipt={"actual_device_hit_tokens": 0, "actual_host_loaded_tokens": 0},
        input_tokens=1000, output_tokens=5, output_budget=64, concurrency=1,
        cache_config_hash="fixture-cache", fingerprint=asdict(ConfigFingerprint()),
        sampling_recipe=SamplingRecipe(.7, .8, 20, 0).record(), system_prompt="actual fixture prompt",
        system_prompt_sha256="fixture-prompt-hash", split="dev", source_files=["fixture-wire"],
        source_sha256={"fixture-wire": "fixture-sha"}, quote_binding={"observed_seq": 1},
        resource_scope={"fixture": True}, resident_model_manifest=None,
        phase_window={"start_definition": "rolling_proposals_compilation_start"})


def test_native_fit_keeps_real_ttft_and_decode_then_adds_control_once(tmp_path):
    row = observed()
    first, decode, control = timing_phases(row)
    assert (first.elapsed_ms, decode.elapsed_ms, control.elapsed_ms) == (40, 50, 10)
    assets = fit_native_assets([row], tmp_path)
    model = CostModel.load(assets["base_artifact"])
    model.configure_sampling(SamplingRecipe(.7, .8, 20, 0), fallback_mean=128,
                             fallback_p90=256, system_prompt=row["system_prompt"])
    model.check_calibration_scope({"fingerprint": row["fingerprint"], "cache_config_hash": "fixture-cache"})
    ids = (1,)*1000
    candidate = PromptCandidate("answer", "REBUILD", 0, (Message("system", row["system_prompt"]),),
        ids, token_hash(ids), frozenset(), frozenset(), output_budget=64)
    quote = CacheQuote("answer", candidate.input_hash, 1000, 999)
    estimate = model.predict_native(candidate, quote)
    assert estimate.calibrated and estimate.cost == 100 and estimate.control_ms == 10
    measurements = [json.loads(line) for line in (tmp_path/"answer-measurements.jsonl").read_text().splitlines()]
    assert measurements[0]["elapsed_ms"] == 40 and measurements[-1]["phase"] == "control"


def test_missing_stream_timings_do_not_emit_native_assets_or_fabricated_phase_cost(tmp_path):
    row = observed("REDUCE")
    row["generation_result"]["ttft_ms"] = None
    with pytest.raises(ValueError, match="unobserved"):
        timing_phases(row)
    assets = fit_native_assets([row], tmp_path)
    assert assets["base_artifact"] is None and not assets["step_artifacts"]
    assert assets["available_kinds"] == [] and assets["excluded"]
