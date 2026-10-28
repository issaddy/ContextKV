"""Session-level heldout labels cannot influence their own length prediction."""
from dataclasses import asdict, replace

import pytest

pytest.importorskip("numpy")

from controller.output_length import OutputLengthModel, candidate_features
from controller.schema import ConfigFingerprint, Message, PromptCandidate, SessionSnapshot, TurnRecord
from profiling.length_fit import LengthCase, build_cases, fit_delta, fit_exploration, fit_regression, predict_cases, predict_delta_cases


def examples():
    cases = []
    for index in range(4):
        for turn in range(2):
            session = SessionSnapshot(f"s{index}", raw_history=(TurnRecord(1, "previous", "answer", (1,) * (100 + 10 * index)),))
            candidates = tuple(PromptCandidate(f"{index}:{turn}:{action}", action, 1,
                               (Message("user", f"\nQuestion:\nHow configure topic {index}?\n"),),
                               tuple(range(200 + 10 * turn + (80 if action == "APPEND" else 0))),
                               f"hash-{index}-{turn}-{action}", frozenset(), frozenset(), fingerprint=ConfigFingerprint())
                               for action in ("APPEND", "REBUILD"))
            targets = {"APPEND": 180 + index * 10 + turn, "REBUILD": 120 + index * 10 + turn}
            cases.append(LengthCase(f"{index}:{turn}", session.session_id, session, candidates,
                                   targets, {a: (n, n) for a, n in targets.items()}, 1024,
                                   {"APPEND": 0, "REBUILD": 0}))
    return cases


def test_ridge_fits_case_arms_and_preserves_scope():
    cases = examples()
    spec = fit_regression(cases, "action_only", .01, "linear", (150, 200))
    model = OutputLengthModel(spec)
    predictions = predict_cases(cases, model, (150, 200))
    assert all(p["predicted_delta_tokens"] > 0 for p in predictions)
    assert spec["target"]["length_semantics"] == "served_output_tokens_capped_not_natural_eos"


def test_duplicate_repetitions_do_not_become_independent_cases():
    case = examples()[0]
    frozen = {"case_id": case.case_id, "conversation_id": case.session_id,
              "snapshot": asdict(case.snapshot), "candidates": [asdict(c) for c in case.candidates],
              "max_new_tokens": 1024}
    rows = []
    for repetition in range(3):
        observations = {c.action: {"input_hash": c.input_hash, "input_tokens": len(c.input_ids),
                         "output_tokens": 1024 if c.action == "APPEND" else 100,
                         "result": {"finish_reason": "length" if c.action == "APPEND" else "stop"}}
                        for c in case.candidates}
        rows.append({"case_id": case.case_id, "conversation_id": case.session_id,
                     "cache_state": "cold", "repetition": repetition, "status": "completed",
                     "measurement_source": "hardware", "observations": observations})
    grouped, failures = build_cases([frozen], rows)
    assert not failures and len(grouped) == 1
    assert grouped[0].targets == {"APPEND": 1024, "REBUILD": 100}
    assert grouped[0].truncated_counts["APPEND"] == 3
    assert candidate_features(grouped[0].candidates[0], grouped[0].snapshot) == candidate_features(case.candidates[0], case.snapshot)
    with pytest.raises(ValueError, match="duplicate"):
        build_cases([frozen], rows + [rows[0]])


def test_outer_session_outputs_cannot_change_own_oof_prediction():
    cases = examples()
    _, before = fit_exploration(cases, priors=(150, 200))
    changed = [replace(c, targets={a: 900 for a in c.targets}, observations={a: (900, 900) for a in c.targets})
               if c.session_id == "s0" else c for c in cases]
    artifacts, after = fit_exploration(changed, priors=(150, 200))
    for kind in ("shared_history", "action_only", "candidate", "paired_delta"):
        first = [(p["predicted_tokens"], p["p90_tokens"], p["error_tokens"])
                 for p in before["models"][kind]["oof_predictions"] if p["session_id"] == "s0"]
        second = [(p["predicted_tokens"], p["p90_tokens"], p["error_tokens"])
                  for p in after["models"][kind]["oof_predictions"] if p["session_id"] == "s0"]
        assert first == second
        assert all(f["heldout_session"] not in f["training_sessions"] for f in after["models"][kind]["folds"])
    assert artifacts["candidate"]["uncertainty"]["confidence_guarantee"] is False
    assert artifacts["candidate"]["metadata"]["default_enabled"] is False


def test_direct_delta_uses_pair_inputs_and_shared_mean_with_range_fallback():
    cases = examples()
    spec = fit_delta(cases, .1)
    predictions = predict_delta_cases(cases, spec, (150, 200))
    assert all(p["predicted_delta_tokens"] == 60 for p in predictions)
    assert spec["deployment"].startswith("offline_exploration")
    extended = replace(cases[0], candidates=(replace(cases[0].candidates[0], input_ids=(1,) * 5000), cases[0].candidates[1]))
    outside = predict_delta_cases([extended], spec, (150, 200))[0]
    assert outside["predicted_delta_tokens"] == 0
    assert not any(outside["within_training_range"].values())


def test_small_session_pool_is_not_misrepresented_as_loso():
    with pytest.raises(ValueError, match="three independent sessions"):
        fit_exploration(examples()[:2])


@pytest.mark.parametrize("prompt", [None, "custom concise policy"])
def test_fit_cli_reuses_effective_serving_prior_and_loads_its_artifact(tmp_path, prompt):
    import asyncio
    from hashlib import sha256
    import json
    from controller.config import RunConfig, create_controller
    from controller.cost_model import CostModel
    from controller.generation import GenerationProtocol
    from controller.serialization import dumps
    from profiling.length_fit import main

    config = RunConfig(state_dir=str(tmp_path / "state"))
    config = replace(config, context=replace(config.context, max_new_tokens=1024, system_prompt=prompt))
    protocol = GenerationProtocol.from_config(config)
    cases_path, pairs_path = tmp_path / "cases.jsonl", tmp_path / "pairs.jsonl"
    frozen, rows = [], []
    for case in examples():
        candidates = [replace(c, messages=(Message("system", protocol.system_prompt), *c.messages))
                      for c in case.candidates]
        frozen.append({"case_id": case.case_id, "conversation_id": case.session_id,
            "snapshot": case.snapshot, "candidates": candidates, "max_new_tokens": case.max_new_tokens})
        rows.append({"case_id": case.case_id, "conversation_id": case.session_id,
            "cache_state": "cold", "repetition": 0, "status": "completed", "measurement_source": "hardware",
            "observations": {c.action: {"input_hash": c.input_hash, "input_tokens": len(c.input_ids),
                "output_tokens": case.targets[c.action], "result": {"finish_reason": "stop"}} for c in candidates}})
    cases_path.write_text("".join(dumps(row) + "\n" for row in frozen))
    pairs_path.write_text("".join(dumps(row) + "\n" for row in rows))
    manifest = {"split": "dev", "cases_sha256": sha256(cases_path.read_bytes()).hexdigest(),
                "generation_protocol": protocol.record()}
    cases_path.with_suffix(".manifest.json").write_text(json.dumps(manifest))
    pairs_manifest = {"split": "dev", "config": asdict(config), "frozen": manifest,
                      "generation_protocol": protocol.record()}
    pairs_path.with_suffix(".manifest.json").write_text(json.dumps(pairs_manifest))
    baseline, output, report = tmp_path / "cost.json", tmp_path / "length.json", tmp_path / "report.json"
    CostModel(default_output_tokens=580, p90_output_tokens=1024).save(baseline,
        metadata={"output_prior_system_prompt": "old verbose prompt"})
    args = ["--cases", str(cases_path), "--pairs", str(pairs_path), "--baseline-cost-artifact", str(baseline),
            "--output", str(output), "--report", str(report)]
    main(args)
    config = replace(config, cost_model=replace(config.cost_model, artifact=str(baseline),
                                                output_length_artifact=str(output)))
    controller = create_controller(config)
    try:
        fitted = json.loads(output.read_text())
        assert controller.cost_model.effective_output_prior == tuple(fitted["metadata"]["global_priors"]) == (128, 256)
        assert fitted["target"]["generation_protocol"] == protocol.record()
        assert controller.cost_model.output_length_model is not None
        assert fitted["metadata"]["training_prior_sampling"]["output_prior_prompt_status"] == "unbound_or_different_prompt"
        with pytest.raises(ValueError, match="generation protocol"):
            create_controller(replace(config, context=replace(config.context, system_prompt="another policy")))
    finally:
        asyncio.run(controller.aclose())
        controller.store.close()
    # Old unconstrained labels must be recollected, not relabeled as serving data.
    pairs_manifest.pop("generation_protocol")
    pairs_path.with_suffix(".manifest.json").write_text(json.dumps(pairs_manifest))
    args[args.index("--output") + 1] = str(tmp_path / "legacy-length.json")
    args[args.index("--report") + 1] = str(tmp_path / "legacy-report.json")
    with pytest.raises(ValueError, match="recollect legacy unconstrained pairs"):
        main(args)
