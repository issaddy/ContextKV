"""CPU protocol fixtures exercise the real Controller transaction and loop."""
from rolling_helpers import setup_controller

import pytest

from policy_helpers import SpecifiedGainFixture
from reading_helpers import material


@pytest.mark.asyncio
async def test_controller_reselects_after_each_read_and_commits_one_qa(tmp_path):
    controller, store, engine, observed, plan = setup_controller(tmp_path, reads=2)
    response = await controller.serve_turn("s", "How?", context_plans=(plan,), request_id="r")
    assert response.read_count == 2 and response.reduce_count == 0
    assert len(observed.states) == 3
    assert [len(state.steps) for state in observed.states] == [0, 1, 2]
    assert observed.paid[-1].input_tokens > observed.paid[1].input_tokens > 0
    assert response.total_input_tokens == response.reader_input_tokens
    assert len(store.get_snapshot("s").raw_history) == store.get_snapshot("s").version == 1
    window = response.answer_phase_window
    assert window["start"] == observed.selection_starts[-1]
    assert window["start"] < window["commit_at"] <= window["end"]
    import json
    committed = next(json.loads(line) for line in (tmp_path/"events.jsonl").read_text().splitlines()
                     if json.loads(line)["event"] == "ledger_committed")
    assert committed["answer_phase_window"] == window
    attempt = store.find_request("s", "r")
    assert attempt.plan.reason == "rolling_plan_gain"
    assert len({row.step_id for row in attempt.reading_outcome.steps}) == 2
    assert all(row.step_id.startswith("policy:") for row in attempt.reading_outcome.steps)
    # The chosen output limit is present in the prompt itself, so each variant
    # must be rendered with that same limit before it is quoted and executed.
    for row in attempt.reading_outcome.steps:
        assert row.task.output_budget == 256
        assert '"extraction_output_budget_tokens":256' in bytes(row.input_ids).decode()
    calls = len(engine.constraints)
    replay = await controller.serve_turn("s", "How?", context_plans=(plan,), request_id="r")
    assert replay.idempotent_replay and len(engine.constraints) == calls
    assert replay.answer_phase_window is None
    await controller.aclose()


@pytest.mark.asyncio
async def test_full_scope_forbids_direct_when_originals_do_not_fit_the_answer(tmp_path):
    from dataclasses import replace
    class NegativeGain(SpecifiedGainFixture):
        def predict(self, features):
            return -.5
    controller, _, _, observed, plan = setup_controller(tmp_path, scope="all_supplied_documents", reads=2,
                                                       gain=NegativeGain())
    plan = replace(plan, budget=replace(plan.budget, body_tokens=0))
    response = await controller.serve_turn("s", "How?", context_plans=(plan,))
    assert response.read_count == 2
    assert len(observed.states[-1].results) == 2
    assert {unit for result in observed.states[-1].results for unit in result.covered_units} == {"0", "1"}
    await controller.aclose()


@pytest.mark.asyncio
async def test_observed_open_need_proposes_priced_multihop_with_real_parent_and_auxiliary_cost(tmp_path):
    import json
    from bench.mock_engine import MockEngine
    from controller.context_schema import EvidencePool
    controller, store, _, observed, plan = setup_controller(tmp_path)
    controller.rolling_policy.output_budgets = (512,)
    block, unit = material("The prerequisite is the current form.", "new")

    class FollowupEngine(MockEngine):
        def __init__(self):
            super().__init__()
            self.model_reads = 0
        async def generate(self, *, response_schema=None, **kwargs):
            if response_schema:
                self.model_reads += 1
                properties = response_schema["properties"]["claims"]["items"]["properties"]
                parents = [] if self.model_reads == 1 else properties["p"]["items"]["enum"]
                value = {"n": properties["n"]["enum"][0], "t": "prerequisite", "e": "item", "r": "requires",
                    "v": "form", "u": "", "c": [], "s": [properties["s"]["items"]["enum"][0]],
                    "p": parents, "f": ([{"n": "n0", "r": "prerequisite", "e": ["item"], "t": "current"}]
                        if self.model_reads == 1 else [])}
                self.response_text = json.dumps({"format_version": response_schema["properties"]["format_version"]["const"],
                    "claims": [value], "support": [] if self.model_reads == 1 else ["n0"],
                    "conflicts": [], "open_needs": []}, separators=(",", ":"))
            else:
                self.response_text = "The prerequisite form is required."
            return await super().generate(**kwargs)

    controller.engine = FollowupEngine()
    calls = []
    async def gap_provider(current, needs, claims):
        calls.append((current, needs, claims))
        return EvidencePool((unit,), source_blocks=(block,),
            auxiliary_usage={"input_tokens": 7, "output_tokens": 2, "compute_ms": 3.0, "models": []})
    response = await controller.serve_turn("s", "How?", context_plans=(plan,), open_need_provider=gap_provider)
    assert response.read_count == 2 and len(calls) == 1
    outcome = store.get_attempt(response.attempt_id).reading_outcome
    assert outcome.steps[-1].task.input_claim_ids == (outcome.claims[0].claim_id,)
    assert outcome.claims[-1].parent_claim_ids == (outcome.claims[0].claim_id,)
    assert response.auxiliary_input_tokens == 7 and response.auxiliary_output_tokens == 2
    assert outcome.open_needs == ()
    assert len(observed.states) == 3
    await controller.aclose()
