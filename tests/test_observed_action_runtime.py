"""CPU native-wire fixtures exercise selection and durable tool accounting, not GPU results."""
from dataclasses import replace
import json

import pytest

from bench.mock_engine import MockEngine
from controller.context_schema import EvidencePool
from controller.plan_policy import PLAN_KINDS, OBSERVED_FEATURE_NAMES, OBSERVED_FEATURE_SCHEMA
from controller.rolling_policy import RollingPlanRuntime
from training.collect_plan_gains import export_observed_packets
from policy_helpers import SpecifiedGainFixture
from reading_helpers import material
from rolling_helpers import NativeObservationFixture, setup_controller


class LearnedFixture(SpecifiedGainFixture):
    """Declared test coefficients/costs with the same strict action and budget schema as the current asset."""
    kinds = {"direct_answer", "read_once", "reduce_once", "gap_retrieval"}

    def __init__(self, reads=1, gap_gain=-.1, reduce_gain=-.1, direct_ms=10, read_ms=20):
        self.reads, self.gap_gain, self.reduce_gain = reads, gap_gain, reduce_gain
        self.direct_ms, self.read_ms = direct_ms, read_ms
        from controller.action_proposals import BOUNDED_CONTINUATION
        self.feature_names = OBSERVED_FEATURE_NAMES
        self.cost_provenance = {"observed_feature_schema": OBSERVED_FEATURE_SCHEMA,
            "prefix_control_scope": "native_excludes_prefix_control"}
        self.provenance = {"observed_feature_schema": OBSERVED_FEATURE_SCHEMA,
            "continuation_protocol": BOUNDED_CONTINUATION,
            "continuation_gap_limits": {"max_units": 2, "max_passages": 2}}
        self.quality_calls = []

    def supports_semantic_kind(self, kind, *, costs=False):
        return kind in self.kinds

    def _kind(self, features):
        kind = next(kind for kind in PLAN_KINDS if features[kind])
        if kind not in self.kinds:
            raise ValueError("fixture has not learned this semantic kind: "+kind)
        return kind

    def predict(self, features):
        kind = self._kind(features)
        self.quality_calls.append(kind)
        return ({"gap_retrieval": self.gap_gain, "reduce_once": self.reduce_gain,
                 "read_once": .8 if features["existing_claims"] < self.reads else -.1,
                 "direct_answer": 0}[kind])

    def predict_costs(self, features):
        kind = self._kind(features)
        ms = self.direct_ms if kind == "direct_answer" else self.read_ms
        return {"branch_complete_cost": {"complete_ms": ms, "gpu_gib_seconds": .002,
            "host_gib_seconds": .003, "total_input_tokens": 10000 if kind == "read_once" else 5000,
            "total_output_tokens": 512 if kind == "read_once" else 64},
            "incremental_complete_cost": {"complete_ms": 0 if kind == "direct_answer" else -100,
            "gpu_gib_seconds": 0, "host_gib_seconds": 0, "total_input_tokens": 0, "total_output_tokens": 0}}


class Observer(NativeObservationFixture):
    def __init__(self, paid_ms=None):
        super().__init__()
        self.paid_ms = paid_ms

    async def observe(self, plans, state, snapshot, engine, cost_model, arrived, usage, *, selection_started):
        native, reusable = await super().observe(plans, state, snapshot, engine, cost_model, arrived, usage,
                                                  selection_started=selection_started)
        extra = state.auxiliary_usage
        paid = replace(native.paid_cost,
            latency_ms=native.paid_cost.latency_ms if self.paid_ms is None else self.paid_ms,
            input_tokens=native.paid_cost.input_tokens+extra["input_tokens"],
            output_tokens=native.paid_cost.output_tokens+extra["output_tokens"])
        self.paid[-1] = paid
        return replace(native, paid_cost=paid), reusable


def configured(tmp_path, *, reads=2, model=None, scope="selected_sources", operation="procedure",
               observer=None, max_steps=64, max_ms=None):
    controller, store, engine, _, plan = setup_controller(tmp_path, reads=reads, scope=scope, operation=operation)
    model, observer = model or LearnedFixture(), observer or Observer()
    runtime = RollingPlanRuntime(model, observer, reader_checkpoint="reader-fixture", reader_profile_id="profile-fixture",
        output_budgets=(256, 512), mode="observed_actions", observation_path=tmp_path / "observations.jsonl",
        gap_retriever_identity={"implementation": "specified test retriever"},
        gap_limits={"max_units": 2, "max_passages": 2})
    controller.rolling_policy = runtime
    plan = replace(plan, budget=replace(plan.budget, max_steps=max_steps, max_complete_ms=max_ms,
                                       lambda_time=.001))
    return controller, store, engine, observer, plan


@pytest.mark.asyncio
async def test_after_each_read_learned_value_selects_stop_or_one_more_read(tmp_path):
    wanted = 1
    controller, store, engine, observed, plan = configured(tmp_path, reads=3, model=LearnedFixture(reads=wanted))
    response = await controller.serve_turn("s", "How?", context_plans=(plan,), request_id="actual-request")
    assert response.read_count == wanted and len(engine.constraints) == wanted+1
    assert [len(state.steps) for state in observed.states] == list(range(wanted+1))
    attempt = store.get_attempt(response.attempt_id)
    assert attempt.plan.reason == "rolling_observed_action_gain_cost"
    records = [json.loads(line) for line in (tmp_path / "observations.jsonl").read_text().splitlines()]
    assert all(record["attempt_id"] == response.attempt_id and record["request_id"] == "actual-request" for record in records)
    assert all("task_id" not in record for record in records)
    assert records[-1]["paid_cost"]["input_tokens"] > 0
    result = export_observed_packets(records, [{"task_id": "real-task-turn-id",
        "cohort_split": "train", "cohort_manifest_sha256": "specified-fixture",
        "response": {"attempt_id": response.attempt_id}}], tmp_path / "packets.jsonl")
    assert result["packets"] == wanted+1
    assert all(json.loads(line)["task_id"] == "real-task-turn-id" for line in
               (tmp_path / "packets.jsonl").read_text().splitlines())
    await controller.aclose()


class BridgeReader(MockEngine):
    def __init__(self, *, question_only=False, repeat=False):
        super().__init__()
        self.reads = 0
        self.question_only, self.repeat = question_only, repeat
        self.read_inputs = []

    async def generate(self, *, response_schema=None, **kwargs):
        if response_schema is not None:
            self.reads += 1
            payload = json.loads(bytes(kwargs["input_ids"]).decode().split("<|user|>\n")[-1].split("\n<|end|>")[0])
            self.read_inputs.append(payload)
            unit = next((unit for unit in payload["units"] if unit["unit_id"] == "new"), payload["units"][-1])
            protocol = payload["fact_protocol"]
            need = next(iter(protocol["needs"]))
            parents = list(protocol["parents"])
            source = next(alias for alias, identity in protocol["sources"].items()
                          if identity == unit["source_spans"][0]["span_id"])
            gap = {"n": need, "r": "start date", "e": ["R3"], "t": "current"}
            if self.question_only and self.reads == 1:
                claims, opened = [], [{**gap, "p": []}]
            else:
                value = {"n": need, "t": "bridge", "e": "R1" if self.reads == 1 else "R3",
                    "r": "parent" if self.reads == 1 else "start date",
                    "v": "R3" if self.reads == 1 else "2001", "u": "", "c": [], "s": [source],
                    "p": parents, "f": [gap] if self.reads == 1 or self.repeat else []}
                claims, opened = [value], []
            self.response_text = json.dumps({"format_version": protocol["format_version"], "claims": claims,
                "support": [] if self.reads == 1 or self.repeat else [need], "conflicts": [], "open_needs": opened},
                separators=(",", ":"))
        else:
            self.response_text = "Supported answer."
        return await super().generate(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("question_only", [False, True])
async def test_selected_gap_is_paid_once_and_new_read_keeps_old_dependency_proof(tmp_path, question_only):
    controller, store, _, observed, plan = configured(tmp_path, reads=1,
        model=LearnedFixture(reads=1 if question_only else 2, gap_gain=.9), max_steps=4)
    engine = BridgeReader(question_only=question_only, repeat=True)
    controller.engine = engine
    block, unit = material("R3 has a registration entry.", "new")
    unit = replace(unit, dependencies=(plan.selected_set.selected_units[0].unit_id,))
    calls = []

    async def provider(current, needs, claims):
        calls.append((current, needs, claims))
        assert bool(needs[0].parent_claim_ids) is not question_only
        return EvidencePool((unit,), (block,), auxiliary_usage={"input_tokens": 7,
            "output_tokens": 2, "compute_ms": 3, "models": []})

    response = await controller.serve_turn("s", "How?", context_plans=(plan,), open_need_provider=provider)
    assert len(calls) == 1 and response.read_count == 2
    outcome = store.get_attempt(response.attempt_id).reading_outcome
    assert sum(record.task.kind == "GAP" for record in outcome.steps) == 1
    assert len(outcome.steps)+1 == plan.budget.max_steps
    newest = outcome.claims[-1]
    assert bool(newest.parent_claim_ids) is not question_only
    assert {span.source_key for span in newest.source_spans} == {
        block.key, plan.selected_set.source_blocks[0].key}
    assert {unit["unit_id"] for unit in engine.read_inputs[-1]["units"]} == {"0", "new"}
    assert response.auxiliary_input_tokens == 7 and response.auxiliary_output_tokens == 2
    assert observed.paid[-1].input_tokens == sum(len(record.input_ids) for record in outcome.steps)+7
    assert outcome.open_needs  # repeating the same relation with a new parent does not trigger another paid GAP
    await controller.aclose()


@pytest.mark.asyncio
async def test_mandatory_coverage_uses_native_fixed_path_without_untrained_whole_plan_gain_or_gap(tmp_path):
    model = LearnedFixture(reads=0, gap_gain=.9)
    controller, _, _, observed, plan = configured(tmp_path, reads=2, scope="all_supplied_documents", model=model)
    controller.engine = BridgeReader(repeat=True)
    calls = []

    async def provider(*args):
        calls.append(args)
        raise AssertionError("full supplied-source scope cannot expand itself")

    response = await controller.serve_turn("s", "How?", context_plans=(plan,), open_need_provider=provider)
    assert response.read_count == 2 and not calls
    assert not {"read_once", "gap_retrieval"}.intersection(model.quality_calls)
    assert len(observed.states[-1].results) == 2
    await controller.aclose()


@pytest.mark.asyncio
async def test_absolute_branch_plus_paid_constrains_budget_even_when_signed_difference_is_negative(tmp_path):
    controller, _, _, _, plan = configured(tmp_path, reads=1, max_ms=1000,
        observer=Observer(paid_ms=900), model=LearnedFixture(direct_ms=100, read_ms=200))
    response = await controller.serve_turn("s", "How?", context_plans=(plan,))
    assert response.read_count == 0
    assert any(item.reason == "complete_latency_budget" for item in controller.rolling_policy.policy.last_evaluations)
    await controller.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("control_ms", [None, 7.0])
async def test_first_read_adds_only_measured_prefix_control_once(tmp_path, control_ms):
    controller, _, _, _, plan = configured(tmp_path, reads=1, model=LearnedFixture(reads=0),
        observer=Observer(paid_ms=0))

    class PrefixControl:
        control_observation_count = 0 if control_ms is None else 3

        def predicted_control_ms(self):
            return control_ms

    controller.observe_prefix = PrefixControl()
    plan = replace(plan, read_layout="source_first")
    response = await controller.serve_turn("s", "How?", context_plans=(plan,))
    assert response.read_count == 0
    policy = controller.rolling_policy.policy
    read = next(item for item in policy.last_evaluations if item.plan_id.endswith(":read_once"))
    direct = next(item for item in policy.last_evaluations if item.plan_id.endswith(":direct_answer"))
    assert read.cost.latency_ms == 20+(control_ms or 0)
    assert direct.cost.latency_ms == 10
    diagnostic = next(item for item in policy.last_option_diagnostics.values() if item["plan_id"] == read.plan_id)
    assert diagnostic["first_read_prefix_control"] == {"predicted_ms": control_ms,
        "source": "actual_prefix_control_ms", "observation_count": PrefixControl.control_observation_count,
        "scope": "first_READ_only"}
    await controller.aclose()
