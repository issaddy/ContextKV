"""CPU protocol fixtures exercise action collection; these are not GPU reward measurements."""
from copy import deepcopy
from dataclasses import replace
import json

import httpx
import pytest
import yaml

from bench.mock_engine import MockEngine
from controller.context_schema import (
    ContextBudget, EvidenceClaim, EvidencePool, EvidenceSet, EvidenceUnit, Need, ReadingOutcome,
    ReadResult, ReadTask, ReductionState, SourceSpan, StepRecord, TaskState, TextRef,
)
from controller.evidence import make_evidence
from controller.execution import ReadingBudgetExceeded, ReadingExecutor
from controller.plan_policy import CompleteCost
from controller.rolling_policy import compile_step
from controller.schema import SessionSnapshot, TurnRecord
from controller.serialization import to_jsonable
from controller.service import Controller
from controller.session_store import SessionStore
from controller.templates import ByteTemplate
from training.collect_plan_gains import collect_pair
from training.plan_gain_protocol import (
    GapRetrievalRequest, ZERO_AUX, build_observed_action_plans, gap_query, observed_state_binding,
)
from training.plan_gain_rollout import GenerationBudget, bounded_engine, execute_branch, observed_rows


class Reader(MockEngine):
    def __init__(self):
        super().__init__()
        self._sampling_params = {"temperature": 0}
        self.inputs = []
        self.empty_reads = False

    async def info(self):
        return replace(await super().info(), engine_kind="sglang")

    async def quote_batch(self, *args, **kwargs):
        batch = await super().quote_batch(*args, **kwargs)
        return replace(batch, gpu_capacity_tokens=128000, active_requests=0, queued_requests=0,
            quotes=tuple(replace(quote, ready_gpu_prefix_tokens=0, host_only_load_tokens=0)
                         for quote in batch.quotes))

    async def generate(self, *, response_schema=None, **kwargs):
        rendered = bytes(kwargs["input_ids"]).decode()
        self.inputs.append((response_schema is not None, rendered))
        if response_schema is not None:
            payload = json.loads(rendered.split("<|user|>\n")[-1].split("\n<|end|>")[0])
            facts = [] if self.empty_reads else [{"need_id": "n", "text": unit["unit_id"]+" qualifies",
                "entity": unit["unit_id"], "relation": "qualifies", "value": "1", "unit": "",
                "conditions": [], "span_ids": [span["span_id"] for span in unit["source_spans"]],
                "parent_claim_ids": [], "followups": ([{"need_id": "n", "missing_relation": "qualifying record",
                    "bound_entities": ["R3"], "temporal_scope": "current"}] if unit["unit_id"] == "R1" else [])}
                for unit in payload["units"]]
            self.response_text = json.dumps({"format_version": 2, "claims": facts,
                "support": [], "conflicts": [], "open_needs": []})
        else:
            self.response_text = "Known qualifying records are supported by the supplied evidence."
        return await super().generate(**kwargs)


class Resources:
    def __init__(self):
        self.online = self.judge = self

    async def start(self):
        pass

    async def close(self):
        pass

    def observation(self):
        return {"source": "specified protocol fixture"}

    async def finish(self, role, start, end):
        return {"gpu_gib_seconds": end-start, "host_gib_seconds": 2*(end-start),
                "window": {"start": start, "end": end}, "complete_window_coverage": True}


def material(name):
    block = make_evidence(name+" qualifies.", name)
    return block, EvidenceUnit(name, (SourceSpan(name, block.key, 0, len(block.text)),), block.text)


async def frozen_fixture(tmp_path):
    from controller.reading import build_read_plan
    template, native = ByteTemplate(), Reader()
    pairs = (material("R1"), material("R2"))
    blocks, units = zip(*pairs)
    state = TaskState("task", "Count qualifying records.", needs=(Need("n", operation="count"),),
        working_history=(TextRef(1, "user", 0, 24),), current_turn_id=2)
    history = (TurnRecord(1, "Use the frozen condition", "Confirmed"),)
    snapshot = SessionSnapshot("capture", raw_history=history)
    plan = build_read_plan(state, EvidenceSet("selected", units, source_blocks=blocks), "selected_sources",
        ContextBudget(window_tokens=16000, read_input_tokens=10000, read_output_tokens=2048,
            output_tokens=256, body_tokens=3000), template=template, snapshot=snapshot,
        read_packing="source_windows")
    store = SessionStore(tmp_path / "state.db")
    initial = store.read_snapshot("capture")
    snapshot = store.sync_history("capture", history, initial.version)
    attempt = store.begin_attempt("capture", "capture", "hash", state.raw_question, snapshot.version)
    empty = ReadingOutcome(plan, auxiliary_usage=dict(ZERO_AUX))
    task = plan.steps[0]
    candidate = compile_step(plan, task, empty, snapshot, template, outputs=(task.output_budget,))[0]
    frozen = await ReadingExecutor(store, native, template).execute_one(
        attempt.attempt_id, snapshot, plan, task, empty, candidate=candidate)
    spent = sum(len(record.input_ids)+len(record.output_ids) for record in frozen.steps)
    paid = CompleteCost(500, 2000, 3000, spent+7, 0)
    checkpoint = template.fingerprint.model_revision
    observed = observed_state_binding(frozen, snapshot, paid_cost=paid,
        native_condition={"cache_state": "specified-fixture", "quote_seq": 1},
        reader_checkpoint=checkpoint, reader_profile_id="fixture-profile")
    observed["resident_model_manifest_digest"] = "specified-fixture"
    request = GapRetrievalRequest(frozen.open_needs, gap_query(state, frozen.open_needs),
        {"implementation": "specified-fixture"}, {"max_units": 1, "max_passages": 1})
    plans, omitted = build_observed_action_plans(frozen, snapshot, template,
        reader_checkpoint=checkpoint, gap_request=request)
    assert not omitted
    service = Controller(store, native, template=template)
    budget = GenerationBudget(64, tmp_path / "calls.jsonl")
    service.engine = bounded_engine(native, budget, tmp_path / "wire.json")
    binding = {"reader_checkpoint": checkpoint, "reader_profile_id": "fixture-profile",
        "scorer_identity": {"kind": "specified-fixture"}, "split": "train", "measurement_kind": "hardware",
        "resident_model_manifest": {"source": "specified-fixture"},
        "comparison_budget": {"window_tokens": 16000, "answer_output_tokens": 256},
        "collection_mode": "observed_actions", "counterfactual_cache_start": "observed", "cohort_split": "train",
        "cohort_manifest_sha256": "specified-fixture"}
    row = {"task_id": "task", "split": "train", "conversation_id": "fixture", "question": state.raw_question,
           "snapshot_id": "demo-v1", "turn": 2, "rollout_policy": "dense5_rebuild_qwen3_32b_nonthinking"}
    preparation = {"shared_cost_id": "specified-preparation", "complete_ms": 0,
        "input_tokens": 0, "output_tokens": 0, "gpu_gib_seconds": 0, "host_gib_seconds": 0}
    return service, native, budget, frozen, snapshot, observed, request, plans, binding, row, preparation


@pytest.mark.asyncio
async def test_frozen_nonempty_state_runs_each_single_action_without_replaying_or_unselected_gap(tmp_path):
    service, native, budget, frozen, snapshot, observed, request, plans, binding, row, prep = await frozen_fixture(tmp_path)
    assert {plan.semantic_kind for plan in plans} == {"direct_answer", "read_once", "gap_retrieval"}
    calls, runs = [], {}

    async def retrieve(plan, registered, claims):
        before = json.loads((tmp_path / "gap_retrieval" / "before.json").read_text())
        assert before["gap_request"] == to_jsonable(registered)
        assert registered == request and claims == frozen.claims
        calls.append(registered)
        block, unit = material("R3")
        return EvidencePool((unit,), (block,), auxiliary_usage={"input_tokens": 3,
            "output_tokens": 2, "compute_ms": 1, "models": []})

    for plan in plans:
        run = await execute_branch(service, plan, frozen, snapshot, row, case_dir=tmp_path / plan.semantic_kind,
            binding=binding, preparation=prep, resources=Resources(), cache_start="observed",
            generation_budget=budget, observed=observed, gap_request=request, gap_provider=retrieve)
        assert run["status"] == "completed" and run["qa_increment"] == 1
        assert frozen.steps[0].step_id not in run["new_step_ids"]
        assert run["cost"]["complete_ms"] == pytest.approx(500+run["actual_branch_execution_cost"]["complete_ms"])
        assert run["actual_branch_execution_cost"]["window"]["end"] <= run["cleanup_cost"]["window"]["end"]
        assert run["observed_state_binding"] == observed
        assert all(step["step_id"] in run["new_step_ids"] for step in run["steps"])
        expected = {"READ": 0, "REDUCE": 0, "GAP_RETRIEVAL": 0, "ANSWER": 1}
        if plan.semantic_kind != "direct_answer":
            expected[{"read_once": "READ", "reduce_once": "REDUCE", "gap_retrieval": "GAP_RETRIEVAL"}[plan.semantic_kind]] = 1
        assert run["new_action_counts"] == expected
        runs[plan.semantic_kind] = run
    assert len(calls) == 1
    assert runs["direct_answer"]["response"]["read_count"] == 1
    assert runs["direct_answer"]["new_action_counts"]["READ"] == 0
    answers = [prompt for structured, prompt in native.inputs if not structured]
    assert len(answers) == 3 and all("Use the frozen condition" in prompt and "R1" in prompt for prompt in answers)
    assert "R3" in answers[-1]
    assert sum(structured for structured, _ in native.inputs) == 2  # one capture READ, one new READ
    direct = runs["direct_answer"]
    candidate = runs["read_once"]
    for run, quality in ((direct, .4), (candidate, .7)):
        run["scoring"] = {"status": "scored", "identity": binding["scorer_identity"]}
        run["metrics"] = {"Q": quality}
    plan = next(plan for plan in plans if plan.semantic_kind == "read_once")
    packet = {"plan_candidate": to_jsonable(plan), "working_state": to_jsonable(frozen),
        "snapshot": to_jsonable(snapshot), "metric": "Q", "normalization": {"scale": 1,
        "offset": 0, "protocol_source": "specified protocol fixture"},
        "scorer_identity": binding["scorer_identity"], "comparison_budget": binding["comparison_budget"]}
    record = collect_pair(packet, direct, candidate)
    assert record["delta_q"] == pytest.approx(.3)
    assert record["incremental_complete_cost"]["complete_ms"] == pytest.approx(
        candidate["actual_branch_execution_cost"]["complete_ms"]-direct["actual_branch_execution_cost"]["complete_ms"])
    assert record["direct_run"]["read_count"] == 0
    await service.aclose()
    service.store.close()


@pytest.mark.asyncio
async def test_empty_new_read_still_gets_a_final_answer_and_prior_paid_budget_is_enforced(tmp_path):
    service, native, budget, frozen, snapshot, observed, request, plans, binding, row, prep = await frozen_fixture(tmp_path)
    native.empty_reads = True
    plan = next(plan for plan in plans if plan.semantic_kind == "read_once")
    run = await execute_branch(service, plan, frozen, snapshot, row, case_dir=tmp_path / "empty-read",
        binding=binding, preparation=prep, resources=Resources(), cache_start="observed",
        generation_budget=budget, observed=observed, gap_request=request,
        gap_provider=None)
    assert run["status"] == "completed" and run["new_action_counts"]["READ"] == 1
    assert run["steps"][0]["result"]["claims"] == []
    assert run["new_action_counts"]["ANSWER"] == 1
    from training.plan_gain_protocol import FixedCounterfactualRuntime
    runtime = FixedCounterfactualRuntime(plan, initial_state=frozen,
        paid_cost=CompleteCost(500, 0, 0, 1000000, 1000000))
    service.store.read_snapshot("over-budget")
    attempt = service.store.begin_attempt("over-budget", "r", "hash", row["question"], 0)
    with pytest.raises(ReadingBudgetExceeded, match="online token allocation"):
        await runtime.run(attempt.attempt_id, snapshot, (plan.context_plan,),
            ReadingExecutor(service.store, service.engine, service.template), service.engine,
            service.template, None, None, arrived=0, auxiliary_usage=dict(ZERO_AUX),
            events=service.events)
    await service.aclose()
    service.store.close()


@pytest.mark.asyncio
async def test_gap_limit_failure_retains_returned_material_and_paid_cost(tmp_path):
    service, _, budget, frozen, snapshot, observed, request, plans, binding, row, prep = await frozen_fixture(tmp_path)
    calls = []

    async def retrieve(plan, registered, claims):
        calls.append(registered)
        blocks, units = zip(material("R3"), material("R4"))
        return EvidencePool(units, blocks, auxiliary_usage={"input_tokens": 11,
            "output_tokens": 13, "compute_ms": 1, "models": []})

    plan = next(plan for plan in plans if plan.semantic_kind == "gap_retrieval")
    run = await execute_branch(service, plan, frozen, snapshot, row, case_dir=tmp_path / "gap-failure",
        binding=binding, preparation=prep, resources=Resources(), cache_start="observed",
        generation_budget=budget, observed=observed, gap_request=request, gap_provider=retrieve)
    assert len(calls) == 1 and run["status"] == "failed" and run["qa_increment"] == 0
    assert run["new_action_counts"]["GAP_RETRIEVAL"] == 1
    assert len(run["gap_result"]["pool"]["units"]) == 2
    assert run["actual_branch_execution_cost"]["total_input_tokens"] == 11
    assert run["actual_branch_execution_cost"]["total_output_tokens"] == 13
    await service.aclose()
    service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("new_facts", [False, True])
async def test_completed_reduction_is_offered_again_only_after_basis_expands(tmp_path, new_facts):
    from controller.reduction import reduce_claims
    service, _, _, frozen, snapshot, observed, _, _, _, _, _ = await frozen_fixture(tmp_path)
    basis = tuple(claim.claim_id for claim in frozen.claims)
    total = reduce_claims("count", frozen.claims, claim_id="completed:claim:0", need_id="n")
    reduction = ReductionState("completed", ("n",), "count", basis, (total.claim_id,), "merged")
    done = ReadResult("completed", (total,), reduction=reduction)
    task = ReadTask("completed", "REDUCE", ("n",), input_claim_ids=basis, operation="count")
    after = replace(frozen, claims=(*frozen.claims, total), results=(*frozen.results, done),
        steps=(*frozen.steps, StepRecord("capture", "completed", task, result=done)))
    if new_facts:
        pending = next(task for task in frozen.plan.steps if task.step_id != frozen.results[0].step_id)
        unit = next(unit for unit in frozen.plan.selected_set.selected_units if unit.unit_id in pending.input_unit_ids)
        fact = EvidenceClaim(pending.step_id+":claim:0", "n", "extract", entity="R2", value="1",
                             source_spans=unit.source_spans)
        read = ReadResult(pending.step_id, (fact,), covered_units=pending.input_unit_ids)
        after = replace(after, claims=(*after.claims, fact), results=(*after.results, read),
            steps=(*after.steps, StepRecord("capture", pending.step_id, pending, result=read)))
    plans, _ = build_observed_action_plans(after, snapshot, service.template,
                                          reader_checkpoint=observed["reader_checkpoint"])
    reductions = [plan for plan in plans if plan.semantic_kind == "reduce_once"]
    assert bool(reductions) is new_facts
    if new_facts:
        new_task = reductions[0].context_plan.steps[-1]
        assert new_task.reduction_basis_claim_ids == (*basis, fact.claim_id)
    await service.aclose()
    service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_material,existing_terminal", [(True, False), (False, False), (True, True)])
async def test_bounded_infer_bridge_reads_terminal_once_and_empty_gap_stops(tmp_path, terminal_material, existing_terminal):
    from training.plan_gain_protocol import BOUNDED_CONTINUATION
    from controller.reading import build_read_plan
    service, native, budget, captured, snapshot, old, _, _, binding, row, prep = await frozen_fixture(tmp_path)
    task_state = replace(captured.plan.task_state, needs=(Need("n", operation="infer"),))
    selected = captured.plan.selected_set if existing_terminal else replace(captured.plan.selected_set,
        selected_units=captured.plan.selected_set.selected_units[:1])
    base = build_read_plan(task_state, selected, "selected_sources", captured.plan.budget,
        template=service.template, snapshot=snapshot, read_packing="source_windows")
    frozen = ReadingOutcome(base, auxiliary_usage=dict(ZERO_AUX))
    observed = observed_state_binding(frozen, snapshot, paid_cost=CompleteCost(500, 0, 0, 7, 0),
        native_condition=old["native_condition"], reader_checkpoint=old["reader_checkpoint"],
        reader_profile_id=old["reader_profile_id"])
    plans, _ = build_observed_action_plans(frozen, snapshot, service.template,
        reader_checkpoint=old["reader_checkpoint"])
    plan = next(plan for plan in plans if plan.semantic_kind == "read_once")
    request = GapRetrievalRequest((), "", {"implementation": "specified-fixture"},
        {"max_units": 1, "max_passages": 1})
    calls = []

    async def retrieve(plan, request, claims):
        calls.append(request)
        assert request.open_needs and request.open_needs[0].missing_relation == "qualifying record"
        block = make_evidence("R3 qualifies. "*220, "R3")
        unit = EvidenceUnit("R3", (SourceSpan("R3", block.key, 0, len(block.text)),), block.text)
        return EvidencePool((unit,) if terminal_material else (), (block,) if terminal_material else (),
            auxiliary_usage=dict(ZERO_AUX))

    if existing_terminal:
        tickets = len(budget.tickets)
        unavailable = await execute_branch(service, plan, frozen, snapshot, row,
            case_dir=tmp_path/"insufficient-continuation-reserve", binding=binding, preparation=prep,
            resources=Resources(), cache_start="observed", generation_budget=GenerationBudget(3, tmp_path/"reserve.jsonl"),
            observed=observed, gap_request=request, gap_provider=retrieve)
        assert unavailable is None and len(budget.tickets) == tickets

    run = await execute_branch(service, plan, frozen, snapshot, row, case_dir=tmp_path/"bridge",
        binding=binding, preparation=prep, resources=Resources(), cache_start="observed",
        generation_budget=budget, observed=observed, gap_request=request, gap_provider=retrieve)
    assert run["status"] == "completed", run["failure"]
    assert len(calls) == (0 if existing_terminal else 1)
    assert run["new_action_counts"] == {"READ": 2 if terminal_material else 1,
        "REDUCE": 0, "GAP_RETRIEVAL": 0 if existing_terminal else 1, "ANSWER": 1}
    if terminal_material:
        assert run["steps"][-1]["task"]["input_claim_ids"]
    assert run["continuation_protocol"] == BOUNDED_CONTINUATION
    assert run["pre_action_quote"]["candidate_id"] is not None
    await service.aclose()
    service.store.close()


def test_bounded_continuation_stops_with_evidence_and_rejects_invalid_actions():
    from types import SimpleNamespace
    from controller.action_proposals import continuation_gap, validate_action_counts
    state = SimpleNamespace(plan=SimpleNamespace(task_state=TaskState("s", "question",
        needs=(Need("n", operation="infer"),))), open_needs=())
    assert continuation_gap(state, used_gap=False) == ()
    validate_action_counts("direct_answer", {"READ": 0, "REDUCE": 0, "GAP_RETRIEVAL": 0, "ANSWER": 1})
    with pytest.raises(ValueError, match="exceed"):
        validate_action_counts("read_once", {"READ": 3, "REDUCE": 0, "GAP_RETRIEVAL": 1, "ANSWER": 1})
    validate_action_counts("read_once", {"READ": 2, "REDUCE": 0, "GAP_RETRIEVAL": 0, "ANSWER": 1})
