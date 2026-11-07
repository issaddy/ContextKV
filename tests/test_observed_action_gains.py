"""CPU protocol fixtures exercise action collection; these are not GPU reward measurements."""
from action_helpers import Resources, material, frozen_fixture
from dataclasses import replace
import json

import pytest

from controller.context_schema import (
    EvidenceClaim, EvidencePool, EvidenceUnit, Need, ReadingOutcome,
    ReadResult, SourceSpan, StepRecord,
)
from controller.evidence import make_evidence
from controller.execution import ReadingBudgetExceeded, ReadingExecutor
from controller.plan_policy import CompleteCost
from controller.serialization import to_jsonable
from training.collect_plan_gains import collect_pair
from training.plan_gain_protocol import (
    GapRetrievalRequest, ZERO_AUX, build_observed_action_plans, observed_state_binding,
)
from training.plan_gain_rollout import GenerationBudget, execute_branch


@pytest.mark.asyncio
async def test_b_official_target_projects_actual_sources_and_keeps_raw_label_identity(tmp_path):
    from copy import deepcopy
    from types import SimpleNamespace
    from training.plan_gain_completion import completion_record
    from training.plan_gain_rollout import JudgeJournal, radbench_scorer_identity, score_branch
    from training.train_plan_gain import working_state_key

    service, _, budget, frozen, snapshot, observed, request, plans, binding, row, prep = await frozen_fixture(tmp_path)
    unseen = make_evidence('UNSEEN OFFLINE GOLD WITNESS', 'unseen-gold')
    snapshot = replace(snapshot, archive=(*snapshot.archive, unseen))
    observed = {**observed, 'state_key': working_state_key(frozen, snapshot)}
    plan = next(plan for plan in plans if plan.semantic_kind == 'read_once')
    branch_dir = tmp_path/'actual-read'
    run = await execute_branch(service, plan, frozen, snapshot, row, case_dir=branch_dir,
        binding=binding, preparation=prep, resources=Resources(), cache_start='observed',
        generation_budget=budget, observed=observed, gap_request=request)
    before_bytes = (branch_dir/'before.json').read_bytes()
    before = json.loads(before_bytes)
    original = deepcopy(run)
    row = {**row, 'reference_answer': 'The original native target.',
        'task_state': {'fallible': 'MUST NOT REACH OFFICIAL PROMPT'},
        'judge_grounding': [{'quote': unseen.text}]}
    records, parsed = [], []

    def prompt(record):
        records.append(record)
        return json.dumps(record)

    def parse(raw):
        parsed.append(raw)
        return .8  # Specified contract result; the author's parser is tested separately.

    contract = SimpleNamespace(prompt=prompt, parse=parse,
        identity={'metric': 'RB_llm', 'source_commit': 'specified official contract fixture',
                  'normalization': 'official_extract_rating_divide_by_10'})
    fingerprint = to_jsonable(service.template.fingerprint)
    identity = radbench_scorer_identity(contract, fingerprint)
    requests = []

    class JudgeClient:
        async def post(self, endpoint, **kwargs):
            if endpoint == '/context/engine/info':
                raw = {'fingerprint': fingerprint}
            else:
                assert endpoint == '/generate'
                requests.append(kwargs['json'])
                raw = {'text': 'Rating: [[8]]', 'meta_info': {
                    'finish_reason': {'type': 'length'}, 'completion_tokens': 5}}
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: raw)

    judge_budget = GenerationBudget(1, tmp_path/'judge-calls.jsonl')
    result = await score_branch(JudgeJournal(JudgeClient(), judge_budget, tmp_path/'judge-wire.jsonl'),
        service.template, row, before, run, contract=contract, path=tmp_path/'rescored/run.json',
        identity=identity, resources=Resources(), judge_window=32768, expected_fingerprint=fingerprint)
    assert len(judge_budget.tickets) == len(requests) == 1 and parsed == ['Rating: [[8]]']
    assert requests[0]['sampling_params'] == {'temperature': 0, 'top_p': 1, 'max_new_tokens': 2048}
    rendered = bytes(requests[0]['input_ids']).decode()
    assert rendered.count('<|user|>') == 1 and '<|system|>' not in rendered
    assert records[0]['targets'] == [{'text': row['reference_answer']}]
    assert records[0]['input'][-1]['text'] == row['question']
    assert records[0]['input'][0]['text'] == original['actual_history'][0]['user_text']
    assert records[0]['contexts'] and all('UNSEEN' not in context['text'] for context in records[0]['contexts'])
    assert 'MUST NOT REACH OFFICIAL PROMPT' not in rendered
    assert result['metrics']['plan_gain_quality'] == .8 and result['quality_components'] == {'RB_llm': .8}
    assert result['scorer_identity'] == original['scorer_identity']
    assert result['actual_native_calls'] == original['actual_native_calls']
    assert result['cost'] == original['cost'] and (branch_dir/'before.json').read_bytes() == before_bytes
    label = completion_record(before, result)
    assert label['scorer_identity'] == identity and label['registered_scorer_identity'] == before['scorer_identity']
    assert label['final_required_point_coverage'] is None
    await service.aclose()
    service.store.close()


@pytest.mark.asyncio
async def test_frozen_nonempty_state_runs_each_single_action_without_replaying_or_unselected_gap(tmp_path):
    service, native, budget, frozen, snapshot, observed, request, plans, binding, row, prep = await frozen_fixture(tmp_path)
    assert {plan.semantic_kind for plan in plans} == {"direct_answer", "read_once", "gap_retrieval"}
    calls, runs = [], {}

    async def retrieve(plan, registered, claims):
        before = json.loads((tmp_path / "gap_retrieval" / "before.json").read_text())
        assert before["gap_request"] == to_jsonable(registered)
        assert registered == request and claims == frozen.claims
        assert {unit.unit_id for unit in plan.selected_set.selected_units} == {"R1"}
        assert {unit.unit_id for unit in frozen.plan.selected_set.selected_units} == {"R1", "R2"}
        assert plan.budget.max_passages == 2  # one completed passage plus one new passage
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
@pytest.mark.parametrize("terminal_material,existing_terminal", [(True, False), (False, False), (True, True)])
@pytest.mark.parametrize("hard_steps", [2, 20])
async def test_bounded_infer_bridge_reads_terminal_once_and_empty_gap_stops(tmp_path, terminal_material, existing_terminal, hard_steps):
    from training.plan_gain_protocol import BOUNDED_CONTINUATION
    from controller.reading import build_read_plan
    service, native, budget, captured, snapshot, old, _, _, binding, row, prep = await frozen_fixture(tmp_path)
    task_state = replace(captured.plan.task_state, needs=(Need("n", operation="infer"),))
    selected = captured.plan.selected_set if existing_terminal else replace(captured.plan.selected_set,
        selected_units=captured.plan.selected_set.selected_units[:1])
    base = build_read_plan(task_state, selected, "selected_sources", captured.plan.budget,
        template=service.template, snapshot=snapshot, read_packing="source_windows")
    base = replace(base, budget=replace(base.budget, max_steps=hard_steps))
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
    assert len(calls) == (0 if existing_terminal or hard_steps == 2 else 1)
    assert run["new_action_counts"] == {"READ": 2 if terminal_material and hard_steps > 2 else 1,
        "REDUCE": 0, "GAP_RETRIEVAL": 0 if existing_terminal or hard_steps == 2 else 1, "ANSWER": 1}
    if hard_steps == 2:
        assert len(run["steps"]) == 1
        assert {claim["entity"] for claim in run["steps"][0]["result"]["claims"]} == {"R1"}
        assert not any(step["step_id"].startswith("bounded-terminal:") for step in run["steps"])
    if terminal_material and hard_steps > 2:
        assert run["steps"][-1]["task"]["input_claim_ids"]
        from training.plan_gain_completion import completion_record
        before = json.loads((tmp_path/"bridge"/"before.json").read_text())
        labelled = completion_record(before, run)
        assert 0 <= labelled["read_source_char_fraction"] <= 1
        assert labelled["read_original_characters"] == labelled["required_original_characters"]
        if existing_terminal:
            from controller.reading import unresolved_needs
            from controller.serialization import decode
            first = decode(StepRecord, run["steps"][0])
            unrelated = replace(first.result.claims[0], claim_id="unrelated:claim:0", need_id="other")
            state = ReadingOutcome(base, results=(first.result,), steps=(first,),
                claims=(*first.result.claims, unrelated), open_needs=unresolved_needs((first.result,)))
            online, _ = build_observed_action_plans(state, snapshot, service.template,
                reader_checkpoint=old["reader_checkpoint"])
            next_read = next(item for item in online if item.semantic_kind == "read_once").context_plan.steps[-1]
            assert next_read.input_claim_ids == tuple(run["steps"][-1]["task"]["input_claim_ids"])
            assert unrelated.claim_id not in next_read.input_claim_ids
    assert run["continuation_protocol"] == BOUNDED_CONTINUATION
    assert run["pre_action_quote"]["candidate_id"] is not None
    await service.aclose()
    service.store.close()


@pytest.mark.asyncio
async def test_gap_for_numeric_need_does_not_continue_an_unrelated_infer_need(tmp_path):
    from training.plan_gain_protocol import gap_query
    service, native, budget, frozen, snapshot, old, _, _, binding, row, prep = await frozen_fixture(tmp_path)
    task_state = replace(frozen.plan.task_state, needs=(*frozen.plan.task_state.needs,
                                                     Need("already-supported", operation="infer")))
    frozen = replace(frozen, plan=replace(frozen.plan, task_state=task_state),
        results=tuple(replace(result, support=("already-supported",)) for result in frozen.results))
    observed = observed_state_binding(frozen, snapshot, paid_cost=CompleteCost(500, 0, 0, 7, 0),
        native_condition=old["native_condition"], reader_checkpoint=old["reader_checkpoint"],
        reader_profile_id=old["reader_profile_id"])
    # Retain the already paid reader tokens in this branch's observed binding.
    observed["paid_cost"]["input_tokens"] += sum(len(step.input_ids) + len(step.output_ids)
                                                  for step in frozen.steps)
    request = GapRetrievalRequest(frozen.open_needs, gap_query(task_state, frozen.open_needs),
        {"implementation": "specified-fixture"}, {"max_units": 1, "max_passages": 1})
    plans, _ = build_observed_action_plans(frozen, snapshot, service.template,
        reader_checkpoint=old["reader_checkpoint"], gap_request=request)
    plan = next(plan for plan in plans if plan.semantic_kind == "gap_retrieval")

    async def retrieve(*args):
        block, unit = material("R3")
        return EvidencePool((unit,), (block,), auxiliary_usage=dict(ZERO_AUX))

    run = await execute_branch(service, plan, frozen, snapshot, row, case_dir=tmp_path/"numeric-gap",
        binding=binding, preparation=prep, resources=Resources(), cache_start="observed",
        generation_budget=budget, observed=observed, gap_request=request, gap_provider=retrieve)
    assert run["status"] == "completed", run["failure"]
    assert run["new_action_counts"] == {"READ": 0, "REDUCE": 0, "GAP_RETRIEVAL": 1, "ANSWER": 1}
    await service.aclose()
    service.store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('stage', ['numeric_select', 'sum'])
async def test_observed_numeric_actions_keep_factory_bound_shared_and_selected_basis(tmp_path, stage):
    from controller.action_proposals import observed_reduction_task
    from controller.plan_policy import complete_plan
    from controller.reading import build_read_plan, numeric_selection_task, parse_read_result
    from training.plan_gain_protocol import FixedCounterfactualRuntime

    service, native, _, captured, snapshot, observed, _, _, _, _, _ = await frozen_fixture(tmp_path)
    needs = (Need('sum', operation='sum'), Need('count', operation='count'))
    base = build_read_plan(replace(captured.plan.task_state, needs=needs),
        replace(captured.plan.selected_set, selected_units=captured.plan.selected_set.selected_units[:1]),
        'selected_sources', captured.plan.budget, template=service.template, snapshot=snapshot)
    read = base.steps[0]
    original = EvidenceClaim(read.step_id+':claim:0', 'count', 'extract',
        text='Inventory-13-029: 29 credits.', entity='Inventory-13-029', value='29', unit='credits',
        source_spans=base.selected_set.selected_units[0].source_spans)
    result = ReadResult(read.step_id, (original,), covered_units=read.input_unit_ids)
    frozen = ReadingOutcome(base, claims=(original,), results=(result,),
        steps=(StepRecord('capture', read.step_id, read, result=result),), auxiliary_usage=dict(ZERO_AUX))
    if stage == 'sum':
        # Both selectors have completed; CPU sum must consume selector-derived facts.
        for need_id in ('sum', 'count'):
            selector = numeric_selection_task(base, frozen.results, frozen.claims, need_id=need_id)
            selected = parse_read_result(selector, json.dumps({'format_version': 7,
                'views': {'n0': {'p0': True}}}), claims=(original,))
            frozen = replace(frozen, claims=(*frozen.claims, *selected.claims),
                results=(*frozen.results, selected),
                steps=(*frozen.steps, StepRecord('capture', selector.step_id, selector, result=selected)))
    task = observed_reduction_task(frozen)
    assert task.operation == stage
    expected = tuple(claim for claim in frozen.claims if claim.claim_id in task.input_claim_ids)
    assert expected and (expected[0].kind == 'derived' if stage == 'sum' else not original.serves('sum'))
    plan = complete_plan(replace(base, steps=(*(record.task for record in frozen.steps), task)), (),
        semantic_kind='reduce_once', reader_checkpoint=observed['reader_checkpoint'])

    class BasisObserved(Exception):
        pass

    class Executor:
        request_compiler = service.request_compiler

        async def execute_one(self, attempt_id, snapshot, plan, actual, state, **kwargs):
            assert actual.input_claim_ids == task.input_claim_ids
            assert tuple(claim for claim in state.claims if claim.claim_id in actual.input_claim_ids) == expected
            raise BasisObserved

    runtime = FixedCounterfactualRuntime(plan, initial_state=frozen, paid_cost=CompleteCost(0, 0, 0, 0, 0))
    with pytest.raises(BasisObserved):
        await runtime.run('basis', snapshot, (plan.context_plan,), Executor(), native, service.template,
            None, None, arrived=0, auxiliary_usage=dict(ZERO_AUX), events=service.events)
    await service.aclose()
    service.store.close()


@pytest.mark.asyncio
async def test_whole_plan_collection_preserves_unpaid_reuse_seed_and_skips_replayed_sources(tmp_path):
    from training.plan_gain_protocol import build_counterfactual_plans

    service, native, budget, frozen, snapshot, _, _, _, binding, row, prep = await frozen_fixture(tmp_path)
    old = frozen.plan.steps[0]
    imported = replace(old, step_id='policy:reuse:specified', input_budget=0, output_budget=0)
    seed = replace(frozen, steps=(), results=(replace(frozen.results[0], step_id=imported.step_id),),
        plan=replace(frozen.plan, steps=(imported, *frozen.plan.steps[1:])))
    plans, _ = build_counterfactual_plans(seed.plan, snapshot, service.template,
        reader_checkpoint=binding['reader_checkpoint'], read_outputs=(2048,), initial_state=seed)
    assert plans
    for planned in plans:
        assert imported.step_id in {step.task.step_id for step in planned.steps}
        assert not any(set(step.task.input_unit_ids) & set(imported.input_unit_ids)
            for step in planned.steps if step.task.kind == 'READ' and step.task.step_id != imported.step_id)
    direct = next(plan for plan in plans if plan.semantic_kind == 'direct_answer')
    assert [step.task.kind for step in direct.steps] == ['READ', 'ANSWER']
    run = await execute_branch(service, direct, seed, snapshot, row, case_dir=tmp_path/'whole-direct',
        binding={**binding, 'collection_mode': 'whole_plan'}, preparation=prep, resources=Resources(),
        cache_start='observed', generation_budget=budget)
    assert run['new_action_counts'] == {'READ': 0, 'REDUCE': 0, 'GAP_RETRIEVAL': 0, 'ANSWER': 1}
    assert run['response']['read_count'] == 0  # reused logical facts do not add physical READ calls
    assert len(run['actual_native_calls']) == 1 and not run['steps']
    assert frozen.claims[0].text in native.inputs[-1][1]
    assert run['attempt']['reading_outcome']['results'][0]['step_id'] == imported.step_id
    await service.aclose()
    service.store.close()


@pytest.mark.asyncio
async def test_counterfactual_terminal_answer_executes_and_records_contextpilot_choice(tmp_path):
    from controller.context_schema import AnswerPlan, EvidenceSet
    from controller.plan_policy import complete_plan
    from controller.request_compiler import RequestCompiler
    from controller.plan_gain_protocol import validate_pre_action_input

    service, native, budget, frozen, snapshot, _, _, _, binding, row, prep = await frozen_fixture(tmp_path)

    class ReverseIndex:
        def propose_orders(self, identities, **scope):
            return (tuple(identities), tuple(reversed(identities)))

        def observe_executed(self, *args, **kwargs):
            return 'specified-executed-order'

    service.request_compiler = RequestCompiler(service.template, ReverseIndex())
    quote = native.quote_batch

    async def prefer_reordered(candidates, fingerprint):
        batch = await quote(candidates, fingerprint)
        choices = {candidate.candidate_id: candidate for candidate in candidates}
        return replace(batch, quotes=tuple(replace(current,
            ready_gpu_prefix_tokens=current.input_len-1 if choices[current.candidate_id].layout_id.endswith(':contextpilot') else 0)
            for current in batch.quotes))

    native.quote_batch = prefer_reordered
    pairs = tuple(material(name) for name in ('left', 'right'))
    blocks, units = zip(*pairs)
    base = replace(frozen.plan, steps=(), answer_plan=AnswerPlan(),
        task_state=replace(frozen.plan.task_state, needs=(Need('n', operation='lookup'),)),
        selected_set=EvidenceSet('small', units, source_blocks=blocks))
    from controller.context_budget import bind_answer_window
    base = bind_answer_window(base, service.template, snapshot, service.policy.system_prompt)
    initial = ReadingOutcome(base, auxiliary_usage=dict(ZERO_AUX))
    plan = complete_plan(base, (), semantic_kind='direct_answer', reader_checkpoint=binding['reader_checkpoint'])
    run = await execute_branch(service, plan, initial, snapshot, row, case_dir=tmp_path/'permuted',
        binding=binding, preparation=prep, resources=Resources(), cache_start='observed', generation_budget=budget)
    chosen = run['pre_action_input']
    assert chosen['layout_id'] == 'existing:contextpilot'
    assert chosen['input_ids'] == run['actual_native_calls'][0]['request']['input_ids']
    assert chosen['input_hash'] == run['attempt']['plan']['candidate']['input_hash']
    validate_pre_action_input(run)
    await service.aclose()
    service.store.close()


@pytest.mark.asyncio
async def test_actual_snapshot_cohort_needs_no_task_labels_and_keeps_native_witness(tmp_path):
    from copy import deepcopy
    from training.collect_plan_gains import export_observed_packets
    from training.plan_gain_rollout import observed_cohort, observed_rows, judge_sources, digest
    from training.reader_data import file_digest
    from controller.plan_gain_protocol import input_protocol
    from controller.context_schema import TextRef
    from controller.resident_models import resident_model_digest, resident_model_identity

    service, native, budget, frozen, snapshot, _, request, plans, binding, row, prep = await frozen_fixture(tmp_path/'capture')
    helper = {'engine_restart_id': 'original-process', 'model_scope': {
        'source_path': '/original/scope.json', 'identity': 'fixed-model', 'record': {'weights': 'sha'}}}
    resident = {'retrieval': {'retriever': {'dense': {'index_sha256': 'fixed-index'},
        'encoder': {'identity': 'fixed-encoder', 'source_path': '/encoder/checkpoint',
            'sdpa_backend': 'math', 'tokenizer': 'fixed-tokenizer', 'version': 'fixed-version'}}},
        'helper_engine_info': helper, 'support_model': {
            'base_url': 'http://127.0.0.1:30005', 'helper_engine_info': deepcopy(helper)},
        'task_model': {'transport': {'base_url': 'http://127.0.0.1:30005'}}}
    restarted = deepcopy(resident)
    restarted['support_model']['base_url'] = 'http://127.0.0.1:30001'
    restarted['task_model']['transport']['base_url'] = 'http://127.0.0.1:30001'
    for engine in (restarted['helper_engine_info'], restarted['support_model']['helper_engine_info']):
        engine['engine_restart_id'] = 'restarted-process'
        engine['model_scope']['source_path'] = '/moved/scope.json'
    assert resident_model_identity(resident) == resident_model_identity(restarted)
    assert resident_model_identity(resident_model_identity(resident)) == resident_model_identity(resident)
    for field in ('identity', 'record'):
        changed = deepcopy(restarted)
        changed['helper_engine_info']['model_scope'][field] = 'different-model'
        assert resident_model_digest(changed) != resident_model_digest(resident)
    for field in ('identity', 'source_path', 'sdpa_backend', 'tokenizer', 'version'):
        changed = deepcopy(restarted)
        changed['retrieval']['retriever']['encoder'][field] = 'different-asset'
        assert resident_model_digest(changed) != resident_model_digest(resident)
    changed = deepcopy(restarted)
    changed['retrieval']['retriever']['dense']['index_sha256'] = 'different-index'
    assert resident_model_digest(changed) != resident_model_digest(resident)
    segment = tmp_path/'capture-generation/measurements/original-segment'
    segment.mkdir(parents=True)
    generation_manifest = segment/'generation_manifest.json'
    generation_manifest.write_text(json.dumps({'retrieval': resident}))
    ready = tuple(step if step.step_id == frozen.steps[0].step_id else replace(step,
        input_budget=frozen.plan.budget.window_tokens-step.output_budget) for step in frozen.plan.steps)
    frozen = replace(frozen, plan=replace(frozen.plan, steps=ready, task_state=replace(frozen.plan.task_state,
        needs=(replace(frozen.plan.task_state.needs[0], anchor_spans=(
            TextRef(2, 'user', 0, len(row['question'])),)),))))
    manifest = tmp_path/'cohort.json'
    # A P3 cohort declares the queue before its model-generated history exists.
    manifest.write_text(json.dumps({'members': [{'task_id': 'task', 'conversation_id': 'fixture',
        'turn': 2, 'split': 'train'}]}))
    cohort_hash = file_digest(manifest)
    blocks = frozen.plan.selected_set.source_blocks
    sources = [{'source_id': block.key.passage_id, 'version': block.key.version, 'text': block.text,
        'source_uri': block.source_uri, 'split': 'train'} for block in blocks]
    span = frozen.plan.selected_set.selected_units[0].source_spans[0]
    block = next(block for block in blocks if block.key == span.source_key)
    witness = {'passage_id': block.key.passage_id, 'start': span.start, 'end': span.start+3,
        'quote': block.text[span.start:span.start+3]}
    raw = {'task_id': 'task', 'conversation_id': 'fixture', 'turn': 2,
        'input': [{'speaker': 'user', 'text': row['question']}],
        'targets': [{'text': '原始 offline reference'}], 'grounding': [witness]}
    capture = {'task_id': 'task', 'conversation_id': 'fixture', 'turn': 2,
        'source_partition': 'train', 'split': 'train', 'measurement_kind': 'hardware',
        'history_origin': 'model_actual', 'cohort_split': 'train', 'cohort_manifest_sha256': cohort_hash,
        'generation_dir': str(tmp_path/'capture-generation'), 'measurement_segment_id': 'original-segment',
        'method': 'B6_native', 'question': row['question'], 'reference': raw,
        'actual_history': to_jsonable(snapshot.raw_history), 'answer': 'specified actual answer',
        'response': {'attempt_id': 'actual-capture'}}
    prior = snapshot.raw_history[0]
    previous = {**capture, 'task_id': 'prior', 'turn': 1, 'question': prior.user_text,
        'answer': prior.assistant_text, 'actual_history': [], 'response': {'attempt_id': 'prior-capture',
            'receipt': {'generation_complete': True, 'input_hash': 'specified actual native prior QA'}}}
    quote = to_jsonable(await service.engine.quote_batch((), service.template.fingerprint))
    packet = {'observation_id': 'actual:decision:1', 'attempt_id': 'actual-capture',
        'decision_seq': 1, 'working_state': to_jsonable(frozen), 'snapshot': to_jsonable(snapshot),
        'input_protocol': input_protocol(service.template.fingerprint),
        'native_condition': quote, 'paid_cost': to_jsonable(CompleteCost(500, 2000, 3000, 10000, 0)),
        'reader_checkpoint': binding['reader_checkpoint'], 'reader_profile_id': binding['reader_profile_id'],
        'resident_model_manifest_digest': 'specified-fixture', 'decision_mode': 'fixed_reading',
        'gap_request': to_jsonable(request),
        'legal_actions': [plan.semantic_kind for plan in plans]}
    output = tmp_path/'packets.jsonl'
    export_observed_packets([packet], [previous, capture], output)
    packets = [json.loads(line) for line in output.read_text().splitlines()]
    assert packets[0]['source_capture_sha256'] == digest(capture)
    assert packets[0]['resident_model_manifest_digest'] == resident_model_digest(resident)
    assert packets[0]['gap_request']['retriever_identity']['resident_model_manifest_digest'] == resident_model_digest(resident)
    assert packets[0]['capture_resident_binding'] == {'recorded_digest': 'specified-fixture',
        'manifest_path': str(generation_manifest.resolve()), 'manifest_sha256': file_digest(generation_manifest)}
    assert packet['resident_model_manifest_digest'] == 'specified-fixture'
    assert packet['gap_request'] == to_jsonable(request)
    failed = {name: value for name, value in capture.items() if name not in {'response', 'generation_dir'}}
    failed.update(status='failed', attempt={'attempt_id': 'failed-capture'},
        state_db=str(tmp_path/'failed-generation/state/session.sqlite'), measurement_segment_id='failed-segment')
    failed_segment = tmp_path/'failed-generation/measurements/failed-segment'
    failed_segment.mkdir(parents=True)
    failed_manifest = failed_segment/'generation_manifest.json'
    failed_manifest.write_text(json.dumps({'retrieval': changed}))
    failed_packet = {**packet, 'observation_id': 'failed:decision:1', 'attempt_id': 'failed-capture'}
    failed_output = tmp_path/'failed-packets.jsonl'
    export_observed_packets([failed_packet], [failed], failed_output)
    exported_failure = json.loads(failed_output.read_text())
    assert exported_failure['source_capture_status'] == 'failed'
    assert exported_failure['source_capture_sha256'] == digest(failed)
    assert exported_failure['resident_model_manifest_digest'] == resident_model_digest(changed)
    assert exported_failure['resident_model_manifest_digest'] != packets[0]['resident_model_manifest_digest']
    assert exported_failure['gap_request']['retriever_identity']['resident_model_manifest_digest'] == resident_model_digest(changed)
    assert exported_failure['capture_resident_binding'] == {'recorded_digest': 'specified-fixture',
        'manifest_path': str(failed_manifest.resolve()), 'manifest_sha256': file_digest(failed_manifest)}
    rows, cohort = observed_cohort([previous, capture], packets, sources, split='train', limit=1,
        cohort_manifest=manifest)
    assert cohort['task_label_dependency'] is False
    assert rows[0]['source_capture']['reference'] == raw
    assert rows[0]['reference_answer'] == raw['targets'][0]['text']
    actual = observed_rows(rows, packets)
    assert actual and actual[0]['task_state'] == to_jsonable(frozen.plan.task_state)

    from controller.events import EventLog
    from controller.serialization import decode
    from training.plan_gain_rollout import read_rows
    from controller.action_proposals import continuation_read
    from training.terminal_observations import restore_read_catalog, terminal_observation

    event_path = tmp_path/'terminal-events.jsonl'
    service.events = EventLog(event_path)
    plans, _ = build_observed_action_plans(frozen, snapshot, service.template,
        reader_checkpoint=binding['reader_checkpoint'], gap_request=request)
    branch = tmp_path/'terminal-read'
    branch_binding = {**binding, 'resident_model_manifest': resident,
        'cohort_manifest_sha256': cohort_hash, 'continuation_gap_limits': request.limits}
    run = await execute_branch(service, next(plan for plan in plans if plan.semantic_kind == 'read_once'),
        frozen, snapshot, actual[0], case_dir=branch, binding=branch_binding, preparation=prep,
        resources=Resources(), cache_start='observed', generation_budget=budget,
        observed=actual[0]['observed_binding'], gap_request=request)
    assert run['status'] == 'completed', run['failure']
    event = next(event for event in read_rows(event_path) if event['event'] == 'plan_gain_experimental_answer')
    before = json.loads((branch/'before.json').read_text())
    start = run['actual_branch_execution_cost']['window']['start']
    end = event['monotonic_timestamp_ns']/1e9
    phase = {'clock': 'client_monotonic', 'samples': [{'timestamp': start-1, 'gpu_gib': 2, 'host_gib': 3},
        {'timestamp': end+1, 'gpu_gib': 7, 'host_gib': 8}]}
    run_path = branch/'run.json'
    terminal = terminal_observation(before, run, actual[0], event, phase, service.template,
        {'run_path': str(run_path), 'run_sha256': file_digest(run_path)})
    assert terminal['attempt_id'] == run['attempt']['attempt_id'] != terminal['source_capture_attempt_id']
    assert terminal['native_condition_scope'] == 'actual_terminal_answer_only'
    assert terminal['requested_families'] == ['read_once', 'reduce_once', 'gap_retrieval']
    assert terminal['paid_cost']['gpu_gib_ms'] == pytest.approx(2000+2000*(end-start))
    assert terminal['paid_cost']['gpu_peak_gib'] == 2  # the later ANSWER-side sample is only a boundary witness
    paid_tokens = sum(terminal['paid_cost'][key] for key in ('input_tokens', 'output_tokens'))
    assert paid_tokens == 10000+len(run['steps'][0]['input_ids'])+len(run['steps'][0]['output_ids'])
    assert terminal['working_state']['plan']['online_tokens_spent'] == paid_tokens
    terminal_rows, _ = observed_cohort([previous, capture], [terminal], sources, split='train', limit=1,
        cohort_manifest=manifest)
    chosen, = observed_rows(terminal_rows, [terminal])
    assert set(chosen['observed_action_kinds']) == {'reduce_once', 'gap_retrieval'}
    terminal_state = decode(ReadingOutcome, terminal['working_state'])
    from action_helpers import material as source_material
    block, unit = source_material('unread-terminal-source')
    selected_set = terminal_state.plan.selected_set
    infer_plan = replace(terminal_state.plan, task_state=replace(terminal_state.plan.task_state,
        needs=(replace(terminal_state.plan.task_state.needs[0], operation='infer'),)),
        selected_set=replace(selected_set, selected_units=(*selected_set.selected_units, unit),
            source_blocks=(*selected_set.source_blocks, block)))
    covered_alias = replace(terminal_state.steps[0].task, step_id='covered-alias')
    unread = replace(covered_alias, step_id='unread-original', input_unit_ids=(unit.unit_id,))
    restored = restore_read_catalog(replace(terminal_state, plan=infer_plan),
        replace(infer_plan, steps=(*infer_plan.steps, covered_alias, unread)))
    assert covered_alias not in restored.plan.steps
    assert continuation_read(restored, restored.plan).step_id == unread.step_id
    plans, _ = build_observed_action_plans(terminal_state, snapshot, service.template,
        reader_checkpoint=binding['reader_checkpoint'])
    from bench.mock_engine import MockEngine
    generate = native.generate

    async def select_quantities(*, response_schema=None, **kwargs):
        if response_schema is not None and response_schema['properties']['format_version']['const'] == 7:
            native.response_text = json.dumps({'format_version': 7, 'views': {
                alias: {parent: rule.get('const', True) for parent, rule in view['properties'].items()}
                for alias, view in response_schema['properties']['views']['properties'].items()}})
            return await MockEngine.generate(native, **kwargs)
        return await generate(response_schema=response_schema, **kwargs)

    native.generate = select_quantities
    reduced = await execute_branch(service, next(plan for plan in plans if plan.semantic_kind == 'reduce_once'),
        terminal_state, snapshot, chosen, case_dir=tmp_path/'terminal-reduce', binding=branch_binding,
        preparation=prep, resources=Resources(), cache_start='observed', generation_budget=budget,
        observed=chosen['observed_binding'], gap_request=request)
    assert reduced['new_action_counts'] == {'READ': 0, 'REDUCE': 1, 'GAP_RETRIEVAL': 0, 'ANSWER': 1}
    assert len(reduced['actual_native_calls']) == 2

    from controller.context_schema import EvidenceSet
    from training.preference_rollout import _judge, P1_RUBRIC

    full_context = 'Context before. Witness excerpt. Exception in the full original after the witness.'
    offline_witness = {'passage_id': 'offline-only', 'start': full_context.index('Witness'),
        'end': full_context.index('Witness')+len('Witness excerpt.'), 'quote': 'Witness excerpt.'}
    original = {'source_id': 'offline-only', 'document_family': 'offline-only', 'version': 'v1',
        'source_uri': 'original://offline-only', 'title': 'Complete offline original', 'text': full_context}
    scope_text = 'A full original block from the explicit applicable source scope.'
    scope_block = replace(make_evidence(scope_text, 'scope-only'), token_ids=tuple(scope_text.encode()))
    scope_unit = EvidenceUnit('scope-only', (SourceSpan('scope-only', scope_block.key, 0, len(scope_text)),),
        scope_text, need_ids=('fallible-model-need',), retrieval_score=.99)
    applicable = EvidenceSet('applicable', (scope_unit,), source_blocks=(scope_block,))
    audit_row = {**actual[0], 'task_state': {'fallible_inference': 'MUST NOT REACH THE JUDGE'},
        'judge_grounding': [*actual[0]['judge_grounding'], offline_witness]}
    material = judge_sources(frozen.plan, applicable, audit_row, [*sources, original], corpus_id='demo')
    assert any(unit.text == witness['quote'] for unit in material.selected_units)
    assert all(block in material.source_blocks for block in (*frozen.plan.selected_set.source_blocks, scope_block))
    assert any(block.text == full_context for block in material.source_blocks)
    assert not any(unit.text == full_context for unit in material.selected_units)
    captured = []

    class JudgeClient:
        async def post(self, endpoint, **kwargs):
            assert endpoint == '/generate'
            rendered = bytes(kwargs['json']['input_ids']).decode()
            payload = json.loads(rendered.split('<|user|>\n')[-1].split('\n<|end|>')[0])
            captured.append(payload)
            assert 'task_state' not in payload and 'MUST NOT REACH THE JUDGE' not in rendered
            assert payload['actual_history'] == audit_row['actual_history']
            assert payload['reference_answer'] == raw['targets'][0]['text']
            blocks = payload['original_sources']
            assert all(set(block) == {'key', 'uri', 'title', 'span', 'text'} for block in blocks)
            assert any(block['text'] == full_context for block in blocks)
            assert any(block['text'] == scope_text for block in blocks)
            assert any(excerpt['text'] == offline_witness['quote'] for excerpt in payload['offline_source_excerpts'])
            assert not any(key in json.dumps(payload) for key in ('token_ids', 'need_ids', 'retrieval_score', 'quality_prediction'))
            response = {key: .5 for key in ('necessary_points', 'grounding', 'conditions_versions', 'task_score')}
            response.update(disposition='partial', rationale='Specified payload fixture only')
            from types import SimpleNamespace
            return SimpleNamespace(raise_for_status=lambda: None,
                json=lambda: {'text': json.dumps(response), 'meta_info': {'completion_tokens': 1}})

    await _judge(JudgeClient(), service.template, audit_row, material, 'Specified actual answer.')
    assert len(captured) == 1 and P1_RUBRIC.endswith('_v3')
    assert actual[0]['source_capture']['actual_history'] == to_jsonable(snapshot.raw_history)

    wrong = deepcopy(packets)
    wrong[0]['attempt_id'] = 'unrelated-attempt'
    with pytest.raises(ValueError, match='exact bound actual capture'):
        observed_cohort([previous, capture], wrong, sources, split='train', limit=1, cohort_manifest=manifest)
    tampered = deepcopy(capture)
    tampered['reference']['grounding'][0]['quote'] = 'not in the source'
    wrong = deepcopy(packets)
    wrong[0]['source_capture_sha256'] = digest(tampered)
    with pytest.raises(ValueError, match='raw offline witness'):
        observed_cohort([previous, tampered], wrong, sources, split='train', limit=1, cohort_manifest=manifest)
    manifest.write_text(json.dumps({'members': [
        {'task_id': 'task', 'conversation_id': 'fixture', 'turn': 2, 'split': 'train'},
        {'task_id': 'other', 'conversation_id': 'fixture', 'turn': 3, 'split': 'dev'}]}))
    with pytest.raises(ValueError, match='conversations overlap'):
        observed_cohort([previous, capture], packets, sources, split='train', limit=1, cohort_manifest=manifest)
    await service.aclose()
    service.store.close()
