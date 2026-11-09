"""Cache paths propose packages; original tokens and the existing packer decide legality."""
from dataclasses import replace
from pathlib import Path

import pytest

from adapters.contextpilot import SourceOrderIndex
from controller.action_proposals import build_observed_action_plans
from controller.context_budget import bind_answer_window, completion_reserve
from controller.context_schema import (ContextBudget, EvidenceSet, Need, ReadResult, ReadTask,
                                       ReadingOutcome, TaskState)
from controller.engine_client import EngineInfo
from controller.reading import build_read_plan, read_output_reservation, regroup_pending_reads
from controller.request_compiler import RequestCompiler
from controller.schema import CompletionReceipt, SessionSnapshot
from controller.templates import ByteTemplate
from reading_helpers import material


def fixture_plan():
    blocks, units = zip(*(material('Material '+name+'. '*150, name) for name in ('a', 'c', 'b', 'd')))
    template, snapshot = ByteTemplate(), SessionSnapshot('s')
    plan = build_read_plan(TaskState('t', 'Compare sources', needs=(Need('n', operation='infer'),)),
        EvidenceSet('s', units, source_blocks=blocks), 'selected_sources',
        ContextBudget(read_input_tokens=6000, read_output_tokens=1000), template=template,
        read_layout='source_first', read_packing='source_windows')
    assert [task.input_unit_ids for task in plan.steps] == [('a', 'c'), ('b', 'd')]
    return plan, template, snapshot




def test_same_package_prefix_does_not_justify_an_extra_split():
    plan, template, snapshot = fixture_plan()
    assert regroup_pending_reads(ReadingOutcome(plan), ('a',), template, snapshot) is None
    assert regroup_pending_reads(ReadingOutcome(plan), ('a', 'c'), template, snapshot) is None
    completed = ReadResult(plan.steps[0].step_id, covered_units=('a', 'c'))
    assert regroup_pending_reads(ReadingOutcome(plan, results=(completed,)), ('a', 'b'), template, snapshot) is None


def test_preferred_packages_include_dependency_closure():
    plan, template, snapshot = fixture_plan()
    a, c, b, d = plan.selected_set.selected_units
    units = (a, replace(c, dependencies=('a',)), b, d)
    changed = build_read_plan(plan.task_state, replace(plan.selected_set, selected_units=units),
        plan.coverage_scope, replace(plan.budget, read_input_tokens=10000), template=template,
        read_layout='source_first', read_packing='source_windows', preferred_groups=(('c', 'b'),))
    assert changed.steps[0].input_unit_ids == ('c', 'b', 'a')
    assert set(changed.steps[-1].input_unit_ids) == {'d'}


def test_regroup_preserves_completed_work_and_redistributes_one_conclusion_budget():
    plan, template, snapshot = fixture_plan()
    block, unit = material('Observed source.', 'done')
    prototype = plan.steps[0]
    finished = replace(prototype, step_id='done', input_unit_ids=('done',))
    pending = tuple(replace(prototype, step_id='pending:'+str(index), input_unit_ids=identities,
        predecessor_step_ids=('done',)) for index, identities in enumerate((('a',), ('c',), ('b', 'd'))))
    reduction = ReadTask('later', 'REDUCE', ('n',),
        predecessor_step_ids=tuple(task.step_id for task in pending))
    plan = replace(plan, selected_set=replace(plan.selected_set,
        selected_units=(*plan.selected_set.selected_units, unit),
        source_blocks=(*plan.selected_set.source_blocks, block)),
        steps=(finished, *pending, reduction), coverage_units=(*plan.coverage_units, 'done'),
        answer_allocation=None)
    plan = bind_answer_window(plan, template, snapshot)
    state = ReadingOutcome(plan, results=(ReadResult('done', covered_units=('done',)),))
    changed = regroup_pending_reads(state, ('a', 'b'), template, snapshot)
    reads = tuple(task for task in changed.steps if task.kind == 'READ')
    assert [task.input_unit_ids for task in reads] == [('done',), ('a', 'b'), ('c', 'd')]
    assert changed.selected_set == plan.selected_set and changed.coverage_units == plan.coverage_units
    assert reads[0] == plan.steps[0]
    assert sum(dict(task.retained_conclusion_tokens)['n'] for task in reads) == dict(
        changed.answer_allocation.need_tokens)['n']
    assert changed.steps[-1].predecessor_step_ids == tuple(task.step_id for task in reads[1:])
    by_unit = {unit.unit_id: unit for unit in changed.selected_set.selected_units}
    for task in reads[1:]:
        candidate, = RequestCompiler(template).compile_read(changed, task,
            tuple(by_unit[identity] for identity in task.input_unit_ids), (), snapshot,
            outputs=(task.output_budget,))
        assert candidate.feasible


def test_regroup_measures_the_original_numeric_scopes_after_source_completion():
    template, snapshot = ByteTemplate(), SessionSnapshot('scopes')
    blocks, units = zip(*(material('Record '+name+' qualifies.', name)
                          for name in ('done', 'a', 'c', 'b', 'd')))
    units = tuple(replace(unit, need_ids=('n',) if unit.unit_id == 'done' else ('n', 'm')) for unit in units)
    state = TaskState('t', 'Count each source population.',
                      needs=(Need('n', operation='count'), Need('m', operation='count')))
    plan = build_read_plan(state, EvidenceSet('s', units, source_blocks=blocks), 'selected_sources',
        ContextBudget(read_input_tokens=16000, read_output_tokens=600), template=template,
        read_layout='source_first', read_packing='source_windows')
    assert [task.input_unit_ids for task in plan.steps] == [('done',), ('a',), ('c',), ('b',), ('d',)]
    assert len(plan.steps[1].shared_numeric_scopes) == 2
    outcome = ReadingOutcome(plan, results=(ReadResult(plan.steps[0].step_id, covered_units=('done',)),))
    preferred = tuple(unit for unit in units if unit.unit_id in {'a', 'b'})
    assert read_output_reservation(state, plan.steps[1], preferred, template.tokenizer) > 600
    assert regroup_pending_reads(outcome, ('a', 'b'), template, snapshot) is None


def test_numeric_completion_reserves_other_declared_reads_without_optional_reductions():
    plan, template, snapshot = fixture_plan()
    plan = replace(plan, task_state=replace(plan.task_state,
        needs=(Need('n', operation='count'), Need('lookup', operation='lookup'))),
        steps=(replace(plan.steps[0], operation='count'), replace(plan.steps[1], need_ids=('lookup',)),
            ReadTask('optional', 'REDUCE', ('lookup',), operation='summarize', output_budget=777)),
        answer_allocation=None)
    plan = bind_answer_window(plan, template, snapshot)
    reserve = completion_reserve(plan, (), plan.steps[0])
    by_unit = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    future_body = sum({by_unit[identity].text: by_unit[identity].body_tokens
                       for identity in plan.steps[1].input_unit_ids}.values())
    assert reserve.future_input_tokens == plan.answer_allocation.fixed_input_tokens+future_body
    assert reserve.future_output_tokens == plan.budget.output_tokens+plan.steps[1].output_budget
    assert reserve.future_steps == 3  # remaining READ, count, ANSWER


def _observed_facts(plan):
    from controller.claim_graph import ClaimGraph
    from controller.context_schema import EvidenceClaim

    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    results = tuple(ReadResult(task.step_id, claims=tuple(EvidenceClaim(task.step_id+':'+identity,
        'n', 'extract', text=units[identity].text, entity=identity, value='1',
        source_spans=units[identity].source_spans) for identity in task.input_unit_ids),
        covered_units=task.input_unit_ids) for task in plan.steps)
    claims = tuple(claim for result in results for claim in result.claims)
    plan = replace(plan, answer_plan=ClaimGraph(claims).answer_plan(plan.task_state, results=results))
    return ReadingOutcome(plan, results=results, claims=claims)


def test_completion_compiles_known_reduce_and_keeps_unconsumed_model_parents():
    plan, template, snapshot = fixture_plan()
    state = _observed_facts(plan)
    reduce = ReadTask('compare', 'REDUCE', ('n',),
        input_claim_ids=tuple(claim.claim_id for claim in state.claims),
        predecessor_step_ids=tuple(result.step_id for result in state.results),
        operation='compare', output_budget=300, retained_conclusion_tokens=plan.answer_allocation.need_tokens)
    plan = replace(state.plan, steps=(*plan.steps, reduce))
    state = replace(state, plan=plan)
    compiler = RequestCompiler(template)
    final, = compiler.compile_answers(snapshot, plan.task_state.raw_question, plan,
        outcome=state, representations=('facts',), actions=('REBUILD',), layouts=('existing',))
    compiled, = compiler.compile_read(plan, reduce, (), state.claims, snapshot, outputs=(300,))
    future = completion_reserve(plan, state.results, None, read_all=True,
        template=template, snapshot=snapshot, claims=state.claims)
    assert future.future_input_tokens == len(compiled.input_ids)+len(final.input_ids)+300
    current = completion_reserve(plan, state.results, reduce, compiled,
        template=template, snapshot=snapshot, claims=state.claims)
    # Unknown model lineage cannot erase its parents in advance.
    assert current.future_input_tokens == len(final.input_ids)+300


@pytest.mark.asyncio
async def test_completion_uses_the_actual_known_cpu_aggregate_frontier(tmp_path):
    from bench.mock_engine import MockEngine
    from controller.execution import ReadingExecutor
    from controller.session_store import SessionStore

    plan, template, snapshot = fixture_plan()
    plan = replace(plan, task_state=replace(plan.task_state, needs=(Need('n', operation='count'),)))
    state = replace(_observed_facts(plan), auxiliary_usage={'input_tokens': 0, 'output_tokens': 0})
    aggregate = ReadTask('count', 'REDUCE', ('n',), operation='count',
        input_claim_ids=tuple(claim.claim_id for claim in state.claims),
        predecessor_step_ids=tuple(result.step_id for result in state.results))
    plan = replace(state.plan, steps=(*plan.steps, aggregate))
    before = completion_reserve(plan, state.results, aggregate,
        template=template, snapshot=snapshot, claims=state.claims)
    store, engine = SessionStore(tmp_path/'aggregate.db'), MockEngine()
    store.read_snapshot(snapshot.session_id)
    attempt = store.begin_attempt(snapshot.session_id, 'count', 'count-hash',
                                  plan.task_state.raw_question, snapshot.version)
    actual = await ReadingExecutor(store, engine, template).execute_one(
        attempt.attempt_id, snapshot, plan, aggregate, replace(state, plan=plan))
    final, = RequestCompiler(template).compile_answers(snapshot, plan.task_state.raw_question,
        actual.plan, outcome=actual, representations=('facts',), actions=('REBUILD',), layouts=('existing',))
    assert actual.results[-1].claims[0].value == 4
    assert before.future_input_tokens == len(final.input_ids)
    assert before.future_input_tokens < completion_reserve(state.plan, state.results, None,
        template=template, snapshot=snapshot, claims=state.claims).future_input_tokens
    store.close()


def test_completion_prices_future_read_envelopes_parent_use_and_selected_output():
    plan, template, snapshot = fixture_plan()
    plan = replace(plan, budget=replace(plan.budget, read_output_tokens=2000),
        steps=tuple(replace(task, output_budget=2000) for task in plan.steps))
    first, future = plan.steps
    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    compiler = RequestCompiler(template)
    current, = compiler.compile_read(plan, first, tuple(units[key] for key in first.input_unit_ids),
                                     (), snapshot, outputs=(2000,))
    short, full = compiler.compile_read(plan, future, tuple(units[key] for key in future.input_unit_ids),
                                        (), snapshot, outputs=(1000, 2000))
    normal = completion_reserve(plan, (), first, current, read_all=True, template=template, snapshot=snapshot)
    selected = completion_reserve(plan, (), first, current, read_all=True, template=template, snapshot=snapshot,
                                  read_inputs={future.step_id: short})
    assert selected.future_output_tokens == normal.future_output_tokens-1000
    assert selected.future_input_tokens == normal.future_input_tokens-1000+len(short.input_ids)-len(full.input_ids)
    structural = completion_reserve(plan, (), first, current, read_all=True,
                                    read_inputs={future.step_id: short})
    assert structural.future_input_tokens == plan.answer_allocation.fixed_input_tokens+len(short.input_ids)
    chained = replace(plan, steps=(first, replace(future, predecessor_step_ids=(first.step_id,))))
    dependent = completion_reserve(chained, (), first, current, read_all=True, template=template, snapshot=snapshot)
    assert dependent.future_input_tokens == normal.future_input_tokens+current.output_budget


@pytest.fixture
def indexed_compiler():
    checkout = Path(__file__).resolve().parents[1]/'vendor/contextpilot'
    if not (checkout/'contextpilot/server/live_index.py').exists():
        pytest.skip('prepared author ContextPilot checkout unavailable')
    return RequestCompiler(ByteTemplate(), SourceOrderIndex(checkout))


def test_executed_path_is_not_a_suffix_cache_and_is_scope_bound(indexed_compiler):
    compiler = indexed_compiler
    plan, _, snapshot = fixture_plan()
    units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    task = replace(plan.steps[0], input_unit_ids=('a', 'b'))
    info = EngineInfo('restart', fingerprint=compiler.template.fingerprint)
    prior = compiler.compile_read(plan, task, (units['a'], units['b']), (), snapshot,
        outputs=(1000,), cache_scope=info)[0]
    receipt = CompletionReceipt('rid', 'completed', 'epoch', 1, 'restart', prior.input_hash, 'stop', True)
    compiler.observe_executed(prior, receipt, cache_config_hash=info.cache_config_hash, engine_restart_id='restart')
    scope = compiler._scope(info)
    assert compiler.index.executed_paths((prior.source_ids[1],), limit=2, **scope) == ()
    assert compiler.index.executed_paths(prior.source_ids, limit=2, **scope) == (prior.source_ids,)
    assert compiler.index.executed_paths(prior.source_ids, limit=2,
        **{**scope, 'engine_restart_id': 'new-process'}) == ()
    original, changed = compiler.read_organizations(ReadingOutcome(plan), snapshot, cache_scope=info, limit=2)
    assert original == plan
    first = changed.steps[0]
    candidates = compiler.compile_read(changed, first, tuple(units[uid] for uid in first.input_unit_ids), (), snapshot,
        outputs=(1000,), cache_scope=info)
    assert candidates[0].source_ids == prior.source_ids
    end = min(prior.source_prefix_tokens, candidates[0].source_prefix_tokens)
    assert end > 0 and candidates[0].input_ids[:end] == prior.input_ids[:end]


def test_captured_organization_keeps_complete_catalog_and_rejects_scope_changes():
    plan, template, snapshot = fixture_plan()
    state = ReadingOutcome(plan)
    changed = regroup_pending_reads(state, ('a', 'b'), template, snapshot)
    branches, _ = build_observed_action_plans(state, snapshot, template, reader_checkpoint='reader',
        read_choices=2, read_organizations=(changed,))
    reads = [branch for branch in branches if branch.semantic_kind == 'read_once']
    assert len(reads) == 2
    assert [step.task.kind for step in reads[1].steps] == ['READ', 'ANSWER']
    assert len(reads[1].context_plan.steps) == 2
    assert reads[1].steps[0].task.input_unit_ids == ('a', 'b')
    assert reads[1].context_plan.steps[1].input_unit_ids == ('c', 'd')
    assert len(branches[0].context_plan.steps) == 2  # DIRECT still knows the complete source scope.
    with pytest.raises(ValueError, match='reading obligations'):
        build_observed_action_plans(state, snapshot, template, reader_checkpoint='reader', read_choices=2,
            read_organizations=(replace(changed, steps=changed.steps[:1]),))
