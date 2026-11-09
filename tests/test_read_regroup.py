"""Cache paths propose packages; original tokens and the existing packer decide legality."""
from dataclasses import replace
from pathlib import Path

import pytest

from adapters.contextpilot import SourceOrderIndex
from controller.action_proposals import build_observed_action_plans
from controller.context_schema import ContextBudget, EvidenceSet, Need, ReadResult, ReadingOutcome, TaskState
from controller.engine_client import EngineInfo
from controller.reading import build_read_plan, regroup_pending_reads
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
