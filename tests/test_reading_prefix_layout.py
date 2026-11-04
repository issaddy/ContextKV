from dataclasses import replace
import json

from controller.context_schema import ContextBudget, ContextPlan, EvidenceSet, EvidenceUnit, Need, ReadTask, SourceSpan, TaskState
from controller.evidence import make_evidence
from controller.reading import build_read_plan, stable_prefix_input_ids, step_messages
from controller.templates import ByteTemplate


def material(version="v1"):
    block = make_evidence("记录 S01：12 kg。", "register", version=version)
    return EvidenceUnit("unit", (SourceSpan("span", block.key, 0, len(block.text)),), block.text)


def plan(unit, question="Count records", need="count"):
    state = TaskState("task", question, needs=(Need(need, operation=need),))
    return ContextPlan("plan", state, EvidenceSet("set", (unit,)), read_layout="source_first")


def test_source_prefix_survives_need_step_and_runtime_alias_changes():
    template, unit = ByteTemplate(), material()
    first = plan(unit)
    task = ReadTask("read:0", "READ", ("count",), ("unit",))
    renamed = replace(unit, unit_id="need:1:unit", source_spans=(replace(unit.source_spans[0], span_id="need:1:span"),))
    second = plan(renamed, "Sum mass in kg", "sum")
    later = ReadTask("hop:4", "READ", ("sum",), (renamed.unit_id,))
    left = step_messages(first, task, (unit,))
    right = step_messages(second, later, (renamed,))
    assert left[:2] == right[:2]
    assert left[2] != right[2]
    rendered = template.render(left)
    prefix = stable_prefix_input_ids(template, first, task, (unit,), rendered)
    assert prefix == template.render(right).input_ids[:len(prefix)]
    assert unit.text in bytes(prefix).decode()
    assert "Count records" not in bytes(prefix).decode()
    aliases = json.loads(right[2].content.split("\nTask:\n", 1)[1])["source_aliases"]
    assert aliases[0]["spans"][0]["span_id"] == "need:1:span"


def test_source_version_and_original_offsets_change_prefix_identity():
    template, original, changed = ByteTemplate(), material(), material("v2")
    task = ReadTask("read:0", "READ", ("count",), ("unit",))
    first = step_messages(plan(original), task, (original,))
    second = step_messages(plan(changed), task, (changed,))
    assert first[1] != second[1]
    shifted = replace(original, source_spans=(replace(original.source_spans[0], start=1),))
    assert first[1] != step_messages(plan(shifted), task, (shifted,))[1]
    assert template.render(first).input_ids != template.render(second).input_ids


def test_canonical_windows_keep_rows_coordinates_and_prefix_across_task_budgets():
    from types import SimpleNamespace
    from adapters.evidence_pool import SourceDirectory
    from controller.source_windows import SOURCE_WINDOW_TOKENS

    template = ByteTemplate()
    text = ''.join(f'Record R{i:02}: mass {i} kg.\n' for i in range(23))
    block = make_evidence(text, 'fixed-table', version='v2')
    directory = SourceDirectory(SimpleNamespace(blocks=(block,), source_metadata={}),
                                None, template.tokenizer)
    units = directory.full_scope((block.key,)).units
    assert ''.join(unit.text for unit in units) == text
    assert len(units) == 3
    for unit in units:
        span = unit.source_spans[0]
        assert text[span.start:span.end] == unit.text
        assert unit.body_tokens <= SOURCE_WINDOW_TOKENS
    selected = EvidenceSet('fixed', units, source_blocks=(block,))
    left = build_read_plan(TaskState('count', 'Count the records', needs=(Need('n', operation='count'),)),
        selected, 'all_supplied_documents', ContextBudget(read_output_tokens=4096), template=template,
        read_layout='source_first', read_packing='source_windows', output_tokens_per_fact=None)
    right = build_read_plan(TaskState('sum', 'Sum the record masses. '+('Use this table. '*25),
        needs=(Need('m', operation='sum'),)), selected, 'all_supplied_documents',
        ContextBudget(read_input_tokens=16000, read_output_tokens=8192), template=template,
        read_layout='source_first', read_packing='source_windows', output_tokens_per_fact=None)
    assert left.coverage_units == right.coverage_units
    assert len(left.steps) == len(right.steps) == len(units)
    for a, b in zip(left.steps, right.steps, strict=True):
        window = tuple(unit for unit in units if unit.unit_id in a.input_unit_ids)
        first = step_messages(left, a, window)
        second = step_messages(right, b, window)
        assert first[:2] == second[:2]
        prefix = stable_prefix_input_ids(template, left, a, window, template.render(first))
        assert prefix == template.render(second).input_ids[:len(prefix)]


def test_canonical_table_windows_keep_shared_context_out_of_data_dependencies():
    from types import SimpleNamespace
    from adapters.evidence_pool import SourceDirectory

    text = ('Table 1: Mass\n| Record | Mass (kg) |\n| --- | --- |\n'
            + ''.join(f'| R{i} | {i} |\n' for i in range(12))
            + 'Note: Approved records only.\n')
    block = replace(make_evidence(text, 'explicit-table', version='v2'),
                    span=(100, 100+len(text)))
    directory = SourceDirectory(SimpleNamespace(blocks=(block,), source_metadata={}),
                                None, ByteTemplate().tokenizer)
    units = directory.full_scope((block.key,)).units
    assert ''.join(unit.text for unit in units) == text
    data = tuple(unit for unit in units if any(f'| R{i} |' in unit.text for i in range(12)))
    assert len(data) == 2
    data_ids = {unit.unit_id for unit in data}
    lookup = {unit.unit_id: unit for unit in units}
    for unit in units:
        span = unit.source_spans[0]
        assert span.source_key == block.key
        assert text[span.start-100:span.end-100] == unit.text
    for unit in data:
        assert not data_ids.intersection(unit.dependencies)
        context = ''.join(lookup[identity].text for identity in unit.dependencies)
        assert 'Table 1: Mass' in context and 'Mass (kg)' in context
        assert 'Note: Approved records only.' in context
        assert unit.table_id is not None
    selected = EvidenceSet('table', units, source_blocks=(block,))
    state = TaskState('count', 'Count all approved records.', needs=(Need('n', operation='count'),))
    reading = build_read_plan(state, selected, 'all_supplied_documents',
        ContextBudget(read_output_tokens=4096), template=ByteTemplate(),
        read_layout='source_first', read_packing='source_windows', output_tokens_per_fact=None)
    assert len(reading.steps) == len(data)
    assert set(reading.coverage_units) == set(lookup)
    for step in reading.steps:
        assert len(set(step.input_unit_ids) & data_ids) == 1
        assert set(lookup) - data_ids <= set(step.input_unit_ids)


def test_committed_windows_do_not_change_original_metadata_or_next_turn_prefixes():
    from types import SimpleNamespace
    from adapters.evidence_pool import SourceDirectory
    from adapters.history_state import restore_source_metadata
    from controller.reading import stable_source_payload

    text = ('Table 1: Mass\n| Record | Mass (kg) |\n| --- | --- |\n| A | 1 |\n| B | 2 |\n\n'
            'Table 2: Mass\n| Record | Mass (kg) |\n| --- | --- |\n| C | 3 |\n| D | 4 |\n')
    block = make_evidence(text, 'two-tables', version='v1')
    corpus = SimpleNamespace(blocks=(block,), source_metadata={})
    directory = SourceDirectory(corpus, None, ByteTemplate().tokenizer)
    before = directory.full_scope((block.key,)).units
    restore_source_metadata(directory, before)
    directory.add_archive((block,))
    after = directory.full_scope((block.key,)).units
    assert stable_source_payload(before) == stable_source_payload(after)
    assert len({unit.table_id for unit in after if unit.table_id is not None}) == 2
    assert 'table_id' not in corpus.source_metadata[block.key]
    assert 'heading' not in corpus.source_metadata[block.key]


def test_full_scope_provider_builds_one_scan_without_support_or_set_search():
    from types import SimpleNamespace
    from unittest.mock import Mock
    from adapters.evidence_pool import SourceDirectory
    from adapters.study_retrieval import StudyEvidenceProvider
    from controller.context_schema import RetrievalState
    from controller.schema import SessionSnapshot

    template = ByteTemplate()
    blocks = tuple(make_evidence(f'Record R{i} weighs {i+1} kg.', f'ledger-{i}') for i in range(2))
    state = TaskState('task', 'Count every record and describe the ledger.', needs=(
        Need('n', operation='count', scope='all_supplied_documents'),
        Need('description', operation='lookup', scope='selected_sources')),
        source_scope=tuple(block.key for block in blocks), current_turn_id=1)
    for coverage, supplied in ((coverage, supplied)
            for coverage in ('all_supplied_documents', 'selected_sources') for supplied in (True, False)):
        corpus = SimpleNamespace(blocks=blocks, source_metadata={})
        interpreter = SimpleNamespace(model=SimpleNamespace(), last_metrics={'task_model_ms': 0},
                                      interpret=lambda turn, memory: state)
        directory = SourceDirectory(corpus, None, template.tokenizer, task_interpreter=interpreter)
        directory.retrieve_pool = Mock(side_effect=AssertionError('full scope has no retrieval choice'))
        serving = SimpleNamespace(corpus_snapshot_id='corpus', corpus_id='corpus', retriever=SimpleNamespace(),
            retrieval_state=lambda snapshot: RetrievalState(0, source_index=directory))
        support = SimpleNamespace(usage={}, predict=Mock(side_effect=AssertionError('scan needs no support search')))
        provider = StudyEvidenceProvider({'study': {'strategy': 'B7'},
            'reading': {'coverage_mode': coverage, 'layout': 'source_first',
                        'output_tokens': 4096, 'output_tokens_per_fact': None}},
            template, serving, directory, support_backend=support)
        provider._rank_supplied = Mock(side_effect=AssertionError('scan needs no relevance ranking'))
        selection = provider._retrieve(SessionSnapshot('session'), state.raw_question, 'request',
            supplied_blocks=blocks if supplied else None, supplied_coverage_scope=coverage)
        assert len(selection.context_plans) == 1
        reading = selection.context_plans[0]
        assert reading.coverage_scope == 'all_supplied_documents'
        assert reading.task_state.needs == state.needs
        assert set(reading.task_state.source_scope) == {block.key for block in blocks}
        assert set(reading.coverage_units) == {unit.unit_id for unit in reading.selected_set.selected_units}
        assert len(reading.steps) == 2
        trace = provider.traces['session', 'request']
        assert trace['support_schedule'] == trace['search_trace'] == ()
        assert not trace['selection_policy_heads_used']
        assert dict(reading.auxiliary_ms)['support'] == dict(reading.auxiliary_ms)['search'] == 0
