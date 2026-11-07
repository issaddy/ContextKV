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
    assert len(left.steps) == len(right.steps) == 1
    assert left.selected_set.selected_units == right.selected_set.selected_units == units
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
    assert len(reading.steps) == 1
    assert set(reading.coverage_units) == set(lookup)
    for step in reading.steps:
        assert set(step.input_unit_ids) & data_ids == data_ids
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

    # A publisher can replace a body without changing its version label. Such
    # sources cannot share window identities or cached document vectors.
    replacement = make_evidence(text.replace('| A | 1 |', '| A | 9 |'), 'two-tables', version='v1')
    directory.add_archive((replacement,))
    changed = directory.full_scope((replacement.key,)).units
    assert not {unit.unit_id for unit in before} & {unit.unit_id for unit in changed}
    assert all(directory.units_by_id[unit.unit_id] == unit for unit in (*before, *changed))


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
        from adapters.history_state import TaskPrediction
        interpreter = SimpleNamespace(model=SimpleNamespace(), last_metrics={'task_model_ms': 0},
                                      last_prediction=TaskPrediction('NEW'), last_read_contracts=(),
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
        assert len(reading.steps) == 1
        trace = provider.traces['session', 'request']
        assert trace['support_schedule'] == trace['search_trace'] == ()
        assert not trace['selection_policy_heads_used']
        assert dict(reading.auxiliary_ms)['support'] == dict(reading.auxiliary_ms)['search'] == 0


def test_request_compiler_binds_read_order_and_actual_encoded_source_geometry():
    from controller.engine_client import EngineInfo
    from controller.prefix_view import actual_source_geometry
    from controller.request_compiler import RequestCompiler
    from controller.schema import CompletionReceipt, SessionSnapshot

    class OrderingIndex:
        def __init__(self):
            self.observed = []
        def propose_orders(self, identities, **scope):
            return (tuple(identities), tuple(reversed(identities)))
        def observe_executed(self, identities, **execution):
            self.observed.append((identities, execution))
            return 'actual-request'

    template = ByteTemplate()
    unit = material()
    other = replace(unit, unit_id='other', text='Quoted "value"\nnext row',
        source_spans=(replace(unit.source_spans[0], span_id='other', start=50, end=73),))
    current = plan(unit)
    current = replace(current, selected_set=EvidenceSet('both', (unit, other)))
    task = ReadTask('read', 'READ', ('count',), ('unit', 'other'), input_budget=32000, output_budget=1024)
    snapshot = SessionSnapshot('session')
    scope = EngineInfo('restart', fingerprint=template.fingerprint)
    index = OrderingIndex()
    compiler = RequestCompiler(template, index)
    variants = compiler.compile_read(current, task, (unit, other), (), snapshot,
                                     outputs=(1024,), cache_scope=scope)
    assert len(variants) == 2 and not index.observed
    assert variants[0].read_unit_order == ('unit', 'other')
    assert variants[1].read_unit_order == ('other', 'unit')
    assert set(variants[0].source_ids) == set(variants[1].source_ids)
    lookup = {unit.unit_id: unit for unit in (unit, other)}
    for candidate in variants:
        ordered = tuple(lookup[identity] for identity in candidate.read_unit_order)
        bound = replace(task, input_unit_ids=candidate.read_unit_order)
        assert template.render(step_messages(current, bound, ordered, snapshot=snapshot)).input_ids == candidate.input_ids
        expected = sum(len(json.dumps(item.text, ensure_ascii=False)[1:-1].encode()) for item in ordered)
        geometry = actual_source_geometry(template, candidate)
        assert geometry['body_tokens'] == candidate.evidence_body_tokens == expected
        assert geometry['causal_prefix_input_ids'] == candidate.input_ids[:geometry['causal_boundary_tokens']]
    selected = variants[1]
    receipt = CompletionReceipt('rid', 'attempt', 'engine', 0, 'restart', selected.input_hash, 'length', True)
    observed = compiler.observe_executed(selected, receipt, cache_config_hash=scope.cache_config_hash,
                                         engine_restart_id=scope.engine_restart_id)
    assert observed['index_request_id'] == 'actual-request'
    assert index.observed[0][0] == selected.source_ids
    assert index.observed[0][1]['input_ids'] == selected.input_ids
    plain = RequestCompiler(template).compile_read(current, task, (unit, other), (), snapshot, outputs=(1024,))
    assert plain[0].input_ids == variants[0].input_ids and plain[0].source_ids == variants[0].source_ids

    from controller.schema import CacheQuote, CacheQuoteBatch
    left, right = variants
    a = CacheQuote(left.candidate_id, left.input_hash, len(left.input_ids), len(left.input_ids),
        ready_gpu_prefix_tokens=10, recoverable_prefix_tokens=len(left.input_ids)-20,
        host_only_load_tokens=20, native_load_estimate=20)
    b = CacheQuote(right.candidate_id, right.input_hash, len(right.input_ids), len(right.input_ids),
        ready_gpu_prefix_tokens=len(right.input_ids)-40)
    batch = CacheQuoteBatch('quote', 1, 1, 'restart', scope.cache_config_hash, 0, 0, (a, b),
                            fingerprint=template.fingerprint)
    assert compiler.choose_variant(variants, batch) is left  # full host restore exposes the complete prefix
    partial = replace(batch, quotes=(replace(a, native_load_estimate=10), b))
    assert compiler.choose_variant(variants, partial) is right  # partial restore leaves the first GPU boundary
    cheaper_host = replace(batch, quotes=(a, replace(b, ready_gpu_prefix_tokens=len(right.input_ids)-20)))
    assert compiler.choose_variant(variants, cheaper_host) is right


def test_request_compiler_answer_reorders_only_actual_excerpts_and_preserves_task():
    from controller.context_schema import TextRef
    from controller.engine_client import EngineInfo
    from controller.prefix_view import actual_source_geometry
    from controller.request_compiler import RequestCompiler
    from controller.schema import ExecutionLedger, Message, SessionSnapshot, TurnRecord
    from controller.templates import longest_common_prefix, verify_evidence_spans

    class ReverseIndex:
        def propose_orders(self, identities, **scope):
            return (tuple(identities), tuple(reversed(identities)))

    template = ByteTemplate()
    body = '记录 S01：12 kg。\n' + 'A complete original source row. ' * 10
    blocks = tuple(make_evidence(body, str(index)) for index in range(2))
    units = tuple(EvidenceUnit(str(index), (SourceSpan(str(index), block.key, 0, len(block.text)),), block.text)
                  for index, block in enumerate(blocks))
    first = TurnRecord(1, 'Earlier question', 'Earlier answer')
    second = TurnRecord(2, 'Different follow-up', 'Different answer')
    state = TaskState('task', 'Extract the source facts.', needs=(Need('extract', operation='extract'),),
        current_turn_id=2, working_history=(TextRef(1, 'user', 0, len(first.user_text)),))
    current = ContextPlan('plan', state, EvidenceSet('sources', units, source_blocks=blocks),
                          read_layout='source_first')
    snapshot = SessionSnapshot('s', raw_history=(first,))
    later = replace(snapshot, raw_history=(first, second))
    next_state = replace(state, raw_question='Extract the measured value.', current_turn_id=3,
        working_history=(TextRef(2, 'user', 0, len(second.user_text)),))
    next_plan = replace(current, task_state=next_state)
    scope = EngineInfo('restart', fingerprint=template.fingerprint)
    native = RequestCompiler(template)
    baseline, = native.compile_answers(snapshot, state.raw_question, current,
                                       layouts=('existing',), actions=('REBUILD',))
    variants = RequestCompiler(template, ReverseIndex()).compile_answers(snapshot, state.raw_question, current,
        layouts=('existing',), actions=('REBUILD',), cache_scope=scope)
    assert len(variants) == 2 and variants[0].input_ids == baseline.input_ids
    assert variants[1].layout_id == 'existing:contextpilot'
    assert variants[0].messages[0] == variants[1].messages[0]
    assert variants[0].messages[1] != variants[1].messages[1]
    assert variants[0].messages[2:-1] == variants[1].messages[2:-1]
    suffix = '\nCurrent task and active conditions:\n'
    assert variants[0].messages[-1].content.split(suffix)[1] == variants[1].messages[-1].content.split(suffix)[1]
    assert set(variants[0].source_ids) == set(variants[1].source_ids)
    assert variants[0].required_blocks == variants[1].required_blocks
    for candidate in variants:
        assert candidate.evidence_body_tokens == len(blocks[0].text.encode())
        assert all(span.message_index == 1 for span in candidate.evidence_spans)
        assert verify_evidence_spans(candidate.messages, candidate.evidence_spans, blocks,
                                     candidate.reference_map) == {block.key for block in blocks}

    compiled, = native.compile_answers(later, next_state.raw_question, next_plan,
                                       layouts=('existing',), actions=('REBUILD',))
    assert baseline.messages[2:-1] != compiled.messages[2:-1]
    boundary = actual_source_geometry(template, baseline)['causal_boundary_tokens']
    assert longest_common_prefix(baseline.input_ids, compiled.input_ids) >= boundary

    control, following = tuple(native.compile_answers(history, task.raw_question,
        replace(current, task_state=task, read_layout='task_first'),
        layouts=('existing',), actions=('REBUILD',))[0]
        for history, task in ((snapshot, state), (later, next_state)))
    assert control.messages[1].content.endswith(first.user_text)
    body_start = actual_source_geometry(template, control)['body_token_ranges'][0][0]
    assert longest_common_prefix(control.input_ids, following.input_ids) < body_start

    direct, followup = tuple(native.compile_answers(history, task.raw_question, retrieved=blocks)[0]
                            for history, task in ((snapshot, state), (later, next_state)))
    assert direct.messages[2:-1] != followup.messages[2:-1]
    assert longest_common_prefix(direct.input_ids, followup.input_ids) >= actual_source_geometry(
        template, direct)['causal_boundary_tokens']
    assert verify_evidence_spans(direct.messages, direct.evidence_spans, blocks,
                                 direct.reference_map) == {block.key for block in blocks}

    prior_block = make_evidence("Earlier reference remains in the committed input.", "prior")
    prior = RequestCompiler(template).compile_answers(snapshot, "First question", retrieved=(prior_block,))[0]
    execution = ExecutionLedger(1, "engine", 0, (*prior.messages, Message("assistant", "First answer")),
        prior.input_ids, (), prior.visible_blocks, template.fingerprint.template_hash,
        template.fingerprint.model_revision, snapshot.corpus_snapshot_id,
        prior.evidence_spans, prior.reference_map)
    committed = replace(snapshot, execution=execution, archive=(prior_block,), reference_map=prior.reference_map)
    plain = RequestCompiler(template).compile_answers(committed, state.raw_question, retrieved=blocks)
    variants = RequestCompiler(template, ReverseIndex()).compile_answers(committed, state.raw_question,
        retrieved=blocks, cache_scope=scope)
    for base in plain:
        original, reordered = (variant for variant in variants if variant.action == base.action)
        assert original.input_ids == base.input_ids
        assert reordered.layout_id == "existing:contextpilot"
        assert reordered.reference_map == base.reference_map and reordered.required_blocks == base.required_blocks
        if base.action == "APPEND":
            assert reordered.messages[:-1] == original.messages[:-1] == execution.messages
        else:
            assert reordered.messages[2:] == original.messages[2:]
        offset = len(prior.evidence_spans) if base.action == "APPEND" else 0
        assert reordered.evidence_spans[:offset] == prior.evidence_spans[:offset]
        assert reordered.source_ids[offset:] == original.source_ids[offset:][::-1]
