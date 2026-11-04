from dataclasses import replace
import json

from controller.context_schema import ContextBudget, ContextPlan, EvidenceSet, EvidenceUnit, Need, ReadTask, SourceSpan, TaskState
from controller.evidence import make_evidence
from controller.reading import build_read_plan, stable_prefix_input_ids, stable_prefix_messages, step_messages
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


def test_source_prefix_is_shared_across_numeric_and_multihop_operations():
    unit = material()
    selected = EvidenceSet("selected", (unit,))
    numeric = build_read_plan(TaskState("numeric", "Count records and sum mass", needs=(
        Need("count", operation="count"), Need("sum", operation="sum"))), selected,
        "selected_sources", ContextBudget(), read_layout="source_first")
    bridge = build_read_plan(TaskState("bridge", "Follow the record's supplier relation", needs=(
        Need("supplier", operation="infer"),)), selected, "selected_sources", ContextBudget(),
        read_layout="source_first")
    left = step_messages(numeric, numeric.steps[0], numeric.selected_set.selected_units)
    right = step_messages(bridge, bridge.steps[0], bridge.selected_set.selected_units)
    assert left[:2] == right[:2]
    assert left[-1].content.split("\nTask:\n", 1)[0] != right[-1].content.split("\nTask:\n", 1)[0]

    # The actual operation, protocol version and task aliases cannot precede sources.
    template = ByteTemplate()
    baseline = stable_prefix_messages(bridge, bridge.steps[0], (unit,))
    prefix = stable_prefix_input_ids(template, bridge, bridge.steps[0], (unit,), template.render(right))
    for operation in ("count", "sum", "extract", "infer"):
        task = replace(bridge.steps[0], operation=operation, step_id="changed:"+operation)
        messages = step_messages(bridge, task, (unit,))
        assert messages[:2] == baseline
        assert template.render(messages).input_ids[:len(prefix)] == prefix
        task_payload = json.loads(messages[-1].content.split("\nTask:\n", 1)[1])
        assert task_payload["source_aliases"][0]["spans"][0]["span_id"] == "span"
        assert "output_schema" in task_payload
    assert "population" not in baseline[0].content


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
