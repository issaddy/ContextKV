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
    aliases = json.loads(right[2].content)["source_aliases"]
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


def test_source_windows_do_not_mix_arbitrary_documents_within_one_cached_read():
    first, second = material(), material("v2")
    second = replace(second, unit_id="other", source_spans=(replace(second.source_spans[0], span_id="other-span"),))
    state = TaskState("task", "Compare both versions", needs=(Need("n"),))
    selected = EvidenceSet("set", (first, second))
    budget = ContextBudget(window_tokens=32768, read_input_tokens=24000, read_output_tokens=4096)
    mixed = build_read_plan(state, selected, "selected_sources", budget)
    separate = build_read_plan(state, selected, "selected_sources", budget,
        read_layout="source_first", read_packing="source_windows")
    assert len(mixed.steps) == 1
    assert tuple(task.input_unit_ids for task in separate.steps) == (("unit",), ("other",))
    assert set(separate.coverage_units) == {"unit", "other"}


def test_read_packing_budget_tracks_visible_needs_when_offline_task_binding_changes():
    unit = material()
    state = TaskState("task", "List records and explain the procedure", needs=(
        Need("new-enumeration", operation="enumerate"), Need("new-procedure", operation="procedure")))
    budget = ContextBudget(window_tokens=16000, read_input_tokens=12000, read_output_tokens=2048)
    current = build_read_plan(state, EvidenceSet("set", (unit,)), "selected_sources", budget)
    old_assignment = replace(unit, need_ids=("old-unrelated-need",))
    rebound = build_read_plan(state, EvidenceSet("set", (old_assignment,)), "selected_sources", budget)
    assert current.steps == rebound.steps
    assert step_messages(current, current.steps[0], (unit,)) == step_messages(
        rebound, rebound.steps[0], (old_assignment,))
