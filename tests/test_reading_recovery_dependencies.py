"""Recovery packets retain source geometry and the headers numeric facts need."""
from dataclasses import replace
import json
import unittest

from controller.context_schema import (
    ContextPlan, EvidenceBlock, EvidenceKey, EvidenceSet, EvidenceUnit, Need,
    NumericReadScope, OpenNeed, ReadResult, ReadTask, ReadingOutcome, SourceSpan, StepRecord, TaskState,
)
from controller.read_recovery import recover_failed_reads, schedule_field_repair
from controller.reading import numeric_aliases, parse_read_result
from controller.templates import ByteTemplate


def fixture(text, *, header=None):
    key = EvidenceKey("c", "company", "body", "2024", "body-hash")
    source = EvidenceUnit("body", (SourceSpan("body:span", key, 0, len(text)),), text,
        dependencies=("header",) if header is not None else (), need_ids=("sum",))
    units = (source,)
    blocks = (EvidenceBlock(key, "fixture", "report", text),)
    if header is not None:
        header_key = replace(key, passage_id="header", content_sha256="header-hash")
        header_unit = EvidenceUnit("header", (SourceSpan("header:span", header_key, 0, len(header)),),
            header, need_ids=("sum",))
        units = (header_unit, source)
        blocks = (EvidenceBlock(header_key, "fixture", "header", header), *blocks)
    scope = NumericReadScope("g", ("sum",), ("year",), ("sum",))
    task = ReadTask("read:original", "READ", ("sum",), tuple(unit.unit_id for unit in units),
        shared_numeric_scopes=(scope,), condition_ids=("year",))
    need = Need("sum", operation="sum", condition_ids=("year",), scope="all_supplied_documents")
    plan = ContextPlan("plan", TaskState("task", "求2024年流动资产合计", needs=(need,)),
        EvidenceSet("evidence", units, source_blocks=blocks), coverage_scope="all_supplied_documents",
        steps=(task,), coverage_units=tuple(unit.unit_id for unit in units))
    return plan, task


def state_for(plan, task, *, truncated=False):
    gap = OpenNeed("sum", "READ recovery required: truncated_output" if truncated else
        "流动资产合计", () if truncated else ("甲公司",), "current" if truncated else "2024")
    result = ReadResult(task.step_id, covered_units=() if truncated else task.input_unit_ids,
        open_needs=(gap,))
    record = StepRecord("attempt", task.step_id, task, input_ids=(1, 2), output_ids=(3,), result=result,
        metrics={"read_failure": "truncated_output", "read_recovery_pending": True} if truncated else {})
    return ReadingOutcome(plan, (result,), (record,), open_needs=(gap,))


def parse_target(plan, task, excerpt):
    by_id = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
    units = tuple(by_id[identity] for identity in task.input_unit_ids)
    _, sources, _ = numeric_aliases(task, units)
    target = next(unit for unit in units if excerpt in unit.text and not unit.unit_id.startswith("repair-header:"))
    alias = next(alias for alias, span_id in sources.items() if span_id in {span.span_id for span in target.source_spans})
    data = {"format_version": 5, "claims": [], "facts": [{"s": [alias], "t": excerpt,
        "g": task.shared_numeric_scopes[0].scope_id, "n": ["n0"], "e": "甲公司",
        "r": "流动资产合计", "v": "100", "u": "万元", "p": []}],
        "support": ["n0"], "conflicts": [], "open_needs": []}
    return parse_read_result(task, json.dumps(data, ensure_ascii=False), units)


class RecoveryDependencyTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = ByteTemplate().tokenizer

    def assert_closed_original_geometry(self, plan, task):
        units = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
        blocks = {block.key: block for block in plan.selected_set.source_blocks}
        inputs = set(task.input_unit_ids)
        for identity in inputs:
            unit = units[identity]
            self.assertTrue(set(unit.dependencies) <= inputs)
            for span in unit.source_spans:
                block = blocks[span.source_key]
                self.assertEqual(unit.text, block.text[span.start:span.end])

    def test_targeted_window_inherits_short_table_entity_period_and_unit_header(self):
        excerpt = "流动资产合计 | 100"
        text = "甲公司 | 2024年 | 单位：万元\n"+"无关说明\n"*700+excerpt+"\n"
        plan, task = fixture(text)
        state = schedule_field_repair(state_for(plan, task), self.tokenizer)
        repair = state.plan.steps[-1]
        self.assertEqual("repair_extract", repair.operation)
        self.assert_closed_original_geometry(state.plan, repair)
        units = {unit.unit_id: unit for unit in state.plan.selected_set.selected_units}
        self.assertTrue(any(identity.startswith("repair-header:") for identity in repair.input_unit_ids))
        self.assertLess(sum(len(units[identity].text) for identity in repair.input_unit_ids), len(text))
        result = parse_target(state.plan, repair, excerpt)
        self.assertEqual(1, len(result.claims))
        self.assertEqual(("100", "万元", ("year",)),
            (result.claims[0].value, result.claims[0].unit, result.claims[0].conditions))
        self.assertFalse(result.open_needs)

    def test_targeted_repair_keeps_existing_dependency_from_another_passage(self):
        excerpt = "流动资产合计 100"
        plan, task = fixture(excerpt, header="甲公司 2024年 单位：万元")
        state = schedule_field_repair(state_for(plan, task), self.tokenizer)
        repair = state.plan.steps[-1]
        self.assertIn("header", repair.input_unit_ids)
        self.assert_closed_original_geometry(state.plan, repair)
        result = parse_target(state.plan, repair, excerpt)
        self.assertEqual("100", result.claims[0].value)
        self.assertFalse(result.open_needs)

    def test_split_tail_preserves_single_line_entity_prefix(self):
        excerpt = "流动资产合计100万元"
        text = "甲公司："+"其他字段10万元；"*18+excerpt+"；总资产500万元。"
        plan, task = fixture(text)
        state = recover_failed_reads(state_for(plan, task, truncated=True), self.tokenizer)
        children = tuple(task for task in state.plan.steps if task.step_id != "read:original")
        self.assertEqual(2, len(children))
        by_id = {unit.unit_id: unit for unit in state.plan.selected_set.selected_units}
        tail = next(task for task in children if any(excerpt in by_id[identity].text for identity in task.input_unit_ids))
        self.assert_closed_original_geometry(state.plan, tail)
        self.assertTrue(any(identity.startswith("repair-header:") for identity in tail.input_unit_ids))
        result = parse_target(state.plan, tail, excerpt)
        self.assertEqual("100", result.claims[0].value)
        self.assertFalse(result.open_needs)

    def test_table_row_with_two_period_columns_stays_unresolved(self):
        excerpt = "流动资产合计 | 100 | 80"
        text = "甲公司 | 2024年 | 2023年 | 单位：万元\n"+"无关说明\n"*700+excerpt+"\n"
        plan, task = fixture(text)
        state = schedule_field_repair(state_for(plan, task), self.tokenizer)
        result = parse_target(state.plan, state.plan.steps[-1], excerpt)
        self.assertFalse(result.claims)
        self.assertTrue(result.open_needs)


if __name__ == "__main__":
    unittest.main()
