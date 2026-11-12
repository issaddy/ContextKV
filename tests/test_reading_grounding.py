"""Source-binding regressions, independent of GPU/model quality measurements."""
from dataclasses import replace
import json
import unittest

from controller.context_schema import (
    EvidenceKey, EvidenceUnit, NumericReadScope, OpenNeed, ReadResult, ReadTask, SourceSpan,
)
from controller.reading import (
    NUMERIC_GROUNDING_GAP_PREFIX, parse_read_result, unresolved_needs,
)


def source(text, identity="u", *, key=None, dependencies=()):
    key = key or EvidenceKey("corpus", "report", "p", "2024", "hash")
    return EvidenceUnit(identity, (SourceSpan(f"{identity}:span", key, 0, len(text)),),
        text, dependencies=dependencies)


def numeric_task(units, *, operations=("sum",)):
    ids = tuple(f"need:{index}" for index in range(len(operations)))
    return ReadTask("read:0", "READ", ids, tuple(unit.unit_id for unit in units),
        shared_numeric_scopes=(NumericReadScope("g0", ids, operations=operations),))


def numeric_output(text, *, entity="甲公司", relation="流动资产合计", value="100", unit="万元",
                   sources=("s0",), needs=("n0",)):
    return {"format_version": 5, "claims": [], "facts": [{"s": list(sources), "t": text,
        "g": "g0", "n": list(needs), "e": entity, "r": relation, "v": value, "u": unit, "p": []}],
        "support": list(needs), "conflicts": [], "open_needs": []}


class NumericSourceBindingTests(unittest.TestCase):
    def parse(self, text, **fact):
        units = (source(text),)
        task = numeric_task(units)
        result = parse_read_result(task, json.dumps(numeric_output(text, **fact), ensure_ascii=False), units)
        return result

    def assert_gap(self, result):
        self.assertEqual((), result.claims)
        self.assertEqual((), result.support)
        self.assertEqual(("u",), result.covered_units)
        self.assertTrue(result.open_needs)
        self.assertTrue(result.open_needs[0].missing_relation.startswith(NUMERIC_GROUNDING_GAP_PREFIX))

    def test_neighboring_field_value_is_bound_to_requested_literal_field(self):
        text = "甲公司：流动资产合计 100 万元；总资产 500 万元；货币资金 20 万元。"
        result = self.parse(text, value="500")
        self.assertEqual("100", result.claims[0].value)
        self.assertEqual("流动资产合计", result.claims[0].relation)
        self.assertEqual("万元", result.claims[0].unit)
        self.assertEqual(text, result.claims[0].text)
        self.assertEqual((), result.support)
        self.assertFalse(result.open_needs)

    def test_ambiguous_or_missing_field_entity_period_and_unit_remain_gaps(self):
        cases = (
            ("甲公司：总资产 500 万元。", {"value": "500"}),
            ("乙公司：流动资产合计 100 万元。", {}),
            ("R1 and R2: weight 3 kg.", {"entity": "R1", "relation": "weight", "value": "3", "unit": "kg"}),
            ("甲公司：净利润 10万元；乙公司：流动资产合计 100万元。", {}),
            ("甲公司：流动资产合计 2024年100万元，2023年80万元。", {"value": "80"}),
            ("甲公司：流动资产合计 2024年100万元，2023年100万元。", {}),
            ("甲公司：流动资产合计 100。", {}),
        )
        for text, fact in cases:
            with self.subTest(text=text):
                self.assert_gap(self.parse(text, **fact))

    def test_repeated_excerpt_without_its_period_cannot_choose_an_occurrence(self):
        excerpt = "甲公司：流动资产合计100万元。"
        unit = source("2023年"+excerpt+"\n2024年"+excerpt)
        result = parse_read_result(numeric_task((unit,)), json.dumps(numeric_output(excerpt)), (unit,))
        self.assert_gap(result)

    def test_zero_is_retained_only_when_explicit_in_source(self):
        result = self.parse("甲公司：流动资产合计 0万元。", value="0")
        self.assertEqual("0", result.claims[0].value)
        self.assertFalse(result.open_needs)
        self.assert_gap(self.parse("甲公司：流动资产合计 未披露。", value="0"))

    def test_original_unit_is_retained_instead_of_model_scale(self):
        result = self.parse("甲公司：流动资产合计 100万元。", unit="元")
        self.assertEqual(("100", "万元"), (result.claims[0].value, result.claims[0].unit))

    def test_uncited_window_cannot_silently_repair_a_wrong_citation(self):
        correct = source("甲公司：流动资产合计 100万元。")
        wrong = source("乙公司：流动资产合计 500万元。", "other")
        units = (wrong, correct)
        result = parse_read_result(numeric_task(units), json.dumps(numeric_output(correct.text)), units)
        self.assertEqual((), result.claims)
        self.assertEqual((), result.support)
        self.assertTrue(result.open_needs)

    def test_explicit_same_source_dependency_supplies_entity_and_unit_headers(self):
        header = source("甲公司 2024年，单位：万元", "header")
        row = source("流动资产合计 100", dependencies=("header",))
        units = (header, row)
        result = parse_read_result(numeric_task(units), json.dumps(numeric_output(row.text,
            sources=("s1",))), units)
        self.assertEqual("100", result.claims[0].value)
        self.assertEqual({"u:span", "header:span"}, {span.span_id for span in result.claims[0].source_spans})

    def test_header_from_another_document_cannot_bind_the_entity(self):
        header = source("甲公司 2024年，单位：万元", "header",
            key=EvidenceKey("corpus", "different", "p", "2024", "hash"))
        row = source("流动资产合计 100", dependencies=("header",))
        units = (header, row)
        result = parse_read_result(numeric_task(units), json.dumps(numeric_output(row.text,
            sources=("s1",))), units)
        self.assertFalse(result.claims)
        self.assertTrue(result.open_needs)

    def test_null_sum_is_a_gap_while_count_can_preserve_missing_quantity(self):
        unit = source("甲公司：流动资产合计 未披露。")
        task = numeric_task((unit,), operations=("count", "sum"))
        data = numeric_output(unit.text, value=None, unit="", needs=("n0",))
        result = parse_read_result(task, json.dumps(data), (unit,))
        self.assertIsNone(result.claims[0].value)
        self.assertEqual((), result.support)
        data = numeric_output(unit.text, value=None, unit="", needs=("n1",))
        self.assert_gap(parse_read_result(task, json.dumps(data), (unit,)))

    def test_regular_sum_protocol_cannot_bypass_source_binding(self):
        unit = source("甲公司：总资产500万元。")
        task = ReadTask("read:0", "READ", ("need:0",), ("u",), operation="sum")
        data = {"format_version": 4, "claims": [{"n": "n0", "t": unit.text,
            "e": "甲公司", "r": "流动资产合计", "v": "500", "u": "万元", "c": [],
            "s": ["s0"], "p": [], "f": []}], "support": ["n0"], "conflicts": [], "open_needs": []}
        self.assert_gap(parse_read_result(task, json.dumps(data), (unit,)))

    def test_shared_numeric_need_cannot_bypass_fact_validation_with_regular_claim(self):
        unit = source("甲公司：总资产500万元。")
        task = numeric_task((unit,))
        data = {"format_version": 5, "facts": [], "claims": [{"n": "n0", "t": unit.text,
            "e": "甲公司", "r": "流动资产合计", "v": "500", "u": "万元", "c": [],
            "s": ["s0"], "p": [], "f": []}], "support": ["n0"], "conflicts": [], "open_needs": []}
        with self.assertRaisesRegex(ValueError, "source-bound facts"):
            parse_read_result(task, json.dumps(data), (unit,))

    def test_later_numeric_window_cannot_erase_another_entity_gap(self):
        earlier = ReadResult("read:earlier", open_needs=(OpenNeed("need:0",
            NUMERIC_GROUNDING_GAP_PREFIX+"流动资产合计", ("乙公司",), "2024"),))
        later = self.parse("甲公司：流动资产合计100万元。")
        self.assertEqual(earlier.open_needs, unresolved_needs((earlier, later)))

    def test_repair_cannot_substitute_neighboring_field_or_reporting_period(self):
        gap = OpenNeed("need:0", "流动资产合计", ("甲公司",), "2024")
        for text, relation in (("甲公司：总资产500万元。", "总资产"),
                               ("甲公司：2023年流动资产合计100万元。", "流动资产合计")):
            with self.subTest(text=text):
                unit = source(text)
                task = replace(numeric_task((unit,)), operation="repair_extract", repair_gaps=(gap,))
                data = numeric_output(text, relation=relation)
                result = parse_read_result(task, json.dumps(data), (unit,))
                self.assertFalse(result.claims)
                self.assertEqual((), result.support)
                self.assertIn(gap, result.open_needs)

    def test_repair_returns_exact_fact_without_claiming_whole_need_support(self):
        text = "甲公司：2024年流动资产合计100万元。"
        unit = source(text)
        gap = OpenNeed("need:0", "流动资产合计", ("甲公司",), "2024")
        task = replace(numeric_task((unit,)), operation="repair_extract", repair_gaps=(gap,))
        result = parse_read_result(task, json.dumps(numeric_output(text)), (unit,))
        self.assertEqual("100", result.claims[0].value)
        self.assertFalse(result.open_needs)
        self.assertEqual((), result.support)


if __name__ == "__main__":
    unittest.main()
