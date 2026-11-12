"""Independent Need views must not wait for unrelated source reading."""
from dataclasses import replace
import unittest

from controller.context_schema import (
    ContextPlan, EvidenceClaim, EvidenceKey, EvidenceSet, EvidenceUnit, Need,
    NumericReadScope, ReadResult, ReadTask, SourceSpan, TaskState,
)
from controller.reading import advance_read_plan, numeric_selection_step_ids, numeric_selection_tasks


class NeedDependencyTests(unittest.TestCase):
    def fixture(self):
        units = tuple(EvidenceUnit(identity,
            (SourceSpan(identity+":span", EvidenceKey("c", identity, "p", "v1", "hash"), 0, len(text)),),
            text, need_ids=(need,)) for identity, need, text in (
                ("numeric", "sum", "R1 weight 3 kg"), ("lookup", "lookup", "unrelated explanation")))
        scope = NumericReadScope("g", ("sum",), operations=("sum",))
        numeric = ReadTask("read:numeric", "READ", ("sum",), ("numeric",), shared_numeric_scopes=(scope,))
        lookup = ReadTask("read:lookup", "READ", ("lookup",), ("lookup",))
        plan = ContextPlan("plan", TaskState("task", "sum weights and explain the policy", needs=(
            Need("sum", operation="sum"), Need("lookup"))), EvidenceSet("evidence", units),
            steps=(numeric, lookup))
        claim = EvidenceClaim("claim:numeric", "sum", "extract", "R1 weight 3 kg", "3", "kg",
            source_spans=units[0].source_spans, entity="R1", relation="weight", version="v1")
        result = ReadResult(numeric.step_id, (claim,), ("numeric",))
        return plan, result, claim

    def test_unrelated_read_does_not_change_numeric_selector_identity(self):
        plan, _, _ = self.fixture()
        independent = replace(plan, steps=plan.steps[:1])
        self.assertEqual(numeric_selection_step_ids(independent), numeric_selection_step_ids(plan))

    def test_zero_coverage_paid_parent_does_not_unlock_numeric_selection(self):
        plan, result, _ = self.fixture()
        paid_failure = replace(result, claims=(), covered_units=())
        self.assertEqual((), tuple(numeric_selection_tasks(plan, (paid_failure,))))

    def test_pending_recovery_child_remains_a_numeric_prerequisite(self):
        plan, result, _ = self.fixture()
        child = replace(plan.steps[0], step_id="read:numeric:recovery", predecessor_step_ids=(result.step_id,))
        plan = replace(plan, steps=(*plan.steps, child))
        self.assertEqual((), tuple(numeric_selection_tasks(plan, (result,))))

    def test_advance_exposes_independent_selection_while_lookup_read_is_pending(self):
        plan, result, _ = self.fixture()
        ready = advance_read_plan(plan, (result,))
        self.assertEqual({"READ", "REDUCE"}, {task.kind for task in ready})
        selection = next(task for task in ready if task.operation == "numeric_select")
        self.assertEqual(("read:numeric",), selection.predecessor_step_ids)
        self.assertEqual(("claim:numeric",), selection.input_claim_ids)

    def test_nonshared_numeric_reduction_uses_only_its_need_reads(self):
        plan, result, _ = self.fixture()
        plan = replace(plan, steps=(replace(plan.steps[0], shared_numeric_scopes=(), operation="sum"),
                                   plan.steps[1]))
        ready = advance_read_plan(plan, (result,))
        reduction = next(task for task in ready if task.kind == "REDUCE")
        self.assertEqual(("read:numeric",), reduction.predecessor_step_ids)
        self.assertEqual("sum", reduction.operation)


if __name__ == "__main__":
    unittest.main()
