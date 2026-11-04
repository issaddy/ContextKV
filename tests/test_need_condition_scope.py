"""Need-specific conditions survive task interpretation and shared reading."""
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest

from adapters.evidence_pool import SourceDirectory
from adapters.history_state import TaskInterpreter, TaskPrediction
from adapters.support_model import query_view
from controller.context_schema import Condition, Need, RetrievalState, TextRef
from controller.reading import numeric_read_scopes
from controller.schema import OnlineTurn, TurnRecord


QUESTION = "Count employees in 2023 and sum revenue in 2024."


def reference(text, *, question=QUESTION, turn=1):
    start = question.index(text)
    return TextRef(turn, "user", start, start+len(text))


def initial_prediction():
    conditions = (
        Condition("year23", value_spans=(reference("2023"),), relation_type="TIME"),
        Condition("year24", value_spans=(reference("2024"),), relation_type="TIME"))
    needs = (
        Need("count", (reference("Count employees"),), "count", condition_ids=("year23",)),
        Need("sum", (reference("sum revenue"),), "sum", condition_ids=("year24",)))
    return TaskPrediction("standalone", conditions=conditions, needs=needs)


def interpret(prediction, *, question=QUESTION, turn=1, memory=None):
    class Model:
        def predict(self, *args):
            return prediction

    class HistoryIndex:
        def select(self, *args, **kwargs):
            return ()

    online = OnlineTurn("session", f"task{turn}", turn, "corpus", question, (), (), "P3")
    return TaskInterpreter(Model(), HistoryIndex()).interpret(online, memory or RetrievalState(0))


def prior_memory():
    return RetrievalState(1, (TurnRecord(1, QUESTION, "Recorded actual answer."),),
                          (interpret(initial_prediction()),))


class NeedConditionScopeTests(unittest.TestCase):
    def test_retrieval_queries_use_only_explicit_active_condition_links(self):
        prediction = initial_prediction()
        for bindings, expected in (
            ((("year23",), ("year24",)), ("2023", "2024")),
            ((("year23",), ("year23",)), ("2023", "2023")),
            ((("year23",), ()), ("2023", "")),
            ((("retired",), ("year24",)), ("", "2024")),
        ):
            with self.subTest(bindings=bindings):
                state = replace(interpret(prediction), needs=tuple(
                    replace(need, condition_ids=ids)
                    for need, ids in zip(prediction.needs, bindings, strict=True)))
                queries = []

                def search(query, *, top_k):
                    queries.append(query)
                    return ()

                index = SimpleNamespace(blocks=(), source_metadata={}, retrieved_keys=lambda contexts: ())
                directory = SourceDirectory(index, search, tokenizer=None)
                pool = directory.retrieve_pool(state, RetrievalState(0))
                self.assertEqual(queries[0], state.query or state.raw_question)
                self.assertEqual(queries[1:], [
                    "Count employees" + ("\n" + expected[0] if expected[0] else ""),
                    "sum revenue" + ("\n" + expected[1] if expected[1] else "")])
                self.assertEqual([row["need_id"] for row in pool.trace], ["", "count", "sum"])

    def test_independent_conditions_remain_separate_in_support_and_shared_reading(self):
        state = interpret(initial_prediction())
        self.assertEqual([need.condition_ids for need in state.needs], [("year23",), ("year24",)])
        queries = [json.loads(query_view(state, need, lambda ref: QUESTION[ref.start:ref.end]))
                   for need in state.needs]
        self.assertEqual([[row["values"] for row in query["conditions"]] for query in queries],
                         [[["2023"]], [["2024"]]])
        self.assertEqual([scope.need_ids for scope in numeric_read_scopes(state, ())], [("count",), ("sum",)])

    def test_explicit_shared_condition_keeps_one_numeric_scope(self):
        prediction = initial_prediction()
        prediction = replace(prediction, conditions=prediction.conditions[:1], needs=tuple(
            replace(need, condition_ids=("year23",)) for need in prediction.needs))
        state = interpret(prediction)
        scopes = numeric_read_scopes(state, ())
        self.assertEqual([(scope.need_ids, scope.condition_ids) for scope in scopes],
                         [(("count", "sum"), ("year23",))])

    def test_unconditioned_need_does_not_acquire_other_need_restrictions(self):
        prediction = initial_prediction()
        state = interpret(replace(prediction, needs=(prediction.needs[0],
            replace(prediction.needs[1], condition_ids=()))))
        self.assertEqual([need.condition_ids for need in state.needs], [("year23",), ()])

    def test_followup_inherits_each_need_own_bindings(self):
        memory = prior_memory()
        state = interpret(TaskPrediction("followup"), question="Repeat the results.", turn=2, memory=memory)
        self.assertEqual(state.needs, memory.prior_task_states[0].needs)

    def test_replace_updates_only_needs_linked_to_the_replaced_condition(self):
        question = "Use 2025 for the employee count."
        replacement = Condition("year25", value_spans=(reference("2025", question=question, turn=2),),
            relation_type="TIME", target_refs=("year23",), supersedes=("year23",), update="REPLACE")
        memory = prior_memory()
        for needs in ((), memory.prior_task_states[0].needs):
            with self.subTest(explicit_needs=bool(needs)):
                state = interpret(TaskPrediction("correction", conditions=(replacement,), needs=needs),
                                  question=question, turn=2, memory=memory)
                self.assertEqual([need.condition_ids for need in state.needs], [("year25",), ("year24",)])
                self.assertEqual({condition.condition_id for condition in state.conditions}, {"year25", "year24"})

    def test_remove_drops_only_the_targeted_binding(self):
        question = "Remove the year restriction from the employee count."
        removal = Condition("remove-year", predicate_span=reference("Remove", question=question, turn=2),
                            target_refs=("year23",), update="REMOVE")
        state = interpret(TaskPrediction("correction", conditions=(removal,)),
                          question=question, turn=2, memory=prior_memory())
        self.assertEqual([need.condition_ids for need in state.needs], [(), ("year24",)])
        self.assertEqual([condition.condition_id for condition in state.conditions], ["year24"])

    def test_added_condition_uses_explicit_need_links(self):
        memory = prior_memory()
        question = "Limit revenue to Germany."
        addition = Condition("germany", value_spans=(reference("Germany", question=question, turn=2),),
                             relation_type="LOCATION")
        first, second = memory.prior_task_states[0].needs
        prediction = TaskPrediction("correction", conditions=(addition,),
            needs=(first, replace(second, condition_ids=("year24", "germany"))))
        state = interpret(prediction, question=question, turn=2, memory=memory)
        self.assertEqual([need.condition_ids for need in state.needs], [("year23",), ("year24", "germany")])


if __name__ == "__main__":
    unittest.main()
