"""Training and retrieval preserve the task meaning across condition and source edits."""
from dataclasses import replace
import unittest

import numpy as np

from adapters.corpus import CorpusIndex
from adapters.evidence_pool import SourceDirectory
from adapters.history_state import TaskInterpreter, TaskPrediction, apply_condition_updates
from adapters.support_model import SetPrediction, SupportModel, SupportPrediction, SupportProfile
from controller.context_schema import (
    Condition, ContextBudget, EvidencePool, EvidenceUnit, Need, OpenNeed, RetrievalState,
    SourceSpan, SupportBundle, SupportScores, TaskState, TextRef,
)
from controller.evidence import make_evidence
from controller.reading import build_read_plan
from controller.schema import OnlineContext, OnlineTurn, SessionSnapshot, TurnRecord
from controller.templates import ByteTemplate
from training.preference_rollout import candidate_sets, remove_premise
from training.task_model import (
    CONDITION_RELATIONS, OPERATIONS, RELATIONS, TEMPORAL_RELATIONS, UPDATES, VERSIONS,
    decode_prediction, encode_task, task_targets,
)


class CharacterTokenizer:
    cls_token_id = 1
    sep_token_id = 2

    def encode(self, text, **kwargs):
        return [ord(character) for character in text]

    def __call__(self, text, **kwargs):
        return {"input_ids": self.encode(text),
                "offset_mapping": [(index, index+1) for index in range(len(text))]}


def perfect_task_output(encoded, prediction):
    targets = task_targets(encoded, prediction)
    sizes = {
        "span": 3, "history": 2, "update": len(UPDATES),
        "condition_relation": len(CONDITION_RELATIONS), "operation": len(OPERATIONS),
        "relation": len(RELATIONS), "version": len(VERSIONS),
        "temporal_relation": len(TEMPORAL_RELATIONS),
        "pointer": len(encoded["prior_conditions"])+1,
        "resume": len(encoded["prior_tasks"])+1,
        "version_pointer": len(encoded["version_turn_ids"])+1,
    }
    return {name: np.eye(size)[np.maximum(np.asarray(targets[name]), 0)][None]
            for name, size in sizes.items()}


def removal_case(question="Remove the platform condition."):
    earlier = "For Linux, how do I file?"
    platform = Condition("platform", value_spans=(TextRef(1, "user", 4, 9),),
                         relation_type="PLATFORM")
    prior = TaskState("filing", earlier, conditions=(platform,),
        needs=(Need("filing-steps", (TextRef(1, "user", 0, len(earlier)),),
                    condition_ids=("platform",)),), current_turn_id=1)
    memory = RetrievalState(1, (TurnRecord(1, earlier, "Use form L."),), (prior,))
    turn = OnlineTurn("conversation", "turn-2", 2, "test", question, (), (), "P3")
    return platform, prior, memory, turn


def test_predicate_only_removal_is_learnable_and_releases_the_active_condition():
    platform, _, memory, turn = removal_case()
    event = Condition("remove-platform", predicate_span=TextRef(2, "user", 0, len(turn.query)),
                      target_refs=("platform",), update="REMOVE")
    expected = TaskPrediction("correction", conditions=(event,))
    encoded = encode_task(CharacterTokenizer(), turn, (), memory)
    targets = task_targets(encoded, expected)
    assert UPDATES.index("REMOVE") in targets["update"]
    assert 1 in targets["pointer"]
    decoded = decode_prediction(encoded, perfect_task_output(encoded, expected), turn)
    assert decoded.conditions[0].value_spans == ()
    assert decoded.conditions[0].predicate_span == event.predicate_span
    assert apply_condition_updates((platform,), decoded.conditions) == ()

    class Model:
        def predict(self, turn, windows, memory):
            return decoded

    class History:
        def select(self, *args, **kwargs):
            return ()

    state = TaskInterpreter(Model(), History()).interpret(turn, memory)
    assert state.task_id == "filing"
    assert state.conditions == ()
    assert state.needs[0].condition_ids == ()
    assert "Linux" not in state.query
    assert "how do I file?" in state.query


def test_removal_anchor_is_not_reused_as_the_key_of_a_new_condition():
    question = "Remove platform; set location to France."
    platform, _, memory, turn = removal_case(question)
    removal = Condition("remove-platform", predicate_span=TextRef(2, "user", 0, 15),
                        target_refs=("platform",), update="REMOVE")
    location_start = question.index("location")
    value_start = question.index("France")
    location = Condition("location", key_span=TextRef(2, "user", location_start, location_start+8),
        value_spans=(TextRef(2, "user", value_start, value_start+6),), relation_type="LOCATION")
    expected = TaskPrediction("correction", conditions=(removal, location),
        needs=(Need("filing-steps", (TextRef(2, "user", 0, len(question)),)),))
    encoded = encode_task(CharacterTokenizer(), turn, (), memory)
    decoded = decode_prediction(encoded, perfect_task_output(encoded, expected), turn)
    active = apply_condition_updates((platform,), decoded.conditions)
    assert len(active) == 1
    assert active[0].key_span == location.key_span
    assert active[0].value_spans == location.value_spans
    assert decoded.needs[0].condition_ids == (active[0].condition_id,)


def test_deleted_premise_removes_dependent_branch_and_compiles_a_real_read():
    specifications = (
        ("eligibility", "Only Linux users are eligible.", ()),
        ("step", "Eligible users submit form L.", ("eligibility",)),
        ("delivery", "Send the completed form to the office.", ("step",)),
        ("alternative", "All users may request general information by phone.", ()),
    )
    blocks = tuple(make_evidence(text, name, corpus_id="test", passage_id=name)
                   for name, text, _ in specifications)
    units = tuple(EvidenceUnit(name, (SourceSpan(name, block.key, 0, len(text)),), text,
                  dependencies=dependencies, retrieval_score=4-index)
                  for index, ((name, text, dependencies), block) in enumerate(zip(specifications, blocks)))
    state = TaskState("filing", "Explain filing and general information.",
                      needs=(Need("filing"),), current_turn_id=1)
    pool = EvidencePool(units, blocks)
    scores = SupportScores(bundles=(SupportBundle("complete", "filing",
        tuple(unit.unit_id for unit in units), 1, 1, disposition="complete"),))

    class Utility:
        def resolve(self, ref):
            return ""

        def predict_set(self, state, selected, supports):
            ids = {unit.unit_id for unit in selected}
            quality = .1 if "eligibility" not in ids else .2 if "step" not in ids else .9
            return SetPrediction(quality)

    template = ByteTemplate()
    budget = ContextBudget(window_tokens=65536, body_tokens=16384, read_input_tokens=32768)
    deleted = candidate_sets(state, pool, scores, Utility(), template.tokenizer, budget)["deleted_premise"]
    assert tuple(unit.unit_id for unit in deleted.selected_units) == ("alternative",)
    plan = build_read_plan(state, deleted, "selected_sources", budget,
                           template=template, snapshot=SessionSnapshot("conversation"))
    assert tuple(task.input_unit_ids for task in plan.steps) == (("alternative",),)
    assert remove_premise(units, "alternative") == units[:3]
    assert units[1].dependencies == ("eligibility",)


def shared_gap_sources():
    bodies = {"shared": "Shared premise.", "decoy": "Partial first need."}
    corpus = CorpusIndex.from_records([
        {"passage_id": key, "document_id": key, "text": text} for key, text in bodies.items()
    ], corpus_id="test", snapshot_id="test")

    def search(query, *, top_k):
        hits = (("shared", .9), ("decoy", .5)) if "relation-one" in query else (("shared", .1),)
        return tuple(OnlineContext(key, bodies[key], score=score) for key, score in hits)

    directory = SourceDirectory(corpus, search, CharacterTokenizer())
    state = TaskState("task", "question", needs=(Need("one"), Need("two")), current_turn_id=1)
    gaps = (OpenNeed("one", "relation-one"), OpenNeed("two", "relation-two"))
    return directory, state, gaps


def test_shared_gap_premise_keeps_both_need_quotas_and_its_highest_score():
    directory, state, gaps = shared_gap_sources()
    pool = directory.retrieve_gap_pool(state, RetrievalState(0), gaps)
    shared = next(unit for unit in pool.units if unit.text == "Shared premise.")
    assert shared.need_ids == ("one", "two")
    assert shared.retrieval_score == .9

    class Backend:
        def __init__(self):
            self.scheduled = []

        def predict(self, examples):
            self.scheduled.extend((example.need_id, tuple(unit.text for unit in example.units))
                                  for example in examples)
            return tuple(SupportPrediction(1, 1, 1, 0, "complete") for _ in examples)

    backend = Backend()
    model = SupportModel(backend, history_resolver=lambda ref: "",
                         profile=SupportProfile(single_candidates_per_need=1))
    model.score_support(state, pool, open_needs=gaps)
    assert backend.scheduled == [("one", ("Shared premise.",)), ("two", ("Shared premise.",))]


def test_gap_extension_preserves_the_existing_need_association():
    directory, state, gaps = shared_gap_sources()
    recalled = directory.retrieve_gap_pool(state, RetrievalState(0), gaps[:1])
    shared = next(unit for unit in recalled.units if unit.text == "Shared premise.")
    previous = replace(recalled, units=(replace(shared, need_ids=("earlier",), retrieval_score=.95),))
    extended = directory.extend_open_needs(state, RetrievalState(0), previous, gaps)
    merged = next(unit for unit in extended.units if unit.text == "Shared premise.")
    assert merged.need_ids == ("earlier", "one", "two")
    assert merged.retrieval_score == .95
    assert merged.source_spans == shared.source_spans


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(function)
        for name, function in globals().items() if name.startswith("test_"))


if __name__ == "__main__":
    unittest.main()
