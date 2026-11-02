from dataclasses import replace
import json

from adapters.support_model import SetPrediction, SupportModel, SupportPrediction
from controller.context_schema import (ContextBudget, EvidenceKey, EvidencePool,
    EvidenceUnit, Need, OpenNeed, SourceSpan, TaskState)
from controller.evidence_planner import (BudgetedEvidencePlanner, SearchProfile,
                                       render_selection, unresolved_needs)
from controller.templates import ByteTokenizer


def unit(name, text, *, dependencies=(), passage=None):
    key = EvidenceKey("corpus", "policy", passage or name, "v2", name)
    return EvidenceUnit(name, (SourceSpan(name, key, 0, len(text)),), text,
        dependencies=dependencies, retrieval_score=1,
        original_passage_ids=(passage or name,), need_ids=("n",))


class ExplicitBackend:
    """Specified test predictions: eligibility and procedure jointly support n."""
    manifest = {"kind": "explicit_test_predictions"}

    def predict(self, inputs):
        result = []
        for example in inputs:
            ids = {unit.unit_id for unit in example.units}
            complete = {"eligibility", "procedure"} <= ids
            result.append(SupportPrediction(1, .4, .95 if complete else .4, .8,
                "complete" if complete else "partial",
                tuple(span for unit in example.units for span in unit.source_spans),
                "next_relation" if not complete else ""))
        return tuple(result)

    def predict_set(self, state, units, supports):
        quality = max((s.joint_support for s in supports), default=0)
        return SetPrediction(quality, "answer" if quality > .9 else "partial")


def test_complementary_bundle_dependency_budget_and_sources():
    eligibility = unit("eligibility", "Eligible Linux residents.", dependencies=("definition",))
    procedure = unit("procedure", "Submit the Linux application.")
    definition = unit("definition", "Resident definition.")
    pool = EvidencePool((eligibility, procedure, definition))
    state = TaskState("task", "Eligibility and procedure?", needs=(Need("n"),))
    model = SupportModel(ExplicitBackend(), history_resolver=lambda ref: "")
    scores = model.score_support(state, pool)
    planner = BudgetedEvidencePlanner(ByteTokenizer(), model)
    sets = planner.search_sets(state, pool, scores, ContextBudget(body_tokens=3000))
    assert sets[0].quality_prediction == .95
    assert {u.unit_id for u in sets[0].selected_units} == {"eligibility", "procedure", "definition"}
    assert sets[0].token_cost == len(render_selection(sets[0].selected_units).encode())
    assert sets[0].need_support[0].disposition == "complete"
    assert len(sets) == 3
    restricted = planner.search_sets(state, pool, scores, ContextBudget(body_tokens=3000, max_passages=1))
    assert all(len({p for u in item.selected_units for p in u.original_passage_ids}) <= 1 for item in restricted)


def test_pair_triple_schedule_is_bounded_and_per_need():
    units = tuple(unit(str(i), "body"+str(i)) for i in range(30))
    state = TaskState("task", "Trace two relations", needs=(Need("n", operation="infer"), Need("m")))
    model = SupportModel(ExplicitBackend(), history_resolver=lambda ref: "")
    scores = model.score_support(state, EvidencePool(units))
    for need_id in ("n", "m"):
        assert len([s for s in scores.singles if s.need_id == need_id]) == 24
        assert len([s for s in scores.bundles if s.need_id == need_id and len(s.unit_ids) == 2]) == 15
        assert 0 < len([s for s in scores.bundles if s.need_id == need_id and len(s.unit_ids) == 3]) <= 6


def test_render_identical_bodies_once_and_keep_version_sources():
    first = unit("a", "A long original body.")
    second = replace(unit("b", first.text), source_spans=(replace(
        first.source_spans[0], source_key=replace(first.source_spans[0].source_key, version="v1")),))
    rendered = render_selection((first, second))
    assert rendered.count(first.text) == 1
    assert '"version":"v1"' in rendered
    assert '"version":"v2"' in rendered


def test_window_budget_and_gap_retrieval_change_search():
    first = unit("eligibility", "Eligibility.")
    second = unit("procedure", "Procedure.")
    state = TaskState("task", "How?", needs=(Need("n"),))
    model = SupportModel(ExplicitBackend(), history_resolver=lambda ref: "")
    pool = EvidencePool((first,))
    planner = BudgetedEvidencePlanner(ByteTokenizer(), model,
        profile=SearchProfile(gap_search_rounds=1),
        measure_context=lambda state, units: 100+sum(len(u.text) for u in units))
    initial = planner.search_sets(state, pool, model.score_support(state, pool),
        ContextBudget(body_tokens=2000))
    assert unresolved_needs(state, initial[0])[0].need_id == "n"
    expanded, scores, sets = planner.search_with_gaps(state, pool,
        model.score_support(state, pool), ContextBudget(body_tokens=2000),
        score_model=model, retrieve_gap=lambda state, gap: (second,))
    assert {u.unit_id for u in expanded.units} == {"eligibility", "procedure"}
    assert sets[0].quality_prediction == .95
    assert not unresolved_needs(state, sets[0])
    overflow = planner.search_sets(state, expanded, scores,
        ContextBudget(body_tokens=2000, window_tokens=110, output_tokens=10))
    assert overflow[0].selected_units == ()


def test_open_need_bindings_augment_support_query_without_rewriting_actual_question():
    class BoundBackend(ExplicitBackend):
        def __init__(self):
            self.inputs = []
        def predict(self, inputs):
            self.inputs.extend(inputs)
            return super().predict(inputs)

    backend = BoundBackend()
    model = SupportModel(backend, history_resolver=lambda ref: "")
    state = TaskState("task", "When was A's parent fund established?", needs=(Need("n"),))
    gap = OpenNeed("n", "fund establishment date", ("B",), "current", ("read0:claim:0",))
    model.score_support(state, EvidencePool((unit("eligibility", "B fund was founded in 1993."),)), open_needs=(gap,))
    query = json.loads(backend.inputs[0].query)
    assert query["question"] == state.raw_question
    assert query["open_needs"][0]["bound_entities"] == ["B"]
    assert query["open_needs"][0]["parent_claim_ids"] == ["read0:claim:0"]


def test_gap_search_keeps_read_premise_for_joint_support_and_charges_only_new_reading():
    old = unit("eligibility", "Old premise. "*300)
    fresh = unit("procedure", "New procedure.")
    state = TaskState("task", "Eligibility and procedure?", needs=(Need("n"),))
    pool = EvidencePool((old, fresh))
    model = SupportModel(ExplicitBackend(), history_resolver=lambda ref: "")
    scores = model.score_support(state, pool)
    tokenizer = ByteTokenizer()
    planner = BudgetedEvidencePlanner(tokenizer, model,
        measure_body=lambda units: len(tokenizer.encode(render_selection(tuple(u for u in units if u.unit_id != "eligibility")))),
        passage_exclusions=frozenset({"eligibility"}))
    result = planner.search_sets(state, pool, scores, ContextBudget(body_tokens=1000, max_passages=1))
    assert {u.unit_id for u in result[0].selected_units} == {"eligibility", "procedure"}
    assert result[0].quality_prediction == .95
    assert result[0].token_cost < 1000
