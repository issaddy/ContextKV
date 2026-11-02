from dataclasses import replace
import json

import numpy as np
import pytest

from adapters.history_state import HistoryIndex, TaskInterpreter, TaskPrediction, retrieval_state
from adapters.study_retrieval import create_study_retriever
from adapters.support_model import SetPrediction, SupportPrediction
from bench.mock_engine import MockEngine
from controller.context_schema import EvidencePool, Need, OpenNeed, TextRef
from controller.evidence import make_evidence
from controller.planner import PlannerKnobs
from controller.schema import OnlineContext, SessionSnapshot, TurnRecord
from controller.service import Controller
from controller.session_store import SessionStore
from controller.templates import ByteTemplate, ByteTokenizer


class OffsetBytes(ByteTokenizer):
    def __call__(self, text, **kwargs):
        result = {"input_ids": self.encode(text)}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(i, i+1) for i, char in enumerate(text) for _ in char.encode()]
        return result


class FixtureRetriever:
    corpus_snapshot_id = "v2"
    manifest = {"kind": "fixture"}
    def candidate_search(self, query, *, top_k=50):
        return (OnlineContext("eligibility", "Eligible Linux applicants", score=1.0),
                OnlineContext("procedure", "Submit the online form", score=.9))[:top_k]
    def retrieve_turn(self, turn):
        return replace(turn, contexts=self.candidate_search(turn.query))


class FixtureTask:
    usage = {"input_tokens": 0, "output_tokens": 0, "elapsed_ms": 0.0}
    def predict(self, turn, windows, memory):
        return TaskPrediction("standalone", needs=(Need("apply",
            anchor_spans=(TextRef(turn.turn, "user", 0, len(turn.query)),), operation="procedure"),))


class FixtureSupport:
    manifest = {"kind": "fixture_support"}
    usage = {"input_tokens": 0, "output_tokens": 0, "elapsed_ms": 0.0}
    def predict(self, examples):
        return tuple(SupportPrediction(1.0, .2, .95 if len(example.units) == 2 else .2, .9,
            "complete" if len(example.units) == 2 else "partial",
            tuple(span for unit in example.units for span in unit.source_spans)) for example in examples)
    def predict_set(self, state, units, supports, *, context=None):
        ids = {unit.unit_id for unit in units}
        return SetPrediction(max((s.joint_support for s in supports if set(s.unit_ids) <= ids), default=0.0))
    def release_task(self, task_id):
        pass
    def close(self):
        pass


class FixtureFrozen:
    tokenizer = OffsetBytes()
    usage = {"input_tokens": 0, "output_tokens": 0, "elapsed_ms": 0.0}
    def encode(self, texts):
        return np.ones((len(texts), 2)) / np.sqrt(2)
    def close(self):
        pass


class CoverageOnlySupport(FixtureSupport):
    def __init__(self):
        self.arities = []
    def predict(self, examples):
        self.arities.extend(len(example.units) for example in examples)
        return super().predict(examples)
    def predict_set(self, *args, **kwargs):
        raise AssertionError("support_coverage issued an online model set/candidate call")


class ReadEngine(MockEngine):
    async def generate(self, *, response_schema=None, **kwargs):
        if response_schema:
            ids = response_schema["properties"]["claims"]["items"]["properties"]["span_ids"]["items"]["enum"]
            self.response_text = json.dumps({"claims": [{"need_id": "apply", "text": "Apply using the form",
                "entity": "applicant", "relation": "procedure", "value": "form", "unit": "", "conditions": [],
                "version": "v2", "span_ids": ids, "parent_claim_ids": []}],
                "support": ["apply"], "conflicts": [], "open_needs": []})
        else:
            self.response_text = "Eligible Linux applicants submit the online form."
        return await super().generate(**kwargs)


def make_provider(tmp_path, strategy="B6"):
    corpus = tmp_path/"corpus.jsonl"
    corpus.write_text("".join(json.dumps(row)+"\n" for row in [
        {"_id": "eligibility", "text": "Eligible Linux applicants", "version": "v2"},
        {"_id": "procedure", "text": "Submit the online form", "version": "v2"}]))
    template = ByteTemplate()
    template.tokenizer = OffsetBytes()
    interpreter = TaskInterpreter(FixtureTask(), HistoryIndex(template.tokenizer,
        lambda texts: np.ones((len(texts), 2))))
    config = {"study": {"strategy": strategy}, "retrieval": {
        "corpus_path": str(corpus), "corpus_id": "manual", "config_path": "unused"},
        "selection": {"body_tokens": 8192, "max_actions": 2, "initial_gap_search_rounds": 0},
        "execution": {"window_tokens": 32768, "output_tokens": 1024},
        "reading": {"enabled": True, "input_target_tokens": 12288, "output_tokens": 2048}}
    return create_study_retriever(config, template, retriever=FixtureRetriever(),
        task_interpreter=interpreter, support_backend=FixtureSupport(), frozen_history=FixtureFrozen()), template


@pytest.mark.asyncio
async def test_factory_connects_task_pair_search_read_and_single_qa_commit(tmp_path):
    provider, template = make_provider(tmp_path)
    selection = await provider.retrieve(SessionSnapshot("s", corpus_snapshot_id="v2"), "How do I apply?", "r")
    assert selection.query_context.task_state.needs[0].operation == "procedure"
    assert len(selection.context_plans) >= 2
    assert len(selection.context_plans[0].selected_set.need_support[0].unit_ids) == 2
    assert selection.context_plans[0].steps[0].kind == "READ"
    controller = Controller(SessionStore(tmp_path/"state.db"), ReadEngine(fingerprint=template.fingerprint),
                            template=template, knobs=PlannerKnobs(strategy="always_rebuild"))
    response = await controller.serve_turn("s", "How do I apply?", request_id="r", corpus_snapshot_id="v2",
        evidence_provider=lambda snapshot: provider.retrieve(snapshot, "How do I apply?", "r"))
    committed = controller.store.get_snapshot("s")
    assert committed.version == 1 and len(committed.raw_history) == 1
    assert response.read_count == 1 and committed.raw_history[0].evidence_set.quality_prediction == .95
    assert provider.traces["s", "r"]["support_scores"]["bundles"]
    await controller.aclose()
    controller.store.close()
    provider.close()


@pytest.mark.asyncio
async def test_lmap_requests_summary_read_and_btoken_skips_learned_support(tmp_path):
    provider, _ = make_provider(tmp_path, "Lmap")
    selection = await provider.retrieve(SessionSnapshot("s", corpus_snapshot_id="v2"), "Summarize", "summary")
    assert all(step.operation == "summarize" for step in selection.context_plans[0].steps if step.kind == "READ")
    provider.close()
    baseline, _ = make_provider(tmp_path, "Btoken")
    selected = await baseline.retrieve(SessionSnapshot("b", corpus_snapshot_id="v2"), "Apply", "b")
    assert not selected.context_plans[0].steps and not selected.context_plans[0].selected_set.need_support
    baseline.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ("B2", "B3", "B5", "B6", "B7", "Lmap"))
async def test_support_coverage_search_uses_group_scores_without_model_set_or_layout_calls(tmp_path, strategy):
    provider, _ = make_provider(tmp_path, strategy)
    backend = CoverageOnlySupport()
    provider.support_backend = backend
    provider.config["support_model"] = {"mode": "teacher"}
    provider.config["selection"]["retained_sets"] = 8
    selected = await provider.retrieve(SessionSnapshot(strategy, corpus_snapshot_id="v2"),
                                       "How do I apply?", "coverage")
    plans = selected.context_plans
    assert len(plans) == 3
    assert all(not plan.candidate_qualities for plan in plans)
    assert provider.traces[strategy, "coverage"]["selection_utility"] == "support_coverage"
    assert set(backend.arities) == ({1} if strategy == "B2" else {1, 2})
    assert plans[0].selected_set.quality_prediction == (.2 if strategy == "B2" else .95)
    provider.close()


@pytest.mark.asyncio
async def test_optional_set_reranking_only_scores_retained_sets(tmp_path):
    provider, _ = make_provider(tmp_path)
    provider.config["selection"]["rerank_retained_sets"] = True
    calls = []
    def final_set(state, units, supports, *, context=None):
        calls.append((tuple(unit.unit_id for unit in units), context))
        return SetPrediction(.9 if len(units) == 2 else .1, "answer" if len(units) == 2 else "partial")
    provider.support_backend.predict_set = final_set
    selected = await provider.retrieve(SessionSnapshot("rerank", corpus_snapshot_id="v2"),
                                       "How do I apply?", "rerank")
    assert len(calls) == len(selected.context_plans) == 3
    assert all(context is None for _, context in calls)
    assert selected.context_plans[0].selected_set.quality_prediction == .9
    assert all(not plan.candidate_qualities for plan in selected.context_plans)
    provider.close()


@pytest.mark.parametrize("shared_encoder", (False, True))
def test_frozen_query_and_history_usage_counts_each_model_delta_once(tmp_path, shared_encoder):
    provider, _ = make_provider(tmp_path)
    task = provider.directory.task_interpreter.model
    history = provider.frozen_history
    query = history if shared_encoder else FixtureFrozen()
    provider.serving.retriever.encoder = query
    models = (task, history, query, provider.support_backend)
    for model in models:
        model.usage = {"input_tokens": 100, "output_tokens": 0, "elapsed_ms": 5.0}
    before = {id(model): dict(model.usage) for _, model in provider._usage_sources()}
    task.usage.update(input_tokens=110, output_tokens=3, elapsed_ms=7.0)
    history.usage.update(input_tokens=120, elapsed_ms=9.0)
    if not shared_encoder:
        query.usage.update(input_tokens=130, elapsed_ms=11.0)
    provider.support_backend.usage.update(input_tokens=140, output_tokens=7, elapsed_ms=13.0)
    result = provider._auxiliary_usage(before, ())
    assert result["input_tokens"] == (70 if shared_encoder else 100)
    assert result["output_tokens"] == 10
    assert result["compute_ms"] == (14.0 if shared_encoder else 20.0)
    assert len(result["models"]) == (3 if shared_encoder else 4)
    encoder_rows = [row for row in result["models"] if "history_document_encoder" in row["roles"]]
    assert encoder_rows[0]["input_tokens"] == 20
    assert ("retrieval_query_encoder" in encoder_rows[0]["roles"]) is shared_encoder
    provider.close()


@pytest.mark.asyncio
async def test_supplied_scope_is_per_request_and_does_not_read_the_dataset_catalog(tmp_path):
    provider, _ = make_provider(tmp_path)
    first = make_evidence("Only source A", "supplied-a", corpus_id="manual", version="v2")
    second = make_evidence("Only source B", "supplied-b", corpus_id="manual", version="v2")
    snapshot = SessionSnapshot("s", corpus_snapshot_id="v2")
    metadata = {first.key: {"document_family": "provided-policy", "valid_from": "2020-01-01",
                            "applicability": {"PLATFORM": "Linux"}, "table_id": "table",
                            "version_order": [2, 1], "current": True}}
    one = await provider.retrieve_supplied(snapshot, "Read every source", "one", (first,), source_metadata=metadata)
    two = await provider.retrieve_supplied(snapshot, "Read every source", "two", (second,))
    for selection, block in ((one, first), (two, second)):
        plan = selection.context_plans[0]
        assert plan.coverage_scope == "all_supplied_documents"
        assert plan.task_state.source_scope == (block.key,)
        assert {span.source_key for unit in plan.selected_set.selected_units for span in unit.source_spans} == {block.key}
    unit = one.context_plans[0].selected_set.selected_units[0]
    assert unit.document_family == "provided-policy" and unit.valid_from == "2020-01-01"
    assert unit.applicability == (("PLATFORM", "Linux"),) and unit.table_id == "table"
    extension = await one.open_need_provider(one.context_plans[0], (OpenNeed("apply", "missing"),), ())
    assert tuple(block.key for block in extension.source_blocks) == (first.key,)
    fresh, _ = make_provider(tmp_path)
    saved = replace(snapshot, raw_history=(TurnRecord(1, "Read", "Answer", evidence_set=one.context_plans[0].selected_set),),
                    archive=(first,))
    retrieval_state(saved, fresh.directory)
    restored = next(unit for unit in fresh.directory.units_by_id.values() if unit.source_spans[0].source_key == first.key)
    assert restored.document_family == "provided-policy" and restored.applicability == (("PLATFORM", "Linux"),)
    assert restored.version_order == (2, 1) and restored.current is True
    fresh.close()
    provider.close()


@pytest.mark.asyncio
async def test_persisted_plan_resumes_scoped_reading_without_task_or_support_replay(tmp_path):
    original, _ = make_provider(tmp_path)
    block = make_evidence("Archived Linux allowance", "old-linux", corpus_id="manual", version="v1")
    selection = await original.retrieve_supplied(SessionSnapshot("resume", corpus_snapshot_id="v2"),
        "Read the supplied policy", "request", (block,), source_metadata={block.key: {
            "document_family": "policy", "applicability": {"PLATFORM": "Linux"}, "current": False}})
    plan = selection.context_plans[0]
    original.close()
    recovered, _ = make_provider(tmp_path)
    def no_replay(*args, **kwargs):
        raise AssertionError("initial task/support model was replayed")
    recovered.directory.task_interpreter.model.predict = no_replay
    recovered.support_backend.predict = no_replay
    extension = await recovered.extend_for_plan(SessionSnapshot("resume", corpus_snapshot_id="v2"),
        plan, (OpenNeed("apply", "missing"),), ())
    assert tuple(item.key for item in extension.source_blocks) == (block.key,)
    assert extension.units[0].document_family == "policy"
    assert extension.units[0].applicability == (("PLATFORM", "Linux"),)
    assert recovered.traces["resume", "request"]["resumed_from_plan"] == plan.plan_id
    recovered.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ("B5", "B6", "B7", "Lmap"))
async def test_gap_selects_joint_old_and_fresh_evidence_without_flooding_old_pool(tmp_path, strategy):
    provider, _ = make_provider(tmp_path, strategy)
    provider.support_backend = CoverageOnlySupport()
    provider.config["support_model"] = {"mode": "teacher"}
    snapshot = SessionSnapshot("gap", corpus_snapshot_id="v2")
    selection = await provider.retrieve(snapshot, "How do I apply?", "gap-request")
    plan = selection.context_plans[0]
    first = plan.selected_set.selected_units[0]
    selected = replace(plan.selected_set, selected_units=(first,), source_blocks=tuple(
        block for block in plan.selected_set.source_blocks if block.key == first.source_spans[0].source_key))
    plan = replace(plan, selected_set=selected, budget=replace(plan.budget, max_passages=2))
    new = make_evidence("Bound applicant must submit a permit", "permit", corpus_id="manual", version="v2")
    provider.directory.add_archive((new,))
    fresh = tuple(unit for unit in provider.directory.units_by_id.values() if unit.source_spans[0].source_key == new.key)
    seen = []
    def recall(state, memory, gaps):
        seen.append(provider.directory.open_need_query(state, gaps[0]))
        return EvidencePool(fresh, (new,), retrieval_ms=2.0)
    provider.directory.retrieve_gap_pool = recall
    def no_task(*args, **kwargs):
        raise AssertionError("gap reran task interpretation")
    provider.directory.task_interpreter.model.predict = no_task
    extension = await provider.extend_for_plan(snapshot, plan,
        (OpenNeed("apply", "permit_requirement", bound_entities=("Bound applicant",),
                  parent_claim_ids=("claim-1",)),), ())
    assert "Bound applicant" in seen[0] and "permit_requirement" in seen[0]
    assert {unit.unit_id for unit in extension.units} == {unit.unit_id for unit in fresh}
    assert {unit.unit_id for unit in extension.selected_set.selected_units} == {first.unit_id, fresh[0].unit_id}
    assert tuple(block.key for block in extension.source_blocks) == (new.key,)
    assert extension.auxiliary_ms[0] == ("retrieval", extension.retrieval_ms)
    assert provider.traces["gap", "gap-request"]["gap_rounds"][-1]["remaining_passages"] == 1
    provider.close()
