from dataclasses import replace
import json

import numpy as np

from adapters.corpus import CorpusIndex
from adapters.evidence_pool import PoolProfile, SourceDirectory
from adapters.history_state import HistoryIndex, TaskInterpreter, TaskPrediction, TeacherTaskModel, apply_condition_updates
from adapters.query_selector import CompletionResult
from controller.context_schema import Condition, Need, RetrievalState, TaskState, TextRef, VersionIntent
from controller.evidence import make_evidence
from controller.schema import OnlineContext, OnlineTurn, QueryContext, TurnRecord


class Characters:
    def __call__(self, text, **kwargs):
        result = {"input_ids": list(range(len(text)))}
        if kwargs.get("return_offsets_mapping"):
            result["offset_mapping"] = [(i, i+1) for i in range(len(text))]
        return result


def encode(texts):
    return np.asarray([[1+text.lower().count("orchid"), 1+text.lower().count("linux")]
                       for text in texts], dtype=np.float32)


def test_condition_correction_preserves_object_and_replaces_platform():
    old_ref = TextRef(1, "user", 10, 17)
    old = Condition("windows", value_spans=(old_ref,), relation_type="PLATFORM")
    need = Need("install", (TextRef(1, "user", 0, 17),), operation="procedure")
    initial = TaskState("install-orchid", "ORCHID on Windows", conditions=(old,), needs=(need,))
    memory = RetrievalState(1, (TurnRecord(1, "ORCHID on Windows", "Actual answer."),), (initial,))
    turn = OnlineTurn("s", "correct", 2, "manual", "改成 Linux", (), (), "P3")
    new = Condition("linux", value_spans=(TextRef(2, "user", 3, 8),), relation_type="PLATFORM",
                    target_refs=("windows",), supersedes=("windows",), update="REPLACE")
    class Model:
        def predict(self, turn, windows, memory):
            return TaskPrediction("correction", conditions=(new,))
    state = TaskInterpreter(Model(), HistoryIndex(Characters(), encode)).interpret(turn, memory)
    assert state.task_id == initial.task_id and state.needs[0].operation == "procedure"
    assert [c.condition_id for c in state.conditions] == ["linux"]
    assert "ORCHID" in state.query and "Linux" in state.query
    assert "Windows" not in state.query
    assert all(ref.turn_id == 1 for ref in state.working_history)
    assert state.needs[0].condition_ids == ("linux",)


def test_condition_compare_keeps_both_branches_and_remove_targets():
    old = Condition("v1")
    new = Condition("v2", update="COMPARE", target_refs=("v1",))
    assert apply_condition_updates((old,), (new,)) == (old, new)
    removed = Condition("remove", update="REMOVE", target_refs=("v1",))
    assert apply_condition_updates((old, new), (removed,)) == (new,)


def test_history_index_retains_remote_question_after_long_answer_and_isolates_sessions():
    index = HistoryIndex(Characters(), encode, unit_tokens=32, candidates=6, view_tokens=192)
    old = TurnRecord(1, "ORCHID install", "Linux details")
    recent = TurnRecord(2, "Unrelated weather", "weather "*1000)
    selected = index.select("ORCHID", RetrievalState(2, (old, recent)))
    assert any(w.ref.turn_id == 1 and "ORCHID" in w.text for w in selected)
    assert sum(w.tokens for w in selected) <= 192
    other = index.select("Different", RetrievalState(1, (TurnRecord(1, "Different", "other answer"),)))
    assert all("ORCHID" not in w.text and "weather" not in w.text for w in other)


def make_directory():
    records = [
        {"_id": "eligibility", "document_id": "manual", "version": "v2", "text": "Linux eligibility",
         "document_family": "ORCHID", "valid_from": "2025-01-01", "applicability": {"platform": "Linux"}},
        {"_id": "steps", "document_id": "manual", "version": "v2", "text": "Install ORCHID\nRun setup",
         "document_family": "ORCHID", "valid_from": "2025-01-01", "dependencies": ["eligibility"]},
        {"_id": "legacy", "document_id": "manual", "version": "v1", "text": "Windows legacy steps",
         "document_family": "ORCHID", "valid_from": "2020-01-01", "valid_to": "2025-01-01"}]
    corpus = CorpusIndex.from_records(records, corpus_id="manual", snapshot_id="snapshot")
    def search(query, top_k):
        return tuple(OnlineContext(b.key.passage_id, b.text, score=1.0) for b in reversed(corpus.blocks))[:top_k]
    return SourceDirectory(corpus, search, Characters(), reference_date="2026-11-02",
                           profile=PoolProfile(primary_pool_size=1))


def test_version_applicability_dependency_closure_and_original_offsets():
    directory = make_directory()
    state = TaskState("t", "Install", needs=(Need("n", (TextRef(1, "user", 0, 7),)),), current_turn_id=1)
    pool = directory.retrieve_pool(state, RetrievalState(0))
    assert {block.key.passage_id for block in pool.source_blocks} == {"steps", "eligibility"}
    assert all(unit.source_spans[0].source_key.version == "v2" for unit in pool.units)
    steps = [unit for unit in pool.units if unit.source_spans[0].source_key.passage_id == "steps"]
    assert steps[1].source_spans[0].start == len("Install ORCHID\n")
    assert steps[0].dependencies and pool.units[-1].applicability == (("platform", "Linux"),)


def test_compare_versions_and_full_scope_ignore_retrieval_quota():
    directory = make_directory()
    state = TaskState("t", "Compare", version_intent=VersionIntent("compare", version_ids=("v1", "v2")))
    pool = directory.full_scope(tuple(directory._blocks))
    assert {u.source_spans[0].source_key.version for u in pool.units} == {"v1", "v2"}
    assert len(pool.units) > directory.profile.primary_pool_size
    historical = replace(state, version_intent=VersionIntent("explicit", version_ids=("v1",)))
    selected = directory.retrieve_pool(historical, RetrievalState(0))
    assert {u.source_spans[0].source_key.version for u in selected.units} == {"v1"}


def test_archive_directory_keeps_old_source_not_in_current_corpus():
    directory = make_directory()
    old = make_evidence("Archived old source", "old", version="v0")
    directory.add_archive((old,))
    state = TaskState("old-task", "Old", source_scope=(old.key,), version_intent=VersionIntent("explicit", version_ids=("v0",)))
    pool = directory.retrieve_pool(state, RetrievalState(0, archive_source_keys=(old.key,)))
    assert pool.source_blocks == (old,)


def test_archive_recall_ranks_remote_source_with_frozen_vectors():
    directory = make_directory()
    directory.archive_encoder = encode
    sources = tuple(make_evidence("Weather records", f"unrelated{i}", version="old") for i in range(20))
    relevant = make_evidence("ORCHID original manual", "remote", version="old")
    directory.add_archive((*sources, relevant))
    ranked = directory.archive_search("ORCHID", tuple(b.key for b in (*sources, relevant)), top_k=12)
    assert ranked[0] == relevant.key and len(ranked) == 12


def test_current_to_previous_selects_prior_family_version_and_compare_retains_both():
    directory = make_directory()
    current = next(key for key in directory._blocks if key.passage_id == "steps")
    old = next(key for key in directory._blocks if key.passage_id == "legacy")
    prior = TaskState("policy", "Current policy", current_turn_id=1)
    memory = RetrievalState(1, (TurnRecord(1, "Current policy", "Current answer", evidence_keys=(current,)),), (prior,))
    before = TaskState("policy", "What about before that?", version_intent=VersionIntent("as_of",
        source_turns=(1,), temporal_relation="before"), current_turn_id=2)
    assert directory._applicable(old, before, memory)
    assert not directory._applicable(current, before, memory)
    compare = replace(before, version_intent=replace(before.version_intent, mode="compare"))
    assert directory._applicable(old, compare, memory) and directory._applicable(current, compare, memory)
    after = replace(before, version_intent=VersionIntent("as_of", version_ids=("v1",), temporal_relation="after"))
    assert directory._applicable(current, after, memory) and not directory._applicable(old, after, memory)


def test_at_date_keeps_at_relation_and_does_not_compile_before_marker():
    question = "Policy on 2026-01-01"
    turn = OnlineTurn("s", "date", 1, "manual", question, (), (), "P3")
    intent = VersionIntent("as_of", time_refs=(TextRef(1, "user", 10, 20),), temporal_relation="at",
                           resolved_version_ids=("v2",))
    class Model:
        def predict(self, turn, windows, memory):
            return TaskPrediction("standalone", version_intent=intent)
    state = TaskInterpreter(Model(), HistoryIndex(Characters(), encode)).interpret(turn, RetrievalState(0))
    assert "BEFORE" not in state.query and state.version_intent.temporal_relation == "at"
    directory = make_directory()
    assert all(directory._applicable(key, state, RetrievalState(0)) == (key.version == "v2") for key in directory._blocks)


def test_declared_semver_order_routes_previous_without_invented_dates():
    corpus = CorpusIndex.from_records([{"_id": version, "document_id": "sdk", "version": version,
        "text": "SDK "+version, "document_family": "sdk", "version_order": order,
        "current": version == "2.0.0"} for version, order in (
            ("1.9.0", [1, 9, 0]), ("1.10.0", [1, 10, 0]), ("2.0.0", [2, 0, 0]))],
        corpus_id="sdk", snapshot_id="sdk-catalog")
    directory = SourceDirectory(corpus, lambda *args, **kwargs: (), Characters())
    latest = next(key for key in directory._blocks if key.version == "2.0.0")
    memory = RetrievalState(1, (TurnRecord(1, "Current SDK", "Version 2.0.0", evidence_keys=(latest,)),),
                            (TaskState("sdk-task", "Current SDK", current_turn_id=1),))
    state = TaskState("sdk-task", "Before that?", version_intent=VersionIntent("as_of", source_turns=(1,),
                        temporal_relation="before"), current_turn_id=2)
    actual = [key.version for key in directory._blocks if directory._applicable(key, state, memory)]
    assert actual == ["1.10.0"]
    assert all(unit.valid_from is None and unit.published_at is None for unit in directory.units_by_id.values())
    assert next(unit for unit in directory.units_by_id.values() if unit.source_spans[0].source_key == latest).version_order == (2, 0, 0)


def test_literal_markdown_rows_keep_header_heading_notes_and_original_unicode_offsets():
    text = "# Linux 限额\nTable 1: Capacity\n| 项目 | Limit (tokens) |\n| --- | --- |\n| 甲 | 12 |\n| 乙 | 20 |\nNote: Linux only.\n"
    corpus = CorpusIndex.from_records([{"_id": "table", "text": text}], corpus_id="table", snapshot_id="v1")
    directory = SourceDirectory(corpus, lambda *args, **kwargs: (), Characters())
    row = next(unit for unit in directory.units_by_id.values() if "| 乙 |" in unit.text)
    deps = [directory.units_by_id[identity].text for identity in row.dependencies]
    assert any("Limit (tokens)" in item for item in deps)
    assert any("Linux 限额" in item for item in deps)
    assert any("Linux only" in item for item in deps)
    assert row.table_id is not None
    span = row.source_spans[0]
    assert text[span.start:span.end] == row.text
    directory.add_archive(corpus.blocks)
    assert directory.units_by_id[row.unit_id].dependencies == row.dependencies


def test_long_prose_prefers_sentence_boundaries_and_overlaps_unbroken_sentences():
    text = "Linux allowance is 10 credits. "*12+"条件"*160+" ends."
    corpus = CorpusIndex.from_records([{"_id": "long", "text": text}], corpus_id="long", snapshot_id="v1")
    directory = SourceDirectory(corpus, lambda *args, **kwargs: (), Characters(), profile=PoolProfile(unit_tokens=64))
    units = tuple(directory.units_by_id.values())
    spans = [unit.source_spans[0] for unit in units]
    assert all(text[span.start:span.end] == unit.text for span, unit in zip(spans, units, strict=True))
    assert all(unit.body_tokens <= 64 for unit in units)
    assert any(left.end > right.start for left, right in zip(spans, spans[1:]))
    assert spans[0].start == 0 and spans[-1].end == len(text)
    assert all(left.end >= right.start for left, right in zip(spans, spans[1:]))
    assert units[0].text.endswith(". ")


def test_teacher_short_pointers_decode_to_stable_ids_and_canonical_catalogue_versions():
    directory = make_directory()
    old = Condition("long-uuid-condition", value_spans=(TextRef(1, "user", 10, 17),), relation_type="PLATFORM")
    prior = TaskState("long-uuid-prior-task", "ORCHID on Windows", conditions=(old,),
        needs=(Need("long-uuid-need", condition_ids=(old.condition_id,)),),
        version_intent=VersionIntent("current", document_families=("ORCHID",), version_ids=("v2",)), current_turn_id=1)
    memory = RetrievalState(1, (TurnRecord(1, "ORCHID on Windows", "Windows answer"),), (prior,), source_index=directory)
    seen = {}
    class Client:
        manifest = {"kind": "test_protocol"}
        def complete(self, messages, *, response_format):
            body = json.loads(messages[-1]["content"])
            schema = response_format["json_schema"]["schema"]
            seen.update(body=body, schema=schema)
            ref = {"turn_id": 2, "speaker": "user", "start": 0, "end": 5}
            version = next(item["version_id"] for item in body["source_versions"] if item["literal_version"] == "v1")
            output = {"relation": "correction", "resume_task_id": "S0", "history_spans": [],
                "conditions": {"c0": {"key_span": ref, "value_spans": [ref],
                    "relation_type": "PLATFORM", "predicate_span": ref, "target_refs": ["C0"],
                    "supersedes": ["C0"], "update": "REPLACE"}}, "condition_removals": [],
                "needs": {"n0": {"anchor_spans": [ref], "subject_refs": [], "operation": "lookup",
                    "condition_ids": ["c0"], "predecessor_need_ids": ["N0"], "scope": "selected_sources"}},
                "version_intent": {"mode": "explicit", "temporal_relation": "at", "document_families": ["F0"],
                    "version_ids": [version], "resolved_version_ids": [], "time_refs": [], "source_turns": [1]}}
            return CompletionResult("completed", json.dumps(output), "stop", 100,
                                    {"completion_tokens": 30}, elapsed_ms=5.0)
    model = TeacherTaskModel(Client())
    result = model.predict(OnlineTurn("s", "current-uuid", 2, "manual", "Linux", (), (), "P3"), (), memory)
    assert result.resume_task_id == prior.task_id
    assert result.conditions[0].condition_id == "current-uuid:c0"
    assert result.conditions[0].target_refs == (old.condition_id,)
    assert result.needs[0].condition_ids == ("current-uuid:c0",)
    assert result.needs[0].predecessor_need_ids == ("long-uuid-need",)
    assert result.version_intent.version_ids == ("v1",) and result.version_intent.document_families == ("ORCHID",)
    assert "long-uuid" not in json.dumps(seen["body"])
    branch = next(item["properties"] for item in seen["schema"]["anyOf"]
        if item["properties"]["resume_task_id"]["enum"] == ["S0"]
        and item["properties"]["conditions"]["required"] == ["c0"])
    assert branch["resume_task_id"]["enum"] == ["S0"]
    needs = branch["needs"]["anyOf"][1]
    need = seen["schema"]["$defs"][needs["properties"]["n0"]["$ref"].rsplit("/", 1)[1]]
    condition_pointer = seen["schema"]["$defs"][need["properties"]["condition_ids"]["$ref"].rsplit("/", 1)[1]]
    assert condition_pointer["items"]["enum"] == ["C0", "c0"]
    assert all(item.startswith("V") for item in seen["schema"]["$defs"]["VersionIntent"]["properties"]["version_ids"]["items"]["enum"])
    assert model.usage == {"tasks": 1, "input_tokens": 100, "output_tokens": 30, "elapsed_ms": 5.0}


def test_as_of_non_iso_wording_uses_final_catalogue_pointer_without_lexical_date_comparison():
    directory = make_directory()
    state = TaskState("policy", "What applied last year?", current_turn_id=1,
        version_intent=VersionIntent("as_of", time_refs=(TextRef(1, "user", 13, 22),),
            temporal_relation="at", resolved_version_ids=("v2",)))
    selected = [key for key in directory._blocks if directory._applicable(key, state, RetrievalState(0))]
    assert selected and all(key.version == "v2" for key in selected)
    unknown = replace(state, version_intent=replace(state.version_intent, resolved_version_ids=()))
    assert not any(directory._applicable(key, unknown, RetrievalState(0)) for key in directory._blocks)
    before = replace(state, version_intent=replace(state.version_intent,
        temporal_relation="before", version_ids=("v2",), resolved_version_ids=("v1",)))
    assert [key.version for key in directory._blocks if directory._applicable(key, before, RetrievalState(0))] == ["v1"]


def test_owned_need_graph_survives_fifth_to_sixth_turn_with_current_predecessors():
    directory = make_directory()
    prior = TaskState("original-task", "Prior actual question", needs=(Need("four:n0"),), current_turn_id=4)
    memory = RetrievalState(4, tuple(TurnRecord(index, f"Actual question {index}", f"Actual answer {index}")
        for index in range(1, 5)), (prior,), source_index=directory)
    captured = {}
    class Client:
        manifest = {"kind": "test_protocol"}
        def complete(self, messages, *, response_format):
            body = json.loads(messages[-1]["content"])
            captured[body["turn_id"]] = (body, response_format["json_schema"]["schema"])
            ref = {"turn_id": body["turn_id"], "speaker": "user", "start": 0, "end": len(body["question"])}
            base = {"anchor_spans": [ref], "subject_refs": [], "operation": "lookup", "condition_ids": [],
                    "predecessor_need_ids": [], "scope": "selected_sources"}
            needs = {"n0": base}
            if body["turn_id"] == 5:
                needs["n1"] = {**base, "predecessor_need_ids": ["n0"]}
            else:
                needs["n0"] = {**base, "predecessor_need_ids": [body["prior_states"][0]["needs"][-1]["need_id"]]}
            value = {"relation": "followup", "resume_task_id": "S0", "history_spans": [],
                "conditions": {}, "condition_removals": [], "needs": needs, "version_intent": {
                    "mode": "current", "temporal_relation": "at", "document_families": [], "version_ids": [],
                    "resolved_version_ids": [], "time_refs": [], "source_turns": []}}
            return CompletionResult("completed", json.dumps(value), "stop", 100, {"completion_tokens": 50}, elapsed_ms=5)
    model = TeacherTaskModel(Client())
    interpreter = TaskInterpreter(model, HistoryIndex(Characters(), encode))
    fifth = interpreter.interpret(OnlineTurn("s", "five", 5, "manual", "Answer both parts", (), (), "P3"), memory)
    assert fifth.needs[1].predecessor_need_ids == ("five:n0",)
    committed = TurnRecord(5, "Answer both parts", "Actual fifth answer",
                           query_context=QueryContext("Answer both parts", "task", task_state=fifth))
    memory = replace(memory, committed_history=(*memory.committed_history, committed),
                     prior_task_states=(*memory.prior_task_states, fifth))
    sixth = interpreter.interpret(OnlineTurn("s", "six", 6, "manual", "Follow that dependency", (), (), "P3"), memory)
    assert sixth.needs[0].predecessor_need_ids == ("five:n1",)
    body, schema = captured[5]
    branch = next(item["properties"] for item in schema["anyOf"] if
        item["properties"]["resume_task_id"]["enum"] == ["S0"] and not item["properties"]["conditions"]["required"])
    owned = branch["needs"]["anyOf"][2]
    assert owned["required"] == ["n0", "n1"] and owned["additionalProperties"] is False
    for index in (0, 1):
        properties = schema["$defs"][owned["properties"][f"n{index}"]["$ref"].rsplit("/", 1)[1]]["properties"]
        allowed = schema["$defs"][properties["predecessor_need_ids"]["$ref"].rsplit("/", 1)[1]]["items"]["enum"]
        assert all(key not in allowed for key in tuple(f"n{i}" for i in range(index, 8)))
    assert len(json.dumps(schema)) < 150000


def test_condition_replacement_and_explicit_removal_share_one_prior_retirement():
    old = Condition("old")
    new = Condition("new", update="REPLACE", target_refs=("old",), supersedes=("old",))
    removal = Condition("remove-event", update="REMOVE", target_refs=("old",))
    assert apply_condition_updates((old,), (new, removal)) == (new,)
