from dataclasses import asdict, replace
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from adapters.conversation import ConversationResolver, apply_frame_updates, frame_query
from adapters.mtrag import OnlineMessage, OnlineTurn
from adapters.query_resolution import PriorContext, PriorSource, snapshot_context
from adapters.query_selector import CompletionResult
from adapters.retrieval import retrieval_query
from controller.candidates import build_candidates
from controller.evidence import make_evidence
from controller.mock_engine import MockEngine
from controller.planner import PlannerKnobs
from controller.schema import QueryContext, SessionSnapshot, TaskFrame
from controller.serialization import decode
from controller.service import Controller, EvidenceSelection
from controller.session_store import SessionStore
from test_candidates import committed


class Client:
    manifest = {"model": "test"}
    def __init__(self, value, *, finish="stop"):
        # Short task-only responses keep the existing resolver tests readable.
        # They still travel through the actual state-update protocol.
        if isinstance(value, dict) and "query" in value:
            value = {**value, "task": {"operation":"replace", "value":value["query"]}, "updates":{}}
            del value["query"]
        if isinstance(value, dict):
            updates = dict(value.get("updates", {}))
            task = ({"operation":"replace", "value":updates.pop("task")} if "task" in updates
                    else {"operation":"keep", "value":""})
            value = {"resume_turn": 0, "task":task, **value, "updates":updates}
            value["task"] = {"requirements": [], **value["task"]}
        self.value, self.finish, self.inputs = value, finish, []
    def complete(self, messages, **kwargs):
        self.inputs.append((messages, kwargs))
        value = self.value
        if (isinstance(value, dict)
                and "frame" in kwargs["response_format"]["json_schema"]["schema"]["properties"]):
            updates = value.get("updates", {})
            frame = value.get("frame", {"task":value["task"]["value"],"topic":updates.get("topic", ""),
                "target":updates.get("target", ""),"conditions":updates.get("conditions", {}),
                "requirements":value["task"]["requirements"]})
            frame = {"requirements": [], **frame}
            value = {k:v for k,v in value.items() if k not in {"task","updates"}} | {"frame":frame}
        return CompletionResult("completed", json.dumps(value), self.finish, 100)


def turn(current, questions=("How does Atlas Cloud store documents?",)):
    history = tuple(m for q in questions for m in (OnlineMessage("user", q),
                      OnlineMessage("assistant", "PhantomStore allows 999 invented items.")))
    return OnlineTurn("s", "opaque-id", len(questions)+1, "corpus", current, history, (), "P3")


def test_attribute_correction_keeps_the_active_task_and_topic():
    frame = TaskFrame("Atlas Store", "documents", "maximum size", (("version","v2"),))
    prior = QueryContext(frame_query(frame), "followup", (1,), frame)
    client = Client({"relation":"correction", "history_turns":[],
                     "updates":{"target":"attachments"}})
    request = turn("No, attachments.", ("What is the size limit?",))
    view = ConversationResolver(client).resolve(request, PriorContext(interpretations=(prior,)))
    intent = decode(QueryContext, view["context"])
    assert intent.task_frame == replace(frame,target="attachments")
    assert intent.history_turns == (1,)  # An inherited state records its actual base.
    assert "maximum size" in view["text"] and "documents" not in view["text"]
    assert request.query == "No, attachments." and prior.task_frame == frame
    body = json.loads(client.inputs[0][0][-1]["content"])
    assert body["previous_turn"]["state"] == {**asdict(frame),"conditions":dict(frame.conditions),
                                                "requirements":list(frame.requirements)}
    assert "prior_interpretation" not in body["previous_turn"]  # Do not duplicate its rendering.


def test_first_request_initializes_compound_needs_then_changes_only_one_condition():
    frame = {"topic":"Atlas CLI", "target":"", "task":"download and install",
             "requirements":["download package", "installation procedure"],
             "conditions":{"version":"v4", "platform":"Linux", "architecture":"x86-64"}}
    client = Client({"frame":frame, "relation":"standalone", "history_turns":[]})
    original = "Give the download package and installation procedure for Atlas CLI v4 on Linux x86-64."
    view = ConversationResolver(client).resolve(turn(original,()), PriorContext())
    initial = decode(QueryContext, view["context"])
    assert view["model_called"] and view["state_mode"] == "initialize"
    assert view["text"] == original and initial.history_turns == ()
    assert initial.task_frame.requirements == ("download package", "installation procedure")
    body = json.loads(client.inputs[0][0][-1]["content"])
    schema = client.inputs[0][1]["response_format"]["json_schema"]["schema"]
    assert body["previous_turn"] is None and schema["properties"]["history_turns"]["maxItems"] == 0
    client.value = {"task":{"operation":"keep", "value":"", "requirements":[]},
                    "updates":{"conditions":{"platform":"Windows"}},
                    "relation":"condition_change", "history_turns":[1], "resume_turn":0}
    changed = ConversationResolver(client).resolve(turn("Use Windows instead.",(original,)),
        PriorContext(interpretations=(initial,)))
    current = decode(QueryContext, changed["context"]).task_frame
    assert current.requirements == initial.task_frame.requirements
    assert dict(current.conditions) == {"version":"v4", "platform":"Windows", "architecture":"x86-64"}
    for need in current.requirements:
        query = frame_query(current, requirement=need)
        assert need in query and "Windows" in query and "v4" in query and "Linux" not in query
    next_task = apply_frame_updates(current,{},task_operation="replace",task_value="supported languages")
    assert next_task.requirements == ()


def test_missing_state_initializes_all_conditions_before_using_later_patches():
    original = "Install Atlas CLI v4 on Linux x86-64 using a private download source."
    frame = {"task":"download package and installation steps", "topic":"Atlas CLI", "target":"installer",
             "conditions":{"version":"v4","platform":"Windows","architecture":"x86-64","source":"private"}}
    client = Client({"frame":frame,"relation":"condition_change","history_turns":[1]})
    view = ConversationResolver(client).resolve(turn("What changes on Windows?",(original,)),PriorContext())
    intent = decode(QueryContext,view["context"])
    assert view["state_mode"] == "initialize" and intent.task_frame.conditions == tuple(sorted(frame["conditions"].items()))
    assert intent.task_frame.task == frame["task"] and intent.task_frame.topic == "Atlas CLI"
    schema = client.inputs[0][1]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["frame"]["required"] == ["task","topic","target","requirements","conditions"]
    assert "updates" not in schema["properties"]
    following = Client({"relation":"condition_change","history_turns":[2],
                        "updates":{"conditions":{"source":"public"}}})
    request = turn("Use the public source instead.",(original,"What changes on Windows?"))
    changed = ConversationResolver(following).resolve(request,PriorContext(interpretations=(None,intent)))
    assert changed["state_mode"] == "patch"
    assert dict(decode(QueryContext,changed["context"]).task_frame.conditions) == {**frame["conditions"],"source":"public"}
    assert following.inputs[0][0][0]["content"] != client.inputs[0][0][0]["content"]


def test_initialization_does_not_accept_a_partial_frame_as_complete_state():
    client = Client({"frame":{"task":"changes on Windows","conditions":{"platform":"Windows"}},
                     "relation":"condition_change","history_turns":[1]})
    request = turn("What changes on Windows?",("Install Atlas CLI v4 on Linux with private endpoints.",))
    view = ConversationResolver(client).resolve(request,PriorContext())
    assert view["status"] == "failed" and view["context"] is None
    assert view["text"] == retrieval_query(request.user_questions) and len(client.inputs) == 1


def test_a_new_information_need_keeps_the_topic_without_regenerating_it():
    frame = TaskFrame("Atlas Store", "Basic and Pro plans", "compare plans")
    prior = QueryContext(frame_query(frame), "followup", (1,), frame)
    client = Client({"relation":"followup", "history_turns":[1],
                     "updates":{"target":"Basic plan","task":"suitability","conditions":{"use":"exploration"}}})
    view = ConversationResolver(client).resolve(turn("Is Basic enough to explore?"),
                                               PriorContext(interpretations=(prior,)))
    intent = decode(QueryContext, view["context"])
    assert intent.task_frame.topic == "Atlas Store" and "compare plans" not in intent.query
    assert intent.task_frame.task == "suitability" and intent.task_frame.conditions == (("use","exploration"),)
    assert len(client.inputs) == 1


def test_condition_changes_and_removals_preserve_other_fields_across_languages():
    original = TaskFrame("晨星计划", "应届生", "申请条件", (("version","2025版"),("region","北京")))
    changed = apply_frame_updates(original, {"target":"毕业三年", "conditions":{"region":"上海"}})
    assert changed.topic == original.topic and changed.task == original.task
    assert "北京" not in frame_query(changed) and "2025版" in frame_query(changed)
    unrestricted = apply_frame_updates(changed, {"conditions":None})
    assert unrestricted.conditions == () and unrestricted.target == "毕业三年"


def test_named_condition_patch_preserves_version_when_connectivity_changes():
    frame = TaskFrame("Atlas SDK","installation","steps",
                      (("version","v4"),("platform","Linux"),("connectivity","offline")))
    changed = apply_frame_updates(frame,{"conditions":{"connectivity":"online"}})
    assert dict(changed.conditions) == {"version":"v4","platform":"Linux","connectivity":"online"}
    assert dict(frame.conditions)["connectivity"] == "offline"
    removed = apply_frame_updates(changed,{"conditions":{"connectivity":None}})
    assert dict(removed.conditions) == {"version":"v4","platform":"Linux"}
    assert "version: v4" in frame_query(removed) and "platform: Linux" in frame_query(removed)


def test_explicit_task_operation_precedes_context_carryover():
    frame = TaskFrame("Atlas Store","attachments","size limit")
    prior = QueryContext(frame_query(frame),"followup",(1,),frame)
    question = "How do I begin using Atlas Store?"
    client = Client({"task":{"operation":"replace","value":"create a service instance"},
                     "relation":"followup","history_turns":[1],"updates":{"target":None}})
    view = ConversationResolver(client).resolve(turn(question),PriorContext(interpretations=(prior,)))
    intent = decode(QueryContext,view["context"])
    assert intent.task_frame == TaskFrame("Atlas Store","","create a service instance")
    assert "size limit" not in view["text"] and "attachments" not in view["text"]
    messages, options = client.inputs[0]
    body = json.loads(messages[-1]["content"])
    assert next(iter(body)) == "current_question" and body["current_question"] == question
    schema = options["response_format"]["json_schema"]["schema"]
    assert next(iter(schema["properties"])) == "task"
    assert "task" not in schema["properties"]["updates"]["properties"]
    kept = apply_frame_updates(frame,{"target":"images"},task_operation="keep",task_value="a different task")
    assert kept.task == "size limit"


def test_legacy_composite_condition_is_preserved_until_explicitly_split():
    value = {"query":"Atlas installation", "relation":"followup", "history_turns":[1],
             "task_frame":{"topic":"Atlas","target":"SDK","task":"installation","conditions":"v4; Linux; offline"}}
    prior = decode(QueryContext,value)
    assert prior.task_frame.conditions == (("legacy","v4; Linux; offline"),)
    assert "v4; Linux; offline" in frame_query(prior.task_frame)
    patched = apply_frame_updates(prior.task_frame,{"conditions":{
        "legacy":None,"version":"v4","platform":"Linux","connectivity":"online"}})
    assert dict(patched.conditions) == {"version":"v4","platform":"Linux","connectivity":"online"}


def test_revisit_resumes_the_selected_task_state_instead_of_the_latest_topic():
    old = TaskFrame("Atlas", "attachments", "maximum size", (("version","v2"),))
    latest = TaskFrame("Nova", "installation", "requirements", (("platform","Windows"),))
    history = ("Atlas file storage?", "And their limits?", "Explain Nova installation.")
    context = PriorContext(interpretations=(None, QueryContext(frame_query(old),"followup",(1,),old),
                                           QueryContext(frame_query(latest),"standalone",(),latest)))
    client = Client({"relation":"revisit","history_turns":[2],"resume_turn":2,
                     "updates":{"conditions":{"version":"v3"}}})
    view = ConversationResolver(client).resolve(turn("Back to the earlier limit, for v3?",history), context)
    assert view["state_base_turn"] == 2 and view["context"]["history_turns"] == (2,)
    assert decode(QueryContext,view["context"]).task_frame == replace(old,conditions=(("version","v3"),))
    assert "Nova" not in view["text"] and "Windows" not in view["text"]


def test_legacy_interpretation_can_initialize_state_without_a_second_call():
    prior = QueryContext("Atlas attachment maximum size?","followup",(1,))
    client = Client({"relation":"followup","history_turns":[1],
                     "updates":{"topic":"Atlas","target":"attachments","task":"upload method"}})
    view = ConversationResolver(client).resolve(turn("How do I upload them?"),
                                               PriorContext(interpretations=(prior,)))
    assert decode(QueryContext,view["context"]).task_frame == TaskFrame("Atlas","attachments","upload method")
    assert view["state_base_turn"] is None and len(client.inputs) == 1
    assert json.loads(client.inputs[0][0][-1]["content"])["previous_turn"]["prior_interpretation"] == prior.query


def test_persistent_task_state_does_not_duplicate_generation_or_change_candidate_tokens():
    frame = TaskFrame("Atlas", "attachments", "size limit", (("version","v2"),))
    intent = QueryContext(frame_query(frame), "correction", (1,), frame)
    blocks = (make_evidence("Original attachment limits, including their conditions.","limits"),)
    snapshot = SessionSnapshot("s")
    first = build_candidates(snapshot,"Initial question",blocks)[0]
    snapshot = committed(first,snapshot,"Initial question",blocks)
    plain = build_candidates(snapshot,"No, attachments.",blocks,query_context=replace(intent,task_frame=None))
    stateful = build_candidates(snapshot,"No, attachments.",blocks,query_context=intent)
    assert [p.input_ids for p in stateful] == [p.input_ids for p in plain]
    assert all('"task_frame"' not in p.messages[-1].content for p in stateful)


@pytest.mark.asyncio
@pytest.mark.parametrize("old_frame", (None, "", "v4; Linux; offline", (("version","v4"),)))
async def test_legacy_intent_request_identity_and_new_state_survive_persistence(tmp_path, old_frame):
    import hashlib
    from controller.serialization import dumps, loads

    legacy = {"query":"Atlas limits?", "relation":"followup", "history_turns":[1]}
    if old_frame is not None:
        legacy["task_frame"] = {"topic":"Atlas","target":"files","task":"size limit","conditions":old_frame}
    intent = loads(QueryContext,json.dumps(legacy))
    assert (intent.task_frame is None) == (old_frame is None)
    store, engine = SessionStore(tmp_path/"state.sqlite3"), MockEngine()
    service = Controller(store,engine)
    evidence = (make_evidence("Original source.","one"),)
    try:
        await service.serve_turn("s","A question",evidence,query_context=intent,request_id="legacy")
        expected = hashlib.sha256(dumps(("A question",evidence,"demo-v1",None,None,legacy)).encode()).hexdigest()
        assert store.find_request("s","legacy").request_hash == expected
        replay = await service.serve_turn("s","A question",evidence,query_context=intent,request_id="legacy")
        assert replay.idempotent_replay
    finally:
        await service.aclose()
        store.close()


def test_model_free_named_questions_keep_the_existing_query_view():
    resolver = ConversationResolver()
    for query in ("How does IBM Cloudant Query work?", "What is AtlasCloud?"):
        view = resolver.resolve(turn(query), PriorContext())
        assert view["text"] == query and not view["model_called"]
        assert view["context"] == asdict(QueryContext(query, "standalone"))
    assert resolver.resolve(turn("First question", ()), PriorContext())["context"] is None


@pytest.mark.parametrize("task,qualifier", [
    ("How do I create tables?", "Use AtlasCloud"),
    ("how to create schema or tables?", "use IBM Cloudant Query"),
    ("如何创建模式和表？", "使用 AtlasCloud"),
])
def test_named_fragment_preserves_the_task_without_an_extra_model_call(task, qualifier):
    client = Client(None)
    view = ConversationResolver(client).resolve(turn(qualifier, ("Earlier unrelated question", task)), PriorContext())
    assert view["text"] == task + "\n" + qualifier
    assert view["context"] == asdict(QueryContext(view["text"], "condition_change", (2,)))
    assert view["condition_query"] == qualifier
    assert not client.inputs and not view["model_called"]


@pytest.mark.parametrize("task,current", [
    ("How do I create tables with AtlasCloud?", "Use NovaCloud"),
    ("How do I create tables?", "Use Python to parse this file instead"),
    ("What is a data lake?", "Use AtlasCloud"),
])
def test_named_replacement_or_new_task_uses_semantic_interpretation(task, current):
    client = Client({"query":"The actual resolved task", "relation":"followup", "history_turns":[1]})
    view = ConversationResolver(client).resolve(turn(current, (task,)), PriorContext())
    assert view["text"] == "The actual resolved task" and view["model_called"]


@pytest.mark.parametrize("query,relation,resolved", [
    ("I meant the warning when installing it with CocoaPods.", "correction", "Atlas Cloud SDK CocoaPods installation warning"),
    ("换到地区 B 呢？", "condition_change", "Atlas Cloud v2 在地区 B 的申请条件是什么？"),
    ("回到最初那个项目，条件是什么？", "revisit", "Atlas Cloud v2 在地区 A 的申请条件是什么？"),
])
def test_contextual_rewrite_preserves_current_question_and_has_bounded_history_provenance(query, relation, resolved):
    client = Client({"query": resolved, "relation": relation, "history_turns": [1],
                     "resume_turn": 1 if relation == "revisit" else 0})
    original = turn(query, ("Atlas Cloud v2 在地区 A 如何申请？", "What about support?"))
    context = PriorContext(interpretations=(QueryContext("first accepted interpretation", "standalone"), None))
    view = ConversationResolver(client).resolve(original, context)
    assert view["text"] == resolved and original.query == query
    assert view["context"]["history_turns"] == ((1,) if relation == "revisit" else (1, 2))
    messages, options = client.inputs[0]
    body = json.loads(messages[-1]["content"])
    assert body["current_question"] == query and body["previous_turn"]["turn"] == 2
    assert body["earlier_user_turns"][0]["prior_interpretation"] == "first accepted interpretation"
    assert "PhantomStore" not in messages[-1]["content"] and "999" not in messages[-1]["content"]
    assert "opaque-id" not in messages[-1]["content"]
    assert options["response_format"]["json_schema"]["schema"]["properties"]["history_turns"]["items"]["enum"] == [1, 2]


def test_qualified_assistant_name_is_context_not_an_answer_fact():
    original = turn("What about its limits?")
    history = (original.history[0], OnlineMessage("assistant", "AtlasCloud holds 999 units. PhantomStore is better."))
    source = make_evidence("AtlasCloud is a document database.", "a")
    client = Client({"query": "AtlasCloud document limits?", "relation": "followup", "history_turns": [1]})
    view = ConversationResolver(client).resolve(replace(original, history=history), PriorContext((PriorSource(source, (1,)),)))
    body = json.loads(client.inputs[0][0][-1]["content"])
    assert [e["text"] for e in body["qualified_topic_names"]] == ["AtlasCloud"]
    assert "999" not in json.dumps(body) and "PhantomStore" not in json.dumps(body)
    assert "PhantomStore" in view["excluded_unqualified_names"]


@pytest.mark.parametrize("value,finish", [
    ({"query":"answer", "relation":"followup", "history_turns":[2]}, "stop"),
    ({"query":"answer", "relation":"unknown", "history_turns":[1]}, "stop"),
    ({"query":"answer", "relation":"followup", "history_turns":[1]}, "length"),
])
def test_invalid_or_truncated_interpretation_uses_current_query_once(value, finish):
    original = turn("What about its limits?")
    client = Client(value, finish=finish)
    view = ConversationResolver(client).resolve(original, PriorContext())
    assert view["status"] == "failed" and view["text"] == retrieval_query(original.user_questions) and view["context"] is None
    assert len(client.inputs) == 1


def test_standalone_state_resets_even_if_the_model_overselects_background_turns():
    current = "What is a data lake?"
    previous = QueryContext("Atlas limits?", "followup", (1,), TaskFrame("Atlas", "files", "size limit", (("version","v2"),)))
    client = Client({"query": current, "relation": "standalone", "history_turns": [1]})
    view = ConversationResolver(client).resolve(turn(current), PriorContext(interpretations=(previous,)))
    assert view["text"] == current and view["context"]["history_turns"] == ()
    assert view["context"]["task_frame"] == asdict(TaskFrame(task=current))


def test_extra_response_metadata_does_not_discard_a_valid_interpretation():
    client = Client({"query":"AtlasCloud limits?", "relation":"followup", "history_turns":[1],
                     "explanation":"Extra transport metadata is not part of the intent."})
    view = ConversationResolver(client).resolve(turn("What about its limits?"), PriorContext())
    assert view["context"] == asdict(QueryContext("AtlasCloud limits?", "followup", (1,), TaskFrame(task="AtlasCloud limits?")))


@pytest.mark.parametrize("question,interpreted", [
    ("how can I improve query", "How can I improve the Atlas Cloud query?"),
    ("Can I address my storage space limits?", "How can I manage Atlas Cloud storage limits?"),
    ("What are the requirements?", "What are the Atlas Cloud requirements?"),
    ("如何增加容量？", "如何增加 Atlas Cloud 的存储容量？"),
])
def test_implicit_followups_do_not_need_a_pronoun_to_receive_interpretation(question, interpreted):
    client = Client({"query":interpreted,"relation":"followup","history_turns":[1]})
    original = turn(question)
    view = ConversationResolver(client).resolve(original, PriorContext())
    assert len(client.inputs) == 1 and view["text"] == interpreted
    assert view["context"] == asdict(QueryContext(interpreted,"followup",(1,),TaskFrame(task=interpreted)))
    assert original.query == question


def test_generic_definition_can_be_resolved_as_a_new_topic():
    question = "What is a data lake?"
    client = Client({"query":question,"relation":"standalone","history_turns":[]})
    view = ConversationResolver(client).resolve(turn(question), PriorContext())
    assert len(client.inputs) == 1 and view["text"] == question
    assert view["context"] == asdict(QueryContext(question,"standalone",task_frame=TaskFrame(task=question)))


def test_conversational_resolution_can_run_without_an_auxiliary_model():
    resolver = ConversationResolver()
    original = turn("What about its limits?")
    view = resolver.resolve(original, PriorContext())
    assert view["text"] == retrieval_query(original.user_questions) and not view["model_called"]
    task = turn("Use AtlasCloud", ("How do I create tables?",))
    assert resolver.resolve(task, PriorContext())["reason"] == "literal_task_continuation"
    assert resolver.manifest["transport"] == {"kind":"no_auxiliary_model"}
    resolver.close()


@pytest.mark.parametrize("client,ambiguous", [(None,False), (Client({}, finish="length"),False),
    (Client({"query":"uncertain", "relation":"ambiguous", "history_turns":[3]}),True)])
def test_history_view_reuses_corrected_conditions_without_restoring_the_old_task(client, ambiguous):
    original = turn("What about the deadline?", (
        "Atlas v1 region A students application?",
        "I meant v2 in region B, for graduates.",
        "What documents do I need?"))
    corrected = QueryContext("Atlas v2 region B graduate application requirements?", "correction", (1,))
    context = PriorContext(interpretations=(None, corrected, None))
    view = ConversationResolver(client).resolve(original, context)
    assert view["text"] == corrected.query + "\nWhat documents do I need?\nWhat about the deadline?"
    assert view["history_view_turns"] == (2, 3)
    assert "region A" not in view["text"] and "v1" not in view["text"]
    if ambiguous:
        assert view["status"] == "ambiguous"
        assert view["context"] == asdict(QueryContext(original.query,"ambiguous",(3,)))
    else:
        assert view["context"] is None  # A history view is not a resolved interpretation.
    assert original.query == "What about the deadline?" and context.interpretations[1] == corrected


def test_ordinary_history_view_starts_at_the_committed_new_topic():
    original = turn("how to create schema or tables?", (
        "Can I address my storage space limits? how can I add or delete storage?",
        "what is Data lakes"))
    context = PriorContext(interpretations=(None, QueryContext("what is Data lakes", "standalone")))
    view = ConversationResolver().resolve(original, context)
    assert view["text"] == "what is Data lakes\nhow to create schema or tables?"
    assert view["history_view_turns"] == (2,)
    assert view["context"] is None and not view["model_called"]


def test_a_new_raw_correction_keeps_precedence_over_the_committed_interpretation():
    original = turn("And what is the deadline?", (
        "Application in region A?", "No, change to region B."))
    context = PriorContext(interpretations=(QueryContext("Atlas graduate application in region A?", "followup"), None))
    view = ConversationResolver().resolve(original, context)
    assert view["text"].splitlines() == ["Atlas graduate application in region A?",
                                        "No, change to region B.", original.query]
    assert view["context"] is None


def test_recent_interpreted_state_preserves_inherited_conditions_without_all_old_topics():
    original = turn("And in region B?", ("old unrelated topic", "another topic", "Atlas v2 region A students?",
                                         "What materials?", "What deadlines?"))
    client = Client({"query":"Atlas v2 students in region B application deadlines?",
                     "relation":"condition_change", "history_turns":[5]})
    state = PriorContext(interpretations=(None, None, None, None,
                         QueryContext("Atlas v2 region A students application deadlines?", "followup", (3, 4))))
    view = ConversationResolver(client).resolve(original, state)
    body=json.loads(client.inputs[0][0][-1]["content"])
    assert [r["turn"] for r in body["earlier_user_turns"]] == [3, 4]
    assert body["previous_turn"]["turn"] == 5
    assert body["previous_turn"]["prior_interpretation"] == state.interpretations[-1].query
    assert view["context"]["history_turns"] == (5,)


@pytest.mark.parametrize("layout,with_intent", [("stable_key", True), ("stable_key", False), ("contextpilot", False)])
def test_source_guide_preserves_reranker_priority_independently_of_body_layout(layout, with_intent):
    from controller.evidence import EvidencePolicy
    primary=make_evidence("PreferredProduct setup steps.", "z")
    secondary=make_evidence("DifferentProduct setup steps.", "a")
    plan=build_candidates(SessionSnapshot("s"), "How to set it up?", (primary, secondary),
        policy=EvidencePolicy(layout=layout),
        query_context=QueryContext("PreferredProduct setup?", "followup", (1,)) if with_intent else None)[0]
    text=plan.messages[-1].content
    guide=json.loads(text.split("Current source guide:",1)[1].splitlines()[1])
    refs={key:ref for ref,key in plan.reference_map}
    assert [row["id"] for row in guide] == [refs[primary.key],refs[secondary.key]]
    assert text.count("Retrieval priority (most relevant first)") == int(with_intent or layout == "contextpilot")
    assert plan.evidence_spans[0].key == (secondary.key if layout == "stable_key" else primary.key)


def test_ambiguous_turn_keeps_source_context_in_both_append_and_rebuild():
    blocks = tuple(make_evidence(f"{name} original limit information.", name,
                   source_uri=f"https://example.org/{name}/limits", title=f"{name} limits")
                   for name in ("Atlas", "Nova"))
    blocks = (replace(blocks[0], parent_section="Atlas account settings"), blocks[1])
    snapshot = SessionSnapshot("s")
    first = build_candidates(snapshot,"Compare Atlas and Nova.",blocks)[0]
    snapshot = committed(first,snapshot,"Compare Atlas and Nova.",blocks)
    query = "How do I raise its limit?"
    view = ConversationResolver(Client({"query":"Atlas limit", "relation":"ambiguous", "history_turns":[1]})).resolve(
        turn(query,("Compare Atlas and Nova.",)),PriorContext())
    intent = QueryContext(**view["context"])
    plans = build_candidates(snapshot,query,blocks,query_context=intent)
    assert {p.action for p in plans if p.feasible} == {"APPEND","REBUILD"}
    for plan in plans:
        message = plan.messages[-1].content
        guide = json.loads(message.split("Current source guide:",1)[1].splitlines()[1])
        assert {row["source"] for row in guide} == {b.source_uri for b in blocks}
        assert next(row for row in guide if row["title"] == "Atlas limits")["section"] == "Atlas account settings"
        assert '"relation":"ambiguous"' in message and "The current referent is unresolved" in message
        assert '"query":"Atlas limit"' not in message
        assert plan.required_blocks == {b.key for b in blocks}
        assert [m.content for m in plan.messages if m.role == "assistant"] == [snapshot.raw_history[0].assistant_text]
        for span in plan.evidence_spans:
            assert plan.messages[span.message_index].content[span.start:span.end] == next(b.text for b in blocks if b.key == span.key)


def test_shared_scope_policy_is_not_duplicated_in_each_interpreted_turn():
    from controller.templates import SYSTEM_PROMPT
    blocks = (make_evidence("Atlas original conditions.","atlas"),)
    snapshot = SessionSnapshot("s")
    intent = QueryContext("Atlas conditions?","standalone")
    first = build_candidates(snapshot,"Atlas conditions?",blocks,query_context=intent)[0]
    snapshot = committed(first,snapshot,"Atlas conditions?",blocks)
    plans = build_candidates(snapshot,"And the limit?",blocks,
        query_context=QueryContext("Atlas limit?","followup",(1,)))
    rule = "If the intended object is unclear"
    for plan in plans:
        assert plan.messages[0].content == SYSTEM_PROMPT
        assert sum(message.content.count(rule) for message in plan.messages) == 1
        assert "Current conversational intent" in plan.messages[-1].content
        assert '"query":"Atlas limit?"' in plan.messages[-1].content


def test_waiting_for_one_sessions_intent_does_not_block_another_sessions_retrieval():
    from adapters.hybrid_retrieval import HybridRetriever
    from adapters.mtrag import OnlineContext
    entered, release = Event(), Event()
    class Resolver:
        def resolve(self, request, context):
            if request.query == "slow":
                entered.set()
                assert release.wait(5)
            return {"text":request.query}
    class Dense:
        manifest={"corpus_sha256":"same"}
        def search(self, query, encoder, **kwargs):
            return (OnlineContext("source",query),)
    sparse=SimpleNamespace(manifest=Dense.manifest,corpus_snapshot_id="snapshot")
    retriever=HybridRetriever(sparse,Dense(),object(),stage="dense",query_mode="conversational",resolver=Resolver())
    with ThreadPoolExecutor(max_workers=2) as pool:
        slow=pool.submit(retriever.retrieve_turn,turn("slow"),prior_context=PriorContext())
        try:
            assert entered.wait(5)
            fast=pool.submit(retriever.retrieve_turn,turn("fast"),prior_context=PriorContext())
            assert fast.result(timeout=2).contexts[0].text == "fast"
        finally:
            release.set()
        assert slow.result(timeout=2).contexts[0].text == "slow"
    assert retriever.requests == 2


@pytest.mark.parametrize("relation", ("condition_change", "correction", "standalone", "followup", "ambiguous"))
def test_intent_label_does_not_disable_cache_reuse_but_current_source_scope_excludes_old_condition(relation):
    shared = make_evidence("Atlas project common rules.", "base")
    a = make_evidence("Region A condition: students may apply.", "a")
    b = make_evidence("Region B condition: workers may apply.", "b")
    snap = SessionSnapshot("s")
    snap = committed(build_candidates(snap, "Region A conditions?", (shared, a))[0], snap,
                     "Region A conditions?", (shared, a))
    intent = QueryContext("Atlas project Region B conditions?", relation, (1,))
    append, rebuild = build_candidates(snap, "What about B?", (shared, b), query_context=intent)
    assert append.feasible and append.required_blocks == rebuild.required_blocks
    assert rebuild.required_blocks == {shared.key, b.key}
    assert snap.raw_history[0].assistant_text in [m.content for m in rebuild.messages]
    assert rebuild.messages[-1].content.endswith("\nQuestion:\nWhat about B?\n")
    assert intent.query in rebuild.messages[-1].content
    assert a.key not in append.required_blocks and a.key in append.visible_blocks
    followup = QueryContext("Atlas project Region A conditions in detail?", "followup", (1,))
    assert build_candidates(snap, "More details?", (shared, a), query_context=followup)[0].feasible


@pytest.mark.parametrize("layout", ("stable_key", "contextpilot"))
def test_explicit_revision_replaces_retrieval_hit_without_mutating_archive(layout):
    from controller.evidence import EvidencePolicy
    v1 = make_evidence("The original v1 limit is 10.", "Atlas", version="v1")
    v2 = make_evidence("The v2 limit is 20.", "Atlas", version="v2")
    snap = SessionSnapshot("s", archive=(v1, v2))
    for hits in ((v2,), (v1, v2)):
        plan = build_candidates(snap, "Use Atlas v1", hits, EvidencePolicy(layout=layout))[0]
        assert plan.required_blocks == {v1.key}
        assert v2.text not in plan.messages[-1].content
        v2_ref = next(ref for ref, key in plan.reference_map if key == v2.key)
        assert v2_ref not in plan.messages[-1].content
    compare = build_candidates(snap, "Compare Atlas v1 and v2", (v2,))[0]
    assert compare.required_blocks == {v1.key, v2.key}
    assert snap.archive == (v1, v2)


@pytest.mark.parametrize("revision", ("v1.1", "v1-rc2", "v1+build1"))
def test_revision_prefix_does_not_select_the_wrong_archived_version(revision):
    old = make_evidence("Original revision.", "Atlas", version="v1")
    new = make_evidence("Requested revision.", "Atlas", version=revision)
    snap = SessionSnapshot("s", archive=(old, new))
    candidate = build_candidates(snap, f"Use Atlas {revision}.", (old, new))[0]
    assert candidate.required_blocks == {new.key}


def test_selected_revisit_restores_older_source_beyond_recent_history_window():
    old = make_evidence("Atlas eligibility condition: students may apply.", "atlas")
    snap = SessionSnapshot("s")
    snap = committed(build_candidates(snap, "Atlas conditions?", (old,))[0], snap,
                     "Atlas conditions?", (old,))
    for index in range(4):
        latest = make_evidence(f"Different project {index}, unrelated condition.", str(index))
        query = f"Describe project {index}."
        snap = committed(build_candidates(snap, query, (latest,))[-1], snap, query, (latest,))
    context = QueryContext("Atlas eligibility conditions?", "revisit", (1,))
    for candidate in build_candidates(snap, "Back to the first one, its conditions?", (latest,), query_context=context):
        assert old.key in candidate.required_blocks <= candidate.visible_blocks
        assert old.text in candidate.messages[-1].content


def test_multiple_selected_antecedents_do_not_restore_the_last_unrelated_source():
    snap = SessionSnapshot("s")
    for name in ("Atlas", "Nova"):
        block = make_evidence(f"{name} application material.", name)
        query = f"{name} requirements?"
        snap = committed(build_candidates(snap, query, (block,))[-1], snap, query, (block,))
    current = make_evidence("Atlas revised material.", "new")
    intent = QueryContext("Atlas application material?", "revisit", (1, 2))
    for plan in build_candidates(snap, "Back to the first one?", (current,), query_context=intent):
        assert plan.required_blocks == {current.key}


@pytest.mark.asyncio
async def test_disjoint_append_commits_new_scope_then_version_change_retires_after_rebuild(tmp_path):
    store, engine = SessionStore(tmp_path/'state.sqlite3'), MockEngine()
    service = Controller(store, engine, knobs=PlannerKnobs(strategy="always_append"))
    old = make_evidence("Old task evidence.", "old")
    current = make_evidence("AtlasCloud table task evidence.", "current")
    revised = make_evidence("Revised AtlasCloud task evidence.", "current", version="v2")
    try:
        first = await service.serve_turn("s", "How do I create tables?", (old,), request_id="one")
        snapshot = store.get_snapshot("s")
        original_epoch = snapshot.current_epoch
        request = turn("Use AtlasCloud", (snapshot.raw_history[0].user_text,))
        intent = QueryContext(**ConversationResolver(Client(None)).resolve(request, PriorContext())["context"])
        second = await service.serve_turn("s", request.query, (current,), query_context=intent, request_id="two")
        snapshot = store.get_snapshot("s")
        assert second.action == "APPEND" and snapshot.current_epoch == original_epoch
        assert set(snapshot.raw_history[-1].evidence_keys) == {current.key}
        assert snapshot.raw_history[-1].query_context == intent
        assert set(first.citation_scope).isdisjoint(second.citation_scope)
        third = await service.serve_turn("s", "Use the revised conditions.", (revised,), request_id="three")
        assert third.action == "REBUILD"
        assert store.get_snapshot("s").current_epoch != original_epoch
        await service.close_session("s")
        assert not store.outbox("s")
        assert store.db.execute("SELECT closed FROM sessions WHERE session_id='s'").fetchone()[0] == 1
    finally:
        await service.aclose()
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("lost_response", (False, True))
@pytest.mark.parametrize("relation", ("condition_change", "ambiguous"))
async def test_interpretation_commits_with_answer_and_idempotent_replay_does_not_rewrite(tmp_path, lost_response, relation):
    from controller.engine_client import UnknownOutcomeError
    store, engine = SessionStore(tmp_path/'state.sqlite3'), MockEngine()
    service = Controller(store, engine, knobs=PlannerKnobs(strategy="always_append"))
    evidence = make_evidence("A condition with its full original source.", "a")
    calls = []
    frame = None if relation == "ambiguous" else TaskFrame("Atlas", "applicants", "eligibility", (("version","v2"),("region","B")))
    intent = QueryContext("And region B?" if relation == "ambiguous" else frame_query(frame), relation, (1,), frame)
    async def provider(snapshot):
        calls.append(snapshot.version)
        return EvidenceSelection((evidence,), (evidence.key,), intent)
    try:
        await service.serve_turn("s", "Original user question", (evidence,), request_id="one")
        if lost_response:
            engine.drop_next_generate_response = True
            with pytest.raises(UnknownOutcomeError):
                await service.serve_turn("s", "And region B?", request_id="two", evidence_provider=provider)
            assert store.get_snapshot("s").raw_history[-1].query_context is None
        answer = await service.serve_turn("s", "And region B?", request_id="two", evidence_provider=provider)
        attempt = store.find_request("s", "two")
        snapshot = store.get_snapshot("s")
        assert attempt.query_context == snapshot.raw_history[-1].query_context == intent
        assert snapshot.raw_history[-1].user_text == "And region B?"
        again = await service.serve_turn("s", "And region B?", request_id="two", evidence_provider=provider)
        assert again.output_ids == answer.output_ids and again.idempotent_replay and calls == [1]
        online = turn("Next", tuple(t.user_text for t in snapshot.raw_history))
        online = replace(online, history=tuple(m for t in snapshot.raw_history for m in (
            OnlineMessage("user", t.user_text), OnlineMessage("assistant", t.assistant_text))))
        assert snapshot_context(online, snapshot).interpretations[-1] == intent
    finally:
        await service.aclose()
        store.close()
