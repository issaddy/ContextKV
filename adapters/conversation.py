"""Conditional conversational query rewriting over committed user context.

Reuses the existing completion transport, actual conversational history and
hybrid retriever. The resulting intent is shared with generation and committed
with its answer; it is never substituted for raw history or source evidence.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import re
import time

from adapters.query_resolution import dialogue_turns
from adapters.query_views import correction_cue, named_user_spans
from adapters.retrieval import retrieval_query
from controller.schema import QueryContext, TaskFrame


VERSION = "conversational-intent-v13"
FRAME_FIELDS = ("topic", "target", "conditions")
RELATIONS = ("standalone", "followup", "correction", "condition_change", "topic_switch", "revisit", "ambiguous", "conversational")
_DEPENDENT = re.compile(
    r"\b(?:it|its|this|that|these|those|they|them|same|previous|earlier|again|instead|otherwise|"
    r"if|not|without)\b|\b(?:what|how)\s+about\b|^\s*(?:and|also|then|for)\b|"
    r"它|这个|那个|前者|后者|之前|刚才|先前|最初|回到|如果|改成|换成|不是|我是说|呢[？?]?$", re.I)
_QUESTION_FORM = re.compile(
    r"^\s*(?:what|how|where|when|why|who|which|is|are|can|does|do)\b|"
    r"^\s*(?:tell me about|explain)\b", re.I)
_FRAGMENT = re.compile(r"^\s*(?:use|using|with)\b|^\s*(?:使用|用|改用|换成)", re.I)
_TASK = re.compile(r"^\s*how\b|^\s*(?:如何|怎么|怎样)", re.I)
_REVISIT = re.compile(r"\b(?:first|original|earlier|previous|back to)\b|最初|最开始|回到|之前|先前|前者|后者", re.I)
SYSTEM = """Interpret the CURRENT user request. Do not answer it or write a free-form rewritten query.
previous_turn is PAST state, not the task the user must continue. Start with the current question.
Question/answer pairs are the actual conversation. Use both sides to identify what 'it',
'that method' or another reference denotes. An earlier answer can establish the discussed
object without proving its factual claims. Current corrections override earlier wording.
conversational: an acknowledgement, closing, greeting or social reaction without a new
information request. Use task.operation keep, empty value/requirements and empty updates;
do not invent a search task or change the active topic. A factual request remains a request
even when phrased as a statement or combined with thanks.
First decide task.operation and task.value, BEFORE choosing inherited context:
replace: the current question asks a new information need or action. Write that requested task.
keep: the current question only corrects an object, changes conditions, or elliptically continues
the previous task. Keep its task unchanged; value is empty if previous state already has a task.
task.requirements lists separately requested information needs when replacing a compound task;
for keep use an empty list, since the host retains all previous requirements. A single need can
use an empty list. Preserve every requested part without adding optional advice or answer facts.
Requirements name information needs, not changing attributes: platform/version belong in conditions.
If no prior state is available, supply the inherited task in value from the prior user question.
Do not classify a new request as keep merely because it is about the same product.
For example, moving from a size limit to an installation procedure replaces the task while the
product can stay unchanged. Correcting 'documents' to 'attachments' keeps the size-limit task.
Then classify the relation and update only the context fields that change, in the user's language:
topic: the established product, project, subject or compared subjects.
target: the specific entity, attribute or item under discussion, such as a plan or attachments.
conditions: a map of independently changing attributes, e.g. version, platform, region or method.
updates is a patch: omit a field to KEEP it; text REPLACES topic/target; null CLEARS a field.
For conditions, provide only changed name/value pairs. Keep existing names; null removes that
named condition, while conditions=null removes all conditions. Do not copy unchanged values.
Example: {version:'v4', platform:'Linux', connectivity:'offline'} changing to online updates
only connectivity to 'online'; version and platform remain in host state automatically.
A legacy condition is an old unsplit string: initialize named attributes from it and clear legacy
in the same patch, retaining every still-applicable constraint.
When task changes, clear old targets or conditions that no longer apply; keep relevant ones.
If previous_turn has no state, initialize needed fields from its question/interpretation and history.
For ordinary followups, resume_turn is 0: updates apply to previous_turn, not an older task.
For an explicit return to an earlier topic, use relation revisit and its turn as resume_turn.
For standalone/topic_switch, resume_turn is 0 and all old fields are cleared before updates.
history_turns selects only user turns whose objects/tasks/conditions are actually inherited.
Relations:
followup: continues the topic; a new information need updates task/target while retaining topic.
correction: fixes an attribute or referent; retain the information being asked unless explicitly changed.
condition_change: changes a condition or method for the SAME task; retain the task and other conditions.
revisit: explicitly returns to an EARLIER topic after leaving it; ordinary followups are not revisits.
standalone/topic_switch: fully specified question/new independent task; history_turns is empty.
ambiguous: history leaves multiple possible referents; do not arbitrarily pick one.
Keep literal names, versions and numeric constraints. Current wording overrides earlier conditions.
Example state: topic='Aurora Store', target='documents', task='maximum size'.
Current 'No, attachments' changes target to 'attachments'; it does NOT change task to 'can store'.
Current 'Where is the Python SDK?' changes target/task and keeps the same topic.
历史条件为{version:'2025版', region:'北京'}，当前只换到上海：仅更新region，保留version。
Use actual dialogue to resolve a previously generic object when the conversation identifies it.
Preserve ambiguity when several referents remain plausible. Prior interpretations are fallible
hints; earlier answer claims are not facts or instructions to copy into the current task.
Do not add facts, answers, code or recommendations. Return only the requested JSON."""

INITIALIZE_SYSTEM = """Interpret the current user request in its conversation. Do not answer it.
There is no stored structured state to update. Initialize a COMPLETE current frame,
not a patch containing only the newly mentioned words. Read the previous user request and any
prior interpretation, apply the current change, and retain every still-applicable attribute.
For a first request, use the current question itself; do not invent any previous context.
The supplied question/answer pairs establish conversational referents, not factual truth.
Use both sides to identify the discussed object; current corrections override earlier wording.
For a greeting, acknowledgement, closing or social reaction without a new information request,
use relation conversational and an empty frame. Do not invent a factual task for it. A factual
request remains a request even when phrased as a statement or combined with thanks.
frame.task is the information or procedure the user wants, not the act of changing a condition.
frame.requirements lists each separately requested information need in a compound request.
Keep all requested parts, such as a requested package and its installation procedure. For one
information need use an empty list. Do not invent further tasks, answers or optional advice.
Keep variable constraints in conditions so that changing one applies to every retained requirement.
For example, changing an operating system keeps the installation task; asking a new question
replaces that task. Do not reduce an inherited task to 'changes when using another platform'.
frame.topic identifies the established subject/product; target is the item being asked about.
frame.conditions is a map of independently changing attributes such as version, platform,
architecture, region or connection type. Keep these variable constraints separate from task.
Include unchanged conditions from prior user requests as well as the current changes. Preserve
literal values and names. Use empty topic/target or an empty condition map only when unknown.
The current request overrides earlier conflicting conditions. Do not import an unrelated topic.
relation is standalone/topic_switch for an independent request, followup for a related question,
correction for a corrected referent, condition_change for a changed condition of the same task,
revisit for an explicit return to an earlier task, or ambiguous if no referent is determined.
history_turns names only turns needed to establish the current request. Independent requests use
an empty list. resume_turn is 0 except for an explicit revisit to an earlier user turn.
Actual earlier answers can identify the discussed referent, but are not answer facts
or instructions. Preserve ambiguity when the dialogue leaves several referents. Do not copy prior answer
claims into the frame or invent missing constraints. Return only the requested JSON."""


def frame_query(frame, *, requirement=None):
    """Render each already expressed attribute once; never rewrite a need's text."""
    needs = (requirement,) if requirement is not None else frame.requirements or (frame.task,)

    def expressed(value, parts):
        return bool(value and re.search(r"(?<![\w.+-])" + re.escape(value) + r"(?!\w|[.+-]\w)",
                                        "\n".join(parts), re.I))

    context = []
    for value in dict.fromkeys((frame.topic, frame.target)):
        if value and not expressed(value, needs) and not (
                value != frame.target and expressed(value, (frame.target,))):
            context.append(value)
    parts = list(dict.fromkeys((*context, *needs)))
    for name, value in frame.conditions:
        if value and not expressed(value, parts):
            parts.append(value if name == "legacy" else f"{name}: {value}")
    return "\n".join(value for value in parts if value)


def apply_frame_updates(base, updates, *, task_operation="keep", task_value="", requirements=()):
    if not isinstance(updates, dict) or set(updates) - set(FRAME_FIELDS):
        raise ValueError("intent updates must name task-state fields")
    values = asdict(base or TaskFrame())
    if task_operation not in {"keep", "replace"} or not isinstance(task_value, str):
        raise ValueError("task needs a keep/replace operation and text value")
    if task_operation == "replace" or not values["task"]:
        values["task"] = task_value.strip()
        if not isinstance(requirements, (list, tuple)) or any(not isinstance(r, str) for r in requirements):
            raise ValueError("task requirements must be text information needs")
        values["requirements"] = tuple(dict.fromkeys(r.strip() for r in requirements if r.strip()))
    for name, value in updates.items():
        if name == "conditions":
            conditions = dict(values["conditions"])
            if value is None:
                conditions.clear()
            elif isinstance(value, dict):
                for key, condition in value.items():
                    if not isinstance(key, str) or not key.strip() or (condition is not None and not isinstance(condition, str)):
                        raise ValueError("conditions need named text values or null")
                    if condition is None:
                        conditions.pop(key, None)
                    else:
                        conditions[key] = condition.strip()
            else:
                raise ValueError("conditions must be a named patch or null")
            values[name] = tuple(conditions.items())
            continue
        if value is not None and not isinstance(value, str):
            raise ValueError("task-state values must be text or null")
        values[name] = value.strip() if value is not None else ""
    frame = TaskFrame(**values)
    if not frame.task:
        raise ValueError("intent state must specify the requested task")
    return frame


def _history_query(turn, context):
    """Use the latest committed interpretation in the existing recent window.

    A resolved question already carries its inherited task and conditions. Do
    not prepend the older raw conditions it replaced. Later raw utterances and
    the current question remain verbatim, including corrections not yet resolved.
    This is a retrieval view, not a new semantic interpretation to commit.
    """
    questions = turn.user_questions
    first = max(0, len(questions) - 3)
    for index in range(len(questions) - 2, first - 1, -1):
        prior = context.interpretations[index] if index < len(context.interpretations) else None
        if prior is not None and prior.relation != "ambiguous":
            if prior.relation == "conversational" and prior.task_frame is None:
                continue
            active_query = frame_query(prior.task_frame) if prior.relation == "conversational" else prior.query
            return "\n".join((active_query, *questions[index + 1:])), tuple(
                range(index + 1, len(questions)))
    return retrieval_query(questions), tuple(range(first + 1, len(questions)))


def _literal_task_continuation(turn, context, fragment):
    """A bare named tool/service can qualify the previous unresolved task.

    Keep both user clauses verbatim. Replacing an already named object or
    interpreting a complete new instruction stays on the existing model path.
    """
    target = turn.query[fragment.end():].strip().rstrip(".?!。！？")
    spans = named_user_spans(target, include_single_title=True)
    if len(spans) != 1 or (spans[0]["start"], spans[0]["end"]) != (0, len(target)):
        return None
    prior = context.interpretations[-1] if context.interpretations else None
    task = prior.query if prior is not None else turn.user_questions[-2]
    if not _TASK.search(task) or named_user_spans(task, include_single_title=True):
        return None
    return QueryContext(task + "\n" + turn.query, "condition_change", (len(turn.user_questions)-1,))


class ConversationResolver:
    def __init__(self, client=None):
        self.client = client

    @property
    def manifest(self):
        return {"version": VERSION, "system_prompt": SYSTEM,
                "initialization_prompt": INITIALIZE_SYSTEM,
                "history": "current question, committed state and complete recent actual QA; explicit revisits can read older user turns and stored states",
                "calls": "one configured interpretation initializes the first task or updates later state; literal task composition still reuses exact clauses; no-client mode retains history handling",
                "self_contained_gate": "the configured model decides whether a named question changes topic or retains conditions; name occurrence alone does not clear active state",
                "condition_view": "literal named task continuation also supplies the unchanged current clause for condition-aware evidence selection",
                "ambiguity": "retain the unresolved current question and relation for generation/commit; history-expanded retrieval does not choose a referent",
                "task_state": "initialize a complete current frame when the previous turn has none; otherwise keep the existing task-first patch protocol; omitted attributes persist, null deletes; independent topics reset and explicit revisits resume",
                "semantics": "derived intent, not factual evidence or a guarantee of correct resolution",
                "conversational_turns": "preserve active state, bypass source retrieval/selection and respond as dialogue",
                "transport": self.client.manifest if self.client is not None else {"kind": "no_auxiliary_model"}}

    def resolve(self, turn, context):
        started = time.perf_counter()
        trace = {"version": VERSION, "text": turn.query, "status": "unchanged", "model_called": False,
                 "snapshot_version": context.snapshot_version, "context": None}

        def finish(reason, *, history_view=False):
            if history_view or trace["status"] == "failed":
                trace["text"], trace["history_view_turns"] = _history_query(turn, context)
            return {**trace, "reason": reason, "elapsed_ms": (time.perf_counter()-started)*1000}

        if not turn.history and self.client is None:
            return finish("first_user_turn")
        fragment = _FRAGMENT.search(turn.query)
        if turn.history and fragment and (intent := _literal_task_continuation(turn, context, fragment)) is not None:
            trace.update(text=intent.query, status="resolved", context=asdict(intent),
                         condition_query=turn.query)
            return finish("literal_task_continuation")
        if self.client is None:
            dependent = _DEPENDENT.search(turn.query) or correction_cue(turn.query) or fragment
            if not dependent and _QUESTION_FORM.search(turn.query) and named_user_spans(turn.query, include_single_title=True):
                trace["context"] = asdict(QueryContext(turn.query, "standalone"))
                return finish("self_contained_current_question")
            return finish("contextual_history_view_without_auxiliary_model", history_view=True)

        history_window = len(turn.user_questions)-1 if _REVISIT.search(turn.query) else 3
        first = max(0, len(turn.user_questions)-1-history_window)
        users = [{"turn": i+1, "question": q} for i, q in enumerate(turn.user_questions[:-1]) if i >= first]
        recent = {row["turn"]: row["answer"] for row in dialogue_turns(turn)}
        for row in users:
            if row["turn"] in recent:
                row["answer"] = recent[row["turn"]]
        for row, interpretation in zip(users, context.interpretations[first:]):
            if interpretation is not None:
                if interpretation.task_frame is not None:
                    row["state"] = {**asdict(interpretation.task_frame),
                                    "conditions": dict(interpretation.task_frame.conditions)}
                else:
                    row["prior_interpretation"] = interpretation.query
                row["relation"] = interpretation.relation
        body = {"current_question": turn.query, "previous_turn": users[-1] if users else None,
                "earlier_turns": users[:-1]}
        initialize = not users or "state" not in users[-1] or users[-1].get("relation") == "ambiguous"
        requirements_schema = {"type": "array", "items": {"type": "string"}}
        schema = {"type": "object", "properties": {
            "task": {"type": "object", "properties": {
                "operation": {"type": "string", "enum": ["keep", "replace"]},
                "value": {"type": "string"}, "requirements": requirements_schema},
                "required": ["operation", "value", "requirements"], "additionalProperties": False},
            "relation": {"type": "string", "enum": list(RELATIONS) if users else ["standalone", "ambiguous", "conversational"]},
            "history_turns": ({"type": "array", "items": {"type": "integer", "enum": [u["turn"] for u in users]}}
                              if users else {"type": "array", "items": {"type": "integer"}, "maxItems": 0}),
            "resume_turn": {"type": "integer", "enum": [0, *(u["turn"] for u in users)]},
            "updates": {"type": "object", "properties": {
                "topic": {"type": ["string", "null"]}, "target": {"type": ["string", "null"]},
                "conditions": {"type": ["object", "null"], "additionalProperties": {"type": ["string", "null"]}}},
                "additionalProperties": False}},
            "required": ["task", "relation", "history_turns", "resume_turn", "updates"], "additionalProperties": False}
        if initialize:
            frame_properties = {name: {"type": "string"} for name in ("task", "topic", "target")}
            frame_properties["requirements"] = requirements_schema
            frame_properties["conditions"] = {"type": "object", "additionalProperties": {"type": "string"}}
            schema["properties"] = {"frame": {"type": "object", "properties": frame_properties,
                "required": list(frame_properties), "additionalProperties": False},
                **{name: schema["properties"][name] for name in ("relation", "history_turns", "resume_turn")}}
            schema["required"] = list(schema["properties"])
        response_format = {"type": "json_schema", "json_schema": {
            "name": "contextkv_conversational_intent", "strict": True, "schema": schema}}
        completion = self.client.complete([{"role": "system", "content": INITIALIZE_SYSTEM if initialize else SYSTEM},
            {"role": "user", "content": json.dumps(body, ensure_ascii=False)}], response_format=response_format)
        trace.update(completion=asdict(completion), model_called=completion.status != "input_over_budget",
                     response_format=response_format,
                     state_mode="initialize" if initialize else "patch")
        if completion.status != "completed" or completion.finish_reason != "stop":
            trace["status"] = "failed"
            return finish("incomplete_interpretation_uses_existing_history_view")
        try:
            value = json.loads(completion.text)
            if value["relation"] not in RELATIONS:
                raise ValueError("invalid intent relation")
            ids = value["history_turns"]
            available_turns = {u["turn"] for u in users}
            if not isinstance(ids, list) or any(type(n) is not int or n not in available_turns for n in ids):
                raise ValueError("interpretation cites an unavailable user turn")
            relation = value["relation"]
            resume = value["resume_turn"]
            if type(resume) is not int or resume not in {0, *available_turns}:
                raise ValueError("intent resumes an unavailable user turn")
            if relation in {"standalone", "topic_switch"}:
                ids = []
            elif relation == "revisit" and not resume:
                relation = "followup"
            frame = None
            if relation != "ambiguous":
                base_turn = resume if relation == "revisit" and resume else users[-1]["turn"] if users else 0
                prior = (context.interpretations[base_turn-1]
                         if 0 < base_turn <= len(context.interpretations) else None)
                base = prior.task_frame if prior is not None and prior.relation != "ambiguous" else None
                if relation in {"standalone", "topic_switch"}:
                    base = None
                elif base_turn:
                    ids = [*ids, base_turn]
                if relation == "conversational":
                    frame = base
                    trace.update(state_base_turn=base_turn if base is not None else None,
                                 task_operation="preserve_conversational_state")
                elif initialize:
                    complete = value["frame"]
                    frame = apply_frame_updates(None, {name: complete[name] for name in FRAME_FIELDS},
                                                task_operation="replace", task_value=complete["task"],
                                                requirements=complete["requirements"])
                    trace.update(state_base_turn=None, task_operation="initialize")
                else:
                    task = value["task"]
                    frame = apply_frame_updates(base, value["updates"],
                                                task_operation=task["operation"], task_value=task["value"],
                                                requirements=task["requirements"])
                    trace.update(state_base_turn=base_turn if base is not None else None,
                                 task_operation=task["operation"], state_updates=value["updates"])
            query = turn.query if relation in {"standalone", "ambiguous", "conversational"} else frame_query(frame)
            resolved = QueryContext(query, relation, tuple(sorted(set(ids))), frame)
            if relation != value["relation"]:
                trace["declared_relation"] = value["relation"]
        except (ValueError, TypeError, KeyError) as error:
            trace.update(status="failed", error=str(error))
            return finish("invalid_interpretation_uses_existing_history_view")
        trace.update(text=resolved.query, status="resolved", context=asdict(resolved))
        if resolved.relation == "ambiguous":
            trace["status"] = "ambiguous"
            return finish("ambiguous_interpretation_uses_existing_history_view", history_view=True)
        return finish("conversational_interpretation")

    def close(self):
        if self.client is not None:
            self.client.close()
