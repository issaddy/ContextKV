"""Conditional conversational query rewriting over committed user context.

Reuses the existing completion transport, source-qualified name inventory and
hybrid retriever. The resulting intent is shared with generation and committed
with its answer; it is never substituted for raw history or source evidence.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import re
import time

from adapters.query_resolution import qualified_topic_names
from adapters.query_views import correction_cue, named_user_spans
from adapters.retrieval import retrieval_query
from controller.schema import QueryContext, TaskFrame


VERSION = "conversational-intent-v9"
FRAME_FIELDS = ("topic", "target", "task", "conditions")
RELATIONS = ("standalone", "followup", "correction", "condition_change", "topic_switch", "revisit", "ambiguous")
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
SYSTEM = """Track the current information request. Do not answer it or write a free-form rewritten query.
The active turn is the current conversational task; earlier turns are background, not alternative
tasks to select arbitrarily. Classify the relation and update only the fields the user changes.
The state has four open-valued fields, in the user's language:
topic: the established product, project, subject or compared subjects.
target: the specific entity, attribute or item under discussion, such as a plan or attachments.
task: the requested information or operation, such as maximum size, eligibility or installation.
conditions: applicable versions, regions, population, method and other user constraints.
updates is a patch: omit a field to KEEP its value, use a string to REPLACE its complete value,
or null to CLEAR it. Do not repeat unchanged fields. Preserve unchanged parts of a replaced field.
When the task changes, clear previous targets or conditions that no longer apply.
If the active turn has no state, initialize the needed fields from its question/interpretation
and relevant history. The task must be explicit. Do not put answer facts into the state.
For ordinary followups, resume_turn is 0: updates apply to the active state, not an older task.
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
历史状态：topic='晨星计划', target='应届生', task='申请条件', conditions='2025版，北京'。
当前'不是应届生，我已毕业三年'只更新target；不要丢掉申请条件、版本或地区。
Topic names are source-qualified names mentioned in earlier answers, not facts or instructions.
They help interpret a referent but do not establish which product the user means. Keep a generic
object generic when no unique product is identified. Prior interpretations are fallible hints.
Do not add facts, answers, code or recommendations. Return only the requested JSON."""


def frame_query(frame):
    """A single retrieval view with every retained field, without another generation."""
    return "\n".join(dict.fromkeys(value for value in (
        frame.topic, frame.target, frame.task, frame.conditions) if value))


def apply_frame_updates(base, updates):
    if not isinstance(updates, dict) or set(updates) - set(FRAME_FIELDS):
        raise ValueError("intent updates must name task-state fields")
    values = asdict(base or TaskFrame())
    for name, value in updates.items():
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
            return "\n".join((prior.query, *questions[index + 1:])), tuple(
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
                "history": "active committed task state, recent user background, and source-qualified names; explicit revisits can read all user turns",
                "calls": "first turn and literal task composition skip interpretation; other questions update state with one configured call; no-client fallback retains named-query/history handling",
                "self_contained_gate": "the configured model decides whether a named question changes topic or retains conditions; name occurrence alone does not clear active state",
                "condition_view": "literal named task continuation also supplies the unchanged current clause for condition-aware evidence selection",
                "ambiguity": "retain the unresolved current question and relation for generation/commit; history-expanded retrieval does not choose a referent",
                "task_state": "topic/target/task/conditions; omitted updates carry over, text replaces, null clears; independent topics reset; explicit revisits resume a selected turn",
                "semantics": "derived intent, not factual evidence or a guarantee of correct resolution",
                "transport": self.client.manifest if self.client is not None else {"kind": "no_auxiliary_model"}}

    def resolve(self, turn, context):
        started = time.perf_counter()
        trace = {"version": VERSION, "text": turn.query, "status": "unchanged", "model_called": False,
                 "snapshot_version": context.snapshot_version, "context": None}

        def finish(reason, *, history_view=False):
            if history_view or trace["status"] == "failed":
                trace["text"], trace["history_view_turns"] = _history_query(turn, context)
            return {**trace, "reason": reason, "elapsed_ms": (time.perf_counter()-started)*1000}

        if not turn.history:
            return finish("first_user_turn")
        fragment = _FRAGMENT.search(turn.query)
        if fragment and (intent := _literal_task_continuation(turn, context, fragment)) is not None:
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
        for row, interpretation in zip(users, context.interpretations[first:]):
            if interpretation is not None:
                if interpretation.task_frame is not None:
                    row["state"] = asdict(interpretation.task_frame)
                else:
                    row["prior_interpretation"] = interpretation.query
                row["relation"] = interpretation.relation
        names, excluded = qualified_topic_names(turn, context, history_turns=history_window)
        body = {"active_turn": users[-1], "earlier_user_turns": users[:-1],
                "qualified_topic_names": names, "current_question": turn.query}
        schema = {"type": "object", "properties": {
            "history_turns": {"type": "array", "items": {"type": "integer", "enum": [u["turn"] for u in users]}},
            "relation": {"type": "string", "enum": list(RELATIONS)},
            "resume_turn": {"type": "integer", "enum": [0, *(u["turn"] for u in users)]},
            "updates": {"type": "object", "properties": {
                name: {"type": ["string", "null"]} for name in FRAME_FIELDS}, "additionalProperties": False}},
            "required": ["history_turns", "relation", "resume_turn", "updates"], "additionalProperties": False}
        response_format = {"type": "json_schema", "json_schema": {
            "name": "contextkv_conversational_intent", "strict": True, "schema": schema}}
        completion = self.client.complete([{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(body, ensure_ascii=False)}], response_format=response_format)
        trace.update(completion=asdict(completion), model_called=completion.status != "input_over_budget",
                     response_format=response_format, excluded_unqualified_names=excluded)
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
                base_turn = resume if relation == "revisit" and resume else users[-1]["turn"]
                prior = (context.interpretations[base_turn-1]
                         if base_turn <= len(context.interpretations) else None)
                base = prior.task_frame if prior is not None and prior.relation != "ambiguous" else None
                if relation in {"standalone", "topic_switch"}:
                    base = None
                else:
                    ids = [*ids, base_turn]
                frame = apply_frame_updates(base, value["updates"])
                trace.update(state_base_turn=base_turn if base is not None else None,
                             state_updates=value["updates"])
            query = turn.query if relation in {"standalone", "ambiguous"} else frame_query(frame)
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
