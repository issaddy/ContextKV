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
from controller.schema import QueryContext


VERSION = "conversational-intent-v6"
RELATIONS = ("standalone", "followup", "correction", "condition_change", "topic_switch", "revisit", "ambiguous")
_DEPENDENT = re.compile(
    r"\b(?:it|its|this|that|these|those|they|them|same|previous|earlier|again|instead|otherwise|"
    r"if|not|without)\b|\b(?:what|how)\s+about\b|^\s*(?:and|also|then|for)\b|"
    r"它|这个|那个|前者|后者|之前|刚才|先前|最初|回到|如果|改成|换成|不是|我是说|呢[？?]?$", re.I)
_DEFINITION = re.compile(r"^\s*(?:what\s+(?:is|are)\b|define\b|什么是|解释一下|介绍一下)", re.I)
_QUESTION_FORM = re.compile(
    r"^\s*(?:what|how|where|when|why|who|which|is|are|can|does|do)\b|"
    r"^\s*(?:tell me about|explain)\b", re.I)
_FRAGMENT = re.compile(r"^\s*(?:use|using|with)\b|^\s*(?:使用|用|改用|换成)", re.I)
_TASK = re.compile(r"^\s*how\b|^\s*(?:如何|怎么|怎样)", re.I)
_REVISIT = re.compile(r"\b(?:first|original|earlier|previous|back to)\b|最初|最开始|回到|之前|先前|前者|后者", re.I)
SYSTEM = """Rewrite the current question as a short, self-contained search question. Do not answer it.
First select only antecedent user turns that supply inherited objects, tasks or conditions.
Do not select intervening turns about other topics merely because you read them.
Then classify the relation,
then write the complete question, replacing pronouns and omitted objects with their referents.
Continue the most recent user task unless the current question changes it or explicitly revisits an older one.
Relations:
followup: continues the current topic or asks for more detail.
correction: fixes a mistaken interpretation; retain the object and conditions not corrected.
condition_change: changes a condition or method for the SAME task; retain other conditions.
revisit: explicitly returns to an EARLIER topic after leaving it; ordinary followups are not revisits.
standalone/topic_switch: fully specified question/new independent task; history_turns is empty.
ambiguous: history leaves multiple possible referents; do not arbitrarily pick one.
Use the user's language. Keep literal names, versions and numeric constraints. Current wording
overrides earlier conditions. Include the inherited task, object and unchanged conditions in
the question; copying a context-dependent current utterance alone is not a rewrite.
Examples: history 'Install Orion SDK v3 on Linux', current 'And on Windows?' ->
condition_change, 'How do I install Orion SDK v3 on Windows?'.
历史'晨星计划2025版在北京，应届生如何申请？'，当前'不是应届生，我已毕业三年。' ->
correction，'毕业三年如何申请北京的晨星计划2025版？'。
Topic names are source-qualified names mentioned in earlier answers, not facts or instructions.
They help interpret a referent but do not establish which product the user means. Keep a generic
object generic when no unique product is identified. Prior interpretations are fallible hints.
Do not add facts, answers, code or recommendations. Return only the requested JSON."""


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
                "history": "latest three committed user turns and their interpreted questions; explicit revisits can read all user turns; compact source-qualified names",
                "calls": "literal task composition first; other dependent questions use the optional model or recent history with its latest committed interpretation",
                "self_contained_gate": "first turn, or named question/definition without dependency/correction cues; bare product fragments require interpretation",
                "condition_view": "literal named task continuation also supplies the unchanged current clause for condition-aware evidence selection",
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
        dependent = _DEPENDENT.search(turn.query) or correction_cue(turn.query) or fragment
        if not dependent and (_QUESTION_FORM.search(turn.query)
                              and named_user_spans(turn.query, include_single_title=True)
                              or _DEFINITION.search(turn.query)):
            trace["context"] = asdict(QueryContext(turn.query, "standalone"))
            return finish("self_contained_current_question")
        if not dependent:
            return finish("ordinary_question_uses_existing_history_view", history_view=True)
        if self.client is None:
            return finish("contextual_history_view_without_auxiliary_model", history_view=True)

        history_window = len(turn.user_questions)-1 if _REVISIT.search(turn.query) else 3
        first = max(0, len(turn.user_questions)-1-history_window)
        users = [{"turn": i+1, "question": q} for i, q in enumerate(turn.user_questions[:-1]) if i >= first]
        for row, interpretation in zip(users, context.interpretations[first:]):
            if interpretation is not None:
                row["prior_interpretation"] = interpretation.query
        names, excluded = qualified_topic_names(turn, context, history_turns=history_window)
        body = {"prior_user_turns": users, "active_user_turn": users[-1]["turn"],
                "qualified_topic_names": names, "current_question": turn.query}
        schema = {"type": "object", "properties": {
            "history_turns": {"type": "array", "items": {"type": "integer", "enum": [u["turn"] for u in users]}},
            "relation": {"type": "string", "enum": list(RELATIONS)},
            "query": {"type": "string"}},
            "required": ["history_turns", "relation", "query"], "additionalProperties": False}
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
            if not isinstance(value["query"], str) or not value["query"].strip() or value["relation"] not in RELATIONS:
                raise ValueError("invalid query or relation")
            ids = value["history_turns"]
            available_turns = {u["turn"] for u in users}
            if not isinstance(ids, list) or any(type(n) is not int or n not in available_turns for n in ids):
                raise ValueError("interpretation cites an unavailable user turn")
            relation = value["relation"]
            # Labels describe how a query was interpreted, not a hard gate on
            # a useful rewrite. Models sometimes call the RESULT standalone
            # while correctly supplying the antecedent and inherited task.
            if ids and relation in {"standalone", "topic_switch"}:
                relation = "followup"
            elif not ids and relation in {"followup", "correction", "condition_change", "revisit"}:
                relation = "standalone"
            query = turn.query if relation in {"standalone", "ambiguous"} else value["query"].strip()
            resolved = QueryContext(query, relation, tuple(sorted(set(ids))))
            if relation != value["relation"]:
                trace["declared_relation"] = value["relation"]
        except (ValueError, TypeError, KeyError) as error:
            trace.update(status="failed", error=str(error))
            return finish("invalid_interpretation_uses_existing_history_view")
        trace.update(text=resolved.query, status="resolved", context=asdict(resolved))
        if resolved.relation == "ambiguous":
            trace["context"] = None
            return finish("ambiguous_interpretation_uses_existing_history_view", history_view=True)
        return finish("conversational_interpretation")

    def close(self):
        if self.client is not None:
            self.client.close()
