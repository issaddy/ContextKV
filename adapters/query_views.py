"""Optional user-only query views; literal cues are not entity/intent classifiers."""
from __future__ import annotations

import re


_CORRECTION = re.compile(r"^\s*(?:no|not)\b|\bi\s+(?:mean|meant)\b|\bi\s+was\s+(?:asking|referring)\b", re.IGNORECASE)


def correction_cue(current_question: str) -> str | None:
    """A lexical cue, not a learned intent/quality classifier; no answers or IDs."""
    match = _CORRECTION.search(current_question)
    return match.group().strip() if match else None


def correction_query_mode(current_question: str) -> str:
    return "last3" if correction_cue(current_question) is not None else "current"


USER_TOPIC_VERSION = "literal-user-topic-v1"
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)*[®™]?")
_DISCOURSE = set("i we you he she they it its this that these those a an the what which who where when why how "
                 "can could should would do does did is are was were will not no yes ok okay great thanks thank "
                 "one last next thing question and or but then so also actually now please tell explain compare use using".split())
_FOLLOWUP = re.compile(r"\b(?:it|its|this|that|these|those|they|them|here|there|same|previous|again)\b|"
                       r"^\s*(?:and|also|then|so)\b|\b(?:one last thing|last question|next question|"
                       r"differences? between|what about|how about)\b", re.IGNORECASE)
_PIVOT = re.compile(r"\b(?:new|different|unrelated)\s+(?:topic|subject)\b|"
                   r"\b(?:switch|switching|change|changing)\s+(?:the\s+)?(?:topic|subject)\b|"
                   r"\b(?:now|instead)\s+(?:let['’]s\s+)?(?:talk|ask|discuss)\b", re.IGNORECASE)
_NEGATIVE = re.compile(r"\b(?:not|no|never|without|except|excluding)\b|n['’]t\b", re.IGNORECASE)


def named_user_spans(text, *, include_single_title=False):
    """Literal Latin-script name-like spans, with no product dictionary or NER.

    Multiword capitalized runs and mixed-case/digit identifiers can be history
    cues. A single title-case word blocks adding an old topic to a current query,
    but is not carried forward as a topic itself. Standalone all-cap acronyms are
    ambiguous and are not carried (e.g. a generic interface/format abbreviation).
    """
    tokens = list(_WORD.finditer(text))
    spans, index = [], 0
    while index < len(tokens):
        token = tokens[index]
        word = token.group().rstrip("®™")
        if not word[0].isupper() or word.lower() in _DISCOURSE:
            index += 1
            continue
        end = index + 1
        while end < len(tokens):
            following = tokens[end].group().rstrip("®™")
            if (not text[tokens[end-1].end():tokens[end].start()].isspace()
                    or not following[0].isupper() or following.lower() in _DISCOURSE):
                break
            end += 1
        if end-index > 1:
            kind = "capitalized_phrase"
        elif re.search(r"[a-z][A-Z]", word) or any(c.isdigit() for c in word):
            kind = "mixed_case_or_digit_identifier"
        elif include_single_title and not word.isupper():
            kind = "single_title_word_current_only"
        else:
            index += 1
            continue
        start_offset, end_offset = token.start(), tokens[end-1].end()
        spans.append({"text": text[start_offset:end_offset], "start": start_offset,
                      "end": end_offset, "kind": kind})
        index = end
    return spans


def user_topic_view(user_questions):
    """Add literal user topic cues to a follow-up, retaining the current query.

    This is an optional lexical experiment, not a resolved standalone rewrite.
    No assistant messages, retrieval results, task IDs, labels or answers enter it.
    Current named wording/pivots stay current; corrections reuse the existing
    last-three-user view. Other follow-ups select all cues from the nearest prior
    eligible user turn, stopping at explicit topic/negation boundaries.
    """
    if not user_questions:
        raise ValueError("retrieval requires the current user question")
    current = user_questions[-1]
    trace = {"version": USER_TOPIC_VERSION, "text": current, "route": "current",
        "reason": "no_followup_cue", "source_user_turns": [len(user_questions)],
        "selected_history_spans": [], "current_named_spans": named_user_spans(current, include_single_title=True)}
    if len(user_questions) == 1:
        trace["reason"] = "first_user_turn"
        return trace
    if _PIVOT.search(current):
        trace["reason"] = "explicit_topic_shift"
        return trace
    if trace["current_named_spans"]:
        trace["reason"] = "current_has_named_wording"
        return trace
    cue = correction_cue(current)
    if cue:
        trace.update(text="\n".join(user_questions[-3:]), route="last3",
            reason="explicit_correction_reuses_last3", correction_cue=cue,
            source_user_turns=list(range(max(1, len(user_questions)-2), len(user_questions)+1)))
        return trace
    if _NEGATIVE.search(current):
        trace["reason"] = "unresolved_current_negation"
        return trace
    if not _FOLLOWUP.search(current):
        return trace
    trace["reason"] = "no_named_user_history"
    for index in range(len(user_questions)-2, -1, -1):
        historical = user_questions[index]
        if _PIVOT.search(historical) or _NEGATIVE.search(historical):
            trace.update(reason="history_topic_or_negation_boundary", boundary_user_turn=index+1)
            break
        spans = named_user_spans(historical)
        if not spans:
            continue
        selected = [{"user_turn": index+1, **span} for span in spans]
        names = list(dict.fromkeys(span["text"] for span in selected))
        trace.update(text=current+"\nUser topic context: "+"; ".join(names), route="user_topic",
            reason="nearest_user_named_spans", source_user_turns=[index+1, len(user_questions)],
            selected_history_spans=selected)
        break
    return trace


def user_topic_manifest():
    return {"version": USER_TOPIC_VERSION, "inputs": "current and prior user questions only",
        "selection": "nearest eligible prior user turn's literal capitalized phrases / mixed-case identifiers",
        "application": "explicit follow-up cues only; current named wording and topic shifts stay current; corrections reuse last3",
        "output": "unchanged current question followed by original user topic spans",
        "extra_model_calls": 0, "semantic_entity_resolution": False}
