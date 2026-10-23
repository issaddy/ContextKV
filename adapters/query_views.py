"""Small optional query routing from the current user's explicit correction cues."""
from __future__ import annotations

import re


_CORRECTION = re.compile(r"^\s*(?:no|not)\b|\bi\s+(?:mean|meant)\b|\bi\s+was\s+(?:asking|referring)\b", re.IGNORECASE)


def correction_cue(current_question: str) -> str | None:
    """A lexical cue, not a learned intent/quality classifier; no answers or IDs."""
    match = _CORRECTION.search(current_question)
    return match.group().strip() if match else None


def correction_query_mode(current_question: str) -> str:
    return "last3" if correction_cue(current_question) is not None else "current"
