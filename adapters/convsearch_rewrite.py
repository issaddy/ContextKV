# SPDX-License-Identifier: Apache-2.0
"""Pure ConvSearch-R1 prompt/format adapter, not source-verified rewriting.

The fixed upstream instruction and Q/A serialization are reused under Apache
2.0; see assets/convsearch_r1/NOTICE.md. No model, HTTP or dataset is loaded.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import json
import re
from typing import Mapping, Sequence

_ASSETS = files("adapters").joinpath("assets/convsearch_r1")
INSTRUCTION = _ASSETS.joinpath("prompt.txt").read_text(encoding="utf-8")
PROTOCOL = json.loads(_ASSETS.joinpath("protocol.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class RewriteResult:
    status: str
    rewrite: str | None
    raw_text: str
    finish_reason: str | None
    reason: str | None = None


def build_messages(current_query: str, actual_history: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Format complete, ordered prior actual QA pairs with the upstream prompt.

    The caller owns the temporal boundary. This function cannot establish that
    supplied answers were generated rather than reference answers; its interface
    intentionally accepts no dataset, target, reference or current answer field.
    """
    if not isinstance(current_query, str) or not current_query.strip():
        raise ValueError("current_query must be nonempty text")
    if len(actual_history) % 2:
        raise ValueError("actual_history must contain complete prior user/assistant pairs")
    context = []
    for index, message in enumerate(actual_history):
        role = "user" if index % 2 == 0 else "assistant"
        if message.get("speaker") != role or not isinstance(message.get("text"), str):
            raise ValueError("actual_history must retain ordered user/assistant speaker/text messages")
        prefix = "Q" if role == "user" else "A"
        context.append(f"{prefix}{index // 2 + 1}: {message['text']}")
    return [{"role": "user", "content": INSTRUCTION.format(context="\n".join(context), query=current_query)}]


def parse_rewrite(raw_text: str, finish_reason: str | None) -> RewriteResult:
    """Parse a completed official tagged rewrite; never invent a fallback query.

    Upstream uses DOTALL searches for both tags. Preserve that extraction while
    rejecting ambiguous repeated/unclosed pairs and truncated generations.
    Surrounding non-tag text is retained in raw_text, not used as the rewrite.
    A complete result is syntactic only: it remains an unverified_rewrite.
    """
    if finish_reason == "length":
        return RewriteResult("truncated", None, raw_text, finish_reason, "output_token_limit")
    if finish_reason != "stop":
        return RewriteResult("invalid_finish", None, raw_text, finish_reason, "generation_did_not_stop")
    matches = {}
    for tag in ("think", "rewrite"):
        match = re.search(rf"<{tag}>(.*?)</{tag}>", raw_text, re.DOTALL)
        if match is None or raw_text.count(f"<{tag}>") != 1 or raw_text.count(f"</{tag}>") != 1:
            return RewriteResult("invalid_format", None, raw_text, finish_reason, "expected_single_think_and_rewrite_pairs")
        matches[tag] = match
    if matches["think"].end() > matches["rewrite"].start():
        return RewriteResult("invalid_format", None, raw_text, finish_reason, "expected_think_before_rewrite")
    rewrite = matches["rewrite"].group(1).strip()
    if not rewrite:
        return RewriteResult("invalid_format", None, raw_text, finish_reason, "empty_rewrite")
    return RewriteResult("complete", rewrite, raw_text, finish_reason)
