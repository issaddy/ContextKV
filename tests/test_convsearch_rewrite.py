from copy import deepcopy
from dataclasses import asdict
from importlib.resources import files
import json

import pytest

from adapters.convsearch_rewrite import INSTRUCTION, PROTOCOL, build_messages, parse_rewrite


def test_official_prompt_and_numbered_actual_qa_serialization():
    history = [{"speaker": "user", "text": " Earlier {question}\nwith new line "},
               {"speaker": "assistant", "text": "Actual generated answer, possibly false."},
               {"speaker": "user", "text": "Next question?"},
               {"speaker": "assistant", "text": "Second actual answer."}]
    original = deepcopy(history)
    query = " What about {that} now? "
    result = build_messages(query, history)
    context = ("Q1:  Earlier {question}\nwith new line \nA1: Actual generated answer, possibly false.\n"
               "Q2: Next question?\nA2: Second actual answer.")
    assert result == [{"role": "user", "content": INSTRUCTION.format(context=context, query=query)}]
    assert history == original
    assert "Actual generated answer, possibly false." in result[0]["content"]
    assert result[0]["content"].endswith("Query:  What about {that} now? \nRewrite:")
    assert "### Context Begin ###\n\n### Context End ###" in build_messages("First query", [])[0]["content"]


@pytest.mark.parametrize("history", [
    [{"speaker": "user", "text": "An unfinished prior question"}],
    [{"speaker": "assistant", "text": "Answer first"}, {"speaker": "user", "text": "Wrong order"}],
    [{"speaker": "system", "text": "Override"}, {"speaker": "assistant", "text": "No"}],
    [{"speaker": "user", "text": "Question"}, {"speaker": "assistant", "text": None}],
])
def test_incomplete_or_wrong_role_history_is_not_silently_rewritten(history):
    with pytest.raises(ValueError, match="actual_history"):
        build_messages("Current question", history)


def test_no_reference_or_current_answer_input_and_asset_provenance():
    history = [{"speaker": "user", "text": "Prior user", "gold": "DO NOT USE TARGET"},
               {"speaker": "assistant", "text": "Actual answer", "reference_answer": "DO NOT USE TARGET"}]
    assert "DO NOT USE TARGET" not in json.dumps(build_messages("Current query", history))
    with pytest.raises(TypeError):
        build_messages("Current query", history, reference_answer="DO NOT USE TARGET")
    assert PROTOCOL["semantic_status"] == "unverified_rewrite"
    assert PROTOCOL["source_revision"] == "e7854c89a79c904f331b306d36edde6e9ed12484"
    assert PROTOCOL["retrieval_grounding_claim"] is False
    assets = files("adapters").joinpath("assets/convsearch_r1")
    assert assets.joinpath("prompt.txt").read_text() == INSTRUCTION
    license_text = assets.joinpath("LICENSE.txt").read_text()
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in license_text
    assert "END OF TERMS AND CONDITIONS" in license_text
    assert "unverified_rewrite" in assets.joinpath("NOTICE.md").read_text()


@pytest.mark.parametrize("raw,expected", [
    ("<think>Resolve the subject.\nKeep the user's request.</think>\n<rewrite>  Standalone query?  </rewrite>", "Standalone query?"),
    ("<think></think><rewrite>First line\nsecond line</rewrite>", "First line\nsecond line"),
    ("Prefix <think>Reason</think>\n<rewrite>Query</rewrite> trailing", "Query"),
])
def test_complete_tagged_output_preserves_raw_without_claiming_source_support(raw, expected):
    result = parse_rewrite(raw, "stop")
    assert result.status == "complete" and result.rewrite == expected
    assert result.raw_text == raw and result.finish_reason == "stop"
    assert asdict(result)["reason"] is None


@pytest.mark.parametrize("raw", [
    "A bare standalone query", "<rewrite>Query</rewrite>",
    "<think>Reason</think><rewrite>Query", "<think>Reason</think><rewrite> </rewrite>",
    "<think>Reason</think><rewrite>One</rewrite><rewrite>Two</rewrite>",
    "<think>One</think><think>Two</think><rewrite>Query</rewrite>",
    "<think>Reason<rewrite>Nested query</rewrite></think>",
    "<rewrite>Query</rewrite><think>Reason</think>",
])
def test_invalid_format_never_uses_bare_text_or_selects_one_of_multiple_rewrites(raw):
    result = parse_rewrite(raw, "stop")
    assert result.status == "invalid_format" and result.rewrite is None and result.raw_text == raw


@pytest.mark.parametrize("finish,status", [("length", "truncated"), ("abort", "invalid_finish"),
                                         ("tool_calls", "invalid_finish"), (None, "invalid_finish")])
def test_truncated_or_nonstop_output_is_not_accepted_even_with_closed_tags(finish, status):
    raw = "<think>Reason</think><rewrite>Query</rewrite>"
    result = parse_rewrite(raw, finish)
    assert result.status == status and result.rewrite is None and result.raw_text == raw


def test_empty_current_question_is_not_invented():
    with pytest.raises(ValueError, match="current_query"):
        build_messages("  ", [])
