"""Identity of the actual B supervision input semantics and reader geometry."""
from __future__ import annotations

from .serialization import to_jsonable
from .source_windows import SOURCE_WINDOW_LINES, SOURCE_WINDOW_TOKENS


def input_protocol(fingerprint):
    """Bind semantics and actual tokenizer/template scope, independently of continuation."""
    return {
        "semantics": "merged_source_windows_actual_coverage_reused_facts_compiled_choice",
        "source_windows": {"tokens": SOURCE_WINDOW_TOKENS, "nonempty_lines": SOURCE_WINDOW_LINES},
        "source_identity": "content_hash_and_original_coordinates",
        "reader_fingerprint": to_jsonable(fingerprint),
    }


def validate_input_protocol(value, *, reader_fingerprint=None):
    if not isinstance(value, dict) or "reader_fingerprint" not in value:
        raise ValueError("plan gain evidence requires the current actual input protocol binding")
    expected = input_protocol(value["reader_fingerprint"] if reader_fingerprint is None else reader_fingerprint)
    if value != expected:
        raise ValueError("plan gain input protocol differs from the actual source/compiler/reader scope")
    return value


def validate_pre_action_input(run):
    """Keep the B cost feature quote attached to the exact chosen native request."""
    native = run.get("pre_action_quote")
    candidate = run.get("pre_action_input")
    if native is None or native["candidate_id"] is None:
        if candidate is not None:
            raise ValueError("non-native first action cannot name a compiled model input")
        return
    if candidate is None or candidate["candidate_id"] != native["candidate_id"]:
        raise ValueError("pre-action quote requires its actual chosen compiled input")
    validate_input_protocol(run.get("input_protocol"), reader_fingerprint=candidate["fingerprint"])
    quotes = [quote for quote in native["batch"]["quotes"]
              if quote["candidate_id"] == candidate["candidate_id"]]
    if len(quotes) != 1 or (quotes[0]["input_hash"] != candidate["input_hash"]
            or quotes[0]["input_len"] != len(candidate["input_ids"])):
        raise ValueError("pre-action quote differs from its chosen compiled input")
    calls = run["actual_native_calls"]
    if calls and calls[0]["request"]["input_ids"] != candidate["input_ids"]:
        raise ValueError("first executed native input differs from the B pre-action choice")
