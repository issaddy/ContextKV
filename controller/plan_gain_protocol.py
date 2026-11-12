"""Identity of the actual B supervision input semantics and reader geometry."""
from __future__ import annotations

from .serialization import to_jsonable
from .source_prompt import SOURCE_ENCODING
from .source_windows import SOURCE_WINDOW_LINES, SOURCE_WINDOW_TOKENS


def input_protocol(fingerprint):
    """Bind the actual compiler, evidence and recovery semantics of each run.

    A protocol change invalidates old frozen branches and their fitted assets.
    Recollect executions under this protocol; rewriting archived metadata does
    not rerun source grounding, dependency closure or bounded recovery.
    """
    from .read_recovery import MAX_SPLIT_DEPTH, MAX_FIELD_REPAIRS, MAX_REPAIR_UNITS, REPAIR_CHAR_WINDOW

    return {
        "schema_version": 2,
        "semantics": "merged_source_windows_actual_coverage_reused_facts_compiled_choice",
        "read_output_reservation": "per_line_fact_wire_and_body_tokens",
        "answer_layout": "sources_before_history_for_source_first",
        "shared_source_protocol": SOURCE_ENCODING,
        "task_projection": "resolved_conditions_and_effective_version_scope_v1",
        "numeric_selection": "source_field_period_column_and_atomic_gap_repair_v3",
        "need_dependencies": "need_local_read_selection_and_reduce_prerequisites_v2",
        "completion_admission": "answerable_direct_or_mandatory_need_closure_v2",
        "read_recovery": {"protocol": "durable_bounded_source_split_and_field_repair_v1",
            "max_split_depth": MAX_SPLIT_DEPTH, "max_field_repairs": MAX_FIELD_REPAIRS,
            "max_repair_units": MAX_REPAIR_UNITS, "repair_char_window": REPAIR_CHAR_WINDOW,
            "failed_read_coverage": "none; paid execution remains in the ledger"},
        "terminal_gaps": "explicit_unresolved_fields_never_fabricated_support_v1",
        "source_windows": {"tokens": SOURCE_WINDOW_TOKENS, "nonempty_lines": SOURCE_WINDOW_LINES},
        "source_identity": "content_hash_and_original_coordinates",
        "reader_fingerprint": to_jsonable(fingerprint),
    }


def validate_input_protocol(value, *, reader_fingerprint=None):
    if not isinstance(value, dict) or "reader_fingerprint" not in value:
        raise ValueError("plan gain evidence requires the current actual input protocol binding")
    expected = input_protocol(value["reader_fingerprint"] if reader_fingerprint is None else reader_fingerprint)
    if value != expected:
        raise ValueError("plan gain input protocol differs from the actual source/compiler/reader scope; "
            "recollect frozen branches and refit under the current protocol, do not relabel archived inputs")
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
