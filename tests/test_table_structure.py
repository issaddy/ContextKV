from dataclasses import replace
from types import SimpleNamespace

import pytest

from adapters.table_focus import focused_annotation
from adapters.table_relations import extract_table_relations, render_table_relations
from adapters.table_structure import inspect_explicit_structure
from bench.native_baseline import resolve_context_policy
from controller.candidates import build_candidates
from controller.evidence import EvidencePolicy, make_evidence
from controller.schema import SessionSnapshot, TurnRecord
from controller.templates import verify_evidence_spans
from tests.test_table_relations import assert_source_spans


UNNUMBERED = """Maximum document size

 Plan File size per document

 Cloud Pak for Data 50 MB
 Premium 50 MB
 Enterprise 10 MB
 Plus (includes Trial) 10 MB
"""


def explicit(text):
    return extract_table_relations(text, mode="explicit")


def test_real_unnumbered_header_is_not_split_or_given_invented_conditions():
    assert extract_table_relations(UNNUMBERED) == ()
    table, = explicit(UNNUMBERED)
    assert table.syntax == "unnumbered_numeric_rows"
    assert table.headers[0].text == "Plan File size per document"
    assert [r.object.text for r in table.relations] == ["Cloud Pak for Data", "Premium", "Enterprise", "Plus (includes Trial)"]
    assert [r.value.text for r in table.relations] == ["50 MB", "50 MB", "10 MB", "10 MB"]
    assert not any(r.parent_path for r in table.relations)
    assert_source_spans(UNNUMBERED, table)


def test_generic_unnumbered_duration_unicode_and_crlf():
    text = "処理時間\r\n\r\nOperation Maximum duration\r\n\r\nRead α 0.7 ms\r\nWrite β 3.2 ms\r\n"
    table, = explicit(text)
    assert table.caption.text == "処理時間"
    assert [r.object.text for r in table.relations] == ["Read α", "Write β"]
    assert_source_spans(text, table)


def test_heading_paths_close_at_same_or_higher_level_and_do_not_inherit_prose():
    text = ("# Region A\n## Version 2 only\n\n" + UNNUMBERED
            + "\n# Region B\n\n" + UNNUMBERED + "\nFor version 4, use a different setting.\n")
    first, second = explicit(text)
    assert [s.text for s in first.relations[0].parent_path] == ["# Region A", "## Version 2 only"]
    assert [s.text for s in second.relations[0].parent_path] == ["# Region B"]
    assert not second.source_notes
    assert_source_spans(text, (first, second))


def test_existing_markdown_table_gets_only_explicit_heading_ancestors():
    text = "# Premium only\n\nNearby prose is not a scope.\n\n| Object | Limit |\n| --- | --- |\n| X | 5 MB |\n"
    legacy, = extract_table_relations(text)
    table, = explicit(text)
    assert not legacy.relations[0].parent_path
    assert [s.text for s in table.relations[0].parent_path] == ["# Premium only"]
    assert table.relations[0].value == legacy.relations[0].value
    assert_source_spans(text, table)


def test_nested_parent_conditions_remain_literal_and_siblings_do_not_leak():
    text = ("# Hosted deployments\n\n- Premium plans only:\n"
            "  - Region A:\n    - Provider Alpha: 48 MB\n    - Other providers: 6 MB\n"
            "- Standard plans:\n  - All providers: 3 MB\n")
    tables = explicit(text)
    assert len(tables) == 3
    assert [p.text for p in tables[0].relations[0].parent_path] == [
        "# Hosted deployments", "Premium plans only:", "Region A:"]
    assert [p.text for p in tables[2].relations[0].parent_path] == ["# Hosted deployments", "Standard plans:"]
    assert tables[0].relations[0].object.text == "Provider Alpha"
    assert tables[0].relations[0].value.text == "48 MB"
    assert_source_spans(text, tables)
    annotation, traces = focused_annotation("Provider Alpha size?", (("E_abc", tables),))
    assert '[E_abc] explicit source parent path: "# Hosted deployments" > "Premium plans only:" > "Region A:"' in annotation
    assert traces[0]["rows"][0]["source"]["parent_path"]
    assert "explicit_source_parent_path=" in render_table_relations("E_abc", tables)


def test_lost_indentation_in_actual_premium_shape_is_not_reconstructed():
    text = ("Managed deployments on IBM Cloud\n\n* Premium plans only:\n\n"
            "* Box: 50 MB\n* All other data sources: 10 MB\n\n* All other plans: 10 MB\n")
    tables, rejected = inspect_explicit_structure(text)
    assert tables == ()
    assert len(rejected) == 3
    assert {r["reason"] for r in rejected} == {"numeric_list_without_explicit_indented_parent"}
    assert_source_spans(text, tuple(r["span"] for r in rejected))


@pytest.mark.parametrize("text", [
    "Maximum size\nPlan File size\nAlpha 4 MB\n",  # Single ambiguous row.
    "Maximum size\nPlan File size\nAlpha 4 MB\nBeta unlimited\n",  # Ragged block.
    "Maximum size\nPlan File size\nAlpha 4 MB 5 MB\nBeta 6 MB 8 MB\n",  # Lost multiple columns.
    "Maximum count\nPlan Count\nAlpha 4 widgets\nBeta 5 widgets\n",  # Unknown units.
    "Here is some ordinary prose.\nPlan File size\nAlpha 4 MB\nBeta 5 MB\n",  # Sentence is not caption.
    "Maximum size\nEach plan is limited to the file size\nAlpha 4 MB\nBeta 5 MB\n",  # Prose is not header.
    "Maximum size\nPlan File size\n* Alpha 4 MB\n* Beta 5 MB\n",  # Lists are not table rows.
    "Maximum size\nPlan File size\nAlpha 4 MB\n* Beta 5 MB\n",  # Mixed list/table is not salvaged.
    "````text\n" + UNNUMBERED + "````\n",  # Fenced code.
    "\n".join("    " + line for line in UNNUMBERED.splitlines()),  # Indented code.
    "- Premium only:\n Alpha: 4 MB\n Beta: 5 MB\n",  # No child bullet.
    "- Premium only:\n - Alpha: 4 MB\n - Beta: 5 MB\n",  # Insufficient child indentation.
    "- Premium only:\n\t- Alpha: 4 MB\n\t- Beta: 5 MB\n",  # Tab nesting left ambiguous.
    "- Example:\n\n      - Cache: 2 MiB\n",  # Four spaces beyond parent content: code, not child list.
])
def test_rejects_structures_whose_relationships_would_need_inference(text):
    assert explicit(text) == ()


@pytest.mark.parametrize("style", ["all_rows", "focus_refs"])
def test_optional_parser_preserves_all_bodies_history_and_legacy_bytes(style):
    block = make_evidence(UNNUMBERED, "document")
    snapshot = SessionSnapshot("s", raw_history=(TurnRecord(1, "Old user", "Old answer"),))
    policy = EvidencePolicy(table_relations=True, table_annotation_style=style)
    old = build_candidates(snapshot, "Premium file size?", (block,), policy)
    explicit_legacy = build_candidates(snapshot, "Premium file size?", (block,), replace(policy, table_structure_mode="legacy"))
    new = build_candidates(snapshot, "Premium file size?", (block,), replace(policy, table_structure_mode="explicit"))
    for before, unchanged, after in zip(old, explicit_legacy, new):
        assert before.messages == unchanged.messages
        assert before.input_ids == unchanged.input_ids and before.input_hash == unchanged.input_hash
        assert before.messages[:3] == after.messages[:3]
        assert before.required_blocks == after.required_blocks
        assert verify_evidence_spans(after.messages, after.evidence_spans, (block,), after.reference_map) == after.required_blocks
        assert UNNUMBERED in after.messages[-1].content
        assert 'object="Premium"' in after.messages[-1].content
        assert after.input_ids != before.input_ids


def test_policy_validation_and_native_config_override():
    with pytest.raises(ValueError, match="table_structure_mode"):
        EvidencePolicy(table_structure_mode="guess")
    with pytest.raises(ValueError, match="structure mode"):
        extract_table_relations("", mode="guess")
    args = SimpleNamespace(context_policy=EvidencePolicy(table_relations=True, table_structure_mode="explicit"))
    assert resolve_context_policy(args).table_structure_mode == args.table_structure_mode == "explicit"
    args.table_structure_mode = "legacy"
    assert resolve_context_policy(args).table_structure_mode == "legacy"
