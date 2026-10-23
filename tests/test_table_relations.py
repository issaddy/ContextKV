from dataclasses import fields, is_dataclass

import pytest

from adapters.table_relations import SourceSpan, extract_table_relations, render_table_relations


# Original table excerpt from the saved MTRAG Cloud snapshot, not an expected answer.
# ibmcld_00558-23465-25360 / ibmcld_12904-23555-25450.
LIMIT_EXCERPT = """
IBM Cloudant JSON documents and requests have the following maximum size limits:



Table 4. Maximum size limits for JSON documents and requests

 Limit Maximum Size 

 Individual Document Size 1 MB 
 Single Attachment Size 10 MB 
 Request Body Size 11 MB 



If you exceed these limits, a 413 response alerts you.

 Locations and tenancy

* Frankfurt‡

‡All IBM Cloudant instances that are deployed from the IBM Cloud Frankfurt region are deployed into EU-managed environments.
"""


def assert_source_spans(text, value):
    if isinstance(value, SourceSpan):
        assert text[value.start:value.end] == value.text
    elif is_dataclass(value):
        for field in fields(value):
            assert_source_spans(text, getattr(value, field.name))
    elif isinstance(value, tuple):
        for item in value:
            assert_source_spans(text, item)


def test_actual_flattened_table_keeps_three_different_objects_and_no_regional_footnote():
    before = LIMIT_EXCERPT
    tables = extract_table_relations(LIMIT_EXCERPT)
    assert len(tables) == 1
    table = tables[0]
    assert table.syntax == "captioned_numeric_rows"
    assert table.headers[0].text == "Limit Maximum Size"  # Lost column boundary stays lost.
    assert [(r.object.text, r.value.text, r.unit.text) for r in table.relations] == [
        ("Individual Document Size", "1 MB", "MB"),
        ("Single Attachment Size", "10 MB", "MB"),
        ("Request Body Size", "11 MB", "MB"),
    ]
    assert table.source_notes == () and all(not r.notes for r in table.relations)
    assert LIMIT_EXCERPT == before
    assert_source_spans(LIMIT_EXCERPT, tables)
    rendered = render_table_relations("E_123abc", tables)
    assert "[E_123abc]" in rendered and "flattened_header=" in rendered
    assert "Frankfurt" not in rendered and "including attachments" not in rendered


def test_different_objects_with_identical_values_and_explicit_total_are_not_merged():
    text = "Table 2. Transfer limits\n\nObject Maximum\n\n Alpha 7 MB\n Beta 7 MB\n Combined Alpha and Beta 12 MB\n"
    table, = extract_table_relations(text)
    assert [r.object.text for r in table.relations] == ["Alpha", "Beta", "Combined Alpha and Beta"]
    assert [r.value.text for r in table.relations] == ["7 MB", "7 MB", "12 MB"]
    assert_source_spans(text, table)


def test_markdown_multiple_columns_and_non_size_units():
    text = "| Operation | Latency (ms) | Rate |\n| :--- | ---: | --- |\n| Read | 2.5 | 80 requests/s |\n| Write | 9 | 25 requests/s |\n"
    table, = extract_table_relations(text)
    assert table.syntax == "markdown"
    assert [(r.object.text, r.column.text, r.value.text, r.unit.text) for r in table.relations] == [
        ("Read", "Latency (ms)", "2.5", "ms"), ("Read", "Rate", "80 requests/s", "requests/s"),
        ("Write", "Latency (ms)", "9", "ms"), ("Write", "Rate", "25 requests/s", "requests/s"),
    ]
    assert_source_spans(text, table)


def test_flattened_non_size_units_and_values_are_generic():
    text = "Table 19: Network latency\n\nOperation Time\n\n Read 0.7 ms\n Write 3.2 ms\n"
    table, = extract_table_relations(text)
    assert [r.value.text for r in table.relations] == ["0.7 ms", "3.2 ms"]
    assert [r.unit.text for r in table.relations] == ["ms", "ms"]
    assert_source_spans(text, table)


def test_adjacent_notes_are_literal_and_only_matching_markers_link_to_rows():
    text = ("| Object | Value |\n| --- | --- |\n| North† | 7 MB |\n| South | 7 MB |\n\n"
            "†For this region only.\n\n‡Deployment location, not a size condition.\n\n"
            "Note: The original source does not state a shared total.\n")
    table, = extract_table_relations(text)
    assert len(table.source_notes) == 3
    assert [n.text for n in table.relations[0].notes] == ["†For this region only."]
    assert not table.relations[1].notes
    assert_source_spans(text, table)
    rendered = render_table_relations("E_ab12", (table,))
    assert rendered.count("marked_source_notes=") == 1
    assert rendered.count("source note (no inferred row scope)=") == 3


def test_markdown_footnote_marker_and_preceding_note_offsets():
    text = "Note: Version A uses a different scale.\n\n| Item | Value |\n| --- | --- |\n| One[^x] | 9 kg |\n\n[^x]: Includes packaging.\n"
    table, = extract_table_relations(text)
    assert len(table.source_notes) == 2
    assert table.relations[0].notes[0].text == "[^x]: Includes packaging."
    assert_source_spans(text, table)


def test_unicode_crlf_escaped_pipe_and_empty_cell_keep_exact_offsets():
    text = "说明\r\n\r\n名称 | 时间 (s)\r\n--- | ---:\r\n甲\\|乙 | 4\r\n| 丙 | |\r\n"
    table, = extract_table_relations(text)
    assert table.relations[0].object.text == "甲\\|乙"
    assert table.relations[1].value.text == ""
    assert_source_spans(text, table)


@pytest.mark.parametrize("text", [
    "Object Maximum\nAlpha 7 MB\nBeta 8 MB\n",  # No table marker in flattened prose.
    "Table 1. Unknown\n\nThing Quantity\n\nAlpha 7 widgets\nBeta 8 widgets\n",  # Unknown units.
    "Table 1. Incomplete\n\nThing Maximum\n\nAlpha 7 MB\n",  # One ambiguous text row.
    "Table 1. Ragged\n\nThing Maximum\n\nAlpha 7 MB\nBeta unlimited\n",
    "Table 1. Multiple values\n\nThing Maximum\n\nAlpha 7 MB 8 MB\nBeta 9 MB 10 MB\n",
    "| A | B |\n| --- | --- |\n| one | 7 MB |\n| two | 8 MB | extra |\n",
    "| A | B |\n| --- | --- |\n| one | 7 MB |\n| | 8 MB |\n",  # No merged-row guessing.
    "| A | |\n| --- | --- |\n| one | 7 MB |\n",
    "```text\n| A | B |\n| --- | --- |\n| one | 7 MB |\n```\n",
    "    | A | B |\n    | --- | --- |\n    | one | 7 MB |\n",
])
def test_unknown_or_ambiguous_tables_are_not_given_invented_relations(text):
    assert extract_table_relations(text) == ()
    assert render_table_relations("E_abc", ()) == ""


def test_multiple_tables_keep_independent_headers_and_no_cross_table_notes():
    text = ("Table 1. First\n\nItem Size\n\n A 7 MB\n B 8 MB\n\n"
            "Table 2. Second\n\nItem Duration\n\n A 4 s\n B 5 s\n")
    tables = extract_table_relations(text)
    assert len(tables) == 2
    assert [t.headers[0].text for t in tables] == ["Item Size", "Item Duration"]
    assert all(not t.source_notes for t in tables)
    assert_source_spans(text, tables)
