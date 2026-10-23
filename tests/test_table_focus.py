import json

import pytest

from adapters._craft_row_text import build_row_texts
from adapters.table_focus import focused_annotation, select_rows
from adapters.table_relations import extract_table_relations
from controller.config import load_config
from controller.evidence import EvidencePolicy


TEXT = "Table 2. Operational ranges\n\nItem Maximum\n\n Width 4 mm\n Duration† 7 ms\n\n†For the narrow lane.\n"


def test_craft_pure_row_builder_retains_table_and_row_mapping():
    rows, meta = build_row_texts([{"table_id": "arbitrary-id", "title": "  Original  title ",
                                  "headers": ["Object", "Maximum"], "rows": [["Width", " 4 mm "]]}])
    assert rows == ["Original title Object: Width Maximum: 4 mm"]
    assert meta == [{"table_id": "arbitrary-id", "row_idx": 0, "text": rows[0]}]


def test_corrective_query_uses_generic_source_fields_and_keeps_original_spans():
    table, = extract_table_relations(TEXT)
    selected, trace = select_rows("Not widths, I meant durations?", table)
    assert [row.object.text for row in selected] == ["Duration†"]
    assert trace["correction_cue"].lower() == "not"
    assert trace["query_used_for_focus"] == "I meant durations?"
    swapped = TEXT.replace(" Width 4 mm\n Duration† 7 ms", " Duration† 7 ms\n Width 4 mm")
    swapped_rows, _ = select_rows("Not widths, I meant durations?", extract_table_relations(swapped)[0])
    assert [row.object.text for row in swapped_rows] == ["Duration†"]
    for row in selected:
        for span in (row.object, row.column, row.value, row.row_span):
            assert TEXT[span.start:span.end] == span.text
    annotation, _ = focused_annotation("Not widths, I meant durations?", (("E_123abc", (table,)),))
    assert '[E_123abc] object="Duration†"' in annotation and 'object="Width"' not in annotation
    assert 'source note with matching row marker: "†For the narrow lane."' in annotation


@pytest.mark.parametrize("query", ["What is the maximum?", "How does it work?", "Not only widths, but also durations?",
                                  "Which are not widths?", "Compare widths and durations."])
def test_unknown_tied_or_multirow_query_preserves_all_source_rows(query):
    table, = extract_table_relations(TEXT)
    rows, _ = select_rows(query, table)
    assert rows == table.relations


def test_matching_ties_and_unscoped_notes_are_not_lost():
    text = "| Object | Rate |\n| --- | --- |\n| North transfer | 3 MB/s |\n| South transfer | 9 MB/s |\n\nNote: No stated regional scope.\n"
    table, = extract_table_relations(text)
    rows, trace = select_rows("What about transfers?", table)
    assert rows == table.relations and trace["selected_indices"] == [0, 1]
    annotation, _ = focused_annotation("What about transfers?", (("E_ffff", (table,)),))
    assert annotation.count('[E_ffff] object=') == 2
    assert "row scope unspecified" in annotation


def test_no_tables_adds_no_prompt_text_or_diagnostics():
    assert focused_annotation("Current question?", (("E_a", ()),)) == ("", [])


def test_style_is_separate_from_boolean_enablement_and_validates_config(tmp_path):
    assert EvidencePolicy().table_relations is False
    assert EvidencePolicy().table_annotation_style == "all_rows"
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"context": {"table_relations": True, "table_annotation_style": "focus_refs"}}))
    assert load_config(path).context == EvidencePolicy(table_relations=True, table_annotation_style="focus_refs")
    with pytest.raises(ValueError, match="table_annotation_style"):
        EvidencePolicy(table_annotation_style="unrecognized")
