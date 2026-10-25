from bench.judge import CITATION_PARSER_VERSION, parse_citations, rule_audit


def test_native_citation_grammar_allows_text_code_links_and_current_grouped_ids():
    import re
    from controller.templates import citation_regex
    pattern = citation_regex(("E_aa", "E_bb"))
    for value in ("", "中文 answer [E_aa].", "[E_aa, E_bb]", "[E_bb,\nE_aa]",
                  "a[0] = [1, [2, 3]]; [Example](https://example.com)",
                  "[[E_aa]]", "[", "[E", "[E[0]]", "[EFI]", "[E.example]", "line\nnext"):
        assert re.fullmatch(pattern, value), value
        assert parse_citations(value) <= {"E_aa", "E_bb"}
    for value in ("[E_old]", "[[E_old]]", "[E[E_old]]", "[E_aa, E_old]", "[E_ab]", "[E_aa_]", "[E_aa"):
        assert not re.fullmatch(pattern, value), value
    assert not re.fullmatch(citation_regex(()), "[E_aa]")
    assert re.fullmatch(citation_regex(()), "No supporting source. [Example](url)")


def test_single_and_grouped_evidence_ids_without_numeric_or_bare_references():
    assert parse_citations("One [E_ab12], group [E_ab12, E_deadbeef].") == {"E_ab12", "E_deadbeef"}
    assert parse_citations("[1] [1, 2] [Source 1] bare E_ab12 https://example.com/E_ab12") == set()


def test_unknown_group_member_is_audited_and_strict_count_is_preserved():
    key = {"document_id": "a"}
    record = {"task_id": "s:1", "answer": "[E_ab12, E_deadbeef] also [E_ab12]",
        "visible_evidence_manifest": [{"key": key, "reference_id": "E_ab12"}],
        "audit_evidence": [{"key": key}]}
    result = rule_audit(record)
    assert result["citation_count"] == 2
    assert result["strict_format_citation_count"] == 1
    assert result["unknown_citations"] == ["E_deadbeef"]
    assert result["flags"] == ["unknown_citations"]
    assert result["citation_parser_version"] == CITATION_PARSER_VERSION
    assert result["semantic_quality_score"] is None


def test_visible_historical_body_is_not_current_citation_permission():
    old,new = {"document_id":"old"},{"document_id":"current"}
    record = {"task_id":"s:2","answer":"Prior claim [E_aa], current [E_bb].",
        "visible_evidence_manifest":[{"key":old,"reference_id":"E_aa"},{"key":new,"reference_id":"E_bb"}],
        "audit_evidence":[{"key":old},{"key":new}],"citation_scope":["E_bb"]}
    result = rule_audit(record)
    assert result["unknown_citations"] == []
    assert result["out_of_scope_citations"] == ["E_aa"]
    assert result["flags"] == ["out_of_scope_citations"]
    # Explicitly restored/revisited evidence is permitted without relabelling it.
    assert rule_audit(record | {"citation_scope":["E_aa","E_bb"]})["flags"] == []
    assert rule_audit(record | {"citation_scope":[]})["out_of_scope_citations"] == ["E_aa","E_bb"]
    assert rule_audit({k:v for k,v in record.items() if k != "citation_scope"})["flags"] == []
