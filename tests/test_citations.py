from bench.judge import CITATION_PARSER_VERSION, parse_citations, rule_audit


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
