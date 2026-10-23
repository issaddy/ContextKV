from copy import deepcopy

from scripts.review_saved_answers import compare_packets, freeze_cohort, make_packet, write_review_packet


def saved_run():
    key = {"document_id": "registry", "version": "v1"}
    answer = "Attachment limit is 10 MB [E_ab12]. Run `ibmcloud cr va scan image`."
    turn = {"task_id": "s:1", "conversation_id": "s", "turn": 1, "protocol": "P2", "status": "success",
        "assistant_text": answer, "online_turn": {"query": "Limit?", "contexts": [], "history": []},
        "result": {"output_ids": [1, 2], "truncated": False,
            "visible_evidence_manifest": [{"key": key, "reference_id": "E_ab12"}],
            "audit_evidence": [{"key": key, "text": "Attachment 50 per account", "source_uri": "security-compliance"}]}}
    return {"workload": {"split": "dev"}, "report": {"turns": [turn]}, "run_id": "q0", "baseline": "B1"}


def test_visible_id_does_not_become_automatic_semantic_support(tmp_path):
    data = saved_run()
    cohort = freeze_cohort(data)
    rows, summary = make_packet(data, cohort)
    row = rows[0]
    assert row["rule_audit"]["unknown_citations"] == []
    assert row["citation_contexts"][0]["support_judgment"] == "pending"
    assert row["manual_review"]["citation_support"] is None
    assert row["literal_hints"][0]["literal_source_hits"] == []
    assert not summary["semantic_quality_evaluated"]
    write_review_packet(tmp_path/"review", cohort, rows, summary)
    assert "Attachment 50 per account" in (tmp_path/"review/review.md").read_text()


def test_changed_missing_and_failed_tasks_are_retained_with_audit_gaps():
    data = saved_run()
    second = deepcopy(data["report"]["turns"][0])
    second["task_id"], second["turn"] = "s:2", 2
    data["report"]["turns"].append(second)
    cohort = freeze_cohort(data)
    candidate = deepcopy(data)
    candidate["report"]["turns"].pop()
    first = candidate["report"]["turns"][0]
    first["status"] = "failed"
    first["result"]["audit_evidence"] = []
    first["online_turn"]["query"] = "different"
    rows, summary = make_packet(candidate, cohort)
    assert [r["status"] for r in rows] == ["failed", "missing"]
    assert summary["status_counts"] == {"failed": 1, "missing": 1}
    assert summary["changed_task_inputs"] == ["s:1"]
    assert summary["full_audit_without_missing_bodies"] == 0
    assert "missing_visible_evidence_body" in rows[0]["rule_audit"]["flags"]


def test_external_controller_audit_is_used_without_reducing_to_current_topk():
    data = saved_run()
    turn = data["report"]["turns"][0]
    full = {"task_id": turn["task_id"], **turn["result"]}
    del turn["result"]["visible_evidence_manifest"]
    del turn["result"]["audit_evidence"]
    rows, summary = make_packet(data, freeze_cohort(data), [full])
    assert rows[0]["visible_blocks"][0]["text"] == "Attachment 50 per account"
    assert summary["full_audit_without_missing_bodies"] == 1


def test_grouped_ids_remain_separate_from_original_strict_rule_and_semantic_support(tmp_path):
    data = saved_run()
    data["report"]["turns"][0]["assistant_text"] = "Limit [E_ab12, E_deadbeef]."
    cohort = freeze_cohort(data)
    rows, summary = make_packet(data, cohort)
    assert summary["turns_with_stable_ids"] == summary["grouped_only_id_turns"] == 1
    assert summary["strict_single_id_turns"] == 0
    assert rows[0]["rule_audit"]["citation_count"] == 2
    assert rows[0]["rule_audit"]["strict_format_citation_count"] == 0
    assert rows[0]["diagnostic_unknown_ids"] == ["E_deadbeef"]
    write_review_packet(tmp_path/"q0", cohort, rows, summary)
    write_review_packet(tmp_path/"q1", cohort, rows, summary)
    comparison = compare_packets([tmp_path/"q0", tmp_path/"q1"])
    assert not comparison["semantic_noninferiority_established"]
    assert comparison["runs"][1]["visible_evidence_changed_vs_first"] == []
