import json

from training.support_data import answer_state_supervision, build_preference_pairs, build_support_examples


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows))


def test_author_grounding_seeds_remain_unlabeled_and_offsets_are_original(tmp_path):
    sources = tmp_path/"sources.jsonl"
    turns = tmp_path/"turns.jsonl"
    text = "资格：居民。步骤：提交表格。"
    write_rows(sources, [{"source_id": "a", "document_family": "policy", "version": "v1", "text": text},
        {"source_id": "b", "document_family": "other", "version": "v1", "text": "其他材料。"}])
    write_rows(turns, [{"task_id": "t", "conversation_id": "c", "split": "train",
        "question": "申请资格和步骤？", "reference_history": [], "snapshot_id": "s",
        "grounding": [{"passage_id": "a", "start": 0, "end": 6, "quote": text[:6]},
                      {"passage_id": "a", "start": 6, "end": len(text), "quote": text[6:]}]}])
    manifest = build_support_examples(turns, sources, tmp_path/"out")
    assert manifest["labeled_examples"] == 0
    assert manifest["candidate_examples"] == 6
    assert manifest["history_kinds"]["actual_history"] == 0
    examples = [json.loads(line) for line in (tmp_path/"out/support_candidates.jsonl").read_text().splitlines()]
    assert all(row["label_origin"] == "pending_teacher" for row in examples)
    joint = next(row for row in examples if row["arity"] == 2 and row["group_kind"] == "member_deletion")
    assert "".join(unit["text"] for unit in joint["units"]) == text
    assert joint["units"][1]["source_spans"][0]["start"] == 6


def test_preferences_compare_actual_answers_with_matching_history_and_budgets(tmp_path):
    rewards = tmp_path/"rewards.jsonl"
    base = {"task_id": "t", "history_id": "actual-h1", "split": "train",
        "reader_model": "qwen3_32b_nonthinking", "body_budget": 2048, "output_budget": 512,
        "task_state": {"task_id": "t"}, "answer": "Fixture answer", "disposition": "answer"}
    write_rows(rewards, [{**base, "quality_components": {"answer": .9, "condition": 1}},
        {**base, "quality_components": {"answer": .5, "condition": .5}},
        {**base, "history_id": "actual-h2", "quality_components": {"answer": .1, "condition": 0}},
        {**base, "body_budget": 4096, "quality_components": {"answer": .2, "condition": 0}}])
    result = build_preference_pairs(rewards, tmp_path/"preferences.jsonl",
                                    weights={"answer": .8, "condition": .2})
    assert result["pairs"] == 1
    pair = json.loads((tmp_path/"preferences.jsonl").read_text())
    assert pair["label_origin"] == "actual_reader_answers"
    assert pair["preferred"]["quality_components"]["answer"] == .9
    assert pair["reward_difference"] > 0


def test_group_absence_cannot_supervise_corpus_unanswerability_without_scope_oracle():
    raw = {"answer_disposition": "unanswerable", "disposition": "background", "quality": .1}
    unknown, origin = answer_state_supervision(raw)
    assert unknown["answer_disposition"] is None
    assert unknown["quality"] is None
    assert origin == "unknown_scope_masked"
    positive, origin = answer_state_supervision(raw, corpus_has_answer=True)
    assert positive["answer_disposition"] == "retrieval_gap"
    negative, origin = answer_state_supervision(raw, corpus_has_answer=False, scope_complete=True)
    assert negative["answer_disposition"] == "unanswerable"
    incomplete, origin = answer_state_supervision(raw, corpus_has_answer=False, scope_complete=False)
    assert incomplete["answer_disposition"] is None
    assert raw["answer_disposition"] == "unanswerable"
