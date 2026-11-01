from collections import defaultdict
import json

import pytest

from bench.span_eval import Passage, load_passages, main, span_coverage


def test_original_span_coverage_uses_parent_offsets_and_unions_overlaps():
    passages = {
        "left": Passage("a", 0, 6, "abcdef"),
        "right": Passage("a", 4, 10, "efghij"),
        "wrong_parent": Passage("b", 0, 10, "abcdefghij"),
        "wrong_text": Passage("a", 2, 8, "XXXXXX"),
    }
    parents = defaultdict(list)
    for item in passages.values():
        parents[item.parent].append(item)
    reference = {"passage_id": "a", "start": 2, "end": 8, "quote": "cdefgh"}
    references = {"q": [reference, reference], "empty": [],
                  "bad": [{"passage_id": "a", "start": 2, "end": 8, "quote": "wrong!"}]}
    predictions = {"q": ["left", "right", "wrong_parent"], "bad": ["left"],
                   "empty": ["left"], "missing": ["left"]}
    result = span_coverage(predictions, references, passages, parents)
    assert result["tasks"]["q"]["aligned_spans"] == 1
    assert result["tasks"]["q"]["character_recall"] == 1
    assert result["tasks"]["q"]["all_aligned_spans_covered"]
    assert result["no_alignable_span_task_ids"] == ["bad"]
    assert result["no_grounding_task_ids"] == ["empty"]
    assert result["missing_reference_task_ids"] == ["missing"]
    partial = span_coverage({"q": ["left", "right"]}, references, passages, parents, k=1)
    assert partial["tasks"]["q"]["character_recall"] == pytest.approx(4 / 6)
    wrong = span_coverage({"q": ["wrong_parent", "wrong_text"]}, references, passages, parents)
    assert wrong["tasks"]["q"]["character_recall"] == 0
    with pytest.raises(ValueError, match="missing from source corpus"):
        span_coverage({"q": ["unknown"]}, references, passages, parents)


def test_literal_span_validation_rejects_bad_offsets(tmp_path):
    path = tmp_path / "passages.jsonl"
    path.write_text(json.dumps({"passage_id": "a", "document_id": "d", "text": "abc", "span": [0, 5]}) + "\n")
    with pytest.raises(ValueError, match="invalid literal passage span"):
        load_passages(path)


def test_cli_rejects_changed_visible_text_with_same_passage_id(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(json.dumps({"passage_id": "a", "document_id": "d", "text": "abc", "span": [0, 3]}) + "\n")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"task_id": "q", "contexts": [{"document_id": "a", "text": "xyz"}]}) + "\n")
    references = tmp_path / "references.jsonl"
    references.write_text(json.dumps({"task_id": "q", "grounding": []}) + "\n")
    with pytest.raises(ValueError, match="visible passage text differs"):
        main(["--input", str(predictions), "--references", str(references), "--corpus", str(corpus),
              "--output", str(tmp_path / "result.json")])
