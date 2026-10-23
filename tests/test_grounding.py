from copy import deepcopy
import json

from bench.grounding import claim_spans, grounding_record, main


def audit(answer="Cloudant attachments allow 10 MB [E_aaaa]."):
    blocks = [{"key": {"document_id": name}, "source_uri": name, "text": text}
              for name, text in (("wrong", "Security Center Attachment 50 per account"),
                                 ("old-visible", "Cloudant single attachment size 10 MB"))]
    return {"task_id": "t", "conversation_id": "s", "answer": answer, "query": "Limit?", "truncated": False,
        "retrieved_evidence": [blocks[0]], "audit_evidence": blocks,
        "visible_evidence_manifest": [{"key": b["key"], "reference_id": ref}
                                      for b, ref in zip(blocks, ("E_aaaa", "E_bbbb"))]}


class ContractScorer:
    """Synthetic scores test routing only, never factual model accuracy."""
    def score_pairs(self, pairs):
        return [{"probability": .9 if "Cloudant" in doc else .1, "status": "scored"} for doc, _ in pairs]


def test_full_visible_old_source_distinguishes_misbinding_without_mutating_answer():
    row = audit()
    old = deepcopy(row)
    result = grounding_record(row, ContractScorer())
    claim = result["claims"][0]
    assert claim["finding"] == "supported_elsewhere_citation_mismatch"
    assert claim["supporting_reference_ids"] == ["E_bbbb"]
    assert len(claim["pairs"]) == 2
    assert result["answer"] == row["answer"] and row == old
    assert result["semantic_quality_score"] is None


def test_offsets_code_tables_and_citation_only_lines_remain_reviewable():
    answer = 'Fact [E_aaaa]. Another fact.\n```python\nx = "Definitely correct."\n```\nA | B | C\nEvidence [E_aaaa]\n'
    units = claim_spans(answer)
    assert all(answer[u["start"]:u["end"]] == u["text"] for u in units)
    assert {u["kind"] for u in units} == {"prose", "code", "table", "ambiguous_citation_binding"}
    scored = grounding_record(audit(answer), ContractScorer())
    assert all(u["pairs"] == [] for u in scored["claims"] if u["kind"] != "prose")


def test_missing_citation_unknown_body_and_model_truncation_are_not_passes():
    uncited = grounding_record(audit("Cloudant attachments allow 10 MB."), ContractScorer())
    assert uncited["claims"][0]["finding"] == "uncited_support_signal"
    row = audit()
    row["audit_evidence"] = row["audit_evidence"][1:]
    assert grounding_record(row, ContractScorer())["claims"][0]["finding"] == "incomplete_model_coverage"
    class Truncated:
        def score_pairs(self, pairs):
            return [{"probability": .99, "status": "model_truncated"} for _ in pairs]
    result = grounding_record(audit(), Truncated())
    assert result["claims"][0]["finding"] == "incomplete_model_coverage"
    assert all(pair["support_signal"] is None for pair in result["claims"][0]["pairs"])


def test_default_cli_dry_run_does_not_construct_model(tmp_path, monkeypatch):
    import bench.grounding as module
    source, output = tmp_path/"audit.jsonl", tmp_path/"pairs.jsonl"
    source.write_text(json.dumps(audit())+"\n")
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run loaded model")
    monkeypatch.setattr(module, "MiniCheckScorer", forbidden)
    main(["--audit", str(source), "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["claims"][0]["finding"] == "dry_run"
    assert all(p["probability"] is None for p in result["claims"][0]["pairs"])
    assert json.loads(output.with_suffix(".manifest.json").read_text())["operation"] == "dry_run"


def test_reference_ids_are_bindings_not_entailment_claims():
    seen = []
    class Scorer(ContractScorer):
        def score_pairs(self, pairs):
            seen.extend(pairs)
            return super().score_pairs(pairs)
    row = audit()
    result = grounding_record(row, Scorer())
    assert all(claim == "Cloudant attachments allow 10 MB ." for _, claim in seen)
    assert result["claims"][0]["citations"] == ["E_aaaa"]
    assert result["claims"][0]["text"] == row["answer"]
