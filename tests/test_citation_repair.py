from copy import deepcopy

import pytest

from bench.citation_repair import repair_record


def example():
    text = "Use the Cloudant SDK [E_aa]."
    return {"task_id": "one", "answer": "# Answer\n" + text + "\nUnchanged.", "claims": [{
        "kind": "prose", "finding": "supported_elsewhere_citation_mismatch",
        "start": 9, "end": 9 + len(text), "text": text, "citations": ["E_aa"],
        "unknown_citations": [], "pairs": [
            {"status": "scored", "cited": True, "reference_id": "E_aa", "probability": .08},
            {"status": "scored", "cited": False, "reference_id": "E_bb", "probability": .96,
             "source_uri": "cloudant"}]}]}


def test_augments_citation_and_retains_original_bindings():
    record = example()
    saved = deepcopy(record)
    result = repair_record(record)
    assert result["repaired_answer"] == record["answer"].replace("[E_aa]", "[E_aa, E_bb]")
    assert result["original_answer"] == record["answer"]
    assert record == saved
    assert not result["quality_validated"] and not result["wording_changed"]
    assert result["changes"][0]["new_source_uri"] == "cloudant"
    assert result["original_citations_preserved"]


def test_compound_claim_preserves_distinct_sources_and_adds_once():
    record = example()
    record["answer"] = "First clause [E_aa], and second clause [E_cc]."
    claim = record["claims"][0]
    claim.update(start=0, end=len(record["answer"]), text=record["answer"], citations=["E_aa", "E_cc"])
    claim["pairs"].append({"status": "scored", "cited": True, "reference_id": "E_cc", "probability": .1})
    result = repair_record(record)
    assert result["repaired_answer"] == "First clause [E_aa], and second clause [E_cc, E_bb]."


@pytest.mark.parametrize("mutation", ["truncated", "inline_code", "uncertain", "unknown", "supported"])
def test_leaves_uncovered_or_uncertain_claims_unchanged(mutation):
    record = example()
    claim = record["claims"][0]
    if mutation == "truncated":
        claim["pairs"][0]["status"] = "model_truncated"
    elif mutation == "inline_code":
        claim["contains_inline_code"] = True
    elif mutation == "uncertain":
        claim["pairs"][1]["probability"] = .75
    elif mutation == "unknown":
        claim["unknown_citations"] = ["E_cc"]
    else:
        claim["pairs"][0]["probability"] = .6
    result = repair_record(record)
    assert result["repaired_answer"] == record["answer"]
    assert not result["changes"]


def test_rejects_stale_offsets_before_applying_changes():
    record = example()
    record["answer"] = "Changed " + record["answer"]
    with pytest.raises(ValueError, match="offsets"):
        repair_record(record)
