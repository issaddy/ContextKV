from dataclasses import replace
import json

from bench.audit_reading_mechanisms import (answer_literals, audit_case, audit_directory,
                                           recorded_computed_lines, source_ledger)
from controller.context_schema import (ContextPlan, EvidenceClaim, EvidenceSet, EvidenceUnit,
    ReadingOutcome, SourceSpan, TaskState)
from controller.evidence import make_evidence
from controller.schema import Message
from controller.serialization import to_jsonable


def material(name, text):
    block = make_evidence(text, name, corpus_id="audit-fixture", version="v1")
    span = SourceSpan("span:"+name, block.key, 0, len(text))
    unit = EvidenceUnit("unit:"+name, (span,), text)
    return block, unit, span


def artifact(case, sources, claims, answer, expected):
    blocks, units, _ = zip(*sources)
    plan = ContextPlan("plan", TaskState("task", "controlled question"),
        EvidenceSet("set", units, source_blocks=blocks), coverage_units=tuple(unit.unit_id for unit in units))
    outcome = ReadingOutcome(plan, claims=tuple(claims), coverage=tuple(
        ("need", block.key.document_id, unit.unit_id, "epoch") for block, unit in zip(blocks, units, strict=True)))
    return {"case": case, "status": "completed", "qa_increment": 1, "version_increment": 1,
        "sources": to_jsonable(blocks), "outcome": to_jsonable(outcome), "expected": expected,
        "answer": answer, "response": {"read_count": len(sources), "reduce_count": 2, "citation_scope": []}}


def test_long_keeps_computed_wrong_mass_and_final_wrong_excerpt_separate():
    source = material("register", "Approved shipment S00: mass 1 kg. Approved shipment S01: mass 2000 g.")
    claims = (EvidenceClaim("count-root", "count", "derived", value=36, operation="count", source_spans=(source[2],)),
        EvidenceClaim("mass-root", "mass", "derived", value="655000", unit="g", operation="sum", source_spans=(source[2],)),
        EvidenceClaim("wrong-read", "mass", "direct", entity="S01", value="1000", unit="g", source_spans=(source[2],)))
    row = artifact("long", [source], claims, "Distinct approved shipments: **3**. Total mass: **6 kg**.",
        {"distinct_shipments": 36, "mass_kg": 666})
    report = audit_case(row)
    assert report["execution"]["single_qa_commit"]
    assert report["coverage"]["required_units_complete"]
    quality = report["quality"]
    assert quality["computed_count_matches_expected"] and not quality["computed_mass_matches_expected"]
    assert quality["computed_root_claims"][1]["mass_g"] == "655000"
    assert quality["final_answer_literals"]["shipment_counts"] == ["3"]
    assert quality["final_answer_literals"]["mass_values_g"] == ["6000"]
    assert not quality["final_unique_count_matches_expected"] and not quality["final_unique_mass_matches_expected"]
    assert quality["extracted_mass_claims"][0]["literal_matches_source"] is False
    assert row["answer"].startswith("Distinct approved shipments: **3**")


def test_source_dedup_and_final_unit_normalization_are_literal_only():
    first = material("register-a", "Approved shipment S00: mass 1 kg. Approved shipment S01: mass 2000 g.")
    repeat = material("register-b", "Repeated registration of approved shipment S00: mass 1000 g; same shipment.")
    ledger = source_ledger({str(index): to_jsonable(source[0]) for index, source in enumerate((first, repeat))})
    assert ledger["unique_shipments"] == 2 and ledger["repeated_registrations"] == 1
    assert ledger["source_mass_g"] == "3000"
    literal = answer_literals("There are **36 distinct approved shipments**, totaling **666,000 grams**.")
    assert literal["shipment_counts"] == ["36"] and literal["mass_values_g"] == ["666000"]
    messages = (Message("user", "Originals\nComputed results for the complete reading scope:\n"
        "count: count = 36; original sources = 12.\nmass: sum = 655000 g; original sources = 12.\n"
        "Citations allowed for this answer: [E_ab].\nQuestion:\nWhat is the total?"),)
    assert recorded_computed_lines(messages) == ["Computed results for the complete reading scope:",
        "count: count = 36; original sources = 12.", "mass: sum = 655000 g; original sources = 12."]


def test_multihop_year_and_grounded_three_source_lineage_are_independent():
    material_ = (material("institute-parent", "Meridian Institute is governed by the Arbor Council."),
        material("parent-fund", "Arbor Council is supported by the Silver Fund."),
        material("fund-year", "Silver Fund was established in 1997."))
    claims = [EvidenceClaim("hop1", "foundation", "direct", material_[0][0].text,
            entity="Meridian Institute", value="Arbor Council", source_spans=(material_[0][2],)),
        EvidenceClaim("hop2", "foundation", "direct", material_[1][0].text,
            entity="Arbor Council", value="Silver Fund", source_spans=(material_[1][2],), parent_claim_ids=("hop1",)),
        EvidenceClaim("hop3", "foundation", "direct", material_[2][0].text,
            entity="Silver Fund", value=1997, source_spans=(material_[2][2],), parent_claim_ids=("hop2",))]
    row = artifact("multihop", material_, claims, "The fund was established in 1997.",
        {"year": 1997, "source_chain": ["institute-parent", "parent-fund", "fund-year"]})
    assert audit_case(row)["quality"]["literal_quality"] == "passed"
    claims[2] = replace(claims[2], parent_claim_ids=())
    row["outcome"]["claims"] = to_jsonable(claims)
    quality = audit_case(row)["quality"]
    assert quality["final_expected_year_present"] and not quality["three_source_claim_chain_observed"]
    claims[2] = replace(claims[2], parent_claim_ids=("hop2",), source_spans=(replace(material_[2][2], end=5),))
    row["outcome"]["claims"] = to_jsonable(claims)
    assert not audit_case(row)["quality"]["three_source_claim_chain_observed"]


def test_offline_directory_preserves_original_and_labels_execution_separately(tmp_path):
    root = tmp_path/"original"
    root.mkdir()
    row = artifact("pair", [material("eligibility", "Residents aged 18 or older; Form W-7.")], [],
        "Residents aged 18 or older use Form W-7.", {"eligibility": "residents aged 18 or older", "form": "W-7"})
    row.update(status="failed", qa_increment=0, version_increment=0, error="recorded failure")
    path = root/"pair.json"
    path.write_text(json.dumps(row))
    before = path.read_bytes()
    summary = audit_directory(root, tmp_path/"audit")
    assert path.read_bytes() == before
    assert summary["benchmark_metric"] is False
    report = json.loads((tmp_path/"audit"/"pair.audit.json").read_text())
    assert report["execution"]["status"] == "failed" and report["quality"]["literal_quality"] == "passed"
    assert report["recorded_final_prompt"]["status"] == "state_db_not_available"
