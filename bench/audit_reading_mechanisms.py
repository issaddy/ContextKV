"""Offline literal/provenance audit of saved controlled mechanism executions.

This reports fixture checks, not a benchmark metric. It never generates an
answer, changes a prompt, repairs a claim, or overwrites the source records.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata

from controller.reduction import UNIT_FACTORS


NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
MASS = re.compile(rf"(?<![\w.])(?P<value>{NUMBER})\s*(?P<unit>kilograms?|milligrams?|grams?|kg|mg|g)\b", re.I)
SHIPMENT = re.compile(rf"\bapproved shipment (?P<entity>S\d+):\s*mass\s*(?P<value>{NUMBER})\s*(?P<unit>kg|mg|g)\b", re.I)
COUNTS = (re.compile(rf"(?<![\w.])(?P<value>{NUMBER})\s+(?:(?:distinct|unique|approved)\s+)*(?:shipments?|shipment records?)\b", re.I),
    re.compile(rf"\b(?:(?:count|number)\s+(?:of\s+)?)?(?:(?:distinct|unique|approved)\s+)*shipments?\s*(?::|=|is|are)\s*(?P<value>{NUMBER})\b", re.I))


def identity(key):
    return json.dumps(key, sort_keys=True, separators=(",", ":"))


def decimal(value):
    return Decimal(str(value).replace(",", ""))


def mass_g(value, unit):
    label = {"kilogram": "kg", "kilograms": "kg", "gram": "g", "grams": "g",
             "milligram": "mg", "milligrams": "mg"}.get(unit.casefold(), unit.casefold())
    if label not in UNIT_FACTORS or UNIT_FACTORS[label][0] != "g":
        return None
    return decimal(value)*UNIT_FACTORS[label][1]


def normalized_text(text):
    return unicodedata.normalize("NFKC", text).translate(str.maketrans("‐‑‒–—", "-----"))


def answer_literals(answer):
    text = re.sub(r"[*`_]", "", normalized_text(answer))
    counts = sorted({decimal(match["value"]) for pattern in COUNTS for match in pattern.finditer(text)})
    masses = sorted({mass_g(match["value"], match["unit"]) for match in MASS.finditer(text)})
    return {"shipment_counts": [str(value) for value in counts], "mass_values_g": [str(value) for value in masses],
            "raw_mass_mentions": [{"text": match.group(), "mass_g": str(mass_g(match["value"], match["unit"]))}
                                   for match in MASS.finditer(text)]}


def artifact_parts(row):
    outcome = row.get("outcome")
    if outcome is not None:
        plan, claims, steps = outcome["plan"], outcome["claims"], outcome["steps"]
    else:
        plan = row.get("context_plan")
        steps = row.get("steps", [])
        claims = [claim for step in steps if step["result"] is not None for claim in step["result"]["claims"]]
    sources = {identity(source["key"]): source for source in row.get("sources", [])}
    if plan is not None:
        sources.update((identity(source["key"]), source) for source in plan["selected_set"]["source_blocks"])
    return outcome, plan, claims, steps, sources


def claim_provenance(claims, sources):
    by_id = {claim["claim_id"]: claim for claim in claims}
    records = []
    for claim in claims:
        spans = []
        for span in claim["source_spans"]:
            key = identity(span["source_key"])
            if key not in sources:
                spans.append({**span, "origin_status": "source_not_recorded"})
                continue
            source = sources[key]
            base = source["span"][0] if source.get("span") is not None else 0
            valid = base <= span["start"] <= span["end"] <= base+len(source["text"])
            spans.append({**span, "origin_status": "valid" if valid else "outside_original",
                "original_quote": source["text"][span["start"]-base:span["end"]-base] if valid else None})
        records.append({**claim, "source_audit": spans,
            "missing_parent_ids": [parent for parent in claim["parent_claim_ids"] if parent not in by_id]})
    return records


def source_ledger(sources):
    records, values = [], defaultdict(set)
    for source in sources.values():
        base = source["span"][0] if source.get("span") is not None else 0
        for match in SHIPMENT.finditer(source["text"]):
            entity = match["entity"].upper()
            number = mass_g(match["value"], match["unit"])
            values[entity].add(number)
            records.append({"entity": entity, "value": match["value"], "unit": match["unit"],
                "mass_g": str(number), "source_key": source["key"], "start": base+match.start(),
                "end": base+match.end(), "quote": match.group()})
    conflicts = {key: [str(value) for value in sorted(group)] for key, group in values.items() if len(group) != 1}
    return {"registrations": records, "unique_shipments": len(values), "repeated_registrations": len(records)-len(values),
        "source_mass_g": str(sum(next(iter(group)) for group in values.values())) if not conflicts else None,
        "conflicting_source_values": conflicts}


def long_audit(row, outcome, plan, claims, sources):
    ledger = source_ledger(sources)
    referenced = {parent for claim in claims for parent in claim["parent_claim_ids"]}
    roots = [claim for claim in claims if claim["kind"] == "derived" and claim["claim_id"] not in referenced]
    computed = [{"claim_id": claim["claim_id"], "need_id": claim["need_id"], "operation": claim["operation"],
        "value": claim["value"], "unit": claim["unit"],
        "mass_g": str(mass_g(claim["value"], claim["unit"])) if claim["operation"] == "sum" and mass_g(claim["value"], claim["unit"]) is not None else None,
        "parent_claim_ids": claim["parent_claim_ids"], "source_spans": claim["source_spans"]} for claim in roots]
    expected_count = decimal(row["expected"]["distinct_shipments"])
    expected_mass = decimal(row["expected"]["mass_kg"])*1000
    counts = [decimal(claim["value"]) for claim in roots if claim["operation"] == "count"]
    sums = [mass_g(claim["value"], claim["unit"]) for claim in roots if claim["operation"] == "sum"]
    literals = answer_literals(row.get("answer", ""))
    final_count = len(literals["shipment_counts"]) == 1 and decimal(literals["shipment_counts"][0]) == expected_count
    final_mass = len(literals["mass_values_g"]) == 1 and decimal(literals["mass_values_g"][0]) == expected_mass
    computed_correct = counts == [expected_count] and sums == [expected_mass]
    source_correct = ledger["unique_shipments"] == expected_count and ledger["source_mass_g"] is not None and decimal(ledger["source_mass_g"]) == expected_mass
    covered = {entry[2] for entry in outcome["coverage"]} if outcome is not None else set()
    source_records_covered = set()
    if plan is not None:
        for unit in plan["selected_set"]["selected_units"]:
            if unit["unit_id"] in covered:
                for span in unit["source_spans"]:
                    source_records_covered.update(record["entity"] for record in ledger["registrations"]
                        if record["source_key"] == span["source_key"] and span["start"] <= record["start"] and record["end"] <= span["end"])
    extracted = defaultdict(set)
    extracted_mass = []
    for claim in claims:
        if claim["kind"] != "derived":
            ids = {value.upper() for value in re.findall(r"(?<![A-Za-z0-9])S\d+(?![A-Za-z0-9])", claim["entity"], re.I)}
            extracted[claim["need_id"]].update(ids)
            normalized = mass_g(claim["value"], claim["unit"])
            if normalized is not None:
                expected = {decimal(record["mass_g"]) for record in ledger["registrations"] if record["entity"] in ids}
                extracted_mass.append({"claim_id": claim["claim_id"], "need_id": claim["need_id"],
                    "entity": claim["entity"], "mass_g": str(normalized),
                    "source_literal_mass_values_g": [str(value) for value in sorted(expected)],
                    "literal_matches_source": expected == {normalized}})
    return {"expected": {"count": str(expected_count), "mass_g": str(expected_mass)}, "original_source_ledger": ledger,
        "computed_root_claims": computed, "computed_count_matches_expected": counts == [expected_count],
        "computed_mass_matches_expected": sums == [expected_mass],
        "computed_quality": "passed" if computed_correct else "failed_or_incomplete",
        "original_source_totals_match_expected": source_correct,
        "final_answer_literals": literals, "final_unique_count_matches_expected": final_count,
        "final_unique_mass_matches_expected": final_mass,
        "final_answer_quality": "passed" if final_count and final_mass else "failed_or_ambiguous",
        "source_records_physically_covered": sorted(source_records_covered),
        "extracted_literal_record_ids_by_need": {key: sorted(value) for key, value in extracted.items()},
        "extracted_mass_claims": extracted_mass,
        "literal_quality": "passed" if source_correct and computed_correct and final_count and final_mass else "failed_or_incomplete",
        "scope": "controlled fixture literal audit; coverage and correct extraction are different observations"}


def multihop_audit(row, claims, provenance):
    expected_year = str(row["expected"]["year"])
    expected_sources = row["expected"]["source_chain"]
    by_id = {claim["claim_id"]: claim for claim in claims}
    missing = {parent for claim in claims for parent in claim["parent_claim_ids"] if parent not in by_id}
    def ancestors(identity_, active=()):
        if identity_ in active:
            return set(), {identity_}
        if identity_ not in by_id:
            return set(), set()
        found, cycles = {identity_}, set()
        for parent in by_id[identity_]["parent_claim_ids"]:
            nodes, bad = ancestors(parent, (*active, identity_))
            found.update(nodes)
            cycles.update(bad)
        return found, cycles
    role_terms = (("Meridian Institute", "Arbor Council"), ("Arbor Council", "Silver Fund"), ("Silver Fund", expected_year))
    witnesses = {}
    source_quotes = {claim["claim_id"]: " ".join(span["original_quote"] for span in claim["source_audit"]
                     if span["origin_status"] == "valid").casefold() for claim in provenance}
    for source_id, terms in zip(expected_sources, role_terms, strict=True):
        witnesses[source_id] = [claim["claim_id"] for claim in claims if any(
            span["source_key"]["document_id"] == source_id for span in claim["source_spans"])
            and all(term.casefold() in " ".join(str(claim[name]) for name in ("text", "entity", "value")).casefold()
                    and term.casefold() in source_quotes[claim["claim_id"]] for term in terms)]
    lineages = []
    for claim in claims:
        if re.search(rf"(?<!\d){re.escape(expected_year)}(?!\d)", str(claim["value"])+" "+claim["text"]):
            nodes, cycles = ancestors(claim["claim_id"])
            documents = {span["source_key"]["document_id"] for node in nodes for span in by_id[node]["source_spans"]}
            lineages.append({"claim_id": claim["claim_id"], "ancestor_claim_ids": sorted(nodes),
                "source_documents": sorted(documents), "cycles": sorted(cycles),
                "three_relation_witnesses_in_lineage": all(set(ids) & nodes for ids in witnesses.values())})
    final_year = re.search(rf"(?<!\d){re.escape(expected_year)}(?!\d)", row.get("answer", "")) is not None
    origin_valid = all(span["origin_status"] == "valid" for claim in provenance for span in claim["source_audit"])
    chain = any(lineage["three_relation_witnesses_in_lineage"] and not lineage["cycles"] for lineage in lineages)
    return {"expected_year": expected_year, "final_expected_year_present": final_year,
        "expected_source_chain": expected_sources, "relation_witness_claims": witnesses,
        "year_claim_lineages": lineages, "missing_parent_ids": sorted(missing), "source_spans_valid": origin_valid,
        "three_source_claim_chain_observed": chain,
        "literal_quality": "passed" if final_year and chain and origin_valid and not missing else "failed_or_incomplete"}


def audit_case(row):
    outcome, plan, claims, steps, sources = artifact_parts(row)
    provenance = claim_provenance(claims, sources)
    response = row.get("response", {})
    coverage = outcome["coverage"] if outcome is not None else []
    report = {"audit_kind": "controlled_fixture_literal_and_provenance", "benchmark_metric": False,
        "case": row["case"], "execution": {"status": row["status"], "qa_increment": row["qa_increment"],
            "version_increment": row["version_increment"], "single_qa_commit": row["qa_increment"] == row["version_increment"] == 1,
            "read_count": response.get("read_count"), "reduce_count": response.get("reduce_count"),
            "error": row.get("error")}, "answer": row.get("answer"), "expected": row["expected"],
        "coverage": {"entries": len(coverage), "covered_units": len({item[2] for item in coverage}),
            "required_units": len(plan["coverage_units"]) if plan is not None else None,
            "required_units_complete": set(plan["coverage_units"]) == {item[2] for item in coverage} if plan is not None else None,
            "covered_source_documents": sorted({item[1] for item in coverage}), "source_documents": sorted({source["key"]["document_id"] for source in sources.values()}),
            "open_needs": outcome["open_needs"] if outcome is not None else None}, "claim_provenance": provenance}
    cited = re.findall(r"\bE_[A-Za-z0-9_]+\b", row.get("answer", ""))
    report["citation_ids"] = {"mentioned": sorted(set(cited)), "allowed": response.get("citation_scope"),
        "outside_allowed": sorted(set(cited)-set(response["citation_scope"])) if "citation_scope" in response else None}
    if row["case"] == "long":
        report["quality"] = long_audit(row, outcome, plan, claims, sources)
    elif row["case"] == "multihop":
        report["quality"] = multihop_audit(row, claims, provenance)
    else:
        answer = normalized_text(row.get("answer", "")).casefold()
        checks = {key: normalized_text(value).casefold() in answer for key, value in row["expected"].items() if key != "eligibility"}
        checks.update(age_18_present=re.search(r"(?<!\d)18(?!\d)", answer) is not None,
                      residence_term_present=re.search(r"\bresiden\w*", answer) is not None)
        report["quality"] = {"expected_term_checks": checks,
            "literal_quality": "passed" if all(checks.values()) else "failed_or_incomplete",
            "scope": "literal terms only; not an entailment or benchmark score"}
    return report


def recorded_computed_lines(messages):
    marker = "Computed results for the complete reading scope:"
    return [line for message in messages if marker in message.content
        for line in (marker+message.content.split(marker, 1)[1]).split("\nCitations allowed for this answer:", 1)[0]
        .split("\nQuestion:", 1)[0].strip().splitlines()]


def recorded_final_prompt(database, row):
    from controller.session_store import Attempt, load_record
    attempt_id = row["response"]["attempt_id"] if "response" in row else row["attempt"]["attempt_id"] if row.get("attempt") is not None else None
    if attempt_id is None:
        return {"status": "no_recorded_attempt"}
    connection = sqlite3.connect(f"file:{Path(database).resolve()}?mode=ro", uri=True)
    payload = connection.execute("SELECT payload FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()[0]
    attempt = load_record(connection, Attempt, payload)
    connection.close()
    if attempt.plan is None:
        return {"status": "no_final_prompt_planned"}
    return {"status": "recorded", "attempt_id": attempt_id, "input_hash": attempt.plan.candidate.input_hash,
            "computed_lines": recorded_computed_lines(attempt.plan.candidate.messages)}


def audit_directory(input_dir, output_dir):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("audit output must be separate from original mechanism records")
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for case in ("pair", "correction", "long", "multihop"):
        path = input_dir/f"{case}.json"
        if not path.exists():
            reports.append({"case": case, "status": "artifact_missing"})
            continue
        row = json.loads(path.read_text())
        report = audit_case(row)
        report["source_artifact"] = {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        database = input_dir/"state"/"sessions.sqlite3"
        report["recorded_final_prompt"] = recorded_final_prompt(database, row) if database.exists() else {"status": "state_db_not_available"}
        (output_dir/f"{case}.audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
        reports.append({"case": case, "execution_status": row["status"], "literal_quality": report["quality"]["literal_quality"]})
    summary = {"audit_kind": "controlled_fixture_literal_and_provenance", "benchmark_metric": False,
        "input_dir": str(input_dir.resolve()), "records_unchanged": True, "cases": reports,
        "literal_status_counts": dict(Counter(row.get("literal_quality", row.get("status")) for row in reports))}
    (output_dir/"manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(audit_directory(args.input_dir, args.output_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
