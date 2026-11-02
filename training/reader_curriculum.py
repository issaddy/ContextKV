"""Grounded reader pairs from reviewed native training retrievals.

The parent reader cohort is read-only. Numeric edits change literal original
coordinates; source-position variants change layout only. Every resulting READ/ANSWER
target is generated and reviewed afresh in the separate curriculum directory.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
import re

import yaml

from training.reader_data import (augment_retrieved_reads, digest, file_digest, history_digest,
    prepare_reader_data, review_reader_traces, rows, write_rows)


_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_UNITS = r"years?|months?|weeks?|days?|hours?|minutes?|seconds?|credits?|tokens?|dollars?|USD|EUR|GBP|percent|%|kilograms?|kg|grams?|g|miles?|kilometers?|km|meters?|m|pounds?|lbs?"
_QUANTITY = re.compile(rf"(?<![\w.])(?:(?P<currency>[$€£])\s*(?P<money>{_NUMBER})(?!\w|[,.]\d)|(?P<number>{_NUMBER})\s*(?P<unit>{_UNITS})(?!\w))", re.I)
_CANONICAL_UNITS = {"$": "USD", "dollar": "USD", "usd": "USD", "€": "EUR", "eur": "EUR",
    "£": "GBP", "gbp": "GBP", "percent": "%", "%": "%", "kilogram": "kg", "kg": "kg",
    "gram": "g", "g": "g", "kilometer": "km", "km": "km", "meter": "m", "m": "m",
    "pound": "lb", "lb": "lb"}


def quantities(text):
    result = []
    for match in _QUANTITY.finditer(text):
        name = "money" if match.group("money") is not None else "number"
        literal = match.group(name)
        unit = match.group("currency") or match.group("unit")
        singular = unit.casefold().removesuffix("s")
        normalized = _CANONICAL_UNITS.get(singular, singular)
        result.append({"start": match.start(name), "end": match.end(name), "literal": literal,
            "quantity_start": match.start(), "quantity_end": match.end(), "quote": match.group(),
            "value": format(Decimal(literal.replace(",", "")), "f"), "unit": normalized})
    return result


def quantity_key(quantity):
    return Decimal(quantity["value"]), quantity["unit"]


def changed_literal(quantity, delta):
    value = Decimal(quantity["value"])+Decimal(str(delta))
    precision = len(quantity["literal"].split(".")[1]) if "." in quantity["literal"] else 0
    return format(value, f",.{precision}f" if "," in quantity["literal"] else f".{precision}f")


def round2_config(config):
    settings = config["reader_curriculum"]
    actual = deepcopy(config)
    actual["reader_training"] = deepcopy(yaml.safe_load(Path(settings["parent_training_config"]).read_text())["reader_training"])
    reader = actual["reader_training"]
    if Path(settings["output_dir"]).resolve() == Path(settings["source_data_dir"]).resolve():
        raise ValueError("curriculum output must be independent of the active parent reader cohort")
    reader["data"]["output_dir"] = settings["output_dir"]
    reader["data"]["reviews_path"] = str(Path(settings["output_dir"])/"reviews.jsonl")
    reader["output_dir"] = settings["reader_output_dir"]
    reader["merged_model_dir"] = settings["reader_merged_dir"]
    reader["initial_adapter"] = settings["initial_adapter"]
    return actual


def numeric_pairs(parent_rows, *, maximum_pairs=128, delta=7):
    """Only edit quantities also present in an accepted, source-verified target."""
    pairs, exclusions = [], []
    for parent in parent_rows:
        reason = ("not_native_train" if parent["split"] != "train" or
            parent["source_partition"] != "native_multidoc2dial_train" else
            "not_training_cohort" if parent["data_split"] != "train" else
            "not_answer" if parent["phase"] != "ANSWER" else
            "target_not_accepted" if parent["supervision"]["status"] != "accepted" else
            "target_not_source_verified" if not parent["supervision"]["source_support_verified"]
                or not parent["supervision"]["correct_task_binding"] else
            "actual_history_binding_mismatch" if history_digest(parent["actual_history"]) != parent["actual_history_sha256"] else None)
        if reason is not None:
            exclusions.append({"example_id": parent["example_id"], "reason": reason})
            continue
        target_quantities = {quantity_key(value) for value in quantities(parent["target_text"])}
        matches = defaultdict(list)
        for source in parent["original_sources"]:
            for value in quantities(source["text"]):
                if quantity_key(value) in target_quantities:
                    matches[quantity_key(value)].append((source, value))
        selected = [(source, value) for candidates in matches.values() if len(candidates) == 1
                    for source, value in candidates]
        if not selected:
            exclusions.append({"example_id": parent["example_id"], "reason":
                "ambiguous_retrieved_quantity" if matches else "no_shared_explicit_quantity"})
            continue
        source, value = selected[0]
        replacement = changed_literal(value, delta)
        if Decimal(replacement.replace(",", "")) == Decimal(value["value"]):
            exclusions.append({"example_id": parent["example_id"], "reason": "numeric_edit_has_no_change"})
            continue
        original = source["text"]
        mutated = original[:value["start"]]+replacement+original[value["end"]:]
        changed_key = {**source["source_key"], "content_sha256": digest_text(mutated)}
        pair_id = "native-numeric:"+digest([parent["example_id"], source["source_key"], value, replacement])[:24]
        mutation = {"task_id": parent["task_id"], "mutation_id": pair_id,
            "source_key": source["source_key"], "start": source["start"]+value["start"],
            "end": source["start"]+value["end"], "original_text": value["literal"], "replacement": replacement}
        pairs.append({"pair_id": pair_id, "kind": "numeric_literal", "task_id": parent["task_id"],
            "parent_example_id": parent["example_id"], "conversation_id": parent["conversation_id"],
            "split": "train", "source_partition": parent["source_partition"], "mutation": mutation,
            "original_quantity": value, "changed_quantity": {**value,
                "value": format(Decimal(replacement.replace(",", "")), "f"), "literal": replacement},
            "original_prompt": parent["messages"], "original_prompt_hash": parent["prompt_token_hash"],
            "parent_reviewed_target": parent["target_text"], "parent_target_origin": parent["target_origin"],
            "actual_history": parent["actual_history"], "actual_history_sha256": parent["actual_history_sha256"],
            "original_source": source, "mutated_source": {**source, "source_key": changed_key,
                "text": mutated, "end": source["start"]+len(mutated), "content_sha256": changed_key["content_sha256"]},
            "target_policy": "fresh prompt-bound teacher target required for both reconstructed controls and mutations"})
        if len(pairs) >= maximum_pairs:
            break
    return pairs, exclusions


def digest_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def _contains_edit_window(trace, pair, *, mutated):
    key = pair["mutated_source"]["source_key"] if mutated else pair["original_source"]["source_key"]
    start = pair["original_source"]["start"]+pair["original_quantity"]["quantity_start"]
    end = pair["original_source"]["start"]+pair["original_quantity"]["quantity_end"]
    if mutated:
        end += len(pair["mutation"]["replacement"])-len(pair["mutation"]["original_text"])
    ids = set(trace["read_task"]["input_unit_ids"])
    return any(unit["unit_id"] in ids and any(span["source_key"] == key
        and span["start"] <= start and end <= span["end"] for span in unit["source_spans"])
        for unit in trace["context_plan"]["selected_set"]["selected_units"])


def position_variants(trace, template):
    """Reorder literal source windows; preserve their original coordinates."""
    from controller.schema import Message
    source_message = json.loads(trace["messages"][1]["content"])
    windows = source_message["original_sources"]
    variants = [("position_reverse", list(reversed(windows)))]
    groups = defaultdict(list)
    for window in windows:
        key = window["source_spans"][0]["source_key"]
        groups[digest(key)].append(window)
    interleaved = [group[index] for index in range(max(map(len, groups.values())))
                   for group in groups.values() if index < len(group)]
    variants.append(("scattered_windows", interleaved))
    result, exclusions = [], []
    for kind, order in variants:
        if order == windows:
            exclusions.append({"example_id": trace["example_id"], "kind": kind, "reason": "layout_has_no_change"})
            continue
        messages = deepcopy(trace["messages"])
        messages[1]["content"] = json.dumps({**source_message, "original_sources": order}, ensure_ascii=False, separators=(",", ":"))
        rendered = template.render(tuple(Message(**message) for message in messages))
        result.append({**trace, "example_id": f"{trace['example_id']}:{kind}", "messages": messages,
            "prompt_token_hash": rendered.input_hash, "prompt_tokens": len(rendered.input_ids),
            "curriculum_kind": kind, "control_example_id": trace["example_id"],
            "original_prompt_hash": trace["prompt_token_hash"], "actual_output": "", "actual_output_ids": []})
    return result, exclusions


def build_curriculum(config, *, template=None):
    settings = config["reader_curriculum"]
    source = Path(settings["source_data_dir"])
    destination = Path(settings["output_dir"])
    actual = round2_config(config)
    parent_rows = rows(source/"train.jsonl")
    pairs, exclusions = numeric_pairs(parent_rows, maximum_pairs=settings["maximum_numeric_pairs"], delta=settings["numeric_delta"])
    chosen = {pair["parent_example_id"] for pair in pairs}
    parents = [{name: value for name, value in row.items() if name not in {
        "target_text", "target_origin", "quality", "supervision", "data_split"}}
        for row in parent_rows if row["example_id"] in chosen]
    destination.mkdir(parents=True, exist_ok=True)
    write_rows(destination/"parent_native_inputs.jsonl", parents)
    write_rows(destination/"traces.jsonl", parents)
    write_rows(destination/"numeric_mutations.jsonl", [pair["mutation"] for pair in pairs])
    actual["reader_training"]["data"].update(paired_mutations_path=str(destination/"numeric_mutations.jsonl"),
                                            retrieved_read_turns=len(parents))
    if template is None:
        from controller.templates import HuggingFaceTemplate
        from engine_patches.contextkv_model_scope import load_model_scope
        scope = load_model_scope(actual["reader_training"]["base_scope"])
        template = HuggingFaceTemplate(scope.model_path, None, local_files_only=True, model_identity=scope.identity)
    augment = augment_retrieved_reads(actual, template=template)
    reads = [row for row in rows(destination/"traces.jsonl") if row["phase"] == "READ"]
    for pair in pairs:
        pair["control_example_ids"] = [row["example_id"] for row in reads
            if row["task_id"] == pair["task_id"] and not row["source_mutations"] and _contains_edit_window(row, pair, mutated=False)]
        pair["changed_example_ids"] = [row["example_id"] for row in reads
            if any(edit["mutation_id"] == pair["pair_id"] for edit in row["source_mutations"])
            and _contains_edit_window(row, pair, mutated=True)]
    from training.reader_constructed_curriculum import constructed_pairs
    constructed_reads, constructed = constructed_pairs(parent_rows, template,
        maximum_cases=settings.get("maximum_constructed_cases", 8),
        excerpt_characters=settings.get("constructed_excerpt_characters", 1200))
    from training.reader_computed_curriculum import computed_answer_pairs
    computed_answers, computed_pairs = computed_answer_pairs(parent_rows, template,
        maximum_cases=settings.get("maximum_computed_cases", 0),
        excerpt_characters=settings.get("computed_excerpt_characters", 256))
    pairs.extend(constructed)
    pairs.extend(computed_pairs)
    positional = []
    for row in reads:
        if row["source_mutations"]:
            continue
        variants, skipped = position_variants(row, template)
        positional.extend(variants)
        exclusions.extend(skipped)
    all_traces = [*reads, *positional, *constructed_reads, *computed_answers]
    by_id = {row["example_id"]: row for row in all_traces}
    for pair in pairs:
        pair["control_prompt_hashes"] = [by_id[identity]["prompt_token_hash"] for identity in pair["control_example_ids"]]
        pair["changed_prompt_hashes"] = [by_id[identity]["prompt_token_hash"] for identity in pair["changed_example_ids"]]
    write_rows(destination/"traces.jsonl", all_traces)
    write_rows(destination/"pairs.prepared.jsonl", pairs)
    write_rows(destination/"construction_exclusions.jsonl", exclusions)
    (destination/"reader_curriculum.yaml").write_text(yaml.safe_dump(actual, sort_keys=False))
    manifest = {"schema_version": 1, "namespace": "native_train_curriculum_round2",
        "numeric_pairs": sum(pair["kind"] == "numeric_literal" for pair in pairs),
        "constructed_pairs": dict(Counter(pair["kind"] for pair in constructed)), "constructed_read_inputs": len(constructed_reads),
        "computed_answer_pairs": len(computed_pairs), "computed_answer_inputs": len(computed_answers),
        "read_inputs": len(reads), "position_inputs": len(positional), "augment": augment,
        "source_train_path": str((source/"train.jsonl").resolve()), "source_train_sha256": file_digest(source/"train.jsonl"),
        "exclusions": dict(Counter(row["reason"] for row in exclusions)), "public_dev_test_sources_read": [],
        "targets": "pending fresh teacher generation; no parent target reused as mutation supervision",
        "reader_config": str((destination/"reader_curriculum.yaml").resolve())}
    (destination/"curriculum_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def audit_curriculum(config):
    """Require observed source-sensitive targets before adding numeric pairs."""
    settings = config["reader_curriculum"]
    destination = Path(settings["output_dir"])
    actual = round2_config(config)
    reviews = {row["example_id"]: row for row in rows(destination/"reviews.jsonl")}
    traces = {row["example_id"]: row for row in rows(destination/"traces.jsonl")}
    records, approved_ids = [], set()
    for pair in rows(destination/"pairs.prepared.jsonl"):
        old, new = quantity_key(pair["original_quantity"]), quantity_key(pair["changed_quantity"])
        required_control = pair.get("all_control_example_ids", pair["control_example_ids"])
        required_changed = pair.get("all_changed_example_ids", pair["changed_example_ids"])
        required_ids = tuple(dict.fromkeys((*required_control, *required_changed)))
        step_statuses = {key: reviews[key]["status"] if key in reviews else "missing_review" for key in required_ids}
        controls = [reviews[key] for key in pair["control_example_ids"] if key in reviews and reviews[key]["status"] == "accepted"]
        changed = [reviews[key] for key in pair["changed_example_ids"] if key in reviews and reviews[key]["status"] == "accepted"]
        control_values = {value for review in controls for value in reviewed_quantities(review, traces[review["example_id"]])}
        changed_values = {value for review in changed for value in reviewed_quantities(review, traces[review["example_id"]])}
        status = ("missing_accepted_control" if not controls else "missing_accepted_changed_target" if not changed else
                  "control_quantity_not_reproduced" if old not in control_values else
                  "target_unchanged" if {review["target_text"] for review in controls} == {review["target_text"] for review in changed} else
                  "changed_quantity_not_reproduced" if new not in changed_values else
                  "mixed_old_and_new_values" if old in changed_values else "source_sensitive_change_verified")
        if pair["kind"] == "record_presence" and controls and changed:
            control_targets = [json.loads(review["target_text"]) for review in controls]
            changed_targets = [json.loads(review["target_text"]) for review in changed]
            control_entities = {claim["entity"] for target in control_targets for claim in target["claims"]}
            changed_claims = [claim for target in changed_targets for claim in target["claims"]]
            changed_entities = {claim["entity"] for claim in changed_claims}
            status = ("no_record_window_contains_spurious_facts" if control_entities else
                "added_record_not_extracted_exactly" if changed_entities != set(pair["expected_record_entities"])
                    or len(changed_claims) != len(pair["expected_record_entities"]) else
                "spurious_record_window_gap" if any(target["open_needs"] or any(claim["followups"]
                    for claim in target["claims"]) for target in (*control_targets, *changed_targets))
                else "source_sensitive_change_verified")
        computed_checks = None
        if pair["kind"] == "computed_full_scope_answer" and controls and changed:
            from training.reader_computed_curriculum import computed_target_check
            computed_checks = {"control": [computed_target_check(review, count=pair["expected_count"], mass_kg=old[0])
                for review in controls], "changed": [computed_target_check(review, count=pair["expected_count"], mass_kg=new[0])
                for review in changed]}
            verified = all(check["full_scope_count_verified"] and check["full_scope_mass_verified"]
                for checks in computed_checks.values() for check in checks)
            status = "source_sensitive_change_verified" if verified else "computed_scope_totals_not_reproduced"
        if status == "source_sensitive_change_verified" and pair["kind"] not in {"numeric_literal", "record_presence", "computed_full_scope_answer"}:
            if control_values != {old}:
                status = "constructed_control_includes_unrequested_quantities"
            elif changed_values != {new}:
                status = "constructed_target_includes_unrequested_quantities"
        for key in required_ids:
            if key not in reviews or reviews[key]["status"] != "accepted":
                continue
            review = reviews[key]
            trace = traces[review["example_id"]]
            if (review["prompt_token_hash"] != trace["prompt_token_hash"] or
                    review["actual_history_sha256"] != pair["actual_history_sha256"] or
                    review["reader_identity"] != trace["reader_identity"]):
                raise ValueError("curriculum teacher target is not bound to its unchanged actual history and new prompt")
        if any(step_statuses[key] != "accepted" for key in required_control):
            status = "incomplete_accepted_control_steps"
        elif any(step_statuses[key] != "accepted" for key in required_changed):
            status = "incomplete_accepted_changed_steps"
        if status == "source_sensitive_change_verified":
            approved_ids.update(required_ids)
        records.append({**pair, "pair_check": status, "fresh_control_targets": controls,
                        "fresh_changed_targets": changed, "computed_target_checks": computed_checks,
                        "fresh_step_review_statuses": step_statuses,
                        "changed_target_was_not_copied": status == "source_sensitive_change_verified"})
    position_checks = []
    for trace in traces.values():
        if "control_example_id" in trace:
            review = reviews.get(trace["example_id"])
            control = reviews.get(trace["control_example_id"])
            status = "missing_accepted_position_pair"
            if review is not None and control is not None and review["status"] == control["status"] == "accepted":
                original_values = reviewed_quantities(control, traces[control["example_id"]])
                changed_values = reviewed_quantities(review, trace)
                status = ("target_numeric_invariance_verified" if original_values and original_values == changed_values else
                    "empty_window_invariance_verified" if not json.loads(control["target_text"])["claims"]
                        and not json.loads(review["target_text"])["claims"] else "target_invariance_not_verified")
            if status in {"target_numeric_invariance_verified", "empty_window_invariance_verified"}:
                approved_ids.add(trace["example_id"])
            position_checks.append({"example_id": trace["example_id"], "control_example_id": trace["control_example_id"],
                                    "kind": trace["curriculum_kind"], "check": status})
    filtered = [row for row in reviews.values() if row["example_id"] in approved_ids]
    write_rows(destination/"curriculum_approved_reviews.jsonl", filtered)
    actual["reader_training"]["data"]["reviews_path"] = str(destination/"curriculum_approved_reviews.jsonl")
    prepared = prepare_reader_data(actual)
    augmented = rows(destination/"train.jsonl")
    source = Path(settings["source_data_dir"])
    parent = rows(source/"train.jsonl")
    validation = rows(source/"validation.jsonl")
    combined = [*parent, *augmented, *validation]
    write_rows(destination/"train.jsonl", [*parent, *augmented])
    write_rows(destination/"validation.jsonl", validation)
    prepared.update(evidence_kind="reviewed_native_train_and_declared_constructed_curriculum",
        fresh_teacher_step_policy="all planned paired steps accepted with exact prompt/history/reader binding, including empty windows",
        training_turns=len({row["task_id"] for row in (*parent, *augmented)}),
        examples=len(parent)+len(augmented)+len(validation),
        turns=len({row["task_id"] for row in combined}),
        phases=dict(Counter(row["phase"] for row in combined)),
        target_origins=dict(Counter(row["target_origin"] for row in combined)),
        source_partitions=dict(Counter(row["source_partition"] for row in combined)),
        data_splits={"train": len(parent)+len(augmented), "validation": len(validation)},
        parent_training_data={"path": str((source/"train.jsonl").resolve()), "sha256": file_digest(source/"train.jsonl")},
        parent_validation_data={"path": str((source/"validation.jsonl").resolve()), "sha256": file_digest(source/"validation.jsonl")},
        curriculum_added_examples=len(augmented), pair_checks=dict(Counter(row["pair_check"] for row in records)),
        position_checks=dict(Counter(row["check"] for row in position_checks)),
        validation_origin="unchanged conversation-heldout native training partition from parent reader; no public dev/test")
    (destination/"data_manifest.json").write_text(json.dumps(prepared, indent=2)+"\n")
    write_rows(destination/"pairs.audited.jsonl", records)
    write_rows(destination/"position_pairs.audited.jsonl", position_checks)
    (destination/"reader_curriculum.yaml").write_text(yaml.safe_dump(actual, sort_keys=False))
    return prepared


def reviewed_quantities(review, trace):
    if trace["phase"] == "ANSWER":
        return {quantity_key(value) for value in quantities(review["target_text"])}
    target = json.loads(review["target_text"])
    values = []
    for claim in target["claims"]:
        values.extend(quantities(claim["text"]))
        values.extend(quantities(f"{claim['value']} {claim['unit']}"))
    return {quantity_key(value) for value in values}


def main(argv=None):
    from functools import partial
    from training.train_reader import measured_command
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "review", "audit"))
    parser.add_argument("--config", default="configs/reader_curriculum.yaml")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text())
    actual = round2_config(config)
    if args.command == "build":
        operation, measured = partial(build_curriculum, config), "augment-reads"
    elif args.command == "audit":
        operation, measured = partial(audit_curriculum, config), "prepare"
    else:
        operation, measured = partial(review_reader_traces, actual, concurrency=args.concurrency), "review"
    result = asyncio.run(measured_command(actual, measured, operation, asynchronous=args.command == "review"))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
