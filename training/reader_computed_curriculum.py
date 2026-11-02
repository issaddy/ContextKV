"""Training-only full-scope computed conclusions with representative excerpts.

Literal records and their typed reductions are declared construction oracles,
not outputs attributed to a reader execution. Fresh teacher calls supply every
ANSWER target. Native parent QA and retrieval coordinates remain attached.
"""
from __future__ import annotations

from decimal import Decimal
import re

from controller.candidates import build_plan_candidates
from controller.context_schema import (AnswerPlan, AnswerPoint, ContextBudget, ContextPlan,
    EvidenceClaim, EvidenceSet, EvidenceUnit, Need, ReadingOutcome, SourceSpan, TaskState, TextRef)
from controller.evidence import make_evidence
from controller.evidence_planner import render_selection
from controller.reduction import reduce_claims
from controller.schema import SessionSnapshot, TurnRecord
from controller.serialization import decode, to_jsonable
from training.reader_data import history_digest


def computed_answer_pairs(parents, template, *, maximum_cases=8, excerpt_characters=256):
    traces, pairs = [], []
    eligible = [row for row in parents if row["phase"] == "ANSWER" and row["data_split"] == "train"
        and row["split"] == "train" and row["source_partition"] == "native_multidoc2dial_train"
        and row["supervision"]["status"] == "accepted" and row["supervision"]["source_support_verified"]
        and row["supervision"]["correct_task_binding"]]
    for case_index, parent in enumerate(eligible[:maximum_cases]):
        if history_digest(parent["actual_history"]) != parent["actual_history_sha256"]:
            raise ValueError("computed curriculum parent actual history differs from its reviewed history")
        pair_id = f"constructed-train:computed-scope:{parent['example_id']}:{case_index}"
        registry = f"Dispatch-QT{case_index:04}"
        quantity = 5+case_index % 3
        weights = [13+4*index+case_index for index in range(quantity)]
        history = tuple(decode(TurnRecord, turn) for turn in parent["actual_history"])
        question = f"Across the complete supplied {registry} registers, count distinct dispatch records and sum their masses in kilograms."
        refs = tuple(TextRef(turn.turn_id, speaker, 0, len(text)) for turn in history
            for speaker, text in (("user", turn.user_text), ("assistant", turn.assistant_text)))
        needs = (Need(pair_id+":count", operation="count", scope="all_supplied_documents"),
                 Need(pair_id+":mass", operation="sum", scope="all_supplied_documents"))
        variant_rows = []
        variant_sources = []
        for variant, delta in (("control", 0), ("changed", 11)):
            actual_weights = [*weights[:-1], weights[-1]+delta]
            fact_records = [(f"{registry}-D{index:02}", mass) for index, mass in enumerate(actual_weights)]
            # The repeated first identity does not add a distinct dispatch.
            groups = (fact_records[:2], [*fact_records[2:], fact_records[0]])
            blocks, units, leaves, coordinates = [], [], [], []
            for slot, records in enumerate(groups):
                parent_source = parent["original_sources"][slot % len(parent["original_sources"])]
                excerpt = parent_source["text"][:excerpt_characters]
                text = excerpt+f"\n\nDeclared {registry} training register {slot}:\n"
                local = []
                for entity, kilograms in records:
                    literal = f"Dispatch record {entity}: approved mass {kilograms} kg."
                    start = len(text)
                    text += literal+"\n"
                    local.append((entity, kilograms, start, len(text)-1, literal))
                block = make_evidence(text, f"{registry}:register:{slot}",
                    corpus_id="constructed_train_computed_scope",
                    version="constructed-v1", source_uri=f"constructed-train-reader://{registry}/register/{slot}")
                blocks.append(block)
                coordinate = {"source_key": to_jsonable(block.key), "start": 0, "end": len(text), "text": text,
                    "content_sha256": block.key.content_sha256, "source_uri": block.source_uri,
                    "native_parent_span": {"source_key": parent_source["source_key"], "start": parent_source["start"],
                        "end": parent_source["start"]+len(excerpt)}, "declared_training_literals": []}
                for index, (entity, kilograms, start, end, literal) in enumerate(local):
                    span = SourceSpan(f"{registry}:{slot}:{index}", block.key, start, end)
                    units.append(EvidenceUnit(f"{registry}:unit:{slot}:{index}", (span,), literal,
                        document_family=registry, retrieval_score=-len(units)))
                    coordinate["declared_training_literals"].append({"start": start, "end": end, "text": literal})
                    for need, value, unit in ((needs[0], 1, ""), (needs[1], kilograms, "kg")):
                        leaves.append(EvidenceClaim(f"{need.need_id}:literal:{slot}:{index}", need.need_id, "direct",
                            literal, value=value, unit=unit, source_spans=(span,), entity=entity,
                            relation="dispatch_record" if need.operation == "count" else "mass", version="constructed-v1"))
                coordinates.append(coordinate)
            roots = tuple(reduce_claims(need.operation, leaves, claim_id=need.need_id+":computed", need_id=need.need_id)
                          for need in needs)
            state = TaskState(pair_id, question, working_history=refs, needs=needs,
                source_scope=tuple(block.key for block in blocks), current_turn_id=parent["turn"], query=question)
            selected = EvidenceSet(pair_id+":"+variant, tuple(units), source_blocks=tuple(blocks))
            answer = AnswerPlan(tuple(AnswerPoint(root.need_id, (root.claim_id,), root.source_spans) for root in roots))
            # The final window intentionally contains fewer records than the
            # complete declared reading scope; the complete ledger stays below.
            body_tokens = len(template.tokenizer.encode(render_selection((units[0],)), add_special_tokens=False))
            budget = ContextBudget(body_tokens=body_tokens, history_tokens=32768, output_tokens=1024)
            plan = ContextPlan(pair_id+":"+variant, state, selected, "all_supplied_documents",
                answer_plan=answer, budget=budget, coverage_units=tuple(unit.unit_id for unit in units), layouts=("existing",))
            coverage = tuple((need.need_id, unit.source_spans[0].source_key.document_id, unit.unit_id, "offline-construction")
                for need in needs for unit in units)
            oracle = ReadingOutcome(plan, claims=(*leaves, *roots), coverage=coverage)
            snapshot = SessionSnapshot(parent["conversation_id"], len(history), history, archive=tuple(blocks))
            candidate = build_plan_candidates(snapshot, question, plan, template, outcome=oracle)[0]
            representative = {unit.unit_id for unit in units if any(span.key == unit.source_spans[0].source_key
                and span.source_start <= unit.source_spans[0].start and unit.source_spans[0].end <= span.source_end
                for span in candidate.evidence_spans)}
            mutations = []
            if variant == "changed":
                original = variant_sources[0][-1]
                literal = next(item for item in original["declared_training_literals"] if fact_records[-1][0] in item["text"])
                number_start = literal["start"]+literal["text"].index("approved mass ")+len("approved mass ")
                mutations = [{"mutation_id": pair_id, "kind": "computed_mass_literal",
                    "source_key": original["source_key"], "mutated_source_key": coordinates[-1]["source_key"],
                    "start": number_start, "end": number_start+len(str(weights[-1])),
                    "original_text": str(weights[-1]), "replacement": str(actual_weights[-1]),
                    "changed_record": fact_records[-1][0], "delta_kg": delta}]
            variant_rows.append({"example_id": pair_id+":"+variant+":ANSWER", "task_id": parent["task_id"],
                "conversation_id": parent["conversation_id"], "turn": parent["turn"], "attempt_id": parent["attempt_id"],
                "question": question, "phase": "ANSWER", "split": "train", "source_partition": "constructed_train_computed_scope",
                "actual_history": parent["actual_history"], "actual_history_sha256": parent["actual_history_sha256"],
                "reader_identity": parent["reader_identity"], "snapshot_id": "reader-round2-computed-v1",
                "messages": to_jsonable(candidate.messages), "prompt_token_hash": candidate.input_hash,
                "prompt_tokens": len(candidate.input_ids), "actual_output": "", "actual_output_ids": [],
                "read_protocol_version": None, "task_state": to_jsonable(state), "context_plan": to_jsonable(plan),
                "curriculum_kind": "computed_full_scope_answer", "pair_id": pair_id,
                "source_mutations": mutations,
                "original_sources": coordinates, "input_claims_oracle": to_jsonable((*leaves, *roots)),
                "trace_origin": {"input_origin": "declared_constructed_train_from_actual_native_retrieval",
                    "parent_example_id": parent["example_id"], "computed_claims": "typed offline construction oracle; not generated READ/REDUCE history"},
                "offline_scope_audit": {"distinct_dispatches": quantity, "mass_g": str(sum(actual_weights)*1000),
                    "representative_unit_ids": sorted(representative), "scope_unit_count": len(units),
                    "representative_scope_is_partial": len(representative) < len(units),
                    "compiled_candidate_feasible": candidate.feasible, "compiled_candidate_reason": candidate.reason},
                "generation_metrics": {"status": "pending_actual_teacher_target", "input_tokens": 0, "output_tokens": 0}})
            variant_sources.append(coordinates)
        traces.extend(variant_rows)
        pairs.append({"pair_id": pair_id, "kind": "computed_full_scope_answer", "task_id": parent["task_id"],
            "parent_example_id": parent["example_id"], "split": "train", "source_partition": "constructed_train_computed_scope",
            "original_prompt": parent["messages"], "original_prompt_hash": parent["prompt_token_hash"],
            "actual_history": parent["actual_history"], "actual_history_sha256": parent["actual_history_sha256"],
            "original_quantity": {"value": str(sum(weights)), "unit": "kg"},
            "changed_quantity": {"value": str(sum(weights)+11), "unit": "kg"}, "expected_count": quantity,
            "control_example_ids": [variant_rows[0]["example_id"]], "changed_example_ids": [variant_rows[1]["example_id"]],
            "constructed_control_sources": variant_sources[0], "constructed_changed_sources": variant_sources[1],
            "target_policy": "fresh teacher ANSWER under exact computed prompt; old answer is not a target"})
    return traces, pairs


def computed_target_check(review, *, count, mass_kg):
    """Audit only the generated target's explicit totals, with SI conversion."""
    from training.reader_curriculum import quantities
    text = re.sub(r"[*`_]", "", review["target_text"])
    counts = {Decimal(match.group(1)) for pattern in (
        r"(?<!\d)(\d+)\s+(?:(?:distinct|unique|approved)\s+)*(?:dispatch records?|dispatches)\b",
        r"\b(?:dispatch records?|dispatches)\s*(?::|=|is|are)\s*(\d+)\b")
        for match in re.finditer(pattern, text, re.I)}
    masses = {Decimal(value["value"])*(1000 if value["unit"] == "kg" else 1)
        for value in quantities(text) if value["unit"] in {"kg", "g"}}
    return {"counts": [str(value) for value in sorted(counts)], "mass_values_g": [str(value) for value in sorted(masses)],
        "full_scope_count_verified": counts == {Decimal(count)},
        "full_scope_mass_verified": masses == {Decimal(mass_kg)*1000}}
