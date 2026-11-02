"""Declared training-only condition/version and three-hop reading pairs."""
from __future__ import annotations

from dataclasses import replace

from adapters.corpus import CorpusIndex
from adapters.evidence_pool import SourceDirectory
from controller.context_schema import (Condition, ContextBudget, EvidenceClaim, EvidenceSet,
    Need, TaskState, TextRef, VersionIntent)
from controller.reading import build_read_plan, step_messages
from controller.schema import SessionSnapshot, TurnRecord
from controller.serialization import decode, to_jsonable


def constructed_pairs(parents, template, *, maximum_cases=8, excerpt_characters=1200):
    """Keep actual native QA; add clearly declared literal training facts.

    Prior hop claims below are offline navigation supervision anchored to these
    constructed source literals. They are never labelled as generated runtime
    claims. All output targets still require fresh teacher calls.
    """
    traces, pairs = [], []
    eligible = [row for row in parents if row["phase"] == "ANSWER" and row["data_split"] == "train"
        and row["source_partition"] == "native_multidoc2dial_train" and row["supervision"]["status"] == "accepted"]
    for case_index, parent in enumerate(eligible[:maximum_cases]):
        native = parent["original_sources"]
        family = f"reader-curriculum-constructed-train-{case_index:04}"
        ledger = f"Ledger-C{case_index:04}"
        history = tuple(decode(TurnRecord, turn) for turn in parent["actual_history"])
        current = parent["turn"]
        refs = tuple(TextRef(turn.turn_id, speaker, 0, len(text)) for turn in history
            for speaker, text in (("user", turn.user_text), ("assistant", turn.assistant_text)))

        def source(slot, fact, *, version="2024", platform=None):
            origin = native[slot % len(native)]
            excerpt = origin["text"][:excerpt_characters]
            body = excerpt+"\n\n"+fact
            return {"passage_id": f"{family}:source:{slot}", "document_id": f"{family}:source:{slot}",
                "version": version, "text": body, "document_family": family,
                "applicability": {"PLATFORM": platform} if platform else {},
                "source_uri": f"constructed-train-reader://{family}/source/{slot}",
                "title": f"{ledger} supplied training record {slot}",
                "derived_from": {"source_key": origin["source_key"], "start": origin["start"],
                    "end": origin["start"]+len(excerpt), "text": excerpt},
                "constructed_literal": {"start": len(excerpt)+2, "end": len(body), "text": fact}}

        def compile_variant(pair_id, variant, kind, question, records, *, separated_hops=False):
            corpus = CorpusIndex.from_records(records, corpus_id="constructed_train_reader_curriculum",
                                               snapshot_id="constructed-train-reader-round2-v1")
            directory = SourceDirectory(corpus, None, template.tokenizer)
            pool = directory.full_scope(tuple(block.key for block in corpus.blocks))
            conditions = ()
            intent = VersionIntent()
            if kind in {"condition_swap", "version_swap"}:
                first = question.index("Windows")
                version = question.index("2024")
                conditions = (Condition(pair_id+":platform", value_spans=(TextRef(current, "user", first, first+7),), relation_type="PLATFORM"),
                    Condition(pair_id+":version", value_spans=(TextRef(current, "user", version, version+4),), relation_type="VERSION"))
                intent = VersionIntent("explicit", document_families=(family,), version_ids=("2024",), resolved_version_ids=("2024",))
            need = Need(pair_id+":need", anchor_spans=(TextRef(current, "user", 0, len(question)),),
                        operation="infer" if separated_hops else "count" if kind == "record_presence" else "lookup",
                        condition_ids=tuple(condition.condition_id for condition in conditions),
                        scope="all_supplied_documents" if kind == "record_presence" else "selected_sources")
            state = TaskState(pair_id, question, refs, conditions, (need,), intent,
                tuple(block.key for block in corpus.blocks), current, query=question)
            snapshot = SessionSnapshot(parent["conversation_id"], len(history), history, archive=corpus.blocks)
            budget = ContextBudget(window_tokens=32768, body_tokens=8192, read_input_tokens=12288,
                                   read_output_tokens=2048, max_steps=512)
            batches = [(tuple(unit for unit in pool.units if unit.source_spans[0].source_key == block.key), (block,))
                       for block in corpus.blocks] if separated_hops else [(pool.units, pool.source_blocks)]
            input_claims = ()
            result = []
            for hop, (units, blocks) in enumerate(batches):
                evidence = EvidenceSet(f"{pair_id}:{variant}:hop{hop}", units, source_blocks=blocks)
                plan = build_read_plan(state, evidence, need.scope, budget, template=template,
                    snapshot=snapshot, input_claims=input_claims, read_layout="source_first", _step_prefix=f"hop{hop}:")
                by_id = {unit.unit_id: unit for unit in plan.selected_set.selected_units}
                for task in plan.steps:
                    if task.kind != "READ":
                        continue
                    messages = step_messages(plan, task, tuple(by_id[identity] for identity in task.input_unit_ids), input_claims, snapshot)
                    prompt = template.render(messages)
                    result.append({"example_id": f"{pair_id}:{variant}:{task.step_id}",
                        "task_id": parent["task_id"], "conversation_id": parent["conversation_id"], "turn": current,
                        "attempt_id": parent["attempt_id"], "step_id": f"{pair_id}:{variant}:{task.step_id}",
                        "question": question, "phase": "READ", "split": "train", "source_partition": "constructed_train_reader_curriculum",
                        "actual_history": parent["actual_history"], "actual_history_sha256": parent["actual_history_sha256"],
                        "reader_identity": parent["reader_identity"], "snapshot_id": corpus.snapshot_id,
                        "messages": to_jsonable(messages), "prompt_token_hash": prompt.input_hash,
                        "prompt_tokens": len(prompt.input_ids), "actual_output": "", "actual_output_ids": [],
                        "read_task": to_jsonable(task), "read_result": None, "output_schema": task.output_schema,
                        "read_protocol_version": 2, "read_layout": "source_first", "task_state": to_jsonable(state),
                        "context_plan": to_jsonable(plan), "curriculum_kind": kind, "pair_id": pair_id,
                        "source_mutations": [] if variant == "control" else [{"mutation_id": pair_id, "kind": kind}],
                        "original_sources": [{"source_key": to_jsonable(block.key), "start": 0, "end": len(block.text),
                            "text": block.text, "content_sha256": block.key.content_sha256, "source_uri": block.source_uri,
                            "native_parent_span": {name: value for name, value in record["derived_from"].items() if name != "text"},
                            "declared_training_literal": {name: value for name, value in record["constructed_literal"].items() if name != "text"}}
                            for block, record in zip(corpus.blocks, records, strict=True)],
                        "trace_origin": {"input_origin": "declared_constructed_train_from_actual_native_retrieval",
                            "parent_example_id": parent["example_id"], "navigation_claims": "offline literal construction oracle; not generated READ history"},
                        "input_claims_oracle": to_jsonable(input_claims),
                        "generation_metrics": {"status": "pending_actual_teacher_target", "input_tokens": 0, "output_tokens": 0}})
                if separated_hops and hop < 2:
                    # Source records declare these exact prior relation literals.
                    record = records[hop]
                    block = corpus.blocks[hop]
                    literal = record["constructed_literal"]
                    span = next(span for unit in pool.units for span in unit.source_spans
                        if span.source_key == block.key and span.start <= literal["start"] and literal["end"] <= span.end)
                    entity = f"Team-T{case_index:04}" if hop == 0 else f"Org-O{case_index:04}"
                    value = f"Org-O{case_index:04}" if hop == 0 else f"Fund-F{case_index:04}"
                    claim = EvidenceClaim(f"{pair_id}:literal:{hop}", need.need_id, "direct",
                        literal["text"], value=value, source_spans=(replace(span, start=literal["start"], end=literal["end"]),),
                        parent_claim_ids=(input_claims[-1].claim_id,) if input_claims else (),
                        entity=entity, relation="belongs_to" if hop == 0 else "funded_by", version=block.key.version)
                    input_claims = (*input_claims, claim)
            return result

        for kind in ("condition_swap", "version_swap", "multihop_terminal", "record_presence"):
            pair_id = f"constructed-train:{family}:{kind}"
            if kind == "condition_swap":
                old, new, unit = 40+case_index, 70+case_index, "credit"
                question = f"For {ledger}, use Windows instead of Linux under policy version 2024. What is its filing allowance in credits?"
                controls = [source(0, f"{ledger} policy version 2024 on Windows: filing allowance is {old} credits.", platform="Windows"),
                    source(1, f"{ledger} policy version 2024 on Linux: filing allowance is {new} credits.", platform="Linux")]
                changed = [source(0, f"{ledger} policy version 2024 on Linux: filing allowance is {old} credits.", platform="Linux"),
                    source(1, f"{ledger} policy version 2024 on Windows: filing allowance is {new} credits.", platform="Windows")]
            elif kind == "version_swap":
                old, new, unit = 90+case_index, 30+case_index, "credit"
                question = f"For {ledger} on Windows, use policy version 2024 instead of 2018. What is the filing allowance in credits?"
                controls = [source(0, f"{ledger} policy version 2024 on Windows: filing allowance is {old} credits.", platform="Windows"),
                    source(1, f"{ledger} policy version 2018 on Windows: filing allowance is {new} credits.", version="2018", platform="Windows")]
                changed = [source(0, f"{ledger} policy version 2018 on Windows: filing allowance is {old} credits.", version="2018", platform="Windows"),
                    source(1, f"{ledger} policy version 2024 on Windows: filing allowance is {new} credits.", platform="Windows")]
            elif kind == "multihop_terminal":
                old, new, unit = 17+case_index, 24+case_index, "year"
                question = f"How many years is the operating mandate of the fund financing Team-T{case_index:04}'s parent organization?"
                controls = [source(0, f"Team-T{case_index:04} belongs to Org-O{case_index:04}."),
                    source(1, f"Org-O{case_index:04} is funded by Fund-F{case_index:04}."),
                    source(2, f"Fund-F{case_index:04} has an operating mandate of {old} years.")]
                changed = [*controls[:2], source(2, f"Fund-F{case_index:04} has an operating mandate of {new} years.")]
            else:
                old, new, unit = 0, 1, "record"
                question = f"Count all unique shipment records in the supplied Registry-R{case_index:04} material."
                note = f"Registry-R{case_index:04} filing notes: this note adds no shipment record."
                controls = [source(0, note)]
                changed = [source(0, note+f"\nShipment record SR-C{case_index:04}-1: route north; delivered amount 1 kg.")]
            control_reads = compile_variant(pair_id, "control", kind, question, controls, separated_hops=kind == "multihop_terminal")
            changed_reads = compile_variant(pair_id, "changed", kind, question, changed, separated_hops=kind == "multihop_terminal")
            traces.extend((*control_reads, *changed_reads))
            all_control_ids = [row["example_id"] for row in control_reads]
            all_changed_ids = [row["example_id"] for row in changed_reads]
            if kind == "multihop_terminal":
                control_reads, changed_reads = control_reads[-1:], changed_reads[-1:]
            pairs.append({"pair_id": pair_id, "kind": kind, "task_id": parent["task_id"],
                "parent_example_id": parent["example_id"], "split": "train", "source_partition": "constructed_train_reader_curriculum",
                "original_prompt": parent["messages"], "original_prompt_hash": parent["prompt_token_hash"],
                "actual_history": parent["actual_history"], "actual_history_sha256": parent["actual_history_sha256"],
                "original_quantity": {"value": str(old), "unit": unit}, "changed_quantity": {"value": str(new), "unit": unit},
                "control_example_ids": [row["example_id"] for row in control_reads],
                "changed_example_ids": [row["example_id"] for row in changed_reads],
                "all_control_example_ids": all_control_ids, "all_changed_example_ids": all_changed_ids,
                "expected_record_entities": [f"SR-C{case_index:04}-1"] if kind == "record_presence" else None,
                "constructed_control_sources": controls, "constructed_changed_sources": changed,
                "target_policy": "fresh teacher generation required; literal expected value is offline pair audit only"})
    return traces, pairs
