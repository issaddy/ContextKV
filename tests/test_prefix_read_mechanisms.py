"""Controlled semantic probe construction; real generation is a GPU experiment."""
from bench.prefix_read_mechanisms import CASES, build_case, material, qualify
from controller.reading import stable_prefix_input_ids, step_messages
from controller.schema import SessionSnapshot, TurnRecord
from controller.templates import ByteTemplate, longest_common_prefix, token_hash


def test_semantic_probe_changes_needs_and_versions_while_preserving_only_eligible_prefixes():
    template = ByteTemplate()
    snapshot = SessionSnapshot("policy-session")
    prefixes, full_hashes = [], []
    needs = []
    for case in CASES:
        plan = build_case(case, snapshot, template)
        task = plan.steps[0]
        units = plan.selected_set.selected_units
        rendered = template.render(step_messages(plan, task, units, snapshot=snapshot))
        prefixes.append(stable_prefix_input_ids(template, plan, task, units, rendered))
        full_hashes.append(rendered.input_hash)
        needs.append(plan.task_state.needs[0].need_id)
        snapshot = SessionSnapshot("policy-session", version=snapshot.version+1,
            raw_history=(*snapshot.raw_history, TurnRecord(len(snapshot.raw_history)+1,
                case.question, "Unit test history; no model quality evidence.")))
    assert needs[0] != needs[1] and prefixes[0] == prefixes[1] == prefixes[3]
    assert full_hashes[0] != full_hashes[1]
    assert prefixes[0] != prefixes[2]
    assert longest_common_prefix(prefixes[0], prefixes[2]) < min(len(prefixes[0]), len(prefixes[2]))
    old, new = material(CASES[0])[0], material(CASES[2])[0]
    assert old.key.document_id == new.key.document_id
    assert old.key.version != new.key.version and old.key.content_sha256 != new.key.content_sha256
    assert "W-7" in old.text and "W-9" in new.text


def test_semantic_qualification_requires_actual_read_epoch_retirement_and_owner_presence():
    prefixes = ((1, 2, 3, 4), (1, 2, 3, 4), (1, 9, 8, 7), (1, 2, 3, 4))
    rows = []
    for index, (case, prefix) in enumerate(zip(CASES, prefixes, strict=True)):
        proof = dict(read_epoch_state="RETIRED", independent_owner_after_read_retire={"reference_present": True},
            full_input_hash=str(index), page_prefix_hash=token_hash(prefix), boundary_tokens=4,
            stable_prefix_input_ids=prefix)
        rows.append(dict(case=case.name, status="completed", qa_increment=1, version_increment=1,
                         quality_literal_checks={"expected": True}, read_proofs=[proof]))
    assert all(qualify(rows, "fixed_ttl").values())
    rows[1]["read_proofs"][0]["independent_owner_after_read_retire"]["reference_present"] = False
    assert not qualify(rows, "fixed_ttl")["independent_owner_survives_read_epoch"]
    assert all(qualify(rows, "native").values())
    assert not qualify(rows[:1], "native")["all_cases_completed"]
