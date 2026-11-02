"""Compile the owned Task wire schema with the deployed native xgrammar."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

from adapters.history_state import TeacherTaskModel, boundary_ref_value
from adapters.query_selector import CompletionResult
from controller.context_schema import Condition, Need, RetrievalState, TaskState, TextRef
from controller.schema import OnlineTurn, TurnRecord


PROBE_QUESTION = "Use the next dependency 😀e\u0301中"


def schema_probe(history_characters=0):
    """Fifth-turn needs own a dependency; sixth-turn aliasing must preserve it."""
    state = TaskState("task-owner", "Linux", conditions=(Condition("five:c0",
        value_spans=(TextRef(5, "user", 0, 5),), relation_type="PLATFORM"),),
        needs=(Need("five:n0", condition_ids=("five:c0",)),
               Need("five:n1", condition_ids=("five:c0",), predecessor_need_ids=("five:n0",))), current_turn_id=5)
    actual_answer = "A"*history_characters if history_characters else "Actual fifth answer"
    memory = RetrievalState(5, (TurnRecord(5, "Linux", actual_answer),), (state,))
    captured = {}
    class CaptureClient:
        manifest = {"kind": "schema_probe_only"}
        def complete(self, messages, *, response_format):
            captured["schema"] = response_format["json_schema"]["schema"]
            captured["wire"] = {"relation": "followup", "resume_task_id": "S0", "history_spans": [
                    {"turn_id": 5, "speaker": "assistant", "boundary_chars": [len(actual_answer)-1, len(actual_answer)-1]}],
                "conditions": {}, "condition_removals": [], "needs": {
                    "n0": {"anchor_spans": [], "subject_refs": [{"turn_id": 5, "speaker": "user", "boundary_chars": [4, 0]}], "operation": "lookup",
                           "condition_ids": ["C0"], "predecessor_need_ids": ["N1"], "scope": "selected_sources"},
                    "n1": {"anchor_spans": [], "subject_refs": [], "operation": "lookup",
                           "condition_ids": ["C0"], "predecessor_need_ids": ["n0"], "scope": "selected_sources"}},
                "version_intent": {"mode": "current", "temporal_relation": "at", "document_families": [],
                    "version_ids": [], "resolved_version_ids": [], "time_refs": [], "source_turns": []}}
            selected = next(branch["properties"] for branch in captured["schema"]["anyOf"] if
                branch["properties"]["resume_task_id"]["enum"] == ["S0"]
                and branch["properties"]["conditions"]["required"] == [])
            captured["wire"] = {name: captured["wire"][name] for name in selected}
            return CompletionResult("completed", json.dumps(captured["wire"]), "stop", 0,
                                    {"completion_tokens": 0})
    prediction = TeacherTaskModel(CaptureClient()).predict(OnlineTurn("probe", "six", 6, "manual",
        PROBE_QUESTION, (), (), "P3"), (), memory)
    assert prediction.needs[0].predecessor_need_ids == ("five:n1",)
    assert prediction.needs[1].predecessor_need_ids == ("six:n0",)
    assert prediction.needs[0].subject_refs == (TextRef(5, "user", 0, 5),)
    return captured["schema"], captured["wire"]


def native_probe(scope_path, *, history_characters=0):
    import xgrammar as xgr
    from transformers import AutoTokenizer
    from engine_patches.contextkv_model_scope import load_model_scope
    schema_started = time.monotonic()
    schema, wire = schema_probe(history_characters)
    schema_build_ms = (time.monotonic()-schema_started)*1000
    scope = load_model_scope(scope_path)
    tokenizer = AutoTokenizer.from_pretrained(scope.model_path, local_files_only=True)
    info = xgr.TokenizerInfo.from_huggingface(tokenizer,
        vocab_size=scope.manifest()["record"]["config"]["vocab_size"])
    started = time.monotonic()
    compiled = xgr.GrammarCompiler(info).compile_json_schema(json.dumps(schema))
    compile_ms = (time.monotonic()-started)*1000
    accepted = xgr.GrammarMatcher(compiled).accept_string(json.dumps(wire))
    invalid = json.loads(json.dumps(wire))
    invalid["needs"]["n0"]["predecessor_need_ids"] = ["n1"]
    forward_accepted = xgr.GrammarMatcher(compiled).accept_string(json.dumps(invalid))
    tests = [{"name": "valid_owned_graph", "accepted": accepted, "expected": True},
             {"name": "forward_need_ref", "accepted": forward_accepted, "expected": False}]
    represented = set()
    for first in range(5):
        for second in range(5):
            candidate = json.loads(json.dumps(wire))
            ref = {"turn_id": 5, "speaker": "user", "boundary_chars": [first, second]}
            candidate["needs"]["n0"]["subject_refs"] = [ref]
            result = xgr.GrammarMatcher(compiled).accept_string(json.dumps(candidate))
            tests.append({"name": f"boundary:{first}:{second}", "accepted": result, "expected": True})
            if result:
                internal = boundary_ref_value(ref)
                represented.add((internal["start"], internal["end"]))
    for first, second in ((-1, 0), (0, 5), (5, 6)):
        candidate = json.loads(json.dumps(wire))
        candidate["needs"]["n0"]["subject_refs"] = [{"turn_id": 5, "speaker": "user", "boundary_chars": [first, second]}]
        tests.append({"name": f"bounds:{first}:{second}",
            "accepted": xgr.GrammarMatcher(compiled).accept_string(json.dumps(candidate)), "expected": False})
    for index, ref in enumerate(({"turn_id": 0, "speaker": "user", "boundary_chars": [0, 0]},
                {"turn_id": 6, "speaker": "assistant", "boundary_chars": [0, 0]},
                {"turn_id": 5, "speaker": "robot", "boundary_chars": [0, 0]},
                {"turn_id": 5, "speaker": "user", "boundary_chars": [.5, 1]})):
        candidate = json.loads(json.dumps(wire))
        candidate["needs"]["n0"]["subject_refs"] = [ref]
        tests.append({"name": f"owner_or_integer:{index}",
            "accepted": xgr.GrammarMatcher(compiled).accept_string(json.dumps(candidate)), "expected": False})
    for anchors in ([], [0], [0, 1, 2]):
        candidate = json.loads(json.dumps(wire))
        candidate["needs"]["n0"]["subject_refs"] = [{"turn_id": 5, "speaker": "user", "boundary_chars": anchors}]
        tests.append({"name": f"array_length:{len(anchors)}",
            "accepted": xgr.GrammarMatcher(compiled).accept_string(json.dumps(candidate)), "expected": False})
    for text in ("😀", "e\u0301"):
        offset = PROBE_QUESTION.index(text)
        ref = {"turn_id": 6, "speaker": "user", "boundary_chars": [offset+len(text)-1, offset]}
        internal = boundary_ref_value(ref)
        assert PROBE_QUESTION[internal["start"]:internal["end"]] == text
        candidate = json.loads(json.dumps(wire))
        candidate["needs"]["n0"]["subject_refs"] = [ref]
        tests.append({"name": "unicode:"+text,
            "accepted": xgr.GrammarMatcher(compiled).accept_string(json.dumps(candidate)), "expected": True})
    candidate = json.loads(json.dumps(wire))
    candidate["needs"]["n0"]["subject_refs"] = [{"turn_id": 5, "speaker": "user", "start": 2, "end": 2}]
    tests.append({"name": "legacy_flat_ref", "accepted": xgr.GrammarMatcher(compiled).accept_string(json.dumps(candidate)), "expected": False})
    failures = [item for item in tests if item["accepted"] != item["expected"]]
    boundaries = [item for item in tests if item["name"].startswith("boundary:")]
    return {"measurement_kind": "native_xgrammar_cpu_probe", "model_forward_executed": False,
        "xgrammar_version": version("xgrammar"), "schema_bytes": len(json.dumps(schema).encode()),
        "schema_build_ms": schema_build_ms, "compile_ms": compile_ms, "probe_history_characters": history_characters,
        "wire_schema": "owned_task_graph_v6_included_boundary_chars",
        "text_ref_owner_branches": len(schema["$defs"]["TextRef"]["anyOf"]),
        "valid_boundary_pairs_accepted": sum(item["accepted"] for item in boundaries),
        "represented_nonempty_intervals": len(represented),
        "malformed_boundary_arrays_rejected": sum(not item["accepted"] for item in tests if item["name"].startswith("array_length:")),
        "unicode_boundary_refs_accepted": sum(item["accepted"] for item in tests if item["name"].startswith("unicode:")),
        "out_of_bounds_spans_rejected": sum(not item["accepted"] for item in tests if item["name"].startswith("bounds:")),
        "invalid_owner_or_noninteger_rejected": sum(not item["accepted"] for item in tests if item["name"].startswith("owner_or_integer:")),
        "fifth_to_sixth_aliasing": "passed",
        "valid_owned_need_graph_accepted": accepted, "unmaterialized_forward_reference_rejected": not forward_accepted,
        "qualification_passed": not failures, "qualification_failures": failures, "qualification_cases": tests}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="configs/model_scopes/qwen3-32b.local.json")
    parser.add_argument("--output")
    parser.add_argument("--history-characters", type=int, default=0)
    args = parser.parse_args(argv)
    result = native_probe(args.scope, history_characters=args.history_characters)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({name: result[name] for name in ("schema_bytes", "schema_build_ms", "compile_ms",
        "text_ref_owner_branches", "probe_history_characters", "valid_owned_need_graph_accepted",
        "unmaterialized_forward_reference_rejected", "valid_boundary_pairs_accepted", "represented_nonempty_intervals",
        "malformed_boundary_arrays_rejected", "unicode_boundary_refs_accepted", "out_of_bounds_spans_rejected", "invalid_owner_or_noninteger_rejected",
        "qualification_passed")}, ensure_ascii=False), flush=True)
    assert result["qualification_passed"], result["qualification_failures"]


if __name__ == "__main__":
    main()
