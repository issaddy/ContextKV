"""Compile the owned Task wire schema with the deployed native xgrammar."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

from adapters.history_state import TeacherTaskModel
from adapters.query_selector import CompletionResult
from controller.context_schema import Condition, Need, RetrievalState, TaskState, TextRef
from controller.schema import OnlineTurn, TurnRecord


def schema_probe():
    """Fifth-turn needs own a dependency; sixth-turn aliasing must preserve it."""
    state = TaskState("task-owner", "Linux", conditions=(Condition("five:c0",
        value_spans=(TextRef(5, "user", 0, 5),), relation_type="PLATFORM"),),
        needs=(Need("five:n0", condition_ids=("five:c0",)),
               Need("five:n1", condition_ids=("five:c0",), predecessor_need_ids=("five:n0",))), current_turn_id=5)
    memory = RetrievalState(5, (TurnRecord(5, "Linux", "Actual fifth answer"),), (state,))
    captured = {}
    class CaptureClient:
        manifest = {"kind": "schema_probe_only"}
        def complete(self, messages, *, response_format):
            captured["schema"] = response_format["json_schema"]["schema"]
            captured["wire"] = {"relation": "followup", "resume_task_id": "S0", "history_spans": [],
                "conditions": {}, "condition_removals": [], "needs": {
                    "n0": {"anchor_spans": [], "subject_refs": [], "operation": "lookup",
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
        "Use the next dependency", (), (), "P3"), (), memory)
    assert prediction.needs[0].predecessor_need_ids == ("five:n1",)
    assert prediction.needs[1].predecessor_need_ids == ("six:n0",)
    return captured["schema"], captured["wire"]


def native_probe(scope_path):
    import xgrammar as xgr
    from transformers import AutoTokenizer
    from engine_patches.contextkv_model_scope import load_model_scope
    schema, wire = schema_probe()
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
    assert accepted and not forward_accepted
    return {"measurement_kind": "native_xgrammar_cpu_probe", "model_forward_executed": False,
        "xgrammar_version": version("xgrammar"), "schema_bytes": len(json.dumps(schema).encode()),
        "compile_ms": compile_ms, "fifth_to_sixth_aliasing": "passed",
        "valid_owned_need_graph_accepted": accepted, "unmaterialized_forward_reference_rejected": not forward_accepted}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", default="configs/model_scopes/qwen3-32b.local.json")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = native_probe(args.scope)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
