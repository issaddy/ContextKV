"""Exercise bound document programs through the real controller on CPU.

The fixtures supply correct field bindings; this measures execution semantics,
not a model's ability to discover those bindings or benchmark answer quality.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from adapters.document_compute import DocumentIndex, DocumentProgram, Field, Projection
from controller import cooperative
from controller.evidence import make_evidence
from controller.schema import EvidencePool, EvidenceUnit, Need, TaskState

ROOT = Path(__file__).resolve().parents[1]
BASE = "8f9fd328458d7df23e63da7f99b01822c2236679"


def baseline_reader():
    source = subprocess.check_output(
        ["git", "show", f"{BASE}:controller/cooperative.py"], cwd=ROOT, text=True
    )
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == "CooperativeController")
    method = next(node for node in cls.body
                  if isinstance(node, ast.AsyncFunctionDef) and node.name == "read_required")
    namespace = dict(vars(cooperative))
    exec(compile(ast.Module(body=[method], type_ignores=[]), "baseline_read_required", "exec"), namespace)
    return namespace["read_required"]


async def dispatch(reader, index, program, *, incomplete=False):
    block = index.blocks[0]
    table = next(iter(index.tables.values()))
    unit = EvidenceUnit("whole-table", (table.source,), block.text, body_tokens=64)
    task = TaskState("fixture", "Group the stores by region.",
                     needs=(Need("need", operation=program.operation),),
                     coverage_scope="all_supplied_documents")
    state = SimpleNamespace(
        query=SimpleNamespace(task=task), reused_claims=(), population_units=(unit,),
        pool=EvidencePool((unit,), (block,)), document_index=index,
        programs=(replace(program, unresolved_conditions=("missing-filter",)) if incomplete else program,),
        computations=(), processed_tables=set(),
    )
    controller = cooperative.CooperativeController.__new__(cooperative.CooperativeController)
    calls = 0

    async def complete(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("a fully bound fixture must not request model generation")

    await reader(controller, state, None, complete)
    return {"computations": len(state.computations), "model_calls": calls,
            "values": [item["value"] for result in state.computations for item in result["results"]]}


async def run():
    text = ("| Store | Region | Revenue |\n|---|---|---|\n"
            "| A | East | 10 |\n| B | West | 20 |\n| C | East | 30 |\n")
    index = DocumentIndex.from_blocks((make_evidence(text, "stores"),))
    table_id = next(iter(index.tables))
    projection = Projection(table_id, (Field("store", 0), Field("region", 1), Field("revenue", 2)))
    previous = baseline_reader()
    rows = []
    for operation in ("count", "sum", "mean", "min", "max", "group"):
        program = DocumentProgram("need", (projection,), (table_id,), operation=operation,
                                  value_field="revenue" if operation in {"sum", "mean", "min", "max"} else "",
                                  entity_fields=("store",), group_by=("region",) if operation == "group" else ())
        direct = index.execute(program)
        before = await dispatch(previous, index, program)
        after = await dispatch(cooperative.CooperativeController.read_required, index, program)
        expected = [claim.value for claim in direct.conclusions]
        rows.append({"operation": operation, "expected_values": expected,
                     "baseline": before, "working_tree": after})
        if operation != "group":
            assert before == after
        assert after["computations"] == 1
        assert after["values"] == expected
        assert before["model_calls"] == after["model_calls"] == 0
    partial = await dispatch(cooperative.CooperativeController.read_required, index, program, incomplete=True)
    assert partial["computations"] == 0
    return {"source_revision": BASE, "measurement": "actual CPU controller and document execution",
            "data": "three synthetic source rows, explicit correct bindings, no model weights",
            "cases": rows, "unresolved_condition": partial}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/cpu_document_execution_20261116.json")
    args = parser.parse_args()
    result = asyncio.run(run())
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
