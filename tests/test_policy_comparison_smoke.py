import json

import pytest

from bench.evidence_policy_comparison import compare, frozen_inputs, frozen_reference
from controller.context_schema import ContextPlan, EvidencePool, EvidenceSet, Need, TaskState
from controller.serialization import to_jsonable
from training.reader_data import write_rows


@pytest.mark.asyncio
async def test_four_cells_keep_actual_e0_task_pool_and_route_own_p3_independently(tmp_path, monkeypatch):
    state = TaskState("actual", "Actual question", needs=(Need("actual-need"),), current_turn_id=1)
    plan = to_jsonable(ContextPlan("initial", state, EvidenceSet("actual-selection", ())))
    record = {"task_id": "task", "status": "completed", "question": "Actual question", "turn": 1,
        "conversation_id": "conversation", "actual_history": [], "context_plan": plan,
        "planning_trace": {"context_plans": [plan]}, "candidate_pool": to_jsonable(EvidencePool(())),
        "verified_task_state": {"must_not_be_used": "oracle"}}
    original, pool, origin = frozen_inputs(record)
    assert original.task_state == state and not pool.units
    assert origin["task_origin"] == "planning_trace.context_plans:matched_plan_id"
    assert "oracle" not in json.dumps(to_jsonable(original))
    assert frozen_reference({"task_id": "task"}, record)["input"] == [{"speaker": "user", "text": "Actual question"}]
    path = tmp_path/"actual.jsonl"
    write_rows(path, [record])
    calls = []
    async def frozen(*args, policy, reader):
        assert args[4][0]["actual_history"] == []
        calls.append((policy, reader, "frozen"))
        return [record]
    async def p3(*args, policy, reader, **kwargs):
        calls.append((policy, reader, "p3"))
        return []
    monkeypatch.setattr("bench.evidence_policy_comparison.frozen_cell", frozen)
    monkeypatch.setattr("bench.evidence_policy_comparison.own_p3_cell", p3)
    result = await compare({}, {}, dataset="diagnostic", split="dev", output=tmp_path/"comparison",
        phase="both", frozen_records=path, limit_states=1, limit_conversations=1)
    assert len(result) == len(calls) == 8
    assert set(calls) == {(policy, reader, phase) for policy in ("P0", "P1") for reader in ("R0", "R1")
                          for phase in ("frozen", "p3")}
