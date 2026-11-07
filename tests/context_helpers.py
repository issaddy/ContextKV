"""Shared CPU protocol fixtures for core property tests."""
from controller.context_schema import (ContextPlan, EvidenceSet,
    EvidenceUnit, Need, SourceSpan, TaskState)
from controller.evidence import make_evidence


def context_plan(question="How?", *, set_id="set", quality=.7):
    block = make_evidence("前文。Linux instructions。后文", "manual")
    start, end = 3, 21
    unit = EvidenceUnit("u", (SourceSpan("span:u", block.key, start, end),),
                        block.text[start:end])
    state = TaskState("task", question, needs=(Need("need"),), current_turn_id=1)
    selected = EvidenceSet(set_id, (unit,), quality_prediction=quality, source_blocks=(block,))
    return ContextPlan("plan:"+set_id, state, selected), block
