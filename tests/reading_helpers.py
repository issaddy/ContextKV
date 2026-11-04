"""Shared CPU protocol fixtures for core property tests."""
import json
from bench.mock_engine import MockEngine
from controller.context_schema import (EvidenceUnit, SourceSpan)
from controller.evidence import make_evidence


class ScriptedReadingEngine(MockEngine):
    def __init__(self, readings):
        super().__init__()
        self.readings = iter(readings)
        self.constraints = []

    async def generate(self, *, response_schema=None, **kwargs):
        self.constraints.append((response_schema, kwargs.get("citation_ids")))
        if response_schema and response_schema["properties"]["format_version"]["const"] == 7:
            self.response_text = json.dumps({"format_version": 7, "views": {
                alias: {parent: decision.get("const", True)
                        for parent, decision in view["properties"].items()}
                for alias, view in response_schema["properties"]["views"]["properties"].items()}})
        else:
            self.response_text = json.dumps(next(self.readings)) if response_schema else "Supported final answer."
        return await super().generate(**kwargs)


def fact(span, entity="item", value="1", unit="", need_id="n"):
    return {"need_id": need_id, "text": entity, "entity": entity, "relation": "quantity",
            "value": value, "unit": unit, "conditions": [], "version": "v1",
            "span_ids": [span], "parent_claim_ids": []}


def read_output(*facts, support=("n",), open_needs=()):
    return {"claims": list(facts), "support": list(support), "conflicts": [],
            "open_needs": list(open_needs)}


def material(text, name="a"):
    block = make_evidence(text, name)
    span = SourceSpan(name, block.key, 0, len(text))
    unit = EvidenceUnit(name, (span,), text, body_tokens=len(text), need_ids=("n",))
    return block, unit
