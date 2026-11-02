"""Frozen-reader support/utility teacher for bootstrapping and data collection."""
from __future__ import annotations

import json
from pathlib import Path
import time

import httpx

from adapters.support_model import SetPrediction, SupportPrediction
from controller.schema import Message
from controller.serialization import to_jsonable


class TeacherSupportBackend:
    def __init__(self, base_url, template, *, batch_size=8, window_tokens=32768,
                 output_tokens=2048, record_path=None):
        self.client = httpx.Client(base_url=base_url, timeout=180)
        self.template = template
        self.batch_size, self.window_tokens = batch_size, window_tokens
        self.output_tokens = output_tokens
        self.record_path = Path(record_path) if record_path else None
        self.manifest = {"backend": "frozen_qwen3_support_utility_teacher",
            "reader_model": "qwen3_32b_nonthinking", "base_url": base_url,
            "batch_size": batch_size, "window_tokens": window_tokens}
        self.usage = {"calls": 0, "examples": 0, "input_tokens": 0, "output_tokens": 0,
                      "compute_ms": 0.0, "support_ms": 0.0, "utility_ms": 0.0, "oversized_examples": 0}
        self.records = []
        self._sets = {}
        self._queries = {}

    def _generate(self, system, payload, schema, kind, output_tokens):
        messages = (Message("system", system), Message("user", json.dumps(payload, ensure_ascii=False)))
        rendered = self.template.render(messages)
        if len(rendered.input_ids)+output_tokens > self.window_tokens:
            self.usage["oversized_examples"] += 1
            return None
        started = time.perf_counter()
        response = self.client.post("/generate", json={"input_ids": list(rendered.input_ids),
            "stream": False, "sampling_params": {"temperature": 0, "max_new_tokens": output_tokens,
                                                 "json_schema": json.dumps(schema)}})
        response.raise_for_status()
        raw = response.json()
        elapsed = (time.perf_counter()-started)*1000
        self.usage["calls"] += 1
        self.usage["input_tokens"] += len(rendered.input_ids)
        self.usage["output_tokens"] += raw["meta_info"]["completion_tokens"]
        self.usage[kind+"_ms"] += elapsed
        self.usage["compute_ms"] += elapsed
        record = {"kind": kind, "raw_output": raw["text"], "meta_info": raw["meta_info"],
                  "input_tokens": len(rendered.input_ids), "elapsed_ms": elapsed,
                  "source_span_catalog": payload.get("span_catalog", {})}
        self.records.append(record)
        if self.record_path:
            self.record_path.parent.mkdir(parents=True, exist_ok=True)
            with self.record_path.open("a") as output:
                output.write(json.dumps(record, ensure_ascii=False)+"\n")
        return json.loads(raw["text"])

    def predict(self, examples):
        results = [None]*len(examples)
        for start in range(0, len(examples), self.batch_size):
            batch = examples[start:start+self.batch_size]
            for example in batch:
                self._queries.setdefault(example.task_id, {})[example.need_id] = example.query
            units = {unit.unit_id: unit for example in batch for unit in (*example.units, *example.dependencies)}
            original_spans = {span.span_id: span for unit in units.values() for span in unit.source_spans}
            spans = {f"s{index}": span for index, span in enumerate(original_spans.values())}
            aliases = {span.span_id: alias for alias, span in spans.items()}
            rendered_units = []
            for unit in units.values():
                rendered = to_jsonable(unit)
                for span in rendered["source_spans"]:
                    span["span_id"] = aliases[span["span_id"]]
                rendered_units.append(rendered)
            scalar = {"type": "number", "minimum": 0, "maximum": 1}
            item = {"type": "object", "properties": {
                **{name: scalar for name in ("applicability", "direct_support", "joint_support", "bridge_value")},
                "disposition": {"type": "string", "enum": ["complete", "partial", "background", "contradict"]},
                "span_ids": {"type": "array", "items": {"type": "string", "enum": list(spans)}},
                "missing_relation": {"type": "string"}},
                "required": ["applicability", "direct_support", "joint_support", "bridge_value",
                             "disposition", "span_ids", "missing_relation"], "additionalProperties": False}
            by_index = {}
            for index, example in enumerate(batch):
                allowed = [aliases[span.span_id] for unit in (*example.units, *example.dependencies)
                           for span in unit.source_spans]
                by_index[str(index)] = {**item, "properties": {**item["properties"],
                    "span_ids": {"type": "array", "items": {"type": "string", "enum": allowed}}}}
            schema = {"type": "object", "properties": {"predictions": {"type": "object",
                "properties": by_index, "required": list(by_index), "additionalProperties": False}},
                "required": ["predictions"], "additionalProperties": False}
            payload = {"queries": {example.need_id: example.query for example in batch},
                "units": rendered_units, "span_catalog": to_jsonable(spans), "examples": [{"index": index,
                    "need_id": example.need_id, "unit_ids": [u.unit_id for u in example.units],
                    "dependency_ids": [u.unit_id for u in example.dependencies]}
                    for index, example in enumerate(batch)]}
            output = self._generate("Score every indicated single/group of original sources for its need. "
                "History identifies the task and conditions, not facts. Distinguish complete, partial, "
                "background and contradict. Applicability tests current conditions and version. "
                "Joint support tests combined premises. A partial source can have high bridge value. "
                "Select supporting original span_ids and name missing relations. Return one prediction "
                "for every index in the requested JSON.", payload, schema, "support", self.output_tokens)
            if output is None:
                continue
            for index_text, value in output["predictions"].items():
                index = int(index_text)
                selected_ids = value.pop("span_ids")
                selected_spans = tuple(spans[span_id] for span_id in selected_ids)
                results[start+index] = SupportPrediction(source_spans=selected_spans, **value)
            self.usage["examples"] += len(batch)
        return tuple(results)

    def predict_set(self, state, units, supports, *, context=(0.0,)*8):
        key = (state.task_id, state.current_turn_id, state.raw_question,
               tuple(unit.unit_id for unit in units), tuple(context))
        if key in self._sets:
            return self._sets[key]
        schema = {"type": "object", "properties": {
            "quality": {"type": "number", "minimum": 0, "maximum": 1},
            "disposition": {"type": "string", "enum": ["answer", "partial", "retrieval_gap", "unanswerable", "underspecified"]}},
            "required": ["quality", "disposition"], "additionalProperties": False}
        output = self._generate("Predict answer quality of this complete evidence set for the current "
            "task and effective conditions. Combine complementary sources, penalize wrong versions, "
            "missing prerequisites and redundancy. History and prior answers are not sources. "
            "Use retrieval_gap when missing evidence could exist outside the selected set. "
            "Return continuous quality and answer disposition in JSON.",
            {"task": to_jsonable(state), "units": to_jsonable(units),
             "working_views": self._queries.get(state.task_id, {}),
             "support_graph": to_jsonable(supports), "candidate_context": context},
            schema, "utility", 256)
        if output is None:
            raise ValueError("teacher utility input exceeds the model window")
        result = SetPrediction(**output)
        self._sets[key] = result
        return result

    def release_task(self, task_id):
        self._queries.pop(task_id, None)
        for key in tuple(self._sets):
            if key[0] == task_id:
                del self._sets[key]

    def close(self):
        self.client.close()
