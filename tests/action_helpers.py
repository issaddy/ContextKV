"""Shared CPU protocol fixtures for core property tests."""
from dataclasses import replace
import json
from bench.mock_engine import MockEngine
from controller.context_schema import (
    ContextBudget, EvidenceSet, EvidenceUnit, Need, ReadingOutcome,
    SourceSpan, TaskState, TextRef,
)
from controller.evidence import make_evidence
from controller.execution import ReadingExecutor
from controller.plan_policy import CompleteCost
from controller.rolling_policy import compile_step
from controller.schema import SessionSnapshot, TurnRecord
from controller.service import Controller
from controller.session_store import SessionStore
from controller.templates import ByteTemplate
from training.plan_gain_protocol import (
    GapRetrievalRequest, ZERO_AUX, build_observed_action_plans, gap_query, observed_state_binding,
)
from training.plan_gain_rollout import GenerationBudget, bounded_engine


class Reader(MockEngine):
    def __init__(self):
        super().__init__()
        self._sampling_params = {"temperature": 0}
        self.inputs = []
        self.empty_reads = False

    async def info(self):
        return replace(await super().info(), engine_kind="sglang")

    async def quote_batch(self, *args, **kwargs):
        batch = await super().quote_batch(*args, **kwargs)
        return replace(batch, gpu_capacity_tokens=128000, active_requests=0, queued_requests=0,
            quotes=tuple(replace(quote, ready_gpu_prefix_tokens=0, host_only_load_tokens=0)
                         for quote in batch.quotes))

    async def generate(self, *, response_schema=None, **kwargs):
        rendered = bytes(kwargs["input_ids"]).decode()
        self.inputs.append((response_schema is not None, rendered))
        if response_schema is not None:
            payload = json.loads(rendered.split("<|user|>\n")[-1].split("\n<|end|>")[0])
            facts = [] if self.empty_reads else [{"need_id": "n", "text": unit["unit_id"]+" qualifies",
                "entity": unit["unit_id"], "relation": "qualifies", "value": "1", "unit": "",
                "conditions": [], "span_ids": [span["span_id"] for span in unit["source_spans"]],
                "parent_claim_ids": [], "followups": ([{"need_id": "n", "missing_relation": "qualifying record",
                    "bound_entities": ["R3"], "temporal_scope": "current"}] if unit["unit_id"] == "R1" else [])}
                for unit in payload["units"]]
            self.response_text = json.dumps({"format_version": 2, "claims": facts,
                "support": [], "conflicts": [], "open_needs": []})
        else:
            self.response_text = "Known qualifying records are supported by the supplied evidence."
        return await super().generate(**kwargs)


class Resources:
    def __init__(self):
        self.online = self.judge = self

    async def start(self):
        pass

    async def close(self):
        pass

    def observation(self):
        return {"source": "specified protocol fixture"}

    async def finish(self, role, start, end):
        return {"gpu_gib_seconds": end-start, "host_gib_seconds": 2*(end-start),
                "window": {"start": start, "end": end}, "complete_window_coverage": True}


def material(name):
    block = make_evidence(name+" qualifies.", name)
    return block, EvidenceUnit(name, (SourceSpan(name, block.key, 0, len(block.text)),), block.text)


async def frozen_fixture(tmp_path):
    from controller.reading import build_read_plan
    template, native = ByteTemplate(), Reader()
    pairs = (material("R1"), material("R2"))
    blocks, units = zip(*pairs)
    state = TaskState("task", "Count qualifying records.", needs=(Need("n", operation="count"),),
        working_history=(TextRef(1, "user", 0, 24),), current_turn_id=2)
    history = (TurnRecord(1, "Use the frozen condition", "Confirmed"),)
    snapshot = SessionSnapshot("capture", raw_history=history)
    plan = build_read_plan(state, EvidenceSet("selected", units, source_blocks=blocks), "selected_sources",
        ContextBudget(window_tokens=16000, read_input_tokens=10000, read_output_tokens=2048,
            output_tokens=256, body_tokens=3000), template=template, snapshot=snapshot,
        read_packing="source_windows")
    store = SessionStore(tmp_path / "state.db")
    initial = store.read_snapshot("capture")
    snapshot = store.sync_history("capture", history, initial.version)
    attempt = store.begin_attempt("capture", "capture", "hash", state.raw_question, snapshot.version)
    empty = ReadingOutcome(plan, auxiliary_usage=dict(ZERO_AUX))
    task = plan.steps[0]
    candidate = compile_step(plan, task, empty, snapshot, template, outputs=(task.output_budget,))[0]
    frozen = await ReadingExecutor(store, native, template).execute_one(
        attempt.attempt_id, snapshot, plan, task, empty, candidate=candidate)
    spent = sum(len(record.input_ids)+len(record.output_ids) for record in frozen.steps)
    paid = CompleteCost(500, 2000, 3000, spent+7, 0)
    checkpoint = template.fingerprint.model_revision
    observed = observed_state_binding(frozen, snapshot, paid_cost=paid,
        native_condition={"cache_state": "specified-fixture", "quote_seq": 1},
        reader_checkpoint=checkpoint, reader_profile_id="fixture-profile")
    observed["resident_model_manifest_digest"] = "specified-fixture"
    request = GapRetrievalRequest(frozen.open_needs, gap_query(state, frozen.open_needs),
        {"implementation": "specified-fixture"}, {"max_units": 1, "max_passages": 1})
    plans, omitted = build_observed_action_plans(frozen, snapshot, template,
        reader_checkpoint=checkpoint, gap_request=request)
    assert not omitted
    service = Controller(store, native, template=template)
    budget = GenerationBudget(64, tmp_path / "calls.jsonl")
    service.engine = bounded_engine(native, budget, tmp_path / "wire.json")
    binding = {"reader_checkpoint": checkpoint, "reader_profile_id": "fixture-profile",
        "scorer_identity": {"kind": "specified-fixture"}, "split": "train", "measurement_kind": "hardware",
        "resident_model_manifest": {"source": "specified-fixture"},
        "comparison_budget": {"window_tokens": 16000, "answer_output_tokens": 256},
        "collection_mode": "observed_actions", "counterfactual_cache_start": "observed", "cohort_split": "train",
        "cohort_manifest_sha256": "specified-fixture"}
    row = {"task_id": "task", "split": "train", "conversation_id": "fixture", "question": state.raw_question,
           "snapshot_id": "demo-v1", "turn": 2, "rollout_policy": "dense5_rebuild_qwen3_32b_nonthinking"}
    preparation = {"shared_cost_id": "specified-preparation", "complete_ms": 0,
        "input_tokens": 0, "output_tokens": 0, "gpu_gib_seconds": 0, "host_gib_seconds": 0}
    return service, native, budget, frozen, snapshot, observed, request, plans, binding, row, preparation
