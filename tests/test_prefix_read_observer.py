"""Source-first callback ownership, recovered-step access counting and value gating."""
import asyncio
from dataclasses import replace
import json
from types import SimpleNamespace

from controller.context_schema import ContextBudget, EvidenceSet, EvidenceUnit, Need, SourceSpan, StepRecord, TaskState
from controller.evidence import make_evidence
from controller.prefix_retention import EmpiricalRetentionModel, PrefixReadObserver, TARGET_DEMAND_RULE
from controller.reading import build_read_plan, step_messages
from controller.schema import CompletionReceipt, ContextEpoch
from controller.session_store import SessionStore
from controller.templates import ByteTemplate, token_hash


class ReadRuntimeChannel:
    def __init__(self, fingerprint):
        self.fingerprint, self.restart_id = fingerprint, "engine-1"
        self.owners, self.calls = {}, []
    async def info(self):
        return SimpleNamespace(engine_restart_id=self.restart_id)
    def inspect(self, body):
        boundary = len(body["input_ids"])//64*64
        path_hash = token_hash(body["input_ids"][:boundary])
        pages = ["scope:"+str(index) for index in range(boundary//64)]
        return dict(state="OBSERVED", engine_restart_id=self.restart_id, cache_scope_hash="scope",
            prefix_hash=path_hash, boundary_tokens=boundary, page_size=64,
            source_attempt_id=body["attempt_id"], sources=body["sources"],
            gpu_page_ids=pages, host_page_ids=pages, reference_present=False,
            pool=dict(gpu_capacity_tokens=65536, free_gpu_tokens=32768,
                host_capacity_tokens=65536, free_host_tokens=32768),
            workload=dict(active_requests=0, queued_requests=0),
            residency=dict(ready_gpu_prefix_tokens=boundary, host_only_load_tokens=0,
                recoverable_prefix_tokens=boundary, inflight_h2d_tokens=0, inflight_d2h_tokens=0))
    async def _post(self, path, body):
        self.calls.append((path, body))
        if path.endswith("/status") and "input_ids" in body:
            return self.inspect(body)
        if path.endswith("/retain"):
            owner_id = body["owner_id"]
            ack = {**self.inspect(body), "owner_id": owner_id, "state": "RETAINED",
                "reference_present": True, "expires_at": 700}
            self.owners[owner_id] = ack
            return ack
        if path.endswith("/release"):
            self.owners.pop(body["owner_id"], None)
            return dict(owner_id=body["owner_id"], state="RELEASED", engine_restart_id=self.restart_id)
        if "owner_id" not in body:
            return dict(prefixes=list(self.owners.values()), engine_restart_id=self.restart_id)
        return self.owners.get(body["owner_id"], dict(owner_id=body["owner_id"], state="RELEASED"))


def material(template, *, turn=1, question="How do I apply?", native_attempt="native-read", version="v1"):
    block = make_evidence("Eligible applicants use the online form.", "source")
    block = replace(block, key=replace(block.key, version=version))
    span = SourceSpan("span", block.key, 0, len(block.text))
    unit = EvidenceUnit("unit", (span,), block.text, need_ids=("n",))
    state = TaskState("task", question, needs=(Need("n"),), current_turn_id=turn)
    plan = build_read_plan(state, EvidenceSet("set", (unit,), source_blocks=(block,)),
        "selected_sources", ContextBudget(), template=template, read_layout="source_first")
    task = plan.steps[0]
    rendered = template.render(step_messages(plan, task, (unit,)))
    receipt = CompletionReceipt(native_attempt, native_attempt, "business-epoch", 1, "engine-1",
        rendered.input_hash, "stop", True, inserted_cache_tokens=len(rendered.input_ids), session_ref_registered=True)
    record = StepRecord("turn:"+str(turn), task.step_id, task,
        ContextEpoch("business", 0, "business-epoch", 1, engine_restart_id="engine-1"),
        input_ids=rendered.input_ids, receipt=receipt)
    return plan, task, (unit,), rendered, record


def test_fixed_ttl_recovery_counts_actual_read_once_and_does_not_extend_ttl(tmp_path):
    async def run():
        template = ByteTemplate()
        engine = ReadRuntimeChannel(template.fingerprint)
        store = SessionStore(tmp_path/"state.sqlite")
        callback = PrefixReadObserver(store, engine, template, {"mode": "fixed_ttl", "ttl_seconds": 30})
        first = await callback(*material(template))
        recovered = await callback(*material(template))
        assert first["owner_id"] == recovered["owner_id"]
        assert len([path for path, _ in engine.calls if path.endswith("/retain")]) == 1
        assert len(store.prefix_accesses()) == 1
        record = store.prefix_owner(first["owner_id"])
        assert record.policy == "fixed_ttl_mechanism" and record.policy_choice["qualification_only"]
        repeated = await callback(*material(template, turn=4, question="Revisit the same policy.", native_attempt="read-again"))
        assert repeated["owner_id"] == first["owner_id"]
        accesses = store.prefix_accesses(first["owner_id"])
        assert len(accesses) == 2
        assert accesses[-1]["payload"]["features"]["prior_access_count"] == 1
        assert accesses[-1]["payload"]["features"]["revisit"]
        changed = await callback(*material(template, turn=5, native_attempt="read-v2", version="v2"))
        assert changed["owner_id"] != first["owner_id"]
        store.close()
    asyncio.run(run())


def model_asset(path, *, native_ms, retained_ms):
    rows = [dict(target_demand_rule=TARGET_DEMAND_RULE, features={"access_state": "first", "native_residency": "device", "pressure_concurrency": "normal:single"},
        cache_scope_hash="scope", horizon_seconds=30, reused=True,
        native_input_hashes=["fixed-input"], retained_input_hashes=["fixed-input"],
        native_reuse_cost_ms=native_ms, retained_reuse_cost_ms=retained_ms,
        native_window_cost_ms=native_ms, retained_window_cost_ms=retained_ms,
        retention_control_ms=0,
        gpu_page_count=1, host_page_count=1, page_size=64,
        native_gpu_token_seconds=0, retained_gpu_token_seconds=0,
        native_host_token_seconds=0, retained_host_token_seconds=0)]
    rows = [{**rows[0], "independent_sample_id": name, "independent_session_ids": [name]} for name in ("session-a", "session-b")]
    path.write_text(json.dumps(EmpiricalRetentionModel.fit(rows).to_dict()))
    return dict(mode="value", model_path=str(path), horizon_seconds=30,
        gpu_budget_tokens=65536, host_budget_tokens=65536, lambda_gpu=0, lambda_host=0)


def test_value_inspects_native_residency_before_protection_and_keeps_measured_gain(tmp_path):
    async def run():
        template = ByteTemplate()
        engine = ReadRuntimeChannel(template.fingerprint)
        store = SessionStore(tmp_path/"state.sqlite")
        settings = model_asset(tmp_path/"paired-model.json", native_ms=100, retained_ms=20)
        result = await PrefixReadObserver(store, engine, template, settings)(*material(template))
        paths = [path for path, _ in engine.calls]
        assert paths[0] == "/context/prefix/status" and paths[-1] == "/context/prefix/retain"
        assert result["retention_choices"][0]["expected_saved_ms"] == 80
        assert store.prefix_owner(result["owner_id"]).policy == "paired_native_value"
        store.close()
    asyncio.run(run())


def test_value_does_not_protect_when_native_already_has_the_same_expected_reuse_cost(tmp_path):
    async def run():
        template = ByteTemplate()
        engine = ReadRuntimeChannel(template.fingerprint)
        store = SessionStore(tmp_path/"state.sqlite")
        settings = model_asset(tmp_path/"paired-model.json", native_ms=20, retained_ms=20)
        result = await PrefixReadObserver(store, engine, template, settings)(*material(template))
        assert result["state"] == "UNRETAINED" and result["policy_choice"]["calibrated"]
        assert not engine.owners and not store.prefix_owners()
        assert all(not path.endswith("/retain") for path, _ in engine.calls)
        store.close()
    asyncio.run(run())


def test_partial_native_reuse_does_not_duplicate_exact_read_target_reward():
    from profiling.retention_trace import exact_target_demand, reuse_pages
    first = {"prefix_input_ids": [1, 2, 3, 4], "evidence_token_ranges": [[0, 4]], "fixed_system_prefix_tokens": 0}
    second = {**first, "prefix_input_ids": [1, 2, 5, 6]}
    future = dict(first)
    receipt = {"actual_device_hit_tokens": 2, "actual_host_loaded_tokens": 0}
    partial = reuse_pages(second, future, receipt, page_size=2, scope="scope")
    assert partial["shared_evidence_page_ids"] and partial["actual_device_evidence_page_ids"]
    assert exact_target_demand(first, future, page_size=2, scope="scope")
    assert not exact_target_demand(second, future, page_size=2, scope="scope")
