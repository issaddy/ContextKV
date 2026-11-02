"""Whole canonical-capture entry accounting, including failed native calls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import time
from urllib.parse import urlsplit

from bench.metrics import occupancy_integral
from bench.native_metrics import NativePoolSampler, pool_series
from bench.study_resources import StudyResourceSampler, integrate_physical, listening_processes
from controller.serialization import to_jsonable
from controller.session_store import Attempt, load_record


def load_resident_provider(path, template):
    import yaml
    from adapters.study_retrieval import create_study_retriever
    config = yaml.safe_load(Path(path).read_text())
    provider = create_study_retriever(config, template)
    return provider, to_jsonable(provider.manifest)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def capture_generation_summary(calls):
    completed = [call for call in calls if "receipt" in call]
    unresolved = [call for call in calls if "receipt" not in call]
    return dict(submitted=len(calls), native_completed=len(completed),
        observed_output_tokens=sum(len(call["result"]["output_ids"]) for call in completed),
        observed_input_tokens=sum(len(call["request"]["input_ids"]) for call in calls),
        output_tokens_complete=not unresolved,
        native_receipts=[call["receipt"] for call in completed],
        unresolved_native_attempts=[dict(attempt_id=call["request"]["attempt_id"],
            engine_session_id=call["request"]["session_id"],
            engine_restart_id=call["request"]["expected_restart_id"],
            input_tokens=len(call["request"]["input_ids"]), error=call.get("error"))
            for call in unresolved])


def capture_attempts(path, started_ns):
    if not path.exists():
        return []
    connection = sqlite3.connect(path.resolve().as_uri()+"?mode=ro", uri=True)
    try:
        attempts = [load_record(connection, Attempt, row[0]) for row in connection.execute(
            "SELECT payload FROM attempts ORDER BY rowid")]
        return [dict(attempt_id=item.attempt_id, session_id=item.session_id,
            request_id=item.request_id, stage=item.stage, error=item.error,
            reading_elapsed_ms=item.reading_elapsed_ms,
            answer_receipt=to_jsonable(item.result.receipt) if item.result is not None else None)
            for item in attempts if item.created_ns >= started_ns]
    finally:
        connection.close()


class CaptureRunAccounting:
    """Bracket controller initialization, all turns, failure and cleanup.

    Server startup precedes this command and belongs to the study launcher.
    Physical occupancy counts the declared device once and PSS process groups;
    CPU process_time covers this controller process, with scope stated in output.
    """
    def __init__(self, config, output_dir, gpu_device, *, native=None, physical=None,
                 resource_gpu_devices=None, resource_process_roots=()):
        self.config, self.output = config, Path(output_dir)
        self.native = native if native is not None else NativePoolSampler(config.engine.base_url,
            gpu_capacity=config.engine.gpu_capacity_tokens,
            host_enabled=config.engine.host_capacity_tokens > 0, model=config.model.name, interval=.1)
        self.physical = physical if physical is not None else StudyResourceSampler(
            tuple(resource_gpu_devices) if resource_gpu_devices is not None else (gpu_device,),
            tuple(sorted(set((*listening_processes((urlsplit(config.engine.base_url).port,)),
                              *resource_process_roots)))), interval=.1)
        self.calls = []
        self.current_source = None
        self.resident_model_manifest = None
        self.resident_model_loading = None
        self.case_outcomes = None

    def record_resident_models(self, path, manifest):
        self.resident_model_manifest = manifest
        self.resident_model_loading = dict(study_config=str(Path(path).resolve()),
            study_config_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            operation="actual create_study_retriever; models kept alive without retrieval-quality measurement",
            controller_role="controller_self_and_descendants; collector PID is provenance, not a reusable identity")

    async def __aenter__(self):
        await self.native.start()
        await self.physical.start()
        self.start = time.monotonic()
        self.started_ns = time.monotonic_ns()
        self.cpu_start = time.process_time()
        write_json(self.output/"capture-entry.json", dict(status="running",
            resource_window={"start": self.start}, clock="client_monotonic"))
        return self

    async def __aexit__(self, error_type, error, traceback):
        end = time.monotonic()
        cpu_end = time.process_time()
        await self.native.stop()
        await self.physical.stop()
        native = self.native.observation()
        physical = self.physical.observation()
        series, coverage = pool_series(native, self.start, end)
        state_path = Path(self.config.state_dir)/"sessions.sqlite3"
        report = dict(format="contextkv_canonical_capture_cost_v1", source="hardware",
            engine_kind=self.config.engine.kind,
            scope="canonical source_first READ/REDUCE/ANSWER collection entry",
            status="completed" if error is None else "failed",
            error=None if error is None else dict(type=error_type.__name__, message=str(error)),
            current_source=self.current_source, resource_window={"start": self.start, "end": end},
            resident_model_manifest=self.resident_model_manifest, case_outcomes=self.case_outcomes,
            resident_model_loading=self.resident_model_loading,
            elapsed_seconds=end-self.start,
            controller_cpu_seconds=cpu_end-self.cpu_start,
            cpu_scope="collector/controller process_time; engine CPU process time is not measured",
            physical_cost=integrate_physical(self.physical.samples, self.start, end),
            native_cost=dict(integrals=occupancy_integral(series), coverage=coverage),
            native_observation=native, physical_observation=physical,
            generations=capture_generation_summary(self.calls),
            attempts=capture_attempts(state_path, self.started_ns),
            wire_path="capture-wire.json", state_database=str(state_path.resolve()),
            cost_scope="controller construction through close, all successful and failed submitted generations, parsing and persisted collection artifacts; dedicated engine startup is external")
        write_json(self.output/"capture-wire.json", self.calls)
        write_json(self.output/"capture-entry.json", report)
        self.entry_manifest = dict(path="capture-entry.json", status=report["status"],
            sha256=hashlib.sha256((self.output/"capture-entry.json").read_bytes()).hexdigest(),
            submitted_generations=len(self.calls), scope=report["scope"])
        trace_path = self.output/"capture-trace.json"
        if trace_path.exists():
            trace = json.loads(trace_path.read_text())
            trace["entry_accounting"] = {**self.entry_manifest, "successful_turns": len(trace["turns"])}
            write_json(trace_path, trace)
        return False
