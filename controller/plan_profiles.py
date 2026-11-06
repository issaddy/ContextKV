"""Reader/cache-bound execution profiles and measured request resource spend."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .execution import auxiliary_tokens
from .plan_policy import (CompleteCost, NativePlanningState, ResourceForecast, StepForecast,
                          native_selection_cost)
from .reading import SOURCE_SYSTEM, step_generation_kind, step_system
from .schema import ConfigFingerprint
from .serialization import decode


def _parent_claim_ceiling(plan, task, state, bound):
    from .claim_capacity import claim_count_ceiling
    completed = {result.step_id: result for result in state.results}
    total = 0
    for parent in plan.steps:
        source = parent.task
        if source.step_id not in task.predecessor_step_ids:
            continue
        if source.step_id in completed:
            total += len(completed[source.step_id].claims)
        elif (plan.context_plan.reading_mode == "typed_reduce" and source.kind == "REDUCE"
              and source.operation in {"union", "count", "sum", "group"}):
            # Each deterministic CPU operation returns one derived claim.
            total += 1
        else:
            total += (claim_count_ceiling((source.output_budget,), bound)
                      if bound is not None else source.output_budget)
    return total


def _prediction_range(row):
    return row.get("prediction_bin", row["input_range"])


def _profile_prediction(row, inputs):
    observed = row.get("observed_input_range", row["input_range"])
    return {"profile_id": row["profile_id"], "input_value": inputs,
        "observed_input_range": observed, "prediction_bin": _prediction_range(row),
        "within_observed_input_range": observed[0] <= inputs <= observed[1],
        "source_profile_ids": row.get("source_profile_ids", [row["profile_id"]]),
        "observed_cache_ranges": row.get("observed_cache_ranges"),
        "queried_cache_state": row.get("queried_cache_state"),
        "cache_conditioning": row["cache_conditioning"],
        "concurrency_inference": row.get("concurrency_inference"),
        "phase_envelope_latency_ms": row["latency_ms"],
        "mean_gpu_gib": row["gpu_gib_ms"]/row["latency_ms"],
        "mean_host_gib": row["host_gib_ms"]/row["latency_ms"],
        "cost_basis": "empirical_profile_prediction; not a new measured phase"}


class ExecutionProfiles:
    """Measured bin profiles; uncovered inputs are not calibrated choices.

    Each row declares its input range, output budget, active request range,
    prefix-ready/host-load ranges, actual prompt hash, hardware observations,
    full-step latency and physical memory integrals. Future inputs are priced
    at their declared input ceiling, with no assumed future cache hit.
    """
    def __init__(self, data):
        if data["schema_version"] != 1 or data["measurement_kind"] != "hardware":
            raise ValueError("execution profiles require measured hardware step observations")
        self.data = data
        self.fingerprint = decode(ConfigFingerprint, data["fingerprint"])
        self.claim_capacity = data.get("claim_capacity")
        if self.claim_capacity is not None:
            from .context_schema import ReadTask
            from .reading import output_schema
            bound = self.claim_capacity
            required = output_schema(ReadTask("capacity", "READ", ("need",)))["properties"]["claims"]["items"]["required"]
            grammar = bound["grammar_contract"]
            if (bound["tokenizer_hash"] != self.fingerprint.tokenizer_hash or
                    bound["required_keys"] != sorted(required) or
                    grammar["protocol_format_version"] != 2 or grammar["backend"] != "xgrammar"):
                raise ValueError("claim capacity must bind the actual tokenizer and native claim-key protocol")
        if not data["cache_config_hash"] or not data["reader_checkpoint"] or not data["reader_profile_id"]:
            raise ValueError("execution profiles require reader and actual cache identity")
        for row in data["rows"]:
            if not row["observation_ids"] or not row["source_files"]:
                raise ValueError("execution profile rows require actual observation/file provenance")
            scope = "answer_until_commit" if row["generation_kind"] == "ANSWER" else "step_wall"
            if row["cost_scope"] != scope:
                raise ValueError("execution profile scope must include the complete step/response tail")
            if row["generation_kind"] == "CPU_REDUCE" and (row["input_measure"] != "claim_count"
                    or not row["operation"]):
                raise ValueError("CPU profiles bind a measured operation and claim-count range")
            if any(row[name] is None or row[name] < 0 for name in (
                    "latency_ms", "gpu_gib_ms", "host_gib_ms", "gpu_peak_gib", "host_peak_gib")):
                raise ValueError("execution profiles require measured complete physical costs")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def bind(self, quotes, reader_checkpoint, reader_profile_id):
        if (reader_checkpoint != self.data["reader_checkpoint"] or
                reader_profile_id != self.data["reader_profile_id"] or
                quotes.fingerprint != self.fingerprint or
                quotes.cache_config_hash != self.data["cache_config_hash"]):
            raise ValueError("execution profile does not match the observed reader/cache fingerprint")

    def row(self, kind, action, inputs, output, concurrency, ready, host, prompt, *, operation=None,
            start_definition=None):
        if not 1 <= concurrency <= 8:
            return None
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        matching = [row for row in self.data["rows"]
            if row["generation_kind"] == kind and row["context_action"] == action
            and row["output_budget"] == output
            and _prediction_range(row)[0] <= inputs <= _prediction_range(row)[1]
            and (kind != "CPU_REDUCE" or row["operation"] == operation)
            and (start_definition is None or row["start_definition"] == start_definition)
            and row["system_prompt_sha256"] == prompt_hash]
        if not matching:
            return None
        observed = [row for row in matching
            if row["ready_gpu_range"][0] <= ready <= row["ready_gpu_range"][1]
            and row["host_load_range"][0] <= host <= row["host_load_range"][1]
            and row["concurrency_range"][0] <= concurrency <= row["concurrency_range"][1]]
        if observed:
            def geometry(row):
                return (_prediction_range(row), row["ready_gpu_range"],
                        row["host_load_range"], row["concurrency_range"])

            narrowest = min(observed, key=lambda row: (
                tuple(bounds[1]-bounds[0] for bounds in geometry(row)), row["profile_id"]))
            matching = [row for row in observed if geometry(row) == geometry(narrowest)]
        else:
            narrowest = min(matching, key=lambda row: (
                _prediction_range(row)[1]-_prediction_range(row)[0], _prediction_range(row), row["profile_id"]))
            matching = [row for row in matching if _prediction_range(row) == _prediction_range(narrowest)]
        observed_q = max(row["concurrency_range"][1] for row in matching)
        multiplier = max(1.0, concurrency/observed_q)
        sources = sorted(row["profile_id"] for row in matching)
        identity = (sources[0] if len(sources) == 1 else
                    "envelope:"+hashlib.sha256(json.dumps(sources).encode()).hexdigest()[:24])
        envelope = {**narrowest, "profile_id": identity,
            "input_range": [min(row["input_range"][0] for row in matching),
                            max(row["input_range"][1] for row in matching)],
            "observed_input_range": [min(row.get("observed_input_range", row["input_range"])[0] for row in matching),
                                     max(row.get("observed_input_range", row["input_range"])[1] for row in matching)],
            **{key: max(row[key] for row in matching)*multiplier
               for key in ("latency_ms", "gpu_gib_ms", "host_gib_ms")},
            **{key: max(row[key] for row in matching) for key in ("gpu_peak_gib", "host_peak_gib")},
            "source_profile_ids": sources, "prediction_estimate": True,
            "cache_conditioning": "measured_cache_and_concurrency_ranges" if observed else "cache_state_envelope",
            "observed_cache_ranges": [{"profile_id": row["profile_id"],
                "observed_input_range": row.get("observed_input_range", row["input_range"]),
                "ready_gpu_range": row["ready_gpu_range"], "host_load_range": row["host_load_range"],
                "concurrency_range": row["concurrency_range"]} for row in matching],
            "queried_cache_state": {"ready_gpu_tokens": ready, "host_load_tokens": host,
                                    "concurrency": concurrency},
            "concurrency_inference": {"observed_max": observed_q, "requested": concurrency,
                "duration_multiplier": multiplier,
                "basis": ("measured cache and concurrency range" if observed else
                    "componentwise measured cache-state envelope; linear duration extrapolation beyond observed concurrency through declared engine concurrency 8; fixed allocated memory peaks")}}
        return envelope


class PhysicalPaidCost:
    """Observe the complete declared device/process footprint since arrival."""
    def __init__(self, sampler):
        self.sampler = sampler
        self._lock = asyncio.Lock()
        self.started = False

    async def start(self):
        async with self._lock:
            if not self.started:
                await self.sampler.start()
                self.started = True

    async def aclose(self):
        if self.started:
            await self.sampler.stop()
            self.started = False

    async def observe(self, arrived, state, usage):
        from bench.study_resources import integrate_physical
        # One new physical observation closes the measured paid window. The
        # next forecast starts here; no missing interval is replaced with zero.
        sample = await asyncio.to_thread(self.sampler.sample, self.sampler.gpu_devices, self.sampler.process_ids)
        self.sampler.samples.append(sample)
        if not set(self.sampler.process_ids) <= {row["pid"] for row in sample["process_pss"]}:
            raise ValueError("declared physical process roots have no current PSS observation")
        end = sample["timestamp"]
        physical = integrate_physical(sorted(self.sampler.samples, key=lambda row: row["timestamp"]), arrived, end)
        if not physical["complete_window_coverage"]:
            raise ValueError("paid resource window has no physical observations on both boundaries")
        self.last_observation = {"clock": "client_monotonic", "start": arrived, "end": end,
            "basis": "complete declared physical device/process footprint during request window",
            "gpu_devices": self.sampler.gpu_devices, "process_roots": self.sampler.process_ids,
            "integration": "left-held sampled physical occupancy", "physical": physical,
            "samples": self.window_samples(arrived, end)}
        if usage is None:
            usage = {"input_tokens": 0, "output_tokens": 0}
        extra = state.auxiliary_usage
        if auxiliary_tokens(usage) is None or auxiliary_tokens(extra) is None:
            raise ValueError("rolling cumulative budgets require measured auxiliary token usage")
        return CompleteCost((end-arrived)*1000,
            physical["gpu_gib_seconds"]*1000, physical["host_gib_seconds"]*1000,
            usage["input_tokens"]+extra["input_tokens"]+sum(len(row.input_ids) for row in state.steps),
            usage["output_tokens"]+extra["output_tokens"]+sum(len(row.output_ids) for row in state.steps),
            physical["gpu_peak_gib"], physical["host_peak_gib"])

    def window(self, started, ended):
        from bench.study_resources import integrate_physical
        result = integrate_physical(sorted(self.sampler.samples, key=lambda row: row["timestamp"]), started, ended)
        if not result["complete_window_coverage"]:
            raise ValueError("current ANSWER overlap requires both measured physical boundaries")
        return result

    def window_samples(self, started, ended):
        rows = sorted(self.sampler.samples, key=lambda row: row["timestamp"])
        return [*[row for row in rows if row["timestamp"] <= started][-1:],
                *(row for row in rows if started < row["timestamp"] < ended),
                *[row for row in rows if row["timestamp"] >= ended][:1]]


class ProfiledNativeObserver:
    def __init__(self, profiles, paid, *, reader_checkpoint, reader_profile_id, sampling,
                 answer_system_prompt, engine_base_url, store=None):
        self.profiles, self.paid, self.store = profiles, paid, store
        self.reader_checkpoint, self.reader_profile_id = reader_checkpoint, reader_profile_id
        if profiles.data["sampling_recipe"] != sampling:
            raise ValueError("execution profiles bind the actual request sampling recipe")
        self.resource_scope = profiles.data["resource_scope"]
        self.resident_models_bound = False
        scope = self.resource_scope
        if (not scope["gpu_devices"] or not scope["process_roots"]
                or sorted(scope["gpu_devices"]) != sorted(paid.sampler.gpu_devices)
                or sorted(scope["process_roots"]) != sorted(paid.sampler.process_ids)
                or scope["host_basis"] != paid.sampler.observation()["host_basis"]
                or scope["native_endpoint"].rstrip("/") != engine_base_url.rstrip("/")):
            raise ValueError("execution resource profile does not match the actual online device/process scope")
        self.answer_system_prompt = answer_system_prompt

    def bind_resident_models(self, manifest):
        from .resident_models import resident_model_identity, resident_model_digest
        actual = resident_model_identity(manifest)
        digest = resident_model_digest(manifest)
        if (actual != resident_model_identity(self.profiles.data["resident_model_manifest"])
                or digest != self.profiles.data["resident_model_manifest_digest"]):
            raise ValueError("execution profile resident models differ from the actually loaded provider")
        self.resident_models_bound = True

    async def start(self):
        if not self.resident_models_bound:
            raise ValueError("rolling resource pricing requires the actual loaded resident model manifest")
        await self.paid.start()
        observed = {(row["index"], row["uuid"]) for row in self.paid.sampler.samples[-1]["devices"]}
        expected = {(row["index"], row["uuid"]) for row in self.resource_scope["devices"]}
        if observed != expected or {index for index, _ in expected} != set(self.resource_scope["gpu_devices"]):
            raise ValueError("execution resource profile GPU UUIDs differ from actual physical sampling")

    async def aclose(self):
        await self.paid.aclose()

    async def observe(self, plans, state, snapshot, engine, cost_model, arrived, auxiliary_usage, *, selection_started):
        candidates = tuple(candidate for plan in plans for step in plan.steps for candidate in step.variants
                           if candidate.feasible)
        quotes = await engine.quote_batch(candidates, self.profiles.fingerprint)
        self.profiles.bind(quotes, self.reader_checkpoint, self.reader_profile_id)
        by_id = {quote.candidate_id: quote for quote in quotes.quotes}
        concurrency = quotes.active_requests+1
        metrics = await engine.metrics()
        admission = {"gpu_capacity_tokens": metrics["pool"]["gpu_capacity_tokens"],
            "page_size": metrics["pool"]["page_size"],
            "native_max_new_tokens_clip": quotes.native_max_new_tokens_clip,
            "profile_service_envelope_ms": max(row["latency_ms"] for row in self.profiles.data["rows"]),
            "basis": "actual native quote workload/clip and actual engine physical token pool"}
        paid = await self.paid.observe(arrived, state, auxiliary_usage)
        overlap = self.paid.window(selection_started, self.paid.last_observation["end"])
        overlap_record = {"start": selection_started, "end": self.paid.last_observation["end"],
            "clock": "client_monotonic", "start_definition": "rolling_proposals_compilation_start",
            "physical": overlap, "samples": self.paid.window_samples(selection_started,
                self.paid.last_observation["end"]),
            "paid_prefix_handling": "already charged in paid_cost; conditional remaining forecast is independent of sunk head",
            "conditional_remaining_resource_forecasts": {}}
        resources, reusable, estimates = {}, {}, {}
        # A real prefix directory is optional; native quote hits remain in the
        # quote itself. No directory means no independently attributed prefix.
        if self.store is not None and candidates:
            from .prefix_retention import PrefixRuntimeClient
            from .prefix_view import prefix_candidate_view
            status = await PrefixRuntimeClient(engine).status(quotes.engine_restart_id)
            metrics = await engine.metrics()
            views = prefix_candidate_view(candidates, quotes, status,
                self.store.prefix_owners(active_only=True), page_size=metrics["pool"]["page_size"])
            reusable = {key: view.prefix_ids for key, view in views.items()}
        for candidate in candidates:
            quote = by_id[candidate.candidate_id]
            row = self.profiles.row(candidate.generation_kind, candidate.action, len(candidate.input_ids),
                candidate.output_budget, concurrency, quote.ready_gpu_prefix_tokens, quote.host_only_load_tokens,
                candidate.messages[0].content, start_definition="rolling_proposals_compilation_start"
                if candidate.generation_kind == "ANSWER" else None)
            if row is not None:
                estimates[candidate.candidate_id] = _profile_prediction(row, len(candidate.input_ids))
                gpu, host = row["gpu_gib_ms"], row["host_gib_ms"]
                if candidate.generation_kind == "ANSWER":
                    timing = cost_model.step_models.get("ANSWER", cost_model)
                    prior = timing.step_output_priors.get(f"ANSWER:{candidate.output_budget}")
                    if (prior is None or "mean_remaining_control_ms" not in prior or
                            timing.calibration_metadata.get("generation_kind") != "ANSWER" or
                            timing.calibration_metadata.get("output_prior_system_prompt") != candidate.messages[0].content or
                            "rolling_proposals_compilation_start" not in timing.calibration_metadata.get("step_control_scope", ())):
                        estimates[candidate.candidate_id].update(
                            unavailable_reason="answer_conditional_remaining_timing_binding_missing")
                        continue
                    remaining = cost_model.predict_native(replace(candidate, predicted_read_ms=0.0,
                        auxiliary_ms=0.0), quote, snapshot, batch=quotes, concurrency=concurrency,
                        control_ms=prior["mean_remaining_control_ms"]-prior["mean_control_ms"])
                    remaining_ms, selection = native_selection_cost(remaining, timing, candidate, quotes,
                        admission, estimates[candidate.candidate_id])
                    if remaining_ms is None:
                        estimates[candidate.candidate_id]["unavailable_reason"] = selection
                        continue
                    mean_gpu, mean_host = row["gpu_gib_ms"]/row["latency_ms"], row["host_gib_ms"]/row["latency_ms"]
                    gpu, host = remaining_ms*mean_gpu, remaining_ms*mean_host
                    prediction = {"method": "empirical_mean_residency_remaining_estimate",
                        "profile_id": row["profile_id"], "native_remaining_estimate": remaining,
                        "remaining_latency_ms": remaining_ms, "selection": selection, "mean_gpu_gib": mean_gpu,
                        "mean_host_gib": mean_host, "gpu_gib_ms": gpu, "host_gib_ms": host,
                        "paid_head_subtracted": False}
                    estimates[candidate.candidate_id]["conditional_remaining_resources"] = prediction
                    overlap_record["conditional_remaining_resource_forecasts"][candidate.candidate_id] = prediction
                resources[candidate.candidate_id] = ResourceForecast(
                    gpu, host, row["gpu_peak_gib"], row["host_peak_gib"],
                    row["profile_id"], quotes.observed_seq,
                    tuple(reusable.get(candidate.candidate_id, ())), True)
        future = {}
        for plan in plans:
            for step in plan.steps:
                task = step.task
                if task.kind == "GAP":
                    future[plan.plan_id, task.step_id] = ()
                    continue
                cpu = (plan.context_plan.reading_mode == "typed_reduce" and task.kind == "REDUCE"
                       and task.operation in {"union", "count", "sum", "group"})
                kind = "CPU_REDUCE" if cpu else step_generation_kind(task.kind, task.operation,
                                                                     plan.context_plan.reading_mode)
                prompt = ("" if cpu else self.answer_system_prompt if task.kind == "ANSWER" else
                    SOURCE_SYSTEM if task.kind == "READ" and plan.context_plan.read_layout == "source_first"
                    else step_system(plan.context_plan, task))
                outputs = (0,) if cpu else (task.output_budget,) if task.kind == "ANSWER" else tuple(
                    output for output in self.profiles.data["output_budgets"] if output <= task.output_budget)
                choices = []
                for output in outputs:
                    known = set(task.predecessor_step_ids) <= {result.step_id for result in state.results}
                    claims_ceiling = len(task.input_claim_ids) if known else _parent_claim_ceiling(
                        plan, task, state, self.profiles.claim_capacity)
                    known_inputs = [len(candidate.input_ids) for candidate in step.variants if candidate.feasible]
                    input_ceiling = max(known_inputs) if known_inputs else task.input_budget
                    row = self.profiles.row(kind, "CPU" if cpu else "REBUILD",
                        claims_ceiling if cpu else input_ceiling, output, concurrency, 0, 0, prompt,
                        operation=task.operation if cpu else None,
                        start_definition="rolling_proposals_compilation_start" if task.kind == "ANSWER" else None)
                    if row is not None:
                        estimates[f"future:{plan.plan_id}:{task.step_id}:{output}"] = _profile_prediction(
                            row, claims_ceiling if cpu else input_ceiling)
                        choices.append(StepForecast("CPU" if cpu else "REBUILD", output,
                            CompleteCost(row["latency_ms"], row["gpu_gib_ms"], row["host_gib_ms"],
                                0 if cpu else input_ceiling, output, row["gpu_peak_gib"], row["host_peak_gib"]),
                            row["profile_id"], True, "cpu_empirical_bin_prediction" if cpu
                            else "future_input_ceiling_empirical_bin_prediction"))
                future[plan.plan_id, task.step_id] = tuple(choices)
        return NativePlanningState(self.reader_checkpoint, self.reader_profile_id, snapshot, quotes,
            cost_model, resources, future, paid, {**self.paid.last_observation,
                "current_answer_phase_overlap": overlap_record,
                "native_admission": admission,
                "execution_profile_predictions": estimates}), reusable


class ObservedNativeObserver:
    def __init__(self, paid, fingerprint, *, reader_checkpoint, reader_profile_id, store=None):
        self.paid, self.fingerprint, self.store = paid, fingerprint, store
        self.reader_checkpoint, self.reader_profile_id = reader_checkpoint, reader_profile_id
        self.resident_model_manifest = None
        self.resident_digest = None

    def bind_resident_models(self, manifest):
        from .resident_models import resident_model_digest
        self.resident_model_manifest = manifest
        self.resident_digest = resident_model_digest(manifest)

    async def start(self):
        await self.paid.start()

    async def aclose(self):
        await self.paid.aclose()

    async def observe(self, plans, state, snapshot, engine, cost_model, arrived, auxiliary_usage, *, selection_started):
        candidates = tuple(candidate for plan in plans for step in plan.steps for candidate in step.variants
                           if candidate.feasible)
        quotes = await engine.quote_batch(candidates, self.fingerprint)
        if quotes.fingerprint != self.fingerprint:
            raise ValueError("native quote does not match the actual reader fingerprint")
        pool = (await engine.metrics())["pool"]
        reusable = {}
        if self.store is not None and candidates:
            from .prefix_retention import PrefixRuntimeClient
            from .prefix_view import prefix_candidate_view
            status = await PrefixRuntimeClient(engine).status(quotes.engine_restart_id)
            views = prefix_candidate_view(candidates, quotes, status,
                self.store.prefix_owners(active_only=True), page_size=pool["page_size"])
            reusable = {key: view.prefix_ids for key, view in views.items()}
        paid = await self.paid.observe(arrived, state, auxiliary_usage)
        admission = {"gpu_capacity_tokens": pool["gpu_capacity_tokens"], "page_size": pool["page_size"],
            "native_max_new_tokens_clip": quotes.native_max_new_tokens_clip,
            "basis": "actual native quote workload/clip and actual engine physical token pool"}
        return NativePlanningState(self.reader_checkpoint, self.reader_profile_id, snapshot, quotes,
            cost_model, {}, {}, paid, {**self.paid.last_observation, "native_admission": admission,
                "resident_model_manifest_digest": self.resident_digest}), reusable


def load_rolling_runtime(settings, *, template, store, sampling, answer_system_prompt, engine_base_url):
    from bench.study_resources import StudyResourceSampler
    from training.train_plan_gain import PlanGainModel
    from .plan_policy import PolicyLimits
    from .rolling_policy import RollingPlanRuntime
    checkpoint, profile = settings["reader_checkpoint"], settings["reader_profile_id"]
    model = PlanGainModel.load(settings["gain_asset"], reader_checkpoint=checkpoint, reader_profile_id=profile,
        reader_fingerprint=template.fingerprint)
    provenance = model.provenance
    if (provenance["label"] not in {"actual_final_quality_minus_actual_direct_answer_quality",
            "actual_completion_weighted_final_utility_minus_direct_utility"}
            or provenance["training_split"] != "train" or provenance["training_records"] < 1
            or not provenance["input_sha256"] or not provenance["counterfactual_runs_retained"]):
        raise ValueError("enabled rolling gains require the actual train-split counterfactual asset")
    if (provenance["label"] == "actual_completion_weighted_final_utility_minus_direct_utility"
            and (provenance["no_answer_utility"] != 0 or not provenance["raw_missing_quality_preserved"])):
        raise ValueError("effective utility assets retain missing raw judge Q and their declared no-QA rule")
    mode = settings.get("mode", "whole_plan")
    if checkpoint != template.fingerprint.model_revision:
        raise ValueError("rolling gains must bind the actual reader checkpoint")
    paid = PhysicalPaidCost(StudyResourceSampler(tuple(settings["gpu_devices"]), tuple(settings["process_ids"]),
                                               interval=settings["sampling_interval_seconds"]))
    prefix_store = store if settings.get("prefix_directory", False) else None
    if mode == "observed_actions":
        observer = ObservedNativeObserver(paid, template.fingerprint, reader_checkpoint=checkpoint,
            reader_profile_id=profile, store=prefix_store)
    else:
        profiles = ExecutionProfiles.load(settings["execution_profile_asset"])
        if profiles.fingerprint != template.fingerprint:
            raise ValueError("execution profiles must bind the actual reader fingerprint")
        observer = ProfiledNativeObserver(profiles, paid, reader_checkpoint=checkpoint, reader_profile_id=profile,
            sampling=sampling, answer_system_prompt=answer_system_prompt, engine_base_url=engine_base_url,
            store=prefix_store)
    return RollingPlanRuntime(model, observer, reader_checkpoint=checkpoint, reader_profile_id=profile,
        output_budgets=settings["output_budgets"], limits=PolicyLimits(**settings.get("limits", {})),
        output_tokens_per_fact=settings.get("output_tokens_per_fact", 64),
        mode=mode, observation_path=settings.get("observation_path"),
        gap_retriever_identity=settings.get("gap_retriever_identity"), gap_limits=settings.get("gap_limits"))


def load_fixed_reading_observer(settings, *, template):
    from bench.study_resources import StudyResourceSampler
    from .rolling_policy import FixedReadingObserver
    if settings["reader_checkpoint"] != template.fingerprint.model_revision:
        raise ValueError("fixed reading observations must bind the actual reader checkpoint")
    paid = PhysicalPaidCost(StudyResourceSampler(tuple(settings["gpu_devices"]), tuple(settings["process_ids"]),
                                               interval=settings["sampling_interval_seconds"]))
    return FixedReadingObserver(settings, paid, template.fingerprint)
