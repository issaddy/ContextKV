"""Freeze development decisions and replay each candidate in an isolated cache fixture.

This is an offline diagnostic, never a production two-generation policy. Each
arm resets the dedicated profiler and warms the same committed execution path;
it does not reconstruct the original concurrent server's physical cache state.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import random
import sqlite3
import statistics
import time
import uuid

from controller.candidates import build_candidates
from controller.config import RunConfig, load_config
from controller.cost_model import CostModel
from controller.engine_client import EngineError, EngineInfo, SGLangEngineClient
from controller.planner import PlannerKnobs, choose_plan
from controller.schema import CacheQuoteBatch, NoFeasiblePlan, PromptCandidate, SessionSnapshot
from controller.serialization import decode, dumps, loads
from controller.session_store import Attempt
from controller.templates import ByteTemplate, HuggingFaceTemplate


@dataclass(frozen=True)
class DecisionCase:
    case_id: str
    conversation_id: str
    request_id: str
    snapshot: SessionSnapshot
    candidates: tuple[PromptCandidate, ...]
    warmup_input_ids: tuple[int, ...]
    max_new_tokens: int
    source_selected_action: str


def select_cases(cases, *, limit=None, selection="database_order", seed=42):
    """Sample pre-turn inputs across conversation/epoch age, never current outputs."""
    if selection == "database_order":
        return tuple(cases[:limit])
    if selection != "stratified":
        raise ValueError("unknown case selection")
    # Age 0/1/2+ distinguishes recent rebuilds from decisions where the default
    # stability gate can open. Each nonempty stratum gets a turn before repeats.
    groups = defaultdict(list)
    for case in cases:
        groups[(case.conversation_id, min(case.snapshot.turns_since_rebuild, 2))].append(case)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for key in keys:
        groups[key].sort(key=lambda case: case.case_id)
        rng.shuffle(groups[key])
    selected = []
    while keys and (limit is None or len(selected) < limit):
        remaining = []
        for key in keys:
            selected.append(groups[key].pop())
            if limit is not None and len(selected) == limit:
                return tuple(selected)
            if groups[key]:
                remaining.append(key)
        keys = remaining
    return tuple(selected)


def freeze_cases(database, manifest, split, template, *, limit=None, selection="database_order", seed=42):
    """Read committed pre-turn inputs only; SQL strips the current answer/result."""
    if limit is not None and limit < 1:
        raise ValueError("case limit must be positive")
    dev, test = set(split["dev"]), set(split["test"])
    selected = set(manifest.get("dataset", {}).get("selected_sessions", ()))
    if not selected or dev & test or not selected <= dev:
        raise ValueError("source run must identify only development conversations")
    if manifest.get("protocol") not in {"P2", "P3"}:
        raise ValueError("decision pairs require generated P2/P3 history, not reference P1 history")
    config = decode(RunConfig, manifest["config"])
    if asdict(template.fingerprint) != manifest["fingerprint"]:
        raise ValueError("freezing tokenizer/template differs from the source run")
    vocabulary = (range(256) if isinstance(template, ByteTemplate) else
                  sorted(set(template.tokenizer.get_vocab().values()) - set(template.tokenizer.all_special_ids)))
    prefix = manifest["run_id"] + ":"
    cases, skipped = [], []
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("BEGIN")
        pending = connection.execute("SELECT session_id FROM attempts WHERE stage NOT IN ('COMMITTED','FAILED')").fetchall()
        if any(sid.startswith(prefix) for (sid,) in pending):
            raise ValueError("finish or recover the source run before freezing decisions")
        rows = connection.execute("SELECT json_remove(payload,'$.result') FROM attempts WHERE stage='COMMITTED' ORDER BY rowid")
        for (payload,) in rows:
            attempt = loads(Attempt, payload)
            if not attempt.session_id.startswith(prefix):
                continue
            conversation = attempt.session_id[len(prefix):]
            if conversation not in selected:
                raise ValueError("source database has run conversations absent from its manifest")
            if attempt.branch_reason in {"recovery", "reference_history_sync"}:
                skipped.append({"attempt_id": attempt.attempt_id, "reason": "source_forced_rebuild"})
                continue
            old = connection.execute("SELECT snapshot FROM ledger_history WHERE session_id=? AND version=?",
                                     (attempt.session_id, attempt.expected_version)).fetchone()
            if old is None:
                if attempt.expected_version != 0:
                    raise ValueError("missing pre-turn ledger snapshot")
                skipped.append({"attempt_id": attempt.attempt_id, "reason": "initial_turn_has_one_candidate"})
                continue
            snapshot = loads(SessionSnapshot, old[0])
            policy = (replace(config.context, layout="contextpilot")
                      if config.planner.strategy == "contextpilot" else config.context)
            candidates = build_candidates(snapshot, attempt.query, attempt.evidence, policy, template,
                                          retrieval_priority=attempt.retrieval_priority,
                                          measure_evidence_redundancy="redundancy_threshold" in {
                                              config.planner.strategy, config.planner.fallback_strategy})
            original = next((c for c in candidates if c.action == attempt.plan.candidate.action), None)
            if original is None or original.input_hash != attempt.plan.candidate.input_hash:
                skipped.append({"attempt_id": attempt.attempt_id, "reason": "source_candidate_changed"})
                continue
            if {c.action for c in candidates if c.feasible} != {"APPEND", "REBUILD"}:
                skipped.append({"attempt_id": attempt.attempt_id, "reason": "not_two_feasible_candidates"})
                continue
            history_ids = snapshot.execution.last_input_ids + snapshot.execution.last_output_ids
            # One extra input token makes the whole committed history materialize
            # without accidentally warming a candidate's next token as well.
            forbidden = {c.input_ids[len(history_ids)] for c in candidates
                         if len(c.input_ids) > len(history_ids)}
            sentinel = next(token for token in vocabulary if token not in forbidden)
            cases.append(DecisionCase(attempt.attempt_id, conversation, attempt.request_id, snapshot,
                                      candidates, history_ids + (sentinel,), policy.max_new_tokens,
                                      attempt.plan.candidate.action))
            if selection == "database_order" and limit is not None and len(cases) >= limit:
                break
    finally:
        connection.close()
    return select_cases(cases, limit=limit, selection=selection, seed=seed), skipped


def write_frozen(database, manifest_path, split_path, output, template, *, limit=None,
                 selection="database_order", seed=42):
    manifest = json.loads(Path(manifest_path).read_text())
    split_bytes = Path(split_path).read_bytes()
    split_hash = sha256(split_bytes).hexdigest()
    if manifest.get("dataset", {}).get("split_manifest_sha256") != split_hash:
        raise ValueError("source run must use this exact frozen development/test split")
    cases, skipped = freeze_cases(database, manifest, json.loads(split_bytes), template,
                                  limit=limit, selection=selection, seed=seed)
    if not cases:
        raise ValueError(f"no unchanged two-candidate development cases: {skipped}")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.with_suffix(".manifest.json").exists():
        raise FileExistsError("frozen bundle already exists; choose a new output")
    payload = "".join(dumps(case) + "\n" for case in cases)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    metadata = {"schema_version": 1, "split": "dev", "cases": len(cases), "skipped": skipped,
                "selection": ("first_eligible_committed_attempts_in_database_order" if selection == "database_order"
                              else "round_robin_conversation_and_epoch_age_0_1_2plus"),
                "selection_seed": seed if selection == "stratified" else None, "case_limit": limit,
                "selected_cases": [{"case_id": case.case_id, "conversation_id": case.conversation_id,
                    "prior_turns": len(case.snapshot.raw_history),
                    "turns_since_rebuild": case.snapshot.turns_since_rebuild,
                    "input_tokens": {c.action: len(c.input_ids) for c in case.candidates}} for case in cases],
                "cases_sha256": sha256(payload.encode()).hexdigest(),
                "source_manifest": str(Path(manifest_path).resolve()),
                "source_manifest_sha256": sha256(Path(manifest_path).read_bytes()).hexdigest(),
                "split_manifest_sha256": split_hash, "source_run_id": manifest["run_id"],
                "source_measurement_kind": manifest["measurement_kind"], "fingerprint": manifest["fingerprint"],
                "source_protocol": manifest["protocol"], "source_database": str(Path(database).resolve()),
                "input_provenance": "committed_attempt_inputs_and_expected_version_ledger; current_result_excluded",
                "warmup_provenance": "previous_execution_input_plus_actual_output_plus_one_common_sentinel"}
    target.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_frozen(path):
    source = Path(path)
    payload = source.read_bytes()
    metadata = json.loads(source.with_suffix(".manifest.json").read_text())
    if metadata.get("split") != "dev" or metadata["cases_sha256"] != sha256(payload).hexdigest():
        raise ValueError("decision cases must be an unchanged frozen development bundle")
    return tuple(loads(DecisionCase, line) for line in payload.decode().splitlines() if line), metadata


def _shape(batch):
    quotes = sorted((q.candidate_id, q.ready_gpu_prefix_tokens, q.recoverable_prefix_tokens,
             q.host_only_load_tokens, q.native_load_estimate, q.inflight_h2d_tokens,
             q.inflight_d2h_tokens, q.native_estimate_uncertain) for q in batch.quotes)
    return batch.free_gpu_tokens, batch.free_host_tokens, quotes


class DecisionPairCollector:
    def __init__(self, engine, model, knobs=None, *, context_limit=32768):
        self.engine, self.model = engine, model
        self.knobs = knobs or PlannerKnobs()
        self.context_limit = context_limit

    async def fixture(self, mode):
        for attempt in range(20):
            try:
                return await self.engine.profile_prepare(mode)
            except EngineError as error:
                if error.code != "ENGINE_BUSY" or attempt == 19:
                    raise
                await asyncio.sleep(.05)

    async def generate(self, ids, max_new_tokens, *, warmup=False):
        info = await self.engine.info()
        sid, attempt = uuid.uuid4().hex, uuid.uuid4().hex
        ack = await self.engine.prepare(sid, uuid.uuid4().hex, info.engine_restart_id)
        try:
            started = time.perf_counter()
            result = await self.engine.generate(input_ids=ids, session_id=sid,
                expected_generation=ack.session_generation, expected_restart_id=info.engine_restart_id,
                attempt_id=attempt, fingerprint=info.fingerprint,
                max_new_tokens=max_new_tokens, ignore_eos=warmup)
            return result, (time.perf_counter() - started) * 1000
        finally:
            await self.engine.retire(sid, ack.session_generation, uuid.uuid4().hex,
                                     info.engine_restart_id, "offline_decision_pair")

    async def measure(self, case, state, *, repetition=0, seed=42):
        info = await self.engine.info()
        calibration_scope = self.model.check_calibration_scope(info)
        if info.engine_kind != "sglang" or not info.profiling_enabled:
            raise ValueError("decision replay requires a dedicated SGLang profiler server")
        if state not in {"cold", "gpu_ready", "host_only"}:
            raise ValueError("unknown replay cache state")
        if any(c.fingerprint != info.fingerprint for c in case.candidates):
            raise ValueError("frozen candidate fingerprint differs from profiler")
        if state != "cold" and len(case.warmup_input_ids) + 1 > self.context_limit:
            raise ValueError("committed warmup path exceeds the profiler context limit")
        order = list(case.candidates)
        random.Random(f"{seed}:{case.case_id}:{state}:{repetition}").shuffle(order)
        observations, first_shape, plan = {}, None, None
        try:
            for candidate in order:
                await self.fixture("cold")
                warmup = None
                if state != "cold":
                    warmed, duration = await self.generate(case.warmup_input_ids, 1, warmup=True)
                    warmup = {"client_complete_ms": duration, "receipt": asdict(warmed.receipt)}
                    await self.fixture("host_only" if state == "host_only" else "settle")
                quote = await self.engine.quote_batch(case.candidates, info.fingerprint)
                if (quote.active_requests or 0) or (quote.queued_requests or 0):
                    raise ValueError("profiler is not isolated: other requests are active or queued")
                if state == "cold" and any(q.raw_match_tokens for q in quote.quotes):
                    raise ValueError("cold fixture retained matching cache")
                if state == "host_only" and (any(q.ready_gpu_prefix_tokens for q in quote.quotes)
                        or not any(q.host_only_load_tokens for q in quote.quotes)):
                    raise ValueError("host-only fixture did not retain an exclusively host prefix")
                if state == "gpu_ready" and not any(q.ready_gpu_prefix_tokens for q in quote.quotes):
                    raise ValueError("warm fixture has no reusable GPU prefix")
                if first_shape is not None and _shape(quote) != first_shape:
                    raise ValueError("pair arms have different prepared cache shapes")
                if plan is None:
                    first_shape = _shape(quote)
                    plan = choose_plan(case.candidates, quote, self.model, case.snapshot, self.knobs,
                                       concurrency=1, control_ms=0)
                    if isinstance(plan, NoFeasiblePlan):
                        raise ValueError("frozen decision has no feasible plan")
                quotes_by_id = {q.candidate_id: q for q in quote.quotes}
                estimates = {c.action: self.model.predict_native(c, quotes_by_id[c.candidate_id], case.snapshot, batch=quote,
                             concurrency=1, control_ms=0)
                             for c in case.candidates}
                observations[candidate.action] = {"input_hash": candidate.input_hash,
                    "input_tokens": len(candidate.input_ids), "prediction": asdict(estimates[candidate.action]),
                    "quote": asdict(quote), "warmup": warmup}
                result, duration = await self.generate(candidate.input_ids, case.max_new_tokens)
                observations[candidate.action].update({
                    "client_complete_ms": duration, "server_elapsed_ms": result.server_elapsed_ms,
                    "output_tokens": len(result.output_ids), "result": asdict(result)})
                if not result.generation_complete or not result.receipt.generation_complete:
                    raise ValueError("incomplete candidate generation")
                if result.receipt.input_hash != candidate.input_hash:
                    raise ValueError("candidate receipt does not describe frozen input")
        except Exception as error:
            return {"case_id": case.case_id, "conversation_id": case.conversation_id,
                    "request_id": case.request_id, "cache_state": state, "repetition": repetition,
                    "status": "failed", "error": f"{type(error).__name__}: {error}",
                    "execution_order": [c.action for c in order], "observations": observations,
                    "selected_action": plan.candidate.action if plan else None,
                    "plan_reason": plan.reason if plan else None, "quality_evaluated": False,
                    "calibration_scope": calibration_scope}
        return {"case_id": case.case_id, "conversation_id": case.conversation_id,
                "request_id": case.request_id, "cache_state": state, "repetition": repetition,
                "status": "completed", "execution_order": [c.action for c in order],
                "selected_action": plan.candidate.action, "plan_reason": plan.reason,
                "observations": observations, "quality_evaluated": False,
                "calibration_scope": calibration_scope,
                "decision_scope": "isolated_single_request_with_one_committed_history_path",
                "prediction_control_ms": 0, "measurement_source": "hardware"}


def analyze_pairs(rows, *, time_field="server_elapsed_ms", tie_ms=1.0):
    """Coverage, sign accuracy and selected-action regret; no quality inference."""
    if tie_ms < 0:
        raise ValueError("tie tolerance must be nonnegative")
    def sign(value):
        return 0 if abs(value) <= tie_ms else (1 if value > 0 else -1)
    groups, failures, reasons = defaultdict(list), Counter(), Counter()
    details = []
    for row in rows:
        if row.get("status") != "completed":
            failures[row.get("error", row.get("status", "failed"))] += 1
            continue
        a, r = (row["observations"][action] for action in ("APPEND", "REBUILD"))
        if a.get(time_field) is None or r.get(time_field) is None:
            failures["missing_comparable_timing"] += 1
            continue
        reasons[row["plan_reason"]] += 1
        calibrated = a["prediction"]["calibrated"] and r["prediction"]["calibrated"]
        predicted = a["prediction"]["cost"] - r["prediction"]["cost"]
        actual = a[time_field] - r[time_field]
        selected = row["observations"][row["selected_action"]][time_field]
        regret = selected - min(a[time_field], r[time_field])
        item = {"case_id": row["case_id"], "conversation_id": row["conversation_id"],
                "cache_state": row["cache_state"], "repetition": row.get("repetition", 0), "calibrated": calibrated,
                "predicted_append_minus_rebuild_ms": predicted, "actual_append_minus_rebuild_ms": actual,
                "sign_correct": sign(predicted) == sign(actual), "selected_regret_ms": regret,
                "outputs": {"APPEND": a["output_tokens"], "REBUILD": r["output_tokens"]}}
        details.append(item)
        groups[row["conversation_id"]].append(item)
    sessions = []
    for sid, items in sorted(groups.items()):
        covered = [item for item in items if item["calibrated"]]
        sessions.append({"conversation_id": sid, "pairs": len(items),
            "coverage": len(covered)/len(items),
            "sign_accuracy_on_calibrated": statistics.mean(i["sign_correct"] for i in covered) if covered else None,
            "mean_selected_regret_ms": statistics.mean(i["selected_regret_ms"] for i in items)})
    covered = [item for item in details if item["calibrated"]]
    return {"timing_scope": time_field, "tie_ms": tie_ms, "completed_pairs": len(details),
            "failed_or_unmeasured_pairs": sum(failures.values()), "failures": dict(failures),
            "calibration_coverage": len(covered)/len(details) if details else None,
            "plan_reasons": dict(reasons), "fallback_pairs": sum(v for k, v in reasons.items() if "fallback" in k),
            "sign_accuracy_on_calibrated": statistics.mean(i["sign_correct"] for i in covered) if covered else None,
            "mean_selected_regret_ms": statistics.mean(i["selected_regret_ms"] for i in details) if details else None,
            "equal_session_mean_regret_ms": statistics.mean(s["mean_selected_regret_ms"] for s in sessions) if sessions else None,
            "sessions": sessions, "pairs": details, "quality_evaluated": False,
            "interpretation": "offline fixture decisions; not original concurrency replay or quality-adjusted oracle"}


def _model_and_knobs(config, cases, artifact):
    common = dict(max_new_tokens=cases[0].max_new_tokens,
                  include_host_cost=config.ablation.host_cost, include_decode_cost=config.ablation.decode_cost,
                  include_control_cost=config.ablation.control_cost)
    if config.cost_model.uncertainty_mode is not None:
        common["uncertainty_mode"] = config.cost_model.uncertainty_mode
    if config.cost_model.admission_estimator is not None:
        common["admission_estimator"] = config.cost_model.admission_estimator
    settings = asdict(config.cost_model)
    for key in ("artifact", "online_calibration", "output_length_artifact", "uncertainty_mode", "admission_estimator"):
        settings.pop(key)
    if artifact:
        model = CostModel.load(artifact, **common)
    elif config.calibration_path:
        rows = [json.loads(line) for line in Path(config.calibration_path).read_text().splitlines() if line]
        model = CostModel.from_measurements(rows, **common, **settings)
    else:
        model = CostModel(**common, **settings)
    if config.cost_model.output_length_artifact:
        from controller.output_length import OutputLengthModel
        model.output_length_model = OutputLengthModel.load(config.cost_model.output_length_artifact,
            expected_fingerprint=cases[0].candidates[0].fingerprint, max_new_tokens=cases[0].max_new_tokens)
    knobs = config.planner if config.ablation.hysteresis else replace(
        config.planner, delta_abs=0, delta_rel=0, z=0, min_stable_turns=0)
    return model, knobs


def rescore_pairs(rows, cases, model, knobs, engine_info):
    """Re-plan saved inputs/quotes only; retain current outcomes for later analysis.

    The first measured arm supplies the decision quote, as in collection. Each
    arm's prediction uses its own saved quote. No current result enters planning.
    """
    info = decode(EngineInfo, engine_info) if isinstance(engine_info, dict) else engine_info
    scope = model.check_calibration_scope(info)
    by_case = {case.case_id: case for case in cases}
    if len(by_case) != len(cases):
        raise ValueError("duplicate frozen case IDs")
    rescored = []
    for source in rows:
        case = by_case.get(source["case_id"])
        if case is None or source["conversation_id"] != case.conversation_id:
            raise ValueError("saved pair case_id/conversation does not match frozen cases")
        row = deepcopy(source)
        if source.get("status") != "completed":
            row["rescoring"] = {"status": "source_failure_preserved"}
            rescored.append(row)
            continue
        if source["request_id"] != case.request_id:
            raise ValueError("saved pair request does not match frozen case")
        if model.max_new_tokens != case.max_new_tokens:
            raise ValueError("rescoring must preserve the frozen generation budget")
        candidates = {candidate.action: candidate for candidate in case.candidates}
        order = source["execution_order"]
        if (set(candidates) != {"APPEND", "REBUILD"} or len(order) != 2
                or set(order) != set(candidates) or set(source["observations"]) != set(candidates)):
            raise ValueError("completed pair requires two frozen arms and their original execution order")
        plans, first_shape = {}, None
        for action in order:
            candidate, observation = candidates[action], source["observations"][action]
            if (observation["input_hash"] != candidate.input_hash
                    or observation["input_tokens"] != len(candidate.input_ids)
                    or observation["result"]["receipt"]["input_hash"] != candidate.input_hash):
                raise ValueError("saved arm input_hash/length differs from frozen candidate")
            quote = decode(CacheQuoteBatch, observation["quote"])
            if quote.fingerprint != info.fingerprint or quote.cache_config_hash != info.cache_config_hash:
                raise ValueError("saved quote differs from recorded engine configuration")
            if first_shape is not None and _shape(quote) != first_shape:
                raise ValueError("saved pair arms have different prepared cache shapes")
            first_shape = _shape(quote)
            # choose_plan validates every candidate/quote identity in this batch.
            plan = choose_plan(case.candidates, quote, model, case.snapshot, knobs,
                               concurrency=1, control_ms=0)
            if isinstance(plan, NoFeasiblePlan):
                raise ValueError("frozen decision has no feasible plan")
            plans[action] = plan
            estimate = next((item for item in plan.predicted if item.candidate_id == candidate.candidate_id), None)
            if estimate is None:
                own_quote = next(item for item in quote.quotes if item.candidate_id == candidate.candidate_id)
                estimate = model.predict_native(candidate, own_quote, case.snapshot, batch=quote,
                                                concurrency=1, control_ms=0)
            row["observations"][action]["prediction"] = asdict(estimate)
        selected = plans[order[0]]
        row["rescoring"] = {"status": "rescored", "source_selected_action": source["selected_action"],
            "source_plan_reason": source["plan_reason"], "source_calibration_scope": source.get("calibration_scope"),
            "source_predictions": {action: deepcopy(arm["prediction"]) for action, arm in source["observations"].items()},
            "decision_quote_arm": order[0], "outcomes_unchanged": True,
            "current_outcomes_used_for_selection": False,
            "current_outcomes_scope": "runtime_feature_inputs_only; fitted model labels must be audited separately"}
        row.update(selected_action=selected.candidate.action, plan_reason=selected.reason,
                   calibration_scope=scope, prediction_control_ms=0)
        rescored.append(row)
    return rescored


def rescore_file(args):
    cases, frozen = load_frozen(args.cases)
    source_path = Path(args.input)
    source_manifest_path = source_path.with_suffix(".manifest.json")
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("status") not in {"complete", "complete_with_failures"}:
        raise ValueError("finish source pair collection before offline rescoring")
    if (source_manifest.get("split") != "dev"
            or source_manifest["frozen"]["cases_sha256"] != frozen["cases_sha256"]):
        raise ValueError("saved pairs were not collected from this frozen development bundle")
    output = Path(args.output)
    targets = (output, output.with_suffix(".manifest.json"), output.with_suffix(".summary.json"))
    if any(path.exists() for path in targets):
        raise FileExistsError("rescored output already exists; choose a new output")
    config = load_config(args.config)
    artifact = args.cost_artifact or config.cost_model.artifact
    if not artifact:
        raise ValueError("offline rescoring requires a frozen cost artifact")
    model, knobs = _model_and_knobs(config, cases, artifact)
    rows = [json.loads(line) for line in source_path.read_text().splitlines() if line]
    records = rescore_pairs(rows, cases, model, knobs, source_manifest["engine"])
    metadata = {"schema_version": 1, "split": "dev", "operation": "offline_rescore",
        "status": "complete_with_failures" if any(row["status"] != "completed" for row in records) else "complete",
        "source_pairs": str(source_path.resolve()), "source_pairs_sha256": sha256(source_path.read_bytes()).hexdigest(),
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": sha256(source_manifest_path.read_bytes()).hexdigest(),
        "source_measurement_manifest": source_manifest,
        "frozen": frozen, "cases_path": str(Path(args.cases).resolve()),
        "config": asdict(config), "config_scope": "cost ablations and planner; candidates/budgets come from frozen cases",
        "cost_artifact": str(Path(artifact).resolve()), "cost_artifact_sha256": sha256(Path(artifact).read_bytes()).hexdigest(),
        "output_length_artifact": config.cost_model.output_length_artifact,
        "output_length_artifact_sha256": sha256(Path(config.cost_model.output_length_artifact).read_bytes()).hexdigest()
            if config.cost_model.output_length_artifact else None,
        "output_length_training_metadata": model.output_length_model.metadata if model.output_length_model else None,
        "engine": source_manifest["engine"], "calibration_scope": model.calibration_scope_report,
        "measurement_source_counts": dict(Counter(row.get("measurement_source", "not_recorded") for row in records)),
        "new_generations": 0, "new_engine_requests": 0, "online_model_updates": False,
        "current_outcomes_used_for_selection": False,
        "current_outcomes_scope": "runtime_feature_inputs_only; fitted model labels must be audited separately",
        "quality_evaluated": False,
        "pressure_concurrency": 1, "prediction_control_ms": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write("".join(dumps(row)+"\n" for row in records))
    output.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    summary = analyze_pairs(records)
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({"output": str(output), "status": metadata["status"], "completed_pairs": summary["completed_pairs"],
                      "new_generations": 0}))


async def collect(args):
    cases, frozen = load_frozen(args.cases)
    if frozen["source_measurement_kind"] == "simulation":
        raise ValueError("hardware decision collection requires a source run with real model tokens")
    config = load_config(args.config)
    artifact = args.cost_artifact or config.cost_model.artifact
    model, knobs = _model_and_knobs(config, cases, artifact)
    fingerprint = cases[0].candidates[0].fingerprint
    engine = SGLangEngineClient(args.base_url or config.engine.base_url, fingerprint,
                                timeout=config.engine.timeout_seconds, stream=False)
    collector = DecisionPairCollector(engine, model, knobs,
                                      context_limit=config.context.max_total_model_tokens)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".manifest.json").exists():
        await engine.aclose()
        raise FileExistsError("decision results already exist; choose a new output")
    metadata = {"schema_version": 1, "split": "dev", "status": "running", "frozen": frozen,
        "cache_states": args.cache_states, "repetitions": args.repetitions, "seed": args.seed,
        "config": asdict(config), "cost_artifact": artifact,
        "cost_artifact_sha256": sha256(Path(artifact).read_bytes()).hexdigest() if artifact else None,
        "output_length_artifact": config.cost_model.output_length_artifact,
        "output_length_artifact_sha256": sha256(Path(config.cost_model.output_length_artifact).read_bytes()).hexdigest()
            if config.cost_model.output_length_artifact else None,
        "quality_evaluated": False, "fixtures_destructive_on_isolated_server": True,
        "online_model_updates": False, "pressure_concurrency": 1,
        "client_complete_scope": "generate_submit_to_complete_response; excludes fixture,prepare,retire,ledger_commit"}
    metadata_path = output.with_suffix(".manifest.json")
    records = []
    try:
        metadata["engine"] = asdict(await engine.info())
        metadata["calibration_scope"] = model.check_calibration_scope(metadata["engine"])
        metadata_path.write_text(json.dumps(metadata, indent=2)+"\n")
        with output.open("x", encoding="utf-8") as stream:
            for case in cases:
                for state in args.cache_states:
                    for repetition in range(args.repetitions):
                        try:
                            row = await collector.measure(case, state, repetition=repetition, seed=args.seed)
                        except Exception as error:
                            row = {"case_id": case.case_id, "conversation_id": case.conversation_id,
                                "cache_state": state, "repetition": repetition, "status": "failed",
                                "error": f"{type(error).__name__}: {error}"}
                        records.append(row)
                        stream.write(dumps(row)+"\n")
                        stream.flush()
        metadata["status"] = "complete_with_failures" if any(r["status"] != "completed" for r in records) else "complete"
        metadata_path.write_text(json.dumps(metadata, indent=2)+"\n")
        summary = analyze_pairs(records)
        output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps({"output": str(output), "status": metadata["status"], "completed_pairs": summary["completed_pairs"]}))
    finally:
        await engine.aclose()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--database", required=True)
    freeze.add_argument("--run-manifest", required=True)
    freeze.add_argument("--split-manifest", required=True)
    freeze.add_argument("--model-path")
    freeze.add_argument("--limit", type=int)
    freeze.add_argument("--selection", choices=("database_order", "stratified"), default="database_order")
    freeze.add_argument("--seed", type=int, default=42)
    freeze.add_argument("--output", required=True)
    run = commands.add_parser("run")
    run.add_argument("--cases", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--base-url", help="Dedicated isolated profiler endpoint; its caches will be reset")
    run.add_argument("--cost-artifact")
    run.add_argument("--cache-states", type=lambda value: tuple(value.split(",")), default=("cold", "gpu_ready", "host_only"))
    run.add_argument("--repetitions", type=int, default=3)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--output", required=True)
    rescore = commands.add_parser("rescore", help="re-score saved pairs with a frozen model; no engine calls")
    rescore.add_argument("--cases", required=True)
    rescore.add_argument("--input", required=True, help="saved pairs JSONL and adjacent .manifest.json")
    rescore.add_argument("--config", required=True)
    rescore.add_argument("--cost-artifact")
    rescore.add_argument("--output", required=True)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("input")
    analyze.add_argument("--time-field", choices=("server_elapsed_ms", "client_complete_ms"), default="server_elapsed_ms")
    analyze.add_argument("--tie-ms", type=float, default=1.0)
    analyze.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        manifest = json.loads(Path(args.run_manifest).read_text())
        config = decode(RunConfig, manifest["config"])
        template = (ByteTemplate(namespace_hash=config.model.namespace_hash) if config.engine.kind == "mock" else
                    HuggingFaceTemplate(args.model_path or config.model.name, config.model.revision,
                                        namespace_hash=config.model.namespace_hash, local_files_only=True))
        print(json.dumps(write_frozen(args.database, args.run_manifest, args.split_manifest, args.output,
                                     template, limit=args.limit, selection=args.selection, seed=args.seed), indent=2))
    elif args.command == "run":
        if args.repetitions < 1:
            parser.error("repetitions must be positive")
        asyncio.run(collect(args))
    elif args.command == "rescore":
        rescore_file(args)
    else:
        rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line]
        Path(args.output).write_text(json.dumps(analyze_pairs(rows, time_field=args.time_field,
                                                           tie_ms=args.tie_ms), indent=2)+"\n")


if __name__ == "__main__":
    main()
