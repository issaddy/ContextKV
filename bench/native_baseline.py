"""Unpatched SGLang B0/B1 baselines, deliberately separate from the research bridge.

B0 submits a complete rebuilt prompt with no session ID. B1 submits the same
prompt policy with one stable native session ID across all context changes.
Neither path quotes, prepares epochs, manufactures receipts, or claims native
close_session provides a scheduler acknowledgement or frees physical KV.
"""
from __future__ import annotations
import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
import os
import re
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
from types import SimpleNamespace
import uuid

import httpx

from adapters.mtrag import CLOUD_COLLECTION, DATASET_COMMIT, OnlineTurn, load_rows
from adapters.corpus import CorpusIndex
from adapters.retrieval import to_evidence
from adapters.retriever_factory import create_retriever
from bench.run_sessions import run_sessions
from bench.quality import export_evaluation
from controller.candidates import CandidatePolicy, build_candidates
from controller.schema import ClientTiming, SessionSnapshot, TurnRecord
from controller.sampling import SamplingRecipe, bound_recipe, request_sampling
from controller.serialization import to_jsonable
from controller.templates import Template
from profiling.common import ENGINE_COMMIT, LEGACY_MODEL_REVISION


class NativeProtocolError(RuntimeError):
    """An incomplete or invalid native response must not enter committed history."""


@dataclass(frozen=True)
class NativeGeneration:
    text: str
    output_ids: tuple[int, ...]
    finish_reason: str
    truncated: bool
    meta_info: dict[str, Any]
    raw_response: dict[str, Any]
    http_elapsed_ms: float
    timing: ClientTiming | None = None


@dataclass(frozen=True)
class NativeTurnResponse:
    assistant_text: str
    baseline: str
    session_id: str
    task_id: str
    native_session_id: str | None
    input_ids: tuple[int, ...]
    input_hash: str
    output_ids: tuple[int, ...]
    finish_reason: str
    truncated: bool
    meta_info: dict[str, Any]
    raw_response: dict[str, Any]
    http_elapsed_ms: float
    visible_evidence_manifest: tuple[Any, ...]
    audit_evidence: tuple[Any, ...]
    branch_reason: str
    measurement_kind: str
    timing: ClientTiming | None = None


@dataclass(frozen=True)
class NativeCloseDispatch:
    session_id: str
    http_status: int
    dispatch_finished_at: float
    acknowledgement: str = "not_scheduler_ack"
    physical_free_observed: bool = False


def validate_engine_manifest(manifest: Mapping[str, Any], baseline: str,
                             template: Template | None = None, *, model_scope=None,
                             sampling: SamplingRecipe | None = None) -> list[str]:
    """Read-only manifest validation, including historical records; not execution permission."""
    errors = []
    if manifest.get("example_only") is True:
        errors.append("example manifest cannot authorize a formal hardware run")
    if baseline not in {"B0", "B1"}:
        return ["baseline must be B0 or B1"]
    expected = {"engine": {"base_commit": ENGINE_COMMIT, "research_patch_applied": False,
                           "tree_backend": "python"},
                "model": {"name": "Qwen/Qwen2.5-7B-Instruct", "revision": LEGACY_MODEL_REVISION,
                          "dtype": "bfloat16", "kv_dtype": "bfloat16"},
                "server": {"enable_session_radix_cache": baseline == "B1",
                           "enable_hierarchical_cache": baseline == "B1",
                           "disable_radix_cache": False, "tp_size": 1, "dp_size": 1}}
    if model_scope is not None:
        expected["model"] = {"name": model_scope.model_path, "revision": None,
            "identity_kind": model_scope.manifest()["record"]["identity_kind"], "identity": model_scope.identity,
            "dtype": "bfloat16", "kv_dtype": "bfloat16"}
        expected["server"]["pp_size"] = 1
        declared = manifest.get("model_scope") or {}
        record = declared.get("record") or {}
        if (declared.get("identity") != model_scope.identity
                or not record.get("model_path")
                or Path(record["model_path"]).resolve() != Path(model_scope.model_path)):
            errors.append("model_scope must bind the same explicit local identity and canonical path")
    if template is not None:
        expected["model"].update(tokenizer_hash=template.fingerprint.tokenizer_hash,
                                 chat_template_hash=template.fingerprint.template_hash)
        if model_scope is None and template.fingerprint.model_revision != LEGACY_MODEL_REVISION:
            errors.append("template must use the pinned real model revision")
        elif model_scope is not None:
            try:
                model_scope.verify_fingerprint(asdict(template.fingerprint))
            except ValueError as error:
                errors.append(str(error))
    if model_scope is not None or sampling is not None or "sampling_recipe" in manifest:
        try:
            bound = bound_recipe(manifest.get("sampling_recipe"))
            requested = sampling or SamplingRecipe()
            legacy_greedy = model_scope is None and "sampling_recipe" not in manifest and requested == SamplingRecipe()
            if bound != requested and not legacy_greedy:
                errors.append("sampling_recipe must explicitly match the native request configuration")
        except (ValueError, TypeError) as error:
            errors.append(str(error))
    for section, fields in expected.items():
        for name, expected_value in fields.items():
            value = manifest.get(section, {}).get(name)
            if value != expected_value or type(value) is not type(expected_value):
                errors.append(f"{section}.{name} must equal {expected_value!r}")
    gpu = manifest.get("runtime", {}).get("gpu_model")
    if not isinstance(gpu, str) or not gpu or gpu.startswith("TO_"):
        errors.append("runtime.gpu_model must identify actual hardware")
    versions = manifest.get("runtime", {}).get("driver_cuda_torch_versions")
    if not isinstance(versions, dict) or any(not isinstance(versions.get(k), str) or not versions[k]
            or versions[k].startswith("TO_") for k in ("driver", "cuda", "torch")):
        errors.append("runtime.driver_cuda_torch_versions must record driver, cuda and torch")
    for section, name in (("runtime", "gpu_memory_bytes"), ("runtime", "actual_kv_pool_tokens"),
                          ("server", "page_size"), ("server", "context_length")):
        value = manifest.get(section, {}).get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"{section}.{name} must be a positive measured integer")
    host = manifest.get("runtime", {}).get("actual_host_pool_bytes")
    if not isinstance(host, int) or isinstance(host, bool) or (host <= 0 if baseline == "B1" else host != 0):
        errors.append("runtime.actual_host_pool_bytes must match the baseline host configuration")
    for name in ("tokenizer_hash", "chat_template_hash"):
        value = manifest.get("model", {}).get(name)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            errors.append(f"model.{name} must be a computed SHA256")
    return errors


class NativeSGLangClient:
    """Only native endpoints, with no retry of ambiguous generation outcomes.

    Optional sampling settings are fixed at construction; the per-call output
    budget remains authoritative. Request-level seeds are deliberately outside
    this interface; their native behavior depends on the inference backend.
    """
    def __init__(self, base_url: str, *, timeout_seconds: float = 120,
                 http_client: httpx.AsyncClient | None = None, audit_prompt_ids: bool = False,
                 stream: bool = False, sampling_params: Mapping[str, Any] | None = None):
        sampling = {"temperature": 0}
        if sampling_params is not None:
            settings = dict(sampling_params)
            allowed = {"temperature", "top_p", "top_k", "min_p"}
            if settings.keys() - allowed:
                raise ValueError("unsupported native sampling settings: " + ", ".join(sorted(settings.keys() - allowed)))
            for name in ("temperature", "top_p", "min_p"):
                if name not in settings:
                    continue
                value = settings[name]
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f"{name} must be a finite number")
                if ((name == "temperature" and value < 0)
                        or (name == "top_p" and not 0 < value <= 1)
                        or (name == "min_p" and not 0 <= value <= 1)):
                    raise ValueError(f"invalid native sampling setting: {name}")
            if "top_k" in settings and (type(settings["top_k"]) is not int
                                       or (settings["top_k"] != -1 and settings["top_k"] < 1)):
                raise ValueError("top_k must be -1 or a positive integer")
            sampling.update(settings)
        self._sampling_params = sampling
        self._owned = http_client is None
        self.http = http_client or httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout_seconds)
        self.audit_prompt_ids = audit_prompt_ids
        self.stream = stream
        self.observations: list[dict[str, Any]] = []

    async def server_info(self) -> dict[str, Any]:
        response = await self.http.get("/server_info")
        if response.status_code == 404:
            response = await self.http.get("/get_server_info")
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise NativeProtocolError("native server_info must be an object")
        if self.stream and result.get("server_args", result).get("incremental_streaming_output", False):
            raise NativeProtocolError("native streaming requires incremental_streaming_output=false (fixed-source default)")
        return result

    async def generate(self, input_ids: Sequence[int], *, max_new_tokens: int,
                       session_id: str | None = None, rid: str | None = None,
                       json_schema: Mapping[str, Any] | None = None) -> NativeGeneration:
        if not input_ids or any(type(token) is not int or token < 0 for token in input_ids):
            raise ValueError("input_ids must be actual nonnegative integer token IDs")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        payload = {"input_ids": list(input_ids), "stream": self.stream,
                   "sampling_params": {**self._sampling_params, "max_new_tokens": max_new_tokens},
                   "return_prompt_token_ids": self.audit_prompt_ids}
        if json_schema is not None:
            if not isinstance(json_schema, Mapping):
                raise ValueError("json_schema must be a schema object")
            # The native SamplingParams field is a JSON string. Compile/validate
            # its grammar in the existing engine; it applies to this call only.
            payload["sampling_params"]["json_schema"] = json.dumps(dict(json_schema), allow_nan=False)
        if session_id is not None:
            payload["session_id"] = session_id
        if rid is not None:
            payload["rid"] = rid
        start = time.monotonic()
        first = last = None
        if self.stream:
            data, first, last = await self._stream_generate(payload)
        else:
            response = await self.http.post("/generate", json=payload)
            try:
                data = response.json()
            except ValueError:
                data = {"invalid_json_body": response.text}
            self.observations.append({"request": deepcopy(payload), "http_status": response.status_code,
                                      "response": deepcopy(data), "http_elapsed_ms": (time.monotonic() - start) * 1000})
            response.raise_for_status()
        response_done = time.monotonic()
        elapsed = (time.monotonic() - start) * 1000
        if not isinstance(data, dict) or data.get("error"):
            raise NativeProtocolError("native generation returned an error/non-object")
        meta = data.get("meta_info")
        if not isinstance(meta, dict):
            raise NativeProtocolError("native generation lacks meta_info")
        reason = meta.get("finish_reason")
        kind = reason.get("type") if isinstance(reason, dict) else reason
        if kind not in {"stop", "length", "eos", "max_tokens"}:
            raise NativeProtocolError(f"native generation is incomplete/aborted: {kind!r}")
        text, ids = data.get("text"), data.get("output_ids")
        if not isinstance(text, str) or not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids):
            raise NativeProtocolError("native generation must return actual text and output_ids")
        if len(ids) > max_new_tokens:
            raise NativeProtocolError("native output exceeds the shared generation budget")
        if "completion_tokens" in meta and meta["completion_tokens"] != len(ids):
            raise NativeProtocolError("native completion token counter disagrees with output_ids")
        if self.audit_prompt_ids and data.get("prompt_token_ids") != list(input_ids):
            raise NativeProtocolError("server prompt_token_ids disagree with submitted input")
        return NativeGeneration(text, tuple(ids), str(kind), kind in {"length", "max_tokens"},
                                deepcopy(meta), deepcopy(data), elapsed,
                                ClientTiming(start, t_submit=start, t_first=first, t_last_token=last,
                                             t_response_done=response_done, token_observation="chunk" if self.stream else "none"))

    async def _stream_generate(self, payload):
        """Read native cumulative SSE snapshots; commit requires terminal + DONE.

        Fixed SGLang defaults incremental_streaming_output=false. CLI verifies
        this against server_info; delta mode must be disabled for this baseline.
        Metadata-only frames do not move either observed token timestamp.
        """
        first = last = None
        final = None
        ids = []
        done = False
        lines = []
        started = time.monotonic()
        observation = {"request": deepcopy(payload), "stream": True, "http_status": None,
                       "stream_frames": 0, "partial_output_tokens": 0, "complete_stream": False}
        try:
            async with self.http.stream("POST", "/generate", json=payload) as response:
                observation["http_status"] = response.status_code
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        lines.append(line[5:].lstrip())
                        continue
                    if line or not lines:
                        continue
                    raw = "\n".join(lines)
                    lines.clear()
                    if raw == "[DONE]":
                        done = True
                        break
                    item = json.loads(raw)
                    observation["stream_frames"] += 1
                    if not isinstance(item, dict) or item.get("error"):
                        raise NativeProtocolError("native stream returned an error/non-object")
                    current = item.get("output_ids", [])
                    if not isinstance(current, list) or len(current) < len(ids):
                        raise NativeProtocolError("native cumulative stream token count decreased")
                    if len(current) > len(ids):
                        at = time.monotonic()
                        first = at if first is None else first
                        last = at
                        ids = current
                        observation["partial_output_tokens"] = len(ids)
                    reason = item.get("meta_info", {}).get("finish_reason")
                    if reason is not None:
                        final = item
                if not done or final is None or lines:
                    raise NativeProtocolError("incomplete native stream: terminal frame and SSE DONE required")
                if final.get("output_ids") != ids:
                    raise NativeProtocolError("terminal native output disagrees with received token sequence")
                observation["response"] = deepcopy(final)
                observation["complete_stream"] = True
                return final, first, last
        except (httpx.HTTPError, ValueError) as exc:
            observation["error"] = f"{type(exc).__name__}: {exc}"
            raise NativeProtocolError("native stream failed; partial answer was not committed") from exc
        except NativeProtocolError as exc:
            observation["error"] = str(exc)
            raise
        finally:
            observation["http_elapsed_ms"] = (time.monotonic() - started) * 1000
            self.observations.append(observation)

    async def close_session(self, session_id: str) -> NativeCloseDispatch:
        response = await self.http.post("/close_session", json={"session_id": session_id})
        response.raise_for_status()
        return NativeCloseDispatch(session_id, response.status_code, time.monotonic())

    async def aclose(self):
        if self._owned:
            await self.http.aclose()


class NativeBaseline:
    """Shared evidence policy and full-history rebuilds over native engine caching.

    State is in-memory workload state, not research epoch/receipt recovery. A
    network-ambiguous generation aborts that session through the common runner.
    """
    def __init__(self, baseline: str, client: NativeSGLangClient, template: Template, *,
                 corpus_snapshot_id: str, policy: CandidatePolicy | None = None,
                 engine_manifest: Mapping[str, Any] | None = None, test_mode: bool = False,
                 run_id: str | None = None, corpus_index: CorpusIndex | None = None,
                 model_scope=None, sampling: SamplingRecipe | None = None):
        if baseline not in {"B0", "B1"}:
            raise ValueError("native baseline must be B0 or B1")
        if not test_mode:
            if model_scope is None:
                raise ValueError("formal native generation requires an explicit Qwen3 model scope")
            errors = validate_engine_manifest(engine_manifest or {}, baseline, template,
                                              model_scope=model_scope, sampling=sampling)
            if errors:
                raise ValueError("invalid engine manifest: " + "; ".join(errors))
            if bound_recipe(client._sampling_params) != (sampling or SamplingRecipe()):
                raise ValueError("native client sampling differs from declared shared configuration")
        self.baseline, self.client, self.template = baseline, client, template
        self.corpus_snapshot_id = corpus_snapshot_id
        self.policy = policy or CandidatePolicy()
        self.corpus_index = corpus_index
        if client.stream and (engine_manifest or {}).get("server", {}).get("incremental_streaming_output", False):
            raise NativeProtocolError("native streaming requires cumulative output; disable incremental_streaming_output")
        if not test_mode and engine_manifest["server"]["context_length"] != self.policy.max_total_model_tokens:
            raise ValueError("native server context length differs from the shared model budget")
        self.engine_manifest = deepcopy(dict(engine_manifest or {}))
        self.model_scope = model_scope
        self.sampling = sampling
        self.test_mode = test_mode
        self.run_id = run_id or uuid.uuid4().hex
        self.snapshots: dict[str, SessionSnapshot] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._native_sessions: set[str] = set()
        self.close_dispatches: list[NativeCloseDispatch] = []
        self.observed_server_info: dict[str, Any] | None = None

    async def verify_server(self) -> dict[str, Any]:
        info = await self.client.server_info()
        self.observed_server_info = deepcopy(info)
        if not self.test_mode:
            observed = info.get("server_args", info)
            if self.model_scope is not None:
                try:
                    self.model_scope.verify_native_settings(SimpleNamespace(**observed))
                except (AttributeError, ValueError) as error:
                    raise NativeProtocolError(f"actual native model settings differ from scope: {error}") from error
                missing = set(self.engine_manifest["server"]) - set(observed)
                if missing:
                    raise NativeProtocolError(f"scoped native server_info lacks required settings: {sorted(missing)}")
                if info.get("max_total_num_tokens") != self.engine_manifest["runtime"]["actual_kv_pool_tokens"]:
                    raise NativeProtocolError("actual native GPU pool differs from scoped engine manifest")
            for name, expected in self.engine_manifest["server"].items():
                # Absent fields remain operator-attested, never falsely verified.
                if name in observed and observed[name] != expected:
                    raise NativeProtocolError(f"server_info differs from engine manifest: {name}")
        return info

    async def __call__(self, turn: OnlineTurn) -> NativeTurnResponse:
        arrived = time.monotonic()
        async with self._locks.setdefault(turn.conversation_id, asyncio.Lock()):
            snapshot = self.snapshots.get(turn.conversation_id, SessionSnapshot(
                turn.conversation_id, corpus_snapshot_id=self.corpus_snapshot_id))
            if turn.turn != snapshot.version + 1:
                raise NativeProtocolError("native baseline requires serial, contiguous turns")
            history = tuple(TurnRecord(i // 2 + 1, turn.history[i].text, turn.history[i + 1].text)
                            for i in range(0, len(turn.history), 2))
            if turn.protocol == "P1":
                committed = {t.turn_id: t for t in snapshot.raw_history}
                history = tuple(replace(t, evidence_keys=committed[t.turn_id].evidence_keys)
                                if t.turn_id in committed and committed[t.turn_id].user_text == t.user_text else t
                                for t in history)
            if turn.protocol != "P1":
                actual = tuple((t.user_text, t.assistant_text) for t in snapshot.raw_history)
                supplied = tuple((t.user_text, t.assistant_text) for t in history)
                if actual != supplied:
                    raise NativeProtocolError("generated history differs from committed native outputs")
                history = snapshot.raw_history
            snapshot = replace(snapshot, raw_history=history, execution=None)
            retrieval_priority = self.corpus_index.retrieved_keys(turn.contexts) if self.corpus_index else None
            retrieved = (self.corpus_index.resolve(retrieval_priority) if self.corpus_index else
                         tuple(to_evidence(c, self.corpus_snapshot_id) for c in turn.contexts))
            # execution=None yields one REBUILD and still uses the same shared
            # current/explicit-revisit/dependency evidence selection and stable IDs.
            candidates = build_candidates(snapshot, turn.query, retrieved, self.policy, self.template,
                                          retrieval_priority=retrieval_priority)
            candidate = next(c for c in candidates if c.action == "REBUILD")
            if not candidate.feasible:
                raise NativeProtocolError(candidate.reason or "infeasible shared prompt")
            native_sid = f"native:{self.run_id}:{turn.conversation_id}" if self.baseline == "B1" else None
            if native_sid is not None:
                self._native_sessions.add(native_sid)
            result = await self.client.generate(candidate.input_ids, max_new_tokens=self.policy.max_new_tokens,
                                                session_id=native_sid, rid=f"{self.run_id}:{turn.task_id}")
            # Candidate construction has already validated this turn's evidence.
            archive = {block.key: block for block in (*snapshot.archive, *retrieved)}
            record = TurnRecord(turn.turn, turn.query, result.text, result.output_ids,
                                result.finish_reason, tuple(sorted(candidate.required_blocks)))
            self.snapshots[turn.conversation_id] = replace(snapshot, version=snapshot.version + 1,
                raw_history=history + (record,), raw_history_version=snapshot.raw_history_version + 1,
                archive=tuple(archive[key] for key in sorted(archive)), reference_map=candidate.reference_map)
            timing = replace(result.timing, t_arrive=arrived, t_commit=time.monotonic()) if result.timing else None
            return NativeTurnResponse(result.text, self.baseline, turn.conversation_id, turn.task_id,
                native_sid, candidate.input_ids, candidate.input_hash, result.output_ids, result.finish_reason,
                result.truncated, result.meta_info, result.raw_response, result.http_elapsed_ms,
                candidate.evidence_spans, tuple(archive[k] for k in sorted(candidate.required_blocks)),
                "reference_history_sync" if turn.protocol == "P1" and turn.history else "baseline_full_rebuild",
                "fake_http_correctness" if self.test_mode else "native_hardware_unvalidated", timing)

    async def close_all(self) -> tuple[NativeCloseDispatch, ...]:
        errors = []
        for session_id in sorted(tuple(self._native_sessions)):
            try:
                dispatch = await self.client.close_session(session_id)
            except Exception as exc:
                errors.append(f"{session_id}: {type(exc).__name__}: {exc}")
                continue
            self.close_dispatches.append(dispatch)
            self._native_sessions.remove(session_id)
        if errors:
            raise NativeProtocolError("native close dispatch failures: " + "; ".join(errors))
        return tuple(self.close_dispatches)


def resolve_context_policy(args):
    """Keep the matrix's complete policy; CLI flags override only explicit values."""
    policy = getattr(args, "context_policy", None)
    if policy is None:
        policy = CandidatePolicy()
    overrides = {}
    for argument, field in (("context_length", "max_total_model_tokens"),
                            ("max_new_tokens", "max_new_tokens"),
                            ("table_relations", "table_relations"),
                            ("table_annotation_style", "table_annotation_style"),
                            ("table_structure_mode", "table_structure_mode")):
        value = getattr(args, argument, None)
        if value is not None:
            overrides[field] = value
    system_prompt = getattr(args, "system_prompt", None)
    if getattr(args, "system_prompt_file", None) is not None:
        with Path(args.system_prompt_file).open("r", encoding="utf-8", newline="") as stream:
            system_prompt = stream.read()
    # Keep the resolved body in existing workload/policy provenance, including
    # intentional leading/trailing whitespace. No separate prompt hash is needed.
    if system_prompt is not None:
        overrides["system_prompt"] = system_prompt
    policy = replace(policy, **overrides)
    args.system_prompt = policy.system_prompt
    args.context_length = policy.max_total_model_tokens
    args.max_new_tokens = policy.max_new_tokens
    args.table_relations = policy.table_relations
    args.table_annotation_style = policy.table_annotation_style
    args.table_structure_mode = policy.table_structure_mode
    return policy


async def _run(args):
    from controller.config import create_model_template, load_config
    shared = load_config(args.run_config) if getattr(args, "run_config", None) else None
    model = shared.model if shared is not None else getattr(args, "model_config", None)
    if model is None or model.scope is None:
        raise NativeProtocolError("formal native generation requires a shared explicit Qwen3 model.scope_path")
    sampling = shared.sampling if shared is not None else getattr(args, "sampling", None)
    if shared is not None:
        args.context_policy = shared.context
    policy = resolve_context_policy(args)
    manifest_path = Path(args.engine_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scope = model.scope
    template = create_model_template(model)
    rows = load_rows(args.input)
    if args.domain == "cloud":
        rows = [r for r in rows if r["Collection"] == CLOUD_COLLECTION]
    if args.split_manifest:
        split = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
        selected = set(split[args.split])
        rows = [r for r in rows if r["conversation_id"] in selected]
    if args.limit_sessions:
        selected = set(sorted({r["conversation_id"] for r in rows})[:args.limit_sessions])
        rows = [r for r in rows if r["conversation_id"] in selected]
    corpus_snapshot = f"mtrag-{DATASET_COMMIT}"
    retriever = create_retriever(args.protocol, config_path=getattr(args, "retriever_config", None),
                                index_path=getattr(args, "retriever_index", None), corpus_path=args.corpus)
    if retriever is not None:
        corpus_snapshot = retriever.corpus_snapshot_id
    corpus_index = (CorpusIndex.from_jsonl(args.evidence_corpus, corpus_id="mtrag", snapshot_id=corpus_snapshot)
                    if getattr(args, "evidence_corpus", None) else None)
    client = NativeSGLangClient(args.base_url, audit_prompt_ids=args.audit_prompt_ids,
                               stream=not getattr(args, "no_stream", False),
                               sampling_params=request_sampling(sampling) if sampling is not None else None)
    baseline = NativeBaseline(args.baseline, client, template, policy=policy,
                              corpus_snapshot_id=corpus_snapshot, engine_manifest=manifest, corpus_index=corpus_index,
                              model_scope=scope, sampling=sampling)
    retrieval_context_provider = None
    if getattr(retriever, "requires_history_context", False):
        from adapters.query_resolution import native_context_provider
        retrieval_context_provider = native_context_provider(baseline)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = None
    close_error = None

    def export_result(item):
        if item.status != "success":
            return
        result = item.result
        export_evaluation(item.online_turn, result.assistant_text,
                          official_path=output / f"official-{baseline.run_id}.jsonl",
                          audit_path=output / f"audit-{baseline.run_id}.jsonl",
                          visible_evidence_manifest=result.visible_evidence_manifest,
                          audit_evidence=result.audit_evidence, input_ids=result.input_ids,
                          output_ids=result.output_ids, branch_reason=result.branch_reason,
                          measurement_kind="unvalidated_hardware_run", truncated=result.truncated)
    try:
        await baseline.verify_server()
        report = await run_sessions(rows, baseline, protocol=args.protocol, concurrency=args.concurrency,
                                    think_time=args.think_time, arrival_rate=args.arrival_rate,
                                    arrival_window=args.arrival_window, drain_timeout=args.drain_timeout,
                                    per_turn_timeout=args.per_turn_timeout, retriever=retriever, seed=args.seed,
                                    retrieval_context_provider=retrieval_context_provider,
                                    on_result=export_result)
    finally:
        cleanup_start = time.monotonic()
        try:
            await baseline.close_all()
        except Exception as exc:
            close_error = f"{type(exc).__name__}: {exc}"
        cleanup_end = time.monotonic()
        await client.aclose()
        payload = {"baseline": args.baseline, "protocol": args.protocol, "run_id": baseline.run_id,
                   "engine_manifest": manifest, "engine_manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(),
                   "engine_commit_provenance": "operator_manifest_not_remote_attestation",
                   "server_info": baseline.observed_server_info, "input_sha256": sha256(Path(args.input).read_bytes()).hexdigest(),
                   "split_manifest_sha256": sha256(Path(args.split_manifest).read_bytes()).hexdigest() if args.split_manifest else None,
                   "template_fingerprint": asdict(template.fingerprint), "policy": asdict(policy),
                   "retrieval": None if retriever is None else retriever.manifest,
                   "workload": vars(args), "report": report, "native_http_observations": client.observations,
                   "client_requested_thread_environment": {key: os.environ.get(key) for key in
                       ("TOKENIZERS_PARALLELISM", "RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                   "close_dispatches": baseline.close_dispatches, "close_error": close_error,
                   "undispatched_native_sessions": sorted(baseline._native_sessions),
                   "cleanup_started_at": cleanup_start, "cleanup_finished_at": cleanup_end,
                   "scheduler_retirement_ack_available": False, "physical_free_measured": False,
                   "gpu_performance_validated": False, "quality_evaluated": False}
        if shared is not None or model is not None or sampling is not None:
            payload["shared_configuration"] = {"model": to_jsonable(model), "sampling_recipe": request_sampling(sampling),
                                               "context": to_jsonable(policy)}
        target = output / f"native-{args.baseline}-{baseline.run_id}.json"
        target.write_text(json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(target.resolve()), "baseline": args.baseline,
                          "success": sum(r.status == "success" for r in report.turns) if report else 0,
                          "close_acknowledgement": "not_scheduler_ack"}))
        if retriever is not None and hasattr(retriever, "close"):
            retriever.close()
    if close_error:
        raise NativeProtocolError(close_error)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="MTRAG RAG.jsonl or an explicit generated subset")
    parser.add_argument("--baseline", choices=["B0", "B1"], required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--engine-manifest", required=True)
    parser.add_argument("--run-config", required=True, help="shared Qwen3 model.scope_path, context and sampling RunConfig")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--system-prompt", help="Override the shared system prompt body")
    prompt.add_argument("--system-prompt-file", help="Read the shared system prompt verbatim from UTF-8 text")
    parser.add_argument("--protocol", choices=["P1", "P2", "P3"], default="P2")
    parser.add_argument("--corpus")
    parser.add_argument("--retriever-index", help="reuse a saved pinned BM25S index")
    parser.add_argument("--retriever-config", help="optional shared P3 hybrid retriever YAML/JSON")
    parser.add_argument("--evidence-corpus", help="original passages with version/condition metadata shared with controller")
    parser.add_argument("--table-relations", action=argparse.BooleanOptionalAction, default=None,
                        help="override literal source table field associations")
    parser.add_argument("--table-annotation-style", choices=["all_rows", "focus_refs"],
                        help="optional annotation style; used only with --table-relations")
    parser.add_argument("--table-structure-mode", choices=["legacy", "explicit"],
                        help="optional literal unnumbered-table/parent-path parser; requires --table-relations")
    parser.add_argument("--domain", choices=["cloud", "all"], default="cloud")
    parser.add_argument("--split-manifest")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--output", default="runs/native")
    parser.add_argument("--limit-sessions", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--think-time", type=float, default=0)
    parser.add_argument("--arrival-rate", type=float)
    parser.add_argument("--arrival-window", type=float)
    parser.add_argument("--drain-timeout", type=float)
    parser.add_argument("--per-turn-timeout", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-length", type=int, help="override the shared context budget (default 32768)")
    parser.add_argument("--max-new-tokens", type=int, help="override the shared output budget (default 512)")
    parser.add_argument("--audit-prompt-ids", action="store_true")
    parser.add_argument("--no-stream", action="store_true", help="disable client token chunk timing (TTFT/TPOT unavailable)")
    args = parser.parse_args(argv)
    report = asyncio.run(_run(args))
    return int(bool(report.unfinished_turns or report.timed_out_sessions or
                    any(turn.status != "success" for turn in report.turns)))


if __name__ == "__main__":
    raise SystemExit(main())
