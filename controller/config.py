"""Validated run configuration and runtime assembly."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import json
from pathlib import Path

import yaml

from .candidates import CandidatePolicy
from .cost_model import CostModel
from .engine_client import SGLangEngineClient
from .events import EventLog
from .planner import PlannerKnobs
from .serialization import decode
from .sampling import SamplingRecipe
from .service import Controller
from .session_store import SessionStore
from .templates import ByteTemplate, HuggingFaceTemplate, SYSTEM_PROMPT


@dataclass(frozen=True)
class GenerationProtocol:
    system_prompt: str
    sampling: SamplingRecipe | None = None

    @classmethod
    def from_config(cls, config):
        # Read only request policy: archived model paths need not still exist.
        config = asdict(config) if is_dataclass(config) else config
        prompt = (config.get("context") or {}).get("system_prompt")
        sampling = config.get("sampling")
        return cls(SYSTEM_PROMPT if prompt is None else prompt,
                   SamplingRecipe(**sampling) if sampling is not None else None)

    def record(self):
        return {"version": 1, "system_prompt": self.system_prompt,
                "sampling_recipe": (self.sampling or SamplingRecipe()).record(),
                "citation_constraint": "required_reference_map_v1"}

@dataclass(frozen=True)
class EngineConfig:
    kind: str = "mock"
    base_url: str = "http://127.0.0.1:30000"
    timeout_seconds: float = 120.0
    stream: bool = True
    gpu_capacity_tokens: int = 65536
    host_capacity_tokens: int = 131072
    page_size: int = 1


@dataclass(frozen=True)
class ModelConfig:
    name: str | None = None
    revision: str | None = None
    namespace_hash: str = "default"
    local_files_only: bool | None = None
    scope_path: str | None = None

    def __post_init__(self):
        scope = None
        if self.scope_path is not None:
            from engine_patches.contextkv_model_scope import load_model_scope
            scope = load_model_scope(self.scope_path)
            if (self.name is not None and str(Path(self.name).resolve()) != scope.model_path
                    or self.revision is not None and self.revision != scope.identity
                    or self.local_files_only is False):
                raise ValueError("model fields conflict with the explicit local scope")
            name, revision, local = scope.model_path, scope.identity, True
        else:
            # CPU simulations have no model. Loading a real model always needs
            # an explicit supported local scope; names are not download defaults.
            name, revision = self.name, self.revision
            local = True if self.local_files_only is None else self.local_files_only
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "local_files_only", local)
        # Resolved once per configuration, not serialized as mutable session state.
        object.__setattr__(self, "_model_scope", scope)

    @property
    def scope(self):
        return self._model_scope


@dataclass(frozen=True)
class AblationConfig:
    host_cost: bool = True
    decode_cost: bool = True
    control_cost: bool = True
    epoch_retirement: bool = True


@dataclass(frozen=True)
class CostSettings:
    artifact: str | None = None
    default_output_tokens: int = 128
    p90_output_tokens: int = 256
    max_normalized_distance: float = .5
    online_calibration: bool = False
    residual_alpha: float = .1
    step_output_priors: dict = field(default_factory=dict)
    step_artifacts: dict = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    engine: EngineConfig = field(default_factory=EngineConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    context: CandidatePolicy = field(default_factory=CandidatePolicy)
    planner: PlannerKnobs = field(default_factory=PlannerKnobs)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    cost_model: CostSettings = field(default_factory=CostSettings)
    state_dir: str = "runs/local"
    calibration_path: str | None = None
    shadow: bool = False
    sampling: SamplingRecipe | None = None
    prefix_retention: dict = field(default_factory=lambda: {"mode": "native"})
    plan_gain: dict | None = None
    state_observation: dict | None = None

    def __post_init__(self):
        if self.engine.kind not in ("mock", "sglang"):
            raise ValueError("engine.kind must be mock or sglang")
        if self.context.max_new_tokens >= self.context.max_total_model_tokens:
            raise ValueError("output budget must be smaller than the total context budget")
        if self.engine.page_size < 1 or min(self.engine.gpu_capacity_tokens, self.engine.host_capacity_tokens) < 0:
            raise ValueError("invalid cache capacities/page size")
        if self.engine.kind == "sglang" and self.model.scope is None:
            raise ValueError("SGLang runs require an explicit Qwen3 model.scope_path")
        if self.model.scope is not None and self.context.max_total_model_tokens > self.model.scope.context_limit:
            raise ValueError("context budget exceeds the explicit model scope")
        if min(self.cost_model.default_output_tokens, self.cost_model.p90_output_tokens) < 1:
            raise ValueError("output priors must be positive")
        if not 0 < self.cost_model.residual_alpha <= 1 or self.cost_model.max_normalized_distance < 0:
            raise ValueError("invalid cost model calibration parameters")
        if self.cost_model.online_calibration and not all((self.ablation.host_cost,
                self.ablation.decode_cost, self.ablation.control_cost)):
            raise ValueError("cost-term ablations require frozen calibration; total residuals would restore excluded costs")
        if self.plan_gain is not None:
            if self.plan_gain["mode"] not in {"rolling_gain", "whole_plan", "observed_actions"} or self.shadow:
                raise ValueError("plan_gain enables reader-bound rolling execution without shadow fallback")
            for name in ("gain_asset", "execution_profile_asset", "reader_checkpoint", "reader_profile_id"):
                if not self.plan_gain[name]:
                    raise ValueError(f"rolling plan policy requires explicit {name}")


def load_config(path: str | Path | None = None) -> RunConfig:
    if path is None:
        return RunConfig()
    return decode(RunConfig, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def apply_study_config(config: RunConfig, study):
    """Use the same execution domain for HTTP serving and study evaluation."""
    from dataclasses import replace
    from .candidates import PLAN_SYSTEM_PROMPT
    execution = study["execution"]
    strategy = study["study"]["strategy"]
    artifact = execution.get("cost_artifact", config.cost_model.artifact)
    return replace(config, context=replace(config.context,
        max_total_model_tokens=execution["window_tokens"], max_new_tokens=execution["output_tokens"],
        system_prompt=config.context.system_prompt or (None if strategy == "B0" else PLAN_SYSTEM_PROMPT)),
        planner=replace(config.planner, strategy="cost" if strategy == "B7" else "always_rebuild",
            lambda_time=execution["lambda_time"], lambda_gpu=execution["lambda_gpu"],
            lambda_host=execution["lambda_host"]),
        cost_model=replace(config.cost_model, artifact=artifact),
        prefix_retention=execution.get("prefix_retention", config.prefix_retention),
        plan_gain=execution.get("plan_gain", config.plan_gain),
        state_observation=execution.get("state_observation", config.state_observation))


def assemble_cost_model(config, *, artifact=None, measurements=None,
                        max_new_tokens=None):
    """Pure assembly from configuration and already-read artifact contents.

    Artifact fitted settings take precedence over unfitted defaults. Explicit
    cost ablations and the effective generation policy apply last.
    Caller-specific model/cache identity checks remain at the caller boundary.
    """
    config = asdict(config) if is_dataclass(config) else config
    settings = dict(config.get("cost_model") or {})
    ablation = config.get("ablation") or {}
    overrides = {"max_new_tokens": max_new_tokens if max_new_tokens is not None else
                 config.get("context", {}).get("max_new_tokens", 512),
                 **{f"include_{name}_cost": ablation.get(f"{name}_cost", True)
                    for name in ("host", "decode", "control")}}
    for name in ("artifact", "online_calibration", "step_artifacts"):
        settings.pop(name, None)
    if settings.get("step_output_priors"):
        overrides["step_output_priors"] = settings.pop("step_output_priors")
    if artifact is not None:
        model = CostModel.from_spec(artifact, **overrides)
    elif measurements is not None:
        model = CostModel.from_measurements(measurements, **settings, **overrides)
    else:
        model = CostModel(**settings, **overrides)
    protocol = GenerationProtocol.from_config(config)
    model.configure_sampling(protocol.sampling,
        fallback_mean=settings.get("default_output_tokens", 128),
        fallback_p90=settings.get("p90_output_tokens", 256), system_prompt=protocol.system_prompt)
    return model


def load_cost_model(config, *, artifact_path=None, max_new_tokens=None):
    """Read declared files once, then use the same pure assembly everywhere."""
    config = asdict(config) if is_dataclass(config) else config
    settings = config.get("cost_model") or {}
    artifact_path = artifact_path or settings.get("artifact")
    calibration = config.get("calibration_path")
    model = assemble_cost_model(config,
        artifact=json.loads(Path(artifact_path).read_text()) if artifact_path else None,
        measurements=[json.loads(line) for line in Path(calibration).read_text().splitlines()
                      if line.strip()] if calibration and not artifact_path else None,
        max_new_tokens=max_new_tokens)
    if settings.get("step_artifacts") or model.step_models:
        from .reading import (READ_SYSTEM, COVERAGE_READ_SYSTEM, REDUCE_SYSTEM,
                              SUMMARY_READ_SYSTEM, SUMMARY_REDUCE_SYSTEM)
        prompts = {"READ": READ_SYSTEM, "REDUCE": REDUCE_SYSTEM,
                   "READ_COVERAGE": COVERAGE_READ_SYSTEM,
                   "READ_SUMMARY": SUMMARY_READ_SYSTEM, "REDUCE_SUMMARY": SUMMARY_REDUCE_SYSTEM}
        for kind, path in settings.get("step_artifacts", {}).items():
            model.step_models[kind] = CostModel.load(path, include_host_cost=model.include_host_cost,
                include_decode_cost=model.include_decode_cost, include_control_cost=model.include_control_cost)
        for kind, child in model.step_models.items():
            child.configure_sampling(GenerationProtocol.from_config(config).sampling,
                fallback_mean=child.default_output_tokens, fallback_p90=child.p90_output_tokens,
                system_prompt=prompts[kind])
    return model


def create_model_template(model: ModelConfig, *, model_path: str | None = None,
                          local_files_only: bool | None = None):
    if model.scope is None:
        raise ValueError("real model loading requires an explicit Qwen3 model.scope_path")
    path = model_path or model.name
    local = model.local_files_only if local_files_only is None else local_files_only
    if not local:
        raise ValueError("explicit local scope cannot enable remote model loading")
    model.scope.verify_files(path)
    template = HuggingFaceTemplate(path, None, model_identity=model.revision,
                                   namespace_hash=model.namespace_hash, local_files_only=True)
    model.scope.verify_fingerprint(asdict(template.fingerprint))
    return template


def create_controller(config: RunConfig) -> Controller:
    if config.engine.kind == "mock":
        from bench.mock_engine import MockEngine
    template = (ByteTemplate(namespace_hash=config.model.namespace_hash) if config.engine.kind == "mock" else
                create_model_template(config.model))
    model = load_cost_model(config)
    if config.model.scope is not None and (config.cost_model.artifact or config.calibration_path):
        manifest = model.calibration_metadata.get("profile_manifest") or {}
        bound = manifest.get("fingerprint") or manifest.get("engine", {}).get("fingerprint")
        cache_binding = manifest.get("engine", {}).get("cache_config_hash")
        if bound != asdict(template.fingerprint) or not isinstance(cache_binding, str) or not cache_binding:
            raise ValueError("explicit model scope requires a cost artifact bound to its actual fingerprint and cache configuration")
    engine = (MockEngine(fingerprint=template.fingerprint, gpu_capacity_tokens=config.engine.gpu_capacity_tokens,
                         host_capacity_tokens=config.engine.host_capacity_tokens, page_size=config.engine.page_size)
              if config.engine.kind == "mock" else
              SGLangEngineClient(config.engine.base_url, template.fingerprint,
                                 timeout=config.engine.timeout_seconds, stream=config.engine.stream, sampling=config.sampling,
                                 **({"model_scope": config.model.scope.manifest()} if config.model.scope else {})))
    state_dir = Path(config.state_dir)
    store = SessionStore(state_dir / "sessions.sqlite3")
    rolling = None
    if config.plan_gain is not None:
        from .plan_profiles import load_rolling_runtime
        from .candidates import PLAN_SYSTEM_PROMPT
        from .sampling import request_sampling
        settings = config.plan_gain
        if settings["mode"] == "rolling_gain":
            settings = {**settings, "mode": "whole_plan"}
        rolling = load_rolling_runtime(settings, template=template, store=store,
            sampling=request_sampling(config.sampling),
            answer_system_prompt=config.context.system_prompt or PLAN_SYSTEM_PROMPT,
            engine_base_url=config.engine.base_url)
    state_observer = None
    if config.state_observation is not None:
        from .plan_profiles import load_fixed_reading_observer
        state_observer = load_fixed_reading_observer(config.state_observation, template=template)
    return Controller(store, engine, template=template,
                      policy=config.context, knobs=config.planner, cost_model=model, shadow=config.shadow,
                      epoch_retirement=config.ablation.epoch_retirement,
                      online_calibration=config.cost_model.online_calibration,
                      prefix_settings=config.prefix_retention,
                      rolling_policy=rolling,
                      state_observer=state_observer,
                      events=EventLog(state_dir / "events.jsonl",
                                      engine_kind=config.engine.kind))
