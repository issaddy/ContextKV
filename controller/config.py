"""Validated run configuration and runtime assembly."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path

import yaml

from .candidates import CandidatePolicy
from .cost_model import CostModel
from .engine_client import SGLangEngineClient
from .events import EventLog
from .mock_engine import MockEngine
from .planner import PlannerKnobs
from .serialization import decode, dumps
from .sampling import SamplingRecipe
from .service import Controller
from .session_store import SessionStore
from .templates import ByteTemplate, HuggingFaceTemplate


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
            name = "Qwen/Qwen2.5-7B-Instruct" if self.name is None else self.name
            revision = "a09a35458c702b33eeacc393d103063234e8bc28" if self.revision is None else self.revision
            local = False if self.local_files_only is None else self.local_files_only
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
    hysteresis: bool = True


@dataclass(frozen=True)
class CostSettings:
    artifact: str | None = None
    output_length_artifact: str | None = None
    uncertainty_mode: str | None = None
    admission_estimator: str | None = None
    default_output_tokens: int = 128
    p90_output_tokens: int = 256
    paired_error_ms: float = 0.0
    max_normalized_distance: float = .5
    online_calibration: bool = False
    residual_alpha: float = .1


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

    def __post_init__(self):
        if self.engine.kind not in ("mock", "sglang"):
            raise ValueError("engine.kind must be mock or sglang")
        if self.context.max_new_tokens >= self.context.max_total_model_tokens:
            raise ValueError("output budget must be smaller than the total context budget")
        if self.engine.page_size < 1 or min(self.engine.gpu_capacity_tokens, self.engine.host_capacity_tokens) < 0:
            raise ValueError("invalid cache capacities/page size")
        if self.engine.kind == "sglang" and self.model.scope is None and len(self.model.revision) != 40:
            raise ValueError("SGLang runs require a pinned model revision SHA")
        if self.model.scope is not None and self.context.max_total_model_tokens > self.model.scope.context_limit:
            raise ValueError("context budget exceeds the explicit model scope")
        if min(self.cost_model.default_output_tokens, self.cost_model.p90_output_tokens) < 1:
            raise ValueError("output priors must be positive")
        if self.cost_model.uncertainty_mode not in (None, "global", "conditional"):
            raise ValueError("uncertainty_mode must be global, conditional or null to use the artifact setting")
        if self.cost_model.admission_estimator not in (None, "strict", "native_workload", "native_budget"):
            raise ValueError("admission_estimator must be strict, native_workload or native_budget")
        if not 0 < self.cost_model.residual_alpha <= 1 or self.cost_model.max_normalized_distance < 0:
            raise ValueError("invalid cost model calibration parameters")
        if self.cost_model.online_calibration and not all((self.ablation.host_cost,
                self.ablation.decode_cost, self.ablation.control_cost)):
            raise ValueError("cost-term ablations require frozen calibration; total residuals would restore excluded costs")

    @property
    def config_hash(self):
        return hashlib.sha256(dumps(self).encode()).hexdigest()


def load_config(path: str | Path | None = None) -> RunConfig:
    if path is None:
        return RunConfig()
    return decode(RunConfig, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def create_model_template(model: ModelConfig, *, model_path: str | None = None,
                          local_files_only: bool | None = None):
    path = model_path or model.name
    local = model.local_files_only if local_files_only is None else local_files_only
    if model.scope is not None:
        if not local:
            raise ValueError("explicit local scope cannot enable remote model loading")
        model.scope.verify_files(path)
    options = {"model_identity": model.revision} if model.scope is not None else {}
    template = HuggingFaceTemplate(path, None if model.scope is not None else model.revision,
                                   namespace_hash=model.namespace_hash, local_files_only=local, **options)
    if model.scope is not None:
        model.scope.verify_fingerprint(asdict(template.fingerprint))
    return template


def create_controller(config: RunConfig) -> Controller:
    template = (ByteTemplate(namespace_hash=config.model.namespace_hash) if config.engine.kind == "mock" else
                create_model_template(config.model))
    common = dict(max_new_tokens=config.context.max_new_tokens,
                  include_host_cost=config.ablation.host_cost, include_decode_cost=config.ablation.decode_cost,
                  include_control_cost=config.ablation.control_cost)
    if config.cost_model.uncertainty_mode is not None:
        common["uncertainty_mode"] = config.cost_model.uncertainty_mode
    if config.cost_model.admission_estimator is not None:
        common["admission_estimator"] = config.cost_model.admission_estimator
    settings = dict(default_output_tokens=config.cost_model.default_output_tokens,
                    p90_output_tokens=config.cost_model.p90_output_tokens,
                    paired_error_ms=config.cost_model.paired_error_ms,
                    max_normalized_distance=config.cost_model.max_normalized_distance,
                    residual_alpha=config.cost_model.residual_alpha)
    model = CostModel(**common, **settings)
    if config.cost_model.artifact:
        model = CostModel.load(config.cost_model.artifact, **common)
    elif config.calibration_path:
        rows = [json.loads(line) for line in Path(config.calibration_path).read_text().splitlines() if line.strip()]
        model = CostModel.from_measurements(rows, **common, **settings)
    if config.model.scope is not None and (config.cost_model.artifact or config.calibration_path):
        manifest = model.calibration_metadata.get("profile_manifest") or {}
        bound = manifest.get("fingerprint") or manifest.get("engine", {}).get("fingerprint")
        cache_binding = manifest.get("engine", {}).get("cache_config_hash")
        if bound != asdict(template.fingerprint) or not isinstance(cache_binding, str) or not cache_binding:
            raise ValueError("explicit model scope requires a cost artifact bound to its actual fingerprint and cache configuration")
    model.configure_sampling(config.sampling, fallback_mean=config.cost_model.default_output_tokens,
                             fallback_p90=config.cost_model.p90_output_tokens)
    if config.cost_model.output_length_artifact:
        from .output_length import OutputLengthModel
        model.output_length_model = OutputLengthModel.load(config.cost_model.output_length_artifact,
            expected_fingerprint=template.fingerprint, max_new_tokens=config.context.max_new_tokens, sampling=config.sampling,
            expected_priors=model.effective_output_prior)
    knobs = config.planner if config.ablation.hysteresis else replace(config.planner,
                delta_abs=0, delta_rel=0, z=0, min_stable_turns=0)
    engine = (MockEngine(fingerprint=template.fingerprint, gpu_capacity_tokens=config.engine.gpu_capacity_tokens,
                         host_capacity_tokens=config.engine.host_capacity_tokens, page_size=config.engine.page_size)
              if config.engine.kind == "mock" else
              SGLangEngineClient(config.engine.base_url, template.fingerprint,
                                 timeout=config.engine.timeout_seconds, stream=config.engine.stream, sampling=config.sampling,
                                 **({"model_scope": config.model.scope.manifest()} if config.model.scope else {})))
    state_dir = Path(config.state_dir)
    return Controller(SessionStore(state_dir / "sessions.sqlite3"), engine, template=template,
                      policy=config.context, knobs=knobs, cost_model=model, shadow=config.shadow,
                      epoch_retirement=config.ablation.epoch_retirement,
                      online_calibration=config.cost_model.online_calibration, sampling=config.sampling,
                      events=EventLog(state_dir / "events.jsonl", config_hash=config.config_hash,
                                      engine_kind=config.engine.kind))
