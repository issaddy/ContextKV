"""Validated run configuration and runtime assembly."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
from pathlib import Path

import yaml

from .candidates import CandidatePolicy
from .cost_model import CostModel
from .generation import GenerationProtocol
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

    @property
    def config_hash(self):
        return hashlib.sha256(dumps(self).encode()).hexdigest()


def load_config(path: str | Path | None = None) -> RunConfig:
    if path is None:
        return RunConfig()
    return decode(RunConfig, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


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
    for name in ("artifact", "online_calibration"):
        settings.pop(name, None)
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
    return assemble_cost_model(config,
        artifact=json.loads(Path(artifact_path).read_text()) if artifact_path else None,
        measurements=[json.loads(line) for line in Path(calibration).read_text().splitlines()
                      if line.strip()] if calibration and not artifact_path else None,
        max_new_tokens=max_new_tokens)


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
    return Controller(SessionStore(state_dir / "sessions.sqlite3"), engine, template=template,
                      policy=config.context, knobs=config.planner, cost_model=model, shadow=config.shadow,
                      epoch_retirement=config.ablation.epoch_retirement,
                      online_calibration=config.cost_model.online_calibration,
                      events=EventLog(state_dir / "events.jsonl", config_hash=config.config_hash,
                                      engine_kind=config.engine.kind))
