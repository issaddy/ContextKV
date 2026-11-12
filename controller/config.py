"""Runtime settings for shared document QA and the native generation engine."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import yaml

from .evidence import EvidencePolicy
from .sampling import SamplingRecipe
from .serialization import decode
from .templates import HuggingFaceTemplate
from .cooperative_request import SYSTEM


@dataclass(frozen=True)
class EngineConfig:
    kind: str = "sglang"
    base_url: str = "http://127.0.0.1:30000"
    timeout_seconds: float = 180.0
    stream: bool = False
    gpu_capacity_tokens: int = 32768
    host_capacity_tokens: int = 61056
    page_size: int = 64


@dataclass(frozen=True)
class ModelConfig:
    name: str | None = None
    revision: str | None = None
    local_files_only: bool | None = None
    scope_path: str | None = None

    def __post_init__(self):
        scope = None
        if self.scope_path is not None:
            from engine_patches.contextkv_model_scope import load_model_scope
            scope = load_model_scope(self.scope_path)
            name, revision, local = scope.model_path, scope.identity, True
        else:
            name, revision = self.name, self.revision
            local = True if self.local_files_only is None else self.local_files_only
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "local_files_only", local)
        object.__setattr__(self, "_model_scope", scope)

    @property
    def scope(self):
        return self._model_scope



@dataclass(frozen=True)
class GenerationProtocol:
    system_prompt: str = SYSTEM
    sampling: SamplingRecipe | None = None

    @classmethod
    def from_config(cls, config):
        value = asdict(config) if isinstance(config, RunConfig) else config
        sampling = value.get("sampling")
        return cls(value.get("context", {}).get("system_prompt") or SYSTEM,
                   SamplingRecipe(**sampling) if sampling else None)

    def record(self):
        return {"system_prompt": self.system_prompt,
                "sampling_recipe": (self.sampling or SamplingRecipe()).record()}


@dataclass(frozen=True)
class RunConfig:
    engine: EngineConfig = field(default_factory=EngineConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    context: EvidencePolicy = field(default_factory=EvidencePolicy)
    state_dir: str = "runs/current"
    sampling: SamplingRecipe | None = None
    dialogue_mode: str = "contextkv"
    cooperative: dict = field(default_factory=dict)


def load_config(path=None):
    return decode(RunConfig, yaml.safe_load(Path(path).read_text())) if path else RunConfig()


def apply_study_config(config, study):
    execution = study["execution"]
    return replace(config,
        context=replace(config.context,
            max_total_model_tokens=execution["window_tokens"],
            max_new_tokens=execution["output_tokens"],
            system_prompt=config.context.system_prompt or SYSTEM),
        cooperative={**execution, **study["shared"].get("runtime", {})},
        dialogue_mode=execution.get("dialogue_mode", "contextkv"))


def create_model_template(model: ModelConfig, *, model_path: str | None = None,
                          local_files_only: bool | None = None):
    path = model_path or model.name
    local = model.local_files_only if local_files_only is None else local_files_only
    template = HuggingFaceTemplate(path, local_files_only=local)
    return template



def create_controller(config):
    from .conversation_store import ConversationStore
    from .cooperative import CooperativeController
    from .engine_client import SGLangEngineClient
    template = create_model_template(config.model)
    engine = SGLangEngineClient(config.engine.base_url,
        timeout=config.engine.timeout_seconds, sampling=config.sampling)
    store = ConversationStore(Path(config.state_dir) / "conversations.sqlite3")
    return CooperativeController(store, engine, template, settings=config.cooperative,
                                  policy=config.context, dialogue_mode=config.dialogue_mode)
