"""Cost assembly shared by serving and offline tools; no engines or state stores."""
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

from .cost_model import CostModel
from .generation import GenerationProtocol
from .output_length import OutputLengthModel


def assemble_cost_model(config, *, artifact=None, measurements=None, length_artifact=None,
                        fingerprint=None, max_new_tokens=None):
    """Pure assembly from configuration and already-read artifact contents.

    Artifact fitted settings take precedence over unfitted defaults. Explicit
    cost ablations/estimators and the effective generation policy apply last.
    Caller-specific model/cache identity checks remain at the caller boundary.
    """
    config = asdict(config) if is_dataclass(config) else config
    settings = dict(config.get("cost_model") or {})
    ablation = config.get("ablation") or {}
    overrides = {"max_new_tokens": max_new_tokens if max_new_tokens is not None else
                 config.get("context", {}).get("max_new_tokens", 512),
                 **{f"include_{name}_cost": ablation.get(f"{name}_cost", True)
                    for name in ("host", "decode", "control")}}
    for name in ("uncertainty_mode", "admission_estimator"):
        value = settings.pop(name, None)
        if value is not None:
            overrides[name] = value
    for name in ("artifact", "output_length_artifact", "online_calibration"):
        settings.pop(name, None)
    if artifact is not None:
        model = CostModel.from_spec(artifact, **overrides)
    elif measurements is not None:
        model = CostModel.from_measurements(measurements, **settings, **overrides)
    else:
        model = CostModel(**settings, **overrides)
    model.configure_for_run(config)
    if length_artifact is not None:
        protocol = GenerationProtocol.from_config(config)
        model.output_length_model = OutputLengthModel.from_spec(length_artifact,
            expected_fingerprint=fingerprint, max_new_tokens=model.max_new_tokens,
            sampling=protocol.sampling, expected_priors=model.effective_output_prior,
            expected_generation_protocol=protocol.record())
    return model


def load_cost_model(config, *, artifact_path=None, fingerprint=None, max_new_tokens=None):
    """Read declared files once, then use the same pure assembly everywhere."""
    config = asdict(config) if is_dataclass(config) else config
    settings = config.get("cost_model") or {}
    artifact_path = artifact_path or settings.get("artifact")
    calibration = config.get("calibration_path")
    length_path = settings.get("output_length_artifact")
    return assemble_cost_model(config,
        artifact=json.loads(Path(artifact_path).read_text()) if artifact_path else None,
        measurements=[json.loads(line) for line in Path(calibration).read_text().splitlines()
                      if line.strip()] if calibration and not artifact_path else None,
        length_artifact=json.loads(Path(length_path).read_text()) if length_path else None,
        fingerprint=fingerprint, max_new_tokens=max_new_tokens)
