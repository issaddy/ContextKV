"""Optional pre-submit length proxy; serving needs only immutable inputs and stdlib.

Training labels never enter feature extraction. The artifact predicts served,
budget-capped output length, not the unobserved natural length of truncated answers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import statistics

from .schema import ConfigFingerprint, PromptCandidate, SessionSnapshot


FEATURE_NAMES = (
    "is_rebuild", "input_tokens", "message_count", "history_turns", "history_qa_chars",
    "history_mean_output_tokens", "history_last_output_tokens", "visible_evidence_count",
    "required_evidence_count", "extra_visible_evidence_count", "evidence_body_chars",
    "question_chars", "question_is_how", "question_requests_detail",
    "rebuild_x_question_chars", "rebuild_x_history_mean_output_tokens",
)
FEATURE_SCHEMA_VERSION = 1


def candidate_features(candidate: PromptCandidate, session: SessionSnapshot | None) -> dict[str, float]:
    """Counts are from the candidate and previously committed QA, never its answer."""
    history = session.raw_history if session else ()
    lengths = [len(turn.assistant_output_ids) for turn in history if turn.assistant_output_ids]
    last_index = len(candidate.messages) - 1
    text = candidate.messages[-1].content if candidate.messages else ""
    body_end = max((span.end for span in candidate.evidence_spans
                    if span.message_index == last_index), default=0)
    marker = "\nQuestion:\n"
    question_start = text.find(marker, body_end)
    question = text[question_start + len(marker):].strip() if question_start >= 0 else text
    words = set(re.findall(r"[a-z]+", question.lower()))
    rebuild = float(candidate.action == "REBUILD")
    history_mean = statistics.mean(lengths) if lengths else 0.0
    values = (
        rebuild, len(candidate.input_ids), len(candidate.messages), len(history),
        sum(len(turn.user_text) + len(turn.assistant_text) for turn in history),
        history_mean, lengths[-1] if lengths else 0,
        len(candidate.visible_blocks), len(candidate.required_blocks),
        len(candidate.visible_blocks - candidate.required_blocks),
        sum(span.end - span.start for span in candidate.evidence_spans), len(question),
        float("how" in words), float(bool(words & {"steps", "explain", "detail", "example", "examples"})),
        rebuild * len(question), rebuild * history_mean,
    )
    return dict(zip(FEATURE_NAMES, map(float, values)))


@dataclass(frozen=True)
class OutputLengthPrediction:
    mean_tokens: int
    p90_tokens: int
    error_tokens: float
    model_kind: str
    within_training_range: bool


class OutputLengthModel:
    """Small frozen residual regression around the existing shared history prior."""

    def __init__(self, specification: dict):
        spec = deepcopy(specification)
        if spec.get("schema_version") != 1 or spec.get("kind") != "contextkv_output_length_ridge_v1":
            raise ValueError("unsupported output-length artifact")
        self.model_kind = spec["model_kind"]
        expected = FEATURE_NAMES if self.model_kind == "candidate" else ("is_rebuild",)
        if self.model_kind not in {"candidate", "action_only"}:
            raise ValueError("unknown output-length model kind")
        schema = spec["feature_schema"]
        if schema["version"] != FEATURE_SCHEMA_VERSION or tuple(schema["names"]) != expected:
            raise ValueError("output-length feature schema mismatch")
        regression = spec["regression"]
        self.names = expected
        self.centers = tuple(map(float, regression["centers"]))
        self.scales = tuple(map(float, regression["scales"]))
        self.coefficients = tuple(map(float, regression["coefficients"]))
        self.intercept = float(regression["intercept"])
        self.target_scale = regression["target_scale"]
        if self.target_scale not in {"linear", "log"}:
            raise ValueError("unsupported output-length target scale")
        if any(len(values) != len(expected) for values in (self.centers, self.scales, self.coefficients)):
            raise ValueError("output-length coefficient dimensions differ from schema")
        if (not all(math.isfinite(x) for x in (*self.centers, *self.scales, *self.coefficients, self.intercept))
                or any(x <= 0 for x in self.scales)):
            raise ValueError("output-length coefficients/scales must be finite, scales positive")
        self.input_range = tuple(spec["training_range"]["input_tokens"])
        self.target_max_new_tokens = int(spec["target"]["max_new_tokens"])
        self.error_tokens = float(spec["uncertainty"]["absolute_error_p90_tokens"])
        self.upper_residual = float(spec["uncertainty"]["positive_residual_p90_tokens"])
        if (len(self.input_range) != 2 or not 0 <= self.input_range[0] <= self.input_range[1]
                or self.target_max_new_tokens < 1
                or not all(math.isfinite(x) and x >= 0 for x in (self.error_tokens, self.upper_residual))):
            raise ValueError("invalid output-length range, budget or residual error")
        self.specification = spec
        self.metadata = spec.get("metadata", {})

    @classmethod
    def load(cls, path: str | Path, *, expected_fingerprint: ConfigFingerprint | None = None,
             max_new_tokens: int | None = None) -> OutputLengthModel:
        model = cls(json.loads(Path(path).read_text()))
        target = model.specification["target"]
        if expected_fingerprint is not None and target["fingerprint"] != asdict(expected_fingerprint):
            raise ValueError("output-length artifact target fingerprint differs from serving")
        if max_new_tokens is not None and target["max_new_tokens"] != max_new_tokens:
            raise ValueError("output-length artifact budget differs from serving")
        return model

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(self.specification, stream, indent=2, allow_nan=False)

    def predict_prior(self, candidate: PromptCandidate, session: SessionSnapshot | None,
                      max_new_tokens: int, shared_prior) -> OutputLengthPrediction:
        """Serving identity/budget are checked once at load; no request-path hash."""
        in_range = self.input_range[0] <= len(candidate.input_ids) <= self.input_range[1]
        if not in_range:
            return OutputLengthPrediction(shared_prior.mean_tokens, shared_prior.p90_tokens,
                                          self.error_tokens, self.model_kind + ":shared_range_fallback", False)
        features = candidate_features(candidate, session)
        adjustment = self.intercept + sum(
            weight * ((features[name] - center) / scale)
            for name, weight, center, scale in zip(self.names, self.coefficients, self.centers, self.scales)
        )
        if self.target_scale == "log":
            # Bounding in log space gives the same served-length clipping without overflow.
            value = math.expm1(min(math.log1p(max_new_tokens),
                                  max(0.0, math.log1p(shared_prior.mean_tokens) + adjustment)))
        else:
            value = shared_prior.mean_tokens + adjustment
        mean = min(max_new_tokens, max(1, round(value)))
        upper = min(max_new_tokens, max(mean, math.ceil(mean + self.upper_residual)))
        return OutputLengthPrediction(mean, upper, self.error_tokens, self.model_kind, True)
