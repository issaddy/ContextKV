"""The optional length model uses pre-submit data and preserves serving contracts."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from controller.output_length import FEATURE_NAMES, OutputLengthModel, candidate_features
from controller.schema import ConfigFingerprint, EvidenceKey, EvidenceSpan, Message, PromptCandidate, SessionSnapshot, TurnRecord


def inputs():
    key = EvidenceKey("corpus", "doc", "passage", "v1", "a" * 64)
    query = "How do I configure this?"
    body = "An embedded\nQuestion:\ninside evidence"
    content = body + "\nQuestion:\n" + query + "\n"
    candidate = PromptCandidate("case:a", "APPEND", 1, (Message("user", content),),
                                tuple(range(100)), "input", frozenset({key}), frozenset({key}),
                                evidence_spans=(EvidenceSpan(key, 0, 0, len(body), "source", (), "E1"),),
                                fingerprint=ConfigFingerprint())
    session = SessionSnapshot("session", raw_history=(TurnRecord(1, "earlier question", "earlier answer", (1,) * 40),))
    return candidate, session, query


def specification():
    candidate, _, _ = inputs()
    return {"schema_version": 1, "kind": "contextkv_output_length_ridge_v1", "model_kind": "candidate",
            "target": {"fingerprint": asdict(candidate.fingerprint), "max_new_tokens": 1024},
            "feature_schema": {"version": 1, "names": list(FEATURE_NAMES)},
            "regression": {"target_scale": "linear", "centers": [0] * len(FEATURE_NAMES),
                           "scales": [1] * len(FEATURE_NAMES), "coefficients": [-20] + [0] * (len(FEATURE_NAMES) - 1),
                           "intercept": 10},
            "training_range": {"input_tokens": [50, 200]},
            "uncertainty": {"absolute_error_p90_tokens": 30, "positive_residual_p90_tokens": 15},
            "metadata": {"training_scope": "development_exploration"}}


def test_features_use_committed_lengths_and_exact_question_boundary():
    candidate, session, query = inputs()
    feature = candidate_features(candidate, session)
    assert feature["input_tokens"] == 100
    assert feature["history_last_output_tokens"] == 40
    assert feature["question_chars"] == len(query)
    assert feature["question_is_how"] == 1
    # Bookkeeping IDs, build timing, and quote opportunity are not answer-length features.
    modified = replace(candidate, candidate_id="other", input_hash="other", build_ms=1e9, theoretical_lcp=99)
    assert candidate_features(modified, session) == feature
    assert "target" not in feature and "output" not in feature


def test_candidate_conditioned_prediction_and_range_fallback():
    candidate, session, _ = inputs()
    model = OutputLengthModel(specification())
    shared = SimpleNamespace(mean_tokens=100, p90_tokens=180)
    append = model.predict_prior(candidate, session, 1024, shared)
    rebuild = model.predict_prior(replace(candidate, action="REBUILD"), session, 1024, shared)
    assert (append.mean_tokens, rebuild.mean_tokens) == (110, 90)
    assert append.p90_tokens == 125 and append.error_tokens == 30
    fallback = model.predict_prior(replace(candidate, input_ids=tuple(range(500))), session, 1024, shared)
    assert (fallback.mean_tokens, fallback.p90_tokens) == (100, 180)
    assert not fallback.within_training_range
    assert fallback.model_kind.endswith("shared_range_fallback")


def test_log_prediction_clips_without_overflow():
    candidate, session, _ = inputs()
    spec = specification()
    spec["regression"].update(target_scale="log", intercept=10000)
    assert OutputLengthModel(spec).predict_prior(candidate, session, 1024, SimpleNamespace(mean_tokens=100)).mean_tokens == 1024
    spec["regression"]["intercept"] = -10000
    assert OutputLengthModel(spec).predict_prior(candidate, session, 1024, SimpleNamespace(mean_tokens=100)).mean_tokens == 1


def test_load_checks_identity_budget_and_feature_schema_once(tmp_path):
    path = tmp_path / "length.json"
    model = OutputLengthModel(specification())
    model.save(path)
    loaded = OutputLengthModel.load(path, expected_fingerprint=ConfigFingerprint(), max_new_tokens=1024)
    assert loaded.metadata["training_scope"] == "development_exploration"
    with pytest.raises(ValueError, match="fingerprint"):
        OutputLengthModel.load(path, expected_fingerprint=ConfigFingerprint(model_revision="wrong"))
    with pytest.raises(ValueError, match="budget"):
        OutputLengthModel.load(path, max_new_tokens=512)
    with pytest.raises(FileExistsError):
        model.save(path)
    spec = specification()
    spec["feature_schema"]["names"][0] = "current_answer_length"
    with pytest.raises(ValueError, match="schema"):
        OutputLengthModel(spec)


def test_serving_import_and_prediction_without_site_packages(tmp_path):
    artifact = tmp_path / "length.json"
    OutputLengthModel(specification()).save(artifact)
    code = """
from types import SimpleNamespace
from controller.output_length import OutputLengthModel
from controller.schema import PromptCandidate, SessionSnapshot
import sys
c = PromptCandidate('id','APPEND',0,(),tuple(range(100)),'hash',frozenset(),frozenset())
p = OutputLengthModel.load(sys.argv[1],max_new_tokens=1024).predict_prior(c,SessionSnapshot('s'),1024,SimpleNamespace(mean_tokens=100,p90_tokens=200))
assert p.mean_tokens == 110
assert 'numpy' not in sys.modules and 'scipy' not in sys.modules
"""
    subprocess.run([sys.executable, "-S", "-c", code, str(artifact)],
                   cwd=Path(__file__).resolve().parents[1], check=True)


def test_nonfinite_artifact_rejected():
    spec = specification()
    spec["regression"]["intercept"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        OutputLengthModel(json.loads(json.dumps(spec)))
