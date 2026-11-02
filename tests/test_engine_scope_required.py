"""Qwen3-only native entry; generic cache protocol fixtures remain reusable."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from engine_patches import contextkv_runtime as native
import test_model_scope as fixtures

local_scope = fixtures.local_scope
ROOT = Path(__file__).resolve().parents[1]


def test_native_ngram_shares_full_cache_and_binds_its_execution_profile(local_scope, monkeypatch):
    ordinary, _ = fixtures.install_native_environment(monkeypatch, local_scope)
    normal = native.get_runtime(ordinary)
    observed = []
    for drafts in (6, 12):
        scheduler, settings = fixtures.install_native_environment(monkeypatch, local_scope)
        scheduler.spec_algorithm = SimpleNamespace(is_none=lambda: False, is_ngram=lambda: True)
        settings.speculative_algorithm = 'NGRAM'
        settings.speculative_num_draft_tokens = drafts
        settings.speculative_ngram_max_bfs_breadth = 1
        runtime = native.get_runtime(scheduler)
        assert runtime.cache is scheduler.tree_cache
        assert runtime.cache_config['decoder']['speculative_num_draft_tokens'] == drafts
        observed.append(runtime.info({})['cache_config_hash'])
        scheduler.tree_cache.session_refs._session_generations = {}
        replacement = native.get_runtime(scheduler)
        assert replacement.info({})['cache_config_hash'] == observed[-1]
        assert replacement.restart_id != runtime.restart_id
    assert len(set([normal.info({})['cache_config_hash'], *observed])) == 3


def test_json_whitespace_configuration_has_independent_calibration_identity(local_scope, monkeypatch):
    identities = []
    for compact in (False, True):
        scheduler, settings = fixtures.install_native_environment(monkeypatch, local_scope)
        settings.grammar_backend = "xgrammar"
        settings.constrained_json_disable_any_whitespace = compact
        runtime = native.get_runtime(scheduler)
        assert runtime.cache_config["decoder"]["constrained_json_disable_any_whitespace"] == compact
        identities.append(runtime.info({})["cache_config_hash"])
    assert identities[0] != identities[1]


def test_native_scheduler_requires_explicit_scope_even_with_valid_old_style_fingerprint(local_scope, monkeypatch):
    scheduler, settings = fixtures.install_native_environment(monkeypatch, local_scope)
    monkeypatch.delenv('CONTEXTKV_MODEL_SCOPE_PATH')
    settings.revision = 'legacy-unscoped-revision'
    with pytest.raises(native.ContextKVError) as caught:
        native.get_runtime(scheduler)
    assert caught.value.code == 'MODEL_SCOPE_REQUIRED'
    assert not hasattr(scheduler, 'contextkv_runtime')


def test_native_scope_rejects_non_qwen3_record_before_protocol_owner(local_scope, monkeypatch):
    scheduler, _ = fixtures.install_native_environment(monkeypatch, local_scope)
    record = json.loads(local_scope.path.read_text())
    record['config']['model_type'] = 'unsupported-family'
    local_scope.path.write_text(json.dumps(record))
    with pytest.raises(native.ContextKVError) as caught:
        native.get_runtime(scheduler)
    assert caught.value.code == 'FINGERPRINT_MISMATCH'
    assert not hasattr(scheduler, 'contextkv_runtime')


@pytest.mark.parametrize('script', ['launch_sglang.sh', 'bootstrap_sglang.sh'])
def test_scripts_reject_model_path_only_before_external_work(tmp_path, script):
    env = {**os.environ, 'CONTEXTKV_MODEL_PATH': str(tmp_path/'retired-model'),
           'CONTEXTKV_FINGERPRINT_JSON': json.dumps(asdict(local_fingerprint())),
           'PATH': str(tmp_path)}
    env.pop('CONTEXTKV_MODEL_SCOPE_PATH', None)
    result = subprocess.run(['/bin/bash', str(ROOT/'scripts'/script)], env=env, capture_output=True, text=True)
    assert result.returncode != 0 and 'CONTEXTKV_MODEL_SCOPE_PATH' in result.stderr
    assert not list(tmp_path.iterdir())


def local_fingerprint():
    from controller.schema import ConfigFingerprint
    return ConfigFingerprint()


def test_bootstrap_reuses_existing_scope_and_has_no_model_download_step():
    script = (ROOT/'scripts/bootstrap_sglang.sh').read_text()
    assert 'snapshot_download' not in script and 'huggingface_hub' not in script
    assert 'create_model_template(model)' in script and 'ModelConfig(scope_path=sys.argv[1])' in script
    assert 'SGLANG_BUILD_RUST_EXTS=none' in script
    assert not (ROOT/'engine_patches/model_manifest.json').exists()


def test_scoped_shell_reaches_only_explicit_native_launch(local_scope, tmp_path):
    checkout = tmp_path/'stock'
    package = checkout/'python/sglang'
    package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    (package/'launch_server.py').write_text('import sys,json; print(json.dumps(sys.argv[1:]))\n')
    env = {**os.environ, 'SGLANG_CHECKOUT': str(checkout), 'CONTEXTKV_MODEL_SCOPE_PATH': str(local_scope.path),
        'CONTEXTKV_FINGERPRINT_JSON': json.dumps(asdict(local_scope.fingerprint)),
        'PATH': str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH']}
    env.pop('CONTEXTKV_MODEL_PATH', None)
    result = subprocess.run(['bash', str(ROOT/'scripts/launch_sglang.sh'), '--port', '30001'],
                            env=env, text=True, capture_output=True, check=True)
    argv = json.loads(result.stdout)
    assert argv[argv.index('--model-path')+1] == local_scope.scope.model_path
    assert argv[argv.index('--tokenizer-path')+1] == local_scope.scope.model_path
    assert '--revision' not in argv
