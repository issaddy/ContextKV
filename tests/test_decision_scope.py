"""Real collection is scope-bound; retained inputs remain readable offline."""
import asyncio
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from controller.config import ModelConfig, RunConfig
from controller.templates import ByteTemplate
from profiling import decision_pairs as pairs
from test_decision_pairs import source_run
from test_model_scope import local_scope as local_scope


def test_recorded_policy_does_not_instantiate_a_retired_runtime(tmp_path):
    database, manifest, split, _, _ = source_run(tmp_path)
    historical = deepcopy(manifest)
    historical['config']['engine']['kind'] = 'sglang'
    historical['config']['model'] = {'name': 'retired-model', 'revision': 'a' * 40}
    cases, _ = pairs.freeze_cases(database, historical, split, ByteTemplate())
    assert len(cases) == 1


def test_freeze_cli_uses_shared_local_scope_factory(tmp_path, local_scope, monkeypatch):
    model = ModelConfig(scope_path=str(local_scope.path))
    manifest = tmp_path / 'manifest.json'
    config = asdict(RunConfig(model=model))
    config['engine']['kind'] = 'sglang'
    manifest.write_text(json.dumps({'config': config}))
    expected, seen = object(), []
    def factory(actual, **kwargs):
        assert actual.scope.identity == local_scope.scope.identity
        seen.append(kwargs)
        return expected
    monkeypatch.setattr(pairs, 'create_model_template', factory)
    def freeze(*args, **kwargs):
        assert args[4] is expected
        return {'scope': 'cpu wiring only'}
    monkeypatch.setattr(pairs, 'write_frozen', freeze)
    pairs.main(['freeze', '--database', 'unused', '--run-manifest', str(manifest),
                '--split-manifest', 'unused', '--output', str(tmp_path / 'cases')])
    assert seen == [{'model_path': None, 'local_files_only': True}]


def test_collection_rejects_foreign_frozen_model_before_engine(tmp_path, local_scope, monkeypatch):
    database, manifest, split, _, _ = source_run(tmp_path)
    cases, _ = pairs.freeze_cases(database, manifest, split, ByteTemplate())
    monkeypatch.setattr(pairs, 'load_frozen', lambda _: (cases, {'source_measurement_kind': 'hardware'}))
    monkeypatch.setattr(pairs, 'SGLangEngineClient', lambda *a, **kw: pytest.fail('created engine for foreign model'))
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'engine': {'kind': 'sglang'}, 'model': {'scope_path': str(local_scope.path)}}))
    with pytest.raises(ValueError, match='fingerprint'):
        asyncio.run(pairs.collect(SimpleNamespace(cases='unused', config=str(config), cost_artifact=None)))
