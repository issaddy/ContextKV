"""Finite owned local-scope matrices, with actual helper contracts and no GPU."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench import b6_joint_sweep as sweep
from bench import experiments
from bench.native_baseline import validate_engine_manifest
from bench.quality_acceptance import evaluator_identity
from controller.config import ModelConfig
from controller.sampling import SamplingRecipe, request_sampling
import test_b6_owned_matrix as owned_fixtures
import test_model_scope as scope_fixtures

matrix = owned_fixtures.matrix
local_scope = scope_fixtures.local_scope
RECIPE = SamplingRecipe(.7, .8, 20, 0)


def scoped_matrix(matrix, local_scope):
    model = ModelConfig(scope_path=str(local_scope.path))
    matrix.config.update(model=asdict(model), sampling=RECIPE.record())
    matrix.config['engine']['gpu_capacity_tokens'] = 16384
    matrix.args.model_path = local_scope.model_path
    matrix.fp = local_scope.fingerprint
    for profile in matrix.spec['server_profiles'].values():
        profile.update(gpu_tokens=16384, hicache_gb=8, mem_fraction_static=.8)
    matrix.spec['server_profiles']['B0'] = {**matrix.spec['server_profiles']['stock'],
        'hicache': False, 'session_radix': False, 'write_policy': None, 'hicache_gb': 0}
    native = {'base_url': matrix.config['engine']['base_url'], 'model_path': model.name,
              'local_files_only': True, 'audit_prompt_ids': True}
    matrix.spec['methods'] = [
        {'id': 'Rebuild', 'server_profile': 'patched', 'overrides': {'planner': {'strategy': 'always_rebuild'}}},
        {'id': 'B1', 'server_profile': 'stock', 'kind': 'native', 'native': {**native, 'baseline': 'B1'}},
        {'id': 'B0', 'server_profile': 'B0', 'kind': 'native', 'native': {**native, 'baseline': 'B0'}}]
    matrix.spec['repetitions'] = [1]
    matrix.spec['reference_method'] = 'B0'
    matrix.spec['tuning'].update(equal_budget=1, quality_constraints={'reference': 'B0', 'epsilon': .01})
    owned_fixtures.write_spec(matrix)
    return model


def fake_scoped_hardware(matrix, local_scope, monkeypatch, *, corrupt=None):
    from controller import config
    from profiling import runtime_manifest
    seen, launched, stopped, templates, state = [], [], [], [], {}
    monkeypatch.setenv('RAYON_NUM_THREADS', '4')
    monkeypatch.setenv('CONTEXTKV_MODEL_SCOPE_PATH', '/unrelated/inherited-scope.json')
    def template(path, revision, **kwargs):
        templates.append((path, revision, kwargs))
        return SimpleNamespace(fingerprint=local_scope.fingerprint)
    # Exercise the real shared factory, including its file/inventory validation.
    monkeypatch.setattr(config, 'HuggingFaceTemplate', template)
    monkeypatch.setattr(sweep.research, 'check_research_checkout', lambda path: {'path': str(path), 'research_patch_applied': True})
    monkeypatch.setattr(sweep.server, 'check_stock', lambda path: {'path': str(path), 'research_patch_applied': False})
    monkeypatch.setattr(sweep.server, 'command', lambda argv, env: str(Path(env['SGLANG_CHECKOUT'])/'python/sglang/__init__.py'))
    def launch(argv, **kwargs):
        state.update(env=kwargs['env'], argv=argv)
        launched.append((argv, kwargs['env']))
        if kwargs['env']['CONTEXTKV_ENABLE_HICACHE'] == '1':
            kwargs['stdout'].write('Allocating kv hierarchical KV host pool: 30464 tokens, 8.00 GB host memory.\n')
            kwargs['stdout'].flush()
        return SimpleNamespace(pid=800+len(launched), returncode=0)
    monkeypatch.setattr(sweep.subprocess, 'Popen', launch)
    monkeypatch.setattr(sweep.server, 'wait_ready', lambda *args: None)
    monkeypatch.setattr(sweep.server, 'stop_owned_server', lambda process: stopped.append(process.pid))
    monkeypatch.setattr(sweep.server, 'get_text', lambda *args: '# fixture')
    async def runtime(**kwargs):
        env = state['env']
        host = env['CONTEXTKV_ENABLE_HICACHE'] == '1'
        info = {**sweep.COMMON_SERVER, 'max_total_num_tokens': 16384,
            'model_path': local_scope.scope.model_path, 'tokenizer_path': local_scope.scope.model_path,
            'revision': None, 'dtype': 'bfloat16', 'kv_cache_dtype': 'bfloat16', 'pp_size': 1,
            'quantization': None, 'json_model_override_args': '{}', 'load_format': 'safetensors',
            'mem_fraction_static': .8, 'enable_session_radix_cache': host, 'enable_hierarchical_cache': host,
            'disable_radix_cache': False, 'hicache_size': 8 if host else 0,
            'hicache_write_policy': env['CONTEXTKV_WRITE_POLICY'], 'hicache_host_memory_mode': 'cache',
            'hicache_io_backend': 'kernel', 'hicache_mem_layout': 'layer_first', 'incremental_streaming_output': False}
        result = {'runtime': {'gpus': [{'index': 0, 'uuid': 'cpu-fixture', 'name': 'fake', 'memory_bytes': 1}],
            'driver_cuda_torch_versions': {'driver': 'fake', 'cuda': 'fake', 'torch': 'fake'}, 'collection_errors': []},
            'server_info': info,
            'engine': {'engine_kind': 'sglang', 'profiling_enabled': True, 'sglang_commit': sweep.server.COMMIT,
                'fingerprint': asdict(local_scope.fingerprint), 'model_scope': local_scope.scope.manifest(),
                'cache_config_hash': 'scope-cache'},
            'pool': {'gpu_capacity_tokens': 16384, 'host_capacity_tokens': 30464},
            'cache_config': {'gpu_capacity_tokens': 16384, 'page_size': 64, 'hicache': True,
                'host_memory_mode': 'cache', 'host_layout': 'layer_first', 'hicache_io_backend': 'kernel',
                'write_policy': env['CONTEXTKV_WRITE_POLICY'], 'tma_transfer_enabled': False, 'sample_pool': True}}
        if corrupt:
            corrupt(result)
        return result
    monkeypatch.setattr(runtime_manifest, 'build_manifest', runtime)
    async def metrics(*args):
        return {'host_admission': {'config': {'mode': 'off'}}, 'pool': {'unique_gpu_tokens': 0, 'unique_host_tokens': 0}}
    monkeypatch.setattr(sweep, 'research_metrics', metrics)
    async def execute(spec, job, directory, *, native_engine_manifest, verified_server):
        seen.append(job)
        assert verified_server['status'] == 'running'
        assert experiments.matrix_config(spec).sampling == RECIPE
        if job['kind'] == 'native':
            manifest = json.loads(native_engine_manifest.read_text())
            assert not validate_engine_manifest(manifest, job['native']['baseline'],
                SimpleNamespace(fingerprint=local_scope.fingerprint), model_scope=local_scope.scope, sampling=RECIPE)
            assert manifest['model']['revision'] is None
            assert manifest['sampling_recipe'] == request_sampling(RECIPE)
            assert manifest['runtime']['actual_kv_pool_tokens'] == 16384
            assert manifest['runtime']['actual_host_pool_tokens'] == (30464 if job['native']['baseline'] == 'B1' else 0)
            if job['native']['baseline'] == 'B1':
                assert manifest['runtime']['kv_bytes_per_token'] == 262144  # explicit head_dim128, not hidden/heads80
        directory.mkdir(parents=True)
        path = directory/'run.json'
        rows = sweep.selected_cohort(spec, 'dev')
        path.write_text(json.dumps({'report': {'turns': [{'task_id': row['task_id'], 'status': 'success'} for row in rows]}}))
        return {**job, 'status': 'completed', 'raw_output': str(path), 'artifact_sha256': {},
            'summary': {'metric_schema_version': 2, 'measurement_kind': 'fake_process_contract', 'timing': {'failure_fraction': 0},
                        'cohort': {'unfinished_turns': 0}}}
    monkeypatch.setattr(sweep, 'execute_job', execute)
    monkeypatch.setattr(experiments, 'paired_latency', lambda *args, **kwargs: {'status': 'not_measured'})
    return seen, launched, stopped, templates


def test_scoped_plan_requires_no_model_http_or_gpu_work(matrix, local_scope, monkeypatch):
    scoped_matrix(matrix, local_scope)
    def forbidden(*args, **kwargs):
        pytest.fail('plan performed execution work')
    monkeypatch.setattr(sweep, 'create_model_template', forbidden)
    monkeypatch.setattr(sweep.subprocess, 'Popen', forbidden)
    plan = sweep.build_matrix_plan(matrix.args)
    assert plan['planned_cells'] == 3 and plan['planned_turns'] == 6
    assert plan['config']['model']['revision'] == local_scope.scope.identity
    assert not matrix.args.output.exists()


@pytest.mark.parametrize('change', ['missing_fraction', 'invalid_fraction', 'mixed_fraction', 'mixed_host', 'mixed_recipe'])
def test_scoped_resource_and_recipe_fail_before_start(matrix, local_scope, change):
    scoped_matrix(matrix, local_scope)
    profiles = matrix.spec['server_profiles']
    if change == 'missing_fraction':
        profiles['patched'].pop('mem_fraction_static')
    elif change == 'invalid_fraction':
        profiles['patched']['mem_fraction_static'] = float('nan')
    elif change == 'mixed_fraction':
        profiles['patched']['mem_fraction_static'] = .7
    elif change == 'mixed_host':
        profiles['patched']['hicache_gb'] = 16
    else:
        matrix.spec['methods'][0]['overrides']['sampling'] = {'temperature': 0}
    owned_fixtures.write_spec(matrix)
    with pytest.raises(ValueError):
        sweep.build_matrix_plan(matrix.args)
    assert not matrix.args.output.exists()


def test_owned_scope_real_factory_manifest_pool_generator_and_cleanup(matrix, local_scope, monkeypatch):
    scoped_matrix(matrix, local_scope)
    seen, launched, stopped, templates = fake_scoped_hardware(matrix, local_scope, monkeypatch)
    assert sweep.execute(matrix.args, sweep.build_matrix_plan(matrix.args)) == 0
    assert len(seen) == len(launched) == len(stopped) == 3
    assert templates == [(local_scope.scope.model_path, None,
        {'namespace_hash': 'default', 'local_files_only': True, 'model_identity': local_scope.scope.identity})]
    assert all(env['CONTEXTKV_MODEL_SCOPE_PATH'] == str(local_scope.path) for _, env in launched)
    assert all(argv[argv.index('--mem-fraction-static')+1] == '0.8' for argv, _ in launched)
    assert all('--revision' not in argv for argv, _ in launched)
    assert [env['CONTEXTKV_HICACHE_GB'] for _, env in launched] == ['8', '8', '0']
    index = json.loads((matrix.args.output/'quality-evidence.template.json').read_text())
    block = json.loads(Path(index['blocks'][0]['quality_config']).read_text())
    for value in block['methods'].values():
        generator = value['generator']
        assert generator['revision'] == local_scope.scope.identity and generator['hf_revision'] is None
        assert generator['identity_kind'] == 'local_snapshot_metadata' and generator['base_family'] == 'Qwen3'
        judge = {'source': 'offline_judge', 'judge': {'model': 'independent-judge', 'revision': 'judge-commit',
                 'prompt': 'frozen-rubric', 'provider': 'local', 'base_family': 'different'}}
        assert evaluator_identity(judge, generator)[0] == 'independent_model_declared'
        judge['judge']['model'] = generator['model']
        assert evaluator_identity(judge, generator)[0] == 'self_judge'


@pytest.mark.parametrize('field,value', [('mem_fraction_static', .15), ('revision', 'fake-hf-sha'),
    ('load_format', 'dummy'), ('hicache_size', 16), ('max_total_num_tokens', 32768)])
def test_actual_scope_or_resource_mismatch_never_reaches_generation(matrix, local_scope, monkeypatch, field, value):
    scoped_matrix(matrix, local_scope)
    matrix.spec['methods'] = [matrix.spec['methods'][0]]
    matrix.spec['reference_method'] = 'Rebuild'
    matrix.spec['tuning']['require_quality_evidence'] = False
    owned_fixtures.write_spec(matrix)
    seen, launched, stopped, _ = fake_scoped_hardware(matrix, local_scope, monkeypatch,
        corrupt=lambda result: result['server_info'].update({field: value}))
    assert sweep.execute(matrix.args, sweep.build_matrix_plan(matrix.args)) == 1
    assert seen == [] and len(launched) == len(stopped) == 1


@pytest.mark.parametrize('binding', ['old_model', 'unknown_sampling', 'matching'])
def test_cost_binding_cannot_reuse_old_model_or_unbound_new_recipe(matrix, local_scope, monkeypatch, binding):
    from controller.cost_model import CostModel, LookupPoint, DecodePoint
    from controller.schema import ConfigFingerprint
    scoped_matrix(matrix, local_scope)
    matrix.spec['methods'] = [matrix.spec['methods'][0]]
    matrix.spec['reference_method'] = 'Rebuild'
    matrix.spec['tuning']['require_quality_evidence'] = False
    artifact = matrix.args.matrix.parent/'cost.json'
    manifest = {'engine': {'fingerprint': asdict(ConfigFingerprint() if binding == 'old_model' else local_scope.fingerprint),
                          'cache_config_hash': 'scope-cache'}, 'cache_config': {'gpu_capacity_tokens': 16384}}
    if binding != 'unknown_sampling':
        manifest['sampling_recipe'] = RECIPE.record()
    CostModel([LookupPoint(0, 512, 1, 10)], [DecodePoint(512, 1, 1)]).save(artifact,
        metadata={'profile_manifest': manifest})
    matrix.spec['methods'][0]['overrides'].update(planner={'strategy': 'cost'}, cost_model={'artifact': str(artifact)})
    owned_fixtures.write_spec(matrix)
    seen, launched, stopped, _ = fake_scoped_hardware(matrix, local_scope, monkeypatch)
    assert sweep.execute(matrix.args, sweep.build_matrix_plan(matrix.args)) == int(binding != 'matching')
    assert len(seen) == int(binding == 'matching')
    assert len(launched) == len(stopped) == 1


def test_owned_native_reuses_shared_model_path_without_duplicate_declaration(matrix, local_scope):
    scoped_matrix(matrix,local_scope)
    for method in matrix.spec['methods']:
        if method.get('kind')=='native':
            method['native'].pop('model_path')
    owned_fixtures.write_spec(matrix)
    plan=sweep.build_matrix_plan(matrix.args)
    assert len(plan['jobs'])==3
    assert plan['config']['model']['name']==str(local_scope.model_path)
