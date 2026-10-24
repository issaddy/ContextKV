"""Synthetic owned-server contracts; no GPU, weights, HTTP, or real test data."""
from argparse import Namespace
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
import yaml

from bench import b6_joint_sweep as sweep
from bench import experiments
from controller.cost_model import CostModel, LookupPoint, DecodePoint
from controller.schema import ConfigFingerprint
from test_native_policy_sweep import MODEL_CONFIG, HOST_LOG


def save(path, value):
    Path(path).write_text(json.dumps(value))


def read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture
def matrix(tmp_path):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    model = tmp_path/'model'
    model.mkdir()
    save(model/'config.json', MODEL_CONFIG)
    fp = ConfigFingerprint(model_revision=sweep.server.MODEL_REVISION, tokenizer_hash='a'*64,
                           template_hash='b'*64, namespace_hash='owned-fixture')
    config = {'model': {'name': str(model), 'revision': sweep.server.MODEL_REVISION,
                       'namespace_hash': fp.namespace_hash, 'local_files_only': True},
        'engine': {'kind': 'sglang', 'base_url': f'http://127.0.0.1:{port}', 'gpu_capacity_tokens': 32768, 'page_size': 64},
        'context': {'max_total_model_tokens': 32768, 'max_new_tokens': 256, 'system_prompt': 'Keep exact shared prompt.\n'},
        'planner': {'strategy': 'always_rebuild', 'concurrency': 2}, 'shadow': False}
    config_path = tmp_path/'unchanged-base.yaml'
    config_path.write_text(yaml.safe_dump(config))
    data, split = tmp_path/'synthetic.jsonl', tmp_path/'split.json'
    rows = [{'conversation_id': f's{i}', 'task_id': f's{i}:1', 'turn': 1, 'Collection': 'synthetic',
             'input': [], 'contexts': []} for i in range(3)]
    data.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    save(split, {'dev': ['s0', 's1'], 'test': ['s2']})
    profiles = {kind: {'kind': kind, 'write_policy': 'write_through', 'hicache': True, 'session_radix': True,
                       'gpu_tokens': 32768, 'hicache_gb': 16, 'checkout': str(tmp_path/kind)}
                for kind in ('patched', 'stock')}
    spec = {'schema_version': 1, 'base_config': str(config_path), 'input': str(data), 'split_manifest': str(split),
        'domain': 'all', 'protocols': ['P2'], 'concurrency': [2], 'seeds': [17], 'repetitions': [1, 2],
        'reference_method': 'Native', 'server_profiles': profiles,
        'methods': [{'id': 'Fixed', 'server_profile': 'patched', 'b6_eligible': True, 'tune': True,
                     'overrides': {'planner': {'strategy': 'fixed_period'}},
                     'candidates': [{'planner': {'period': 2}}, {'planner': {'period': 4}}]},
                    {'id': 'Native', 'server_profile': 'stock', 'kind': 'native',
                     'native': {'baseline': 'B1', 'base_url': config['engine']['base_url'],
                                'model_path': str(model), 'local_files_only': True, 'audit_prompt_ids': True}}],
        'tuning': {'equal_budget': 2, 'require_quality_evidence': True,
                   'quality_constraints': {'reference': 'Native', 'epsilon': .01}}}
    path = tmp_path/'matrix.yaml'
    path.write_text(yaml.safe_dump(spec))
    args = Namespace(matrix=path, stage='dev', frozen=None, output=tmp_path/'new-output', port=port,
                     model_path=model, startup_timeout=1, job_timeout=1)
    return SimpleNamespace(args=args, spec=spec, config=config, config_path=config_path, fp=fp, rows=rows)


def write_spec(fixture):
    fixture.args.matrix.write_text(yaml.safe_dump(fixture.spec))
    fixture.config_path.write_text(yaml.safe_dump(fixture.config))


def freeze(fixture, *, simulation=False):
    # Synthetic boundary fixture, not a scored hardware development selection.
    frozen = {'provenance': experiments.provenance(fixture.spec), 'simulation_only': simulation,
        'b6': 'Fixed', 'winners': {'Fixed': {'candidate_index': 1, 'candidate_overrides': {'planner': {'period': 4}}},
                                 'Native': {'candidate_index': 0, 'candidate_overrides': {}}}}
    path = fixture.args.matrix.parent/'synthetic-frozen.json'
    save(path, frozen)
    fixture.args.frozen = path
    return frozen


def fake_hardware(fixture, monkeypatch, *, runtime_cache='cache-32', incomplete_job=None):
    from profiling import runtime_manifest
    seen, launched, stopped, current = [], [], [], {}
    monkeypatch.setenv('RAYON_NUM_THREADS', '4')
    def template(model, revision, **kwargs):
        assert model == fixture.config['model']['name'] and revision == sweep.server.MODEL_REVISION
        assert kwargs['namespace_hash'] == fixture.fp.namespace_hash
        return SimpleNamespace(fingerprint=fixture.fp)
    monkeypatch.setattr(sweep, 'create_model_template', lambda model: template(model.name, model.revision, namespace_hash=model.namespace_hash))
    monkeypatch.setattr(sweep.research, 'check_research_checkout', lambda path: {'path': str(path), 'research_patch_applied': True})
    monkeypatch.setattr(sweep.server, 'check_stock', lambda path: {'path': str(path), 'research_patch_applied': False})
    monkeypatch.setattr(sweep.server, 'command', lambda argv, env: str(Path(env['SGLANG_CHECKOUT'])/'python/sglang/__init__.py'))
    def launch(argv, **kwargs):
        current.update(env=kwargs['env'], argv=argv)
        launched.append(argv)
        kwargs['stdout'].write(HOST_LOG+'\n')
        kwargs['stdout'].flush()
        return SimpleNamespace(pid=900+len(launched), returncode=0)
    monkeypatch.setattr(sweep.subprocess, 'Popen', launch)
    monkeypatch.setattr(sweep.server, 'wait_ready', lambda *a: None)
    monkeypatch.setattr(sweep.server, 'stop_owned_server', lambda process: stopped.append(process.pid))
    monkeypatch.setattr(sweep.server, 'get_text', lambda *a: '# synthetic metrics')
    async def runtime(**kwargs):
        env, argv = current['env'], current['argv']
        capacity = int(argv[argv.index('--max-total-tokens')+1])
        return {'runtime': {'gpus': [{'index': 0, 'uuid': 'fixture', 'name': 'fake', 'memory_bytes': 1}],
                           'driver_cuda_torch_versions': {'driver': 'fake', 'cuda': 'fake', 'torch': 'fake'}, 'collection_errors': []},
            'server_info': {**sweep.COMMON_SERVER, 'max_total_tokens': capacity, 'max_total_num_tokens': capacity,
                'model_path': env['CONTEXTKV_MODEL_PATH'], 'tokenizer_path': env['CONTEXTKV_MODEL_PATH'],
                'revision': sweep.server.MODEL_REVISION, 'dtype': 'bfloat16', 'kv_cache_dtype': 'bfloat16',
                'enable_session_radix_cache': True, 'enable_hierarchical_cache': True, 'disable_radix_cache': False,
                'hicache_size': 16, 'hicache_write_policy': env['CONTEXTKV_WRITE_POLICY'], 'hicache_host_memory_mode': 'cache',
                'hicache_io_backend': 'kernel', 'hicache_mem_layout': 'layer_first', 'incremental_streaming_output': False},
            'engine': {'engine_kind': 'sglang', 'profiling_enabled': True, 'sglang_commit': sweep.server.COMMIT,
                       'fingerprint': asdict(fixture.fp), 'cache_config_hash': runtime_cache},
            'pool': {'gpu_capacity_tokens': capacity, 'host_capacity_tokens': 279040},
            'cache_config': {'gpu_capacity_tokens': capacity, 'page_size': 64, 'hicache': True,
                'host_memory_mode': 'cache', 'host_layout': 'layer_first', 'hicache_io_backend': 'kernel',
                'write_policy': env['CONTEXTKV_WRITE_POLICY'], 'tma_transfer_enabled': False, 'sample_pool': True}}
    monkeypatch.setattr(runtime_manifest, 'build_manifest', runtime)
    async def metrics(*a):
        return {'host_admission': {'config': {'mode': 'off'}}, 'pool': {'unique_gpu_tokens': 0, 'unique_host_tokens': 0}}
    monkeypatch.setattr(sweep, 'research_metrics', metrics)
    async def execute(spec, job, directory, *, native_engine_manifest, verified_server):
        seen.append(job)
        assert verified_server['profile']['gpu_tokens'] == 32768 and verified_server['status'] == 'running'
        assert experiments.matrix_config(spec).context.system_prompt == 'Keep exact shared prompt.\n'
        assert experiments.matrix_config(spec).context.max_new_tokens == 256
        if job['kind'] == 'native':
            assert read(native_engine_manifest)['runtime']['actual_kv_pool_tokens'] == 32768
        else:
            assert native_engine_manifest is None
        directory.mkdir(parents=True)
        rows = sweep.selected_cohort(spec, 'test' if job['split'] == 'test' else 'dev')
        turns = [{'task_id': row['task_id'], 'status': 'success'} for row in rows]
        if job['job_id'] == incomplete_job:
            turns.pop()
        path = directory/'synthetic-run.json'
        save(path, {'report': {'turns': turns}})
        return {**job, 'status': 'completed', 'raw_output': str(path), 'artifact_sha256': {},
            'summary': {'measurement_kind': 'synthetic_process_contract', 'timing': {'failure_fraction': 0},
                        'cohort': {'unfinished_turns': int(job['job_id'] == incomplete_job)}}}
    monkeypatch.setattr(sweep, 'execute_job', execute)
    monkeypatch.setattr(experiments, 'paired_latency', lambda *a, **kw: {'status': 'not_measured_fixture'})
    return seen, launched, stopped


def cost(fixture, *, cache='cache-32', capacity=32768, points=True):
    path = fixture.args.matrix.parent/'cost.json'
    model = CostModel([LookupPoint(0, 512, 1, 10)] if points else [],
                      [DecodePoint(512, 1, 1)] if points else [])
    model.save(path, metadata={'profile_manifest': {'run_id': 'synthetic-binding',
        'engine': {'fingerprint': asdict(fixture.fp), 'cache_config_hash': cache},
        'cache_config': {'gpu_capacity_tokens': capacity}}})
    return str(path)


def test_generic_plan_keeps_existing_paths_prompt_budget_and_has_no_launch_side_effects(matrix, monkeypatch, capsys):
    original = matrix.args.matrix.read_bytes(), matrix.config_path.read_bytes()
    def forbidden(*a, **kw):
        pytest.fail('plan reached a process/model/source check')
    monkeypatch.setattr(sweep, 'execute', forbidden)
    monkeypatch.setattr(sweep.server, 'command', forbidden)
    assert sweep.main(['--matrix', str(matrix.args.matrix), '--output', str(matrix.args.output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan['matrix'] == matrix.spec and plan['config']['context']['max_new_tokens'] == 256
    assert plan['planned_cells'] == 6 and plan['planned_turns'] == 12
    assert plan['config']['context']['system_prompt'] == matrix.config['context']['system_prompt']
    assert not matrix.args.output.exists()
    assert original == (matrix.args.matrix.read_bytes(), matrix.config_path.read_bytes())


@pytest.mark.parametrize('field', ['model', 'endpoint', 'context', 'profile_mode', 'capacity', 'mock', 'native_endpoint', 'native_model_missing', 'mixed_capacity'])
def test_unsupported_or_cross_job_resource_changes_fail_before_launch(matrix, field):
    method = matrix.spec['methods'][0]
    if field == 'model':
        method['overrides']['model'] = {'namespace_hash': 'different'}
    elif field == 'endpoint':
        method['overrides']['engine'] = {'base_url': 'http://127.0.0.1:9999'}
    elif field == 'context':
        matrix.config['context']['max_total_model_tokens'] = 8192
    elif field == 'profile_mode':
        matrix.spec['server_profiles']['patched']['host_admission'] = 'conditioned'
    elif field == 'capacity':
        matrix.spec['server_profiles']['patched']['gpu_tokens'] = 16384
    elif field == 'mock':
        method['overrides']['engine'] = {'kind': 'mock'}
    elif field == 'native_endpoint':
        matrix.spec['methods'][1]['native']['base_url'] = 'http://127.0.0.1:9999'
    elif field == 'native_model_missing':
        matrix.spec['methods'][1]['native'].pop('model_path')
    else:
        matrix.spec['server_profiles']['patched']['gpu_tokens'] = 16384
        method['overrides']['engine'] = {'gpu_capacity_tokens': 16384}
    write_spec(matrix)
    with pytest.raises(ValueError):
        sweep.build_matrix_plan(matrix.args)
    assert not matrix.args.output.exists()


def test_repeated_dev_starts_real_declared_capacity_and_writes_job_id_quality_blocks(matrix, monkeypatch):
    seen, launched, stopped = fake_hardware(matrix, monkeypatch)
    originals = matrix.args.matrix.read_bytes(), matrix.config_path.read_bytes()
    plan = sweep.build_matrix_plan(matrix.args)
    assert sweep.execute(matrix.args, plan) == 0
    assert len(seen) == len(launched) == len(stopped) == 6
    assert all(argv[argv.index('--max-total-tokens')+1] == '32768' for argv in launched)
    assert all(argv[argv.index('--random-seed')+1] == '42' for argv in launched)
    assert {job['seed'] for job in seen} == {17}  # workload seed is not the fixed engine RNG setting
    assert originals == (matrix.args.matrix.read_bytes(), matrix.config_path.read_bytes())
    assert not (matrix.args.output/'matrix.yaml').exists() and not (matrix.args.output/'controller.yaml').exists()
    index = read(matrix.args.output/'quality-evidence.template.json')
    assert index['schema_version'] == 2 and len(index['blocks']) == 2
    for entry in index['blocks']:
        config = read(entry['quality_config'])
        expected = {job['job_id'] for job in seen if job['repetition'] == entry['repetition']}
        assert set(config['methods']) == expected and config['reference'].startswith('Native-c0-')
        assert config['frozen_protocol'] is None and all(value['scores'] is None for value in config['methods'].values())


def test_failed_cohort_keeps_denominator_and_cleans_then_continues(matrix, monkeypatch):
    plan = sweep.build_matrix_plan(matrix.args)
    missing = plan['jobs'][0]['job_id']
    seen, launched, stopped = fake_hardware(matrix, monkeypatch, incomplete_job=missing)
    assert sweep.execute(matrix.args, plan) == 1
    assert len(seen) == len(launched) == len(stopped) == 6
    completion = read(matrix.args.output/'completion.json')
    assert completion['expected_turns_per_job'] == 2 and completion['status'] == 'generation_failures'
    rows = [json.loads(line) for line in (matrix.args.output/'matrix-results/results.jsonl').read_text().splitlines()]
    assert len(rows) == 6 and rows[0]['server_execution']['status'] == 'generation_failures'


def test_frozen_test_uses_only_winning_candidate_and_actual_test_cohort(matrix, monkeypatch):
    freeze(matrix)
    matrix.args.stage = 'test'
    seen, launched, stopped = fake_hardware(matrix, monkeypatch)
    plan = sweep.build_matrix_plan(matrix.args)
    assert plan['cohort_turns'] == 1 and plan['selected_sessions'] == ['s2']
    assert sweep.execute(matrix.args, plan) == 0
    assert len(seen) == len(launched) == len(stopped) == 4
    assert sum(job.get('method_aliases') == ['B6'] for job in seen) == 2
    aliases = read(matrix.args.output/'matrix-results/method-aliases.json')
    assert aliases['B6']['canonical_method'] == 'Fixed'
    assert aliases['B6']['job_ids'] == [job['job_id'] for job in seen if job['method'] == 'Fixed']
    assert all(job['candidate_index'] == 1 and job['overrides']['planner']['period'] == 4
               for job in seen if job['kind'] == 'controller')
    assert all(job['split'] == 'test' for job in seen)
    assert not (matrix.args.output/'quality-evidence.template.json').exists()


@pytest.mark.parametrize('invalid', ['missing', 'changed_candidate', 'simulation'])
def test_invalid_frozen_test_stops_before_source_model_or_owned_start(matrix, monkeypatch, invalid):
    if invalid != 'missing':
        freeze(matrix, simulation=invalid == 'simulation')
    if invalid == 'changed_candidate':
        matrix.spec['methods'][0]['candidates'][0]['planner']['period'] = 8
        write_spec(matrix)
    matrix.args.stage = 'test'
    def forbidden(*a, **kw):
        pytest.fail('invalid frozen selection reached server/model/source work')
    monkeypatch.setattr(sweep, 'make_owned_executor', forbidden)
    monkeypatch.setattr(sweep.server, 'command', forbidden)
    with pytest.raises(ValueError, match='frozen|simulation-only'):
        sweep.build_matrix_plan(matrix.args)
    assert not matrix.args.output.exists()


def test_matched_ablation_inherits_parent_profile_and_frozen_candidate(matrix, monkeypatch):
    matrix.spec['methods'].append({'id': 'no_retire', 'ablation_of': 'Fixed',
        'overrides': {'ablation': {'epoch_retirement': False}}})
    write_spec(matrix)
    freeze(matrix)
    matrix.args.stage = 'dev_ablation'
    plan = sweep.build_matrix_plan(matrix.args)
    assert len(plan['jobs']) == 2 and all(j['server_profile'] == 'patched' for j in plan['jobs'])
    assert all(j['overrides']['planner']['period'] == 4 and not j['overrides']['ablation']['epoch_retirement'] for j in plan['jobs'])
    seen, _, stopped = fake_hardware(matrix, monkeypatch)
    assert sweep.execute(matrix.args, plan) == 0
    assert len(seen) == len(stopped) == 2 and all(j['method'] == 'no_retire' and j['split'] == 'dev' for j in seen)
    assert not (matrix.args.output/'quality-evidence.template.json').exists()


def test_matched_ablation_cannot_change_declared_hardware(matrix):
    matrix.spec['server_profiles']['other'] = deepcopy(matrix.spec['server_profiles']['patched'])
    matrix.spec['methods'].append({'id': 'no_retire', 'ablation_of': 'Fixed', 'server_profile': 'other',
                                  'overrides': {'ablation': {'epoch_retirement': False}}})
    with pytest.raises(ValueError, match='inherit the parent server_profile'):
        experiments.matrix_jobs(matrix.spec)


@pytest.mark.parametrize('strategy,shadow,needs_cost', [('cost', False, True), ('cost', True, True),
    ('warm_idle', False, False), ('warm_idle', True, False), ('stable_layout', False, False), ('fixed_period', False, False)])
def test_cost_scope_requirement_follows_merged_strategy_not_presence_of_quote(matrix, monkeypatch, strategy, shadow, needs_cost):
    monkeypatch.setenv("RAYON_NUM_THREADS", "4")
    method = matrix.spec['methods'][0]
    method['overrides'].update(planner={'strategy': strategy}, shadow=shadow,
                               cost_model={'artifact': str(matrix.args.matrix.parent/'absent.json')})
    write_spec(matrix)
    plan = sweep.build_matrix_plan(matrix.args)
    required = [r for job_id, r in plan['job_requirements'].items() if job_id.startswith('Fixed-')]
    assert all(r['needs_cost'] is needs_cost for r in required)
    assert all(r['needs_quote'] is (strategy in {'cost', 'warm_idle'}) for r in required)
    if needs_cost:
        with pytest.raises(ValueError, match='own bound artifact before startup'):
            sweep.execute(matrix.args, plan)
        assert not matrix.args.output.exists()


@pytest.mark.parametrize('cache,points', [('wrong-cache', True), ('cache-32', False), ('cache-32', True)])
def test_actual_cost_scope_rejects_mismatch_or_empty_model_before_generation(matrix, monkeypatch, cache, points):
    matrix.spec['methods'] = [matrix.spec['methods'][0]]
    matrix.spec['tuning']['require_quality_evidence'] = False
    matrix.spec['reference_method'] = 'Fixed'
    method = matrix.spec['methods'][0]
    method['overrides'].update(planner={'strategy': 'cost'}, cost_model={'artifact': cost(matrix, cache=cache, points=points)})
    write_spec(matrix)
    seen, launched, stopped = fake_hardware(matrix, monkeypatch)
    plan = sweep.build_matrix_plan(matrix.args)
    expected = int(cache != 'cache-32' or not points)
    assert sweep.execute(matrix.args, plan) == expected
    assert len(launched) == len(stopped) == 4
    assert len(seen) == (0 if expected else 4)
    if expected:
        rows = [json.loads(line) for line in (matrix.args.output/'matrix-results/results.jsonl').read_text().splitlines()]
        assert len(rows) == 4 and all(row['status'] == 'failed' for row in rows)


def test_relative_checkout_paths_are_resolved_only_for_execution(matrix, monkeypatch):
    import os
    for profile in matrix.spec['server_profiles'].values():
        profile['checkout'] = os.path.relpath(profile['checkout'], sweep.ROOT)
    write_spec(matrix)
    original = matrix.args.matrix.read_bytes()
    seen, launched, stopped = fake_hardware(matrix, monkeypatch)
    assert sweep.execute(matrix.args, sweep.build_matrix_plan(matrix.args)) == 0
    assert len(seen) == len(launched) == len(stopped) == 6
    assert matrix.args.matrix.read_bytes() == original


def test_arrival_window_that_can_omit_sessions_is_outside_full_cohort_contract(matrix):
    matrix.spec.update(arrival_rate=.01, arrival_window=.001)
    write_spec(matrix)
    with pytest.raises(ValueError, match='full-cohort mode'):
        sweep.build_matrix_plan(matrix.args)


def test_actual_model_mismatch_stops_before_any_generation_and_still_cleans(matrix, monkeypatch):
    from profiling import runtime_manifest
    seen, launched, stopped = fake_hardware(matrix, monkeypatch)
    original = runtime_manifest.build_manifest
    async def mismatch(**kwargs):
        value = await original(**kwargs)
        value['server_info']['model_path'] = '/another/model'
        return value
    monkeypatch.setattr(runtime_manifest, 'build_manifest', mismatch)
    assert sweep.execute(matrix.args, sweep.build_matrix_plan(matrix.args)) == 1
    assert not seen and len(launched) == len(stopped) == 6


def test_same_checkout_cannot_be_relabeled_stock_and_patched(matrix):
    matrix.spec['server_profiles']['stock']['checkout'] = matrix.spec['server_profiles']['patched']['checkout']
    write_spec(matrix)
    with pytest.raises(ValueError, match='both patched and clean stock'):
        sweep.build_matrix_plan(matrix.args)


def test_declared_capacity_mismatch_cost_artifact_fails_before_owned_start(matrix, monkeypatch):
    monkeypatch.setenv('RAYON_NUM_THREADS', '4')
    matrix.spec['methods'][0]['overrides'].update(planner={'strategy': 'cost'},
        cost_model={'artifact': cost(matrix, capacity=16384)})
    write_spec(matrix)
    plan = sweep.build_matrix_plan(matrix.args)
    def forbidden(*a, **kw):
        pytest.fail('known cost capacity mismatch reached source/model/server')
    monkeypatch.setattr(sweep, 'make_owned_executor', forbidden)
    monkeypatch.setattr(sweep.server, 'check_stock', forbidden)
    monkeypatch.setattr(sweep.research, 'check_research_checkout', forbidden)
    with pytest.raises(ValueError, match='own bound artifact before startup'):
        sweep.execute(matrix.args, plan)
    assert not matrix.args.output.exists()
