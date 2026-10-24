"""Owned-server and file contracts: fake processes, no GPU/model/network work."""
from argparse import Namespace
import asyncio
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench import b6_joint_sweep as sweep
from bench import experiments
from controller.schema import ConfigFingerprint
from test_controller_capacity_sweep import runtime
from test_native_policy_sweep import HOST_LOG, MODEL_CONFIG


def arguments(tmp_path):
    config = tmp_path/'base.yaml'
    config.write_text('{}\n')
    return Namespace(config=config, model_path=tmp_path/'model', patched_checkout=tmp_path/'patched',
        stock_checkout=tmp_path/'stock', input=tmp_path/'input.jsonl', split_manifest=tmp_path/'split.json',
        output=tmp_path/'output', port=0, startup_timeout=1, job_timeout=1)


def test_plan_is_finite_actual_policy_matrix_and_has_no_execution_side_effects(tmp_path, monkeypatch, capsys):
    def forbidden(*a, **kw):
        pytest.fail('plan cannot launch, connect, load weights or read GPU state')
    monkeypatch.setattr(sweep, 'execute', forbidden)
    monkeypatch.setattr(sweep.server, 'command', forbidden)
    config = tmp_path/'base.yaml'
    config.write_text('{}\n')
    output = tmp_path/'uncreated'
    assert sweep.main(['--config', str(config), '--model-path', 'absent', '--output', str(output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert not output.exists() and not plan['execute']
    assert (plan['planned_cells'], plan['planned_turns']) == (13, 533)
    spec, config = plan['matrix'], plan['config']
    assert Counter(m['family'] for m in spec['methods']) == {'Rebuild': 3, 'Fixed4096': 3, 'Redundancy40': 3, 'B1': 3, 'B0_reference': 1}
    assert len(spec['server_profiles']) == 7
    assert spec['protocols'] == ['P2'] and spec['concurrency'] == [4] and spec['seeds'] == [42]
    assert 'retriever_config' not in spec and config['context']['system_prompt'] is None
    assert config['context']['max_new_tokens'] == 1024 and not config['shadow']
    assert config['cost_model']['artifact'] is config['cost_model']['output_length_artifact'] is None
    assert all(m['native']['audit_prompt_ids'] for m in spec['methods'] if m.get('kind') == 'native')
    assert not any(m.get('native', {}).get('engine_manifest') for m in spec['methods'])
    assert spec['tuning']['require_quality_evidence'] and 'quality_eligible' not in spec['tuning']
    assert plan['future_repeat_orders'][1] == plan['future_repeat_orders'][0][1:]+plan['future_repeat_orders'][0][:1]
    assert all(p['write_policy'] in sweep.server.POLICIES for p in spec['server_profiles'].values() if p['hicache'])
    assert spec['server_profiles']['stock_B0']['hicache'] is False
    assert spec['server_profiles']['stock_B0']['session_radix'] is False
    assert 'disk/compiler caches' in plan['server_initial_state']


@pytest.mark.parametrize('verified', [None, {'status': 'running', 'server_profile': 'other', 'profile': {}}])
def test_generic_executor_rejects_owned_profile_before_generation_or_output(tmp_path, monkeypatch, verified):
    from controller import cli
    async def forbidden(*a):
        pytest.fail('unverified profile reached generation')
    monkeypatch.setattr(cli, 'benchmark', forbidden)
    spec = {'server_profiles': {'patched_write_back': {'write_policy': 'write_back'}}}
    job = {'kind': 'controller', 'server_profile': 'patched_write_back', 'overrides': {}}
    with pytest.raises(ValueError, match='verified owned-server'):
        asyncio.run(experiments.execute_job(spec, job, tmp_path/'uncreated', verified_server=verified))
    assert not (tmp_path/'uncreated').exists()


def test_runtime_checks_scheduling_and_hardware_not_just_cache_hash():
    machine = {'gpus': [{'index': 0, 'uuid': 'fixture', 'name': 'fake', 'memory_bytes': 1}],
        'driver_cuda_torch_versions': {'driver': 'fake', 'cuda': 'fake', 'torch': 'fake'}, 'collection_errors': []}
    assert sweep.check_common_runtime(dict(sweep.COMMON_SERVER), machine)['cache_config_hash_is_not_runtime_equivalence_proof']
    with pytest.raises(ValueError, match='settings differ'):
        sweep.check_common_runtime({**sweep.COMMON_SERVER, 'attention_backend': 'triton'}, machine)
    with pytest.raises(ValueError, match='could not be observed'):
        sweep.check_common_runtime(dict(sweep.COMMON_SERVER), {**machine, 'collection_errors': ['not observed']})


@pytest.mark.parametrize('failed_first_job', [False, True])
def test_owned_execution_binds_real_profiles_continues_failures_and_cleans_each_process(tmp_path, monkeypatch, failed_first_job):
    from adapters import mtrag
    from profiling import runtime_manifest
    args = arguments(tmp_path)
    args.model_path.mkdir()
    (args.model_path/'config.json').write_text(json.dumps(MODEL_CONFIG))
    sessions = [f's{i}' for i in range(5)]
    args.split_manifest.write_text(json.dumps({'dev': sessions, 'test': ['heldout']}))
    rows = [{'conversation_id': sessions[i%5], 'task_id': f't{i}', 'turn': i//5+1,
        'Collection': mtrag.CLOUD_COLLECTION} for i in range(41)]
    args.input.write_text(''.join(json.dumps({**r, 'input': [], 'contexts': []})+'\n' for r in rows + [
        {'conversation_id': 'heldout', 'task_id': 'heldout:1', 'turn': 1, 'Collection': mtrag.CLOUD_COLLECTION}]))
    monkeypatch.setenv('RAYON_NUM_THREADS', '4')
    fp = ConfigFingerprint(tokenizer_hash='a'*64, template_hash='b'*64)
    monkeypatch.setattr(sweep, 'create_model_template', lambda model: SimpleNamespace(fingerprint=fp))
    monkeypatch.setattr(mtrag, 'load_rows', lambda path: rows)
    monkeypatch.setattr(sweep.research, 'check_research_checkout', lambda path: {'path': str(path), 'commit': sweep.server.COMMIT})
    monkeypatch.setattr(sweep.server, 'check_stock', lambda path: {'path': str(path), 'commit': sweep.server.COMMIT})
    monkeypatch.setattr(sweep.server, 'command', lambda argv, env: str(Path(env['SGLANG_CHECKOUT'])/'python/sglang/__init__.py'))
    active, stopped, observed = {}, [], []
    def launch(argv, **kwargs):
        active.clear()
        active.update(kwargs['env'])
        if active['CONTEXTKV_ENABLE_HICACHE'] == '1':
            kwargs['stdout'].write(HOST_LOG+'\n')
            kwargs['stdout'].flush()
        return SimpleNamespace(pid=100+len(stopped), returncode=0)
    monkeypatch.setattr(sweep.subprocess, 'Popen', launch)
    monkeypatch.setattr(sweep.server, 'stop_owned_server', lambda process: stopped.append(process.pid))
    monkeypatch.setattr(sweep.server, 'wait_ready', lambda *a: None)
    monkeypatch.setattr(sweep.server, 'get_text', lambda *a: '# fake metrics\n')
    async def manifest(**kwargs):
        result = runtime()
        result['engine']['fingerprint'] = asdict(fp)
        result['cache_config']['write_policy'] = active['CONTEXTKV_WRITE_POLICY']
        result['server_info'].update(sweep.COMMON_SERVER)
        result['server_info'].update(max_total_num_tokens=16384,
            model_path=str(args.model_path), tokenizer_path=str(args.model_path), revision=sweep.server.MODEL_REVISION,
            dtype='bfloat16', kv_cache_dtype='bfloat16',
            enable_session_radix_cache=active['CONTEXTKV_ENABLE_SESSION_RADIX'] == '1',
            enable_hierarchical_cache=active['CONTEXTKV_ENABLE_HICACHE'] == '1',
            disable_radix_cache=False, incremental_streaming_output=False,
            hicache_write_policy=active['CONTEXTKV_WRITE_POLICY'], hicache_host_memory_mode='cache',
            hicache_io_backend='kernel', hicache_mem_layout='layer_first')
        result['runtime'] = {'gpus': [{'index': 0, 'uuid': 'fixture', 'name': 'fake', 'memory_bytes': 1}],
            'driver_cuda_torch_versions': {'driver': 'fake', 'cuda': 'fake', 'torch': 'fake'}, 'collection_errors': []}
        return result
    monkeypatch.setattr(runtime_manifest, 'build_manifest', manifest)
    async def metrics(*a):
        return {'host_admission': {'config': {'mode': 'off'}}, 'pool': {'unique_gpu_tokens': 0, 'unique_host_tokens': 0}}
    monkeypatch.setattr(sweep, 'research_metrics', metrics)
    async def job(spec, entry, directory, *, native_engine_manifest, verified_server):
        assert verified_server['status'] == 'running'
        assert verified_server['profile'] == spec['server_profiles'][entry['server_profile']]
        observed.append(entry['method'])
        if failed_first_job and len(observed) == 1:
            raise RuntimeError('injected generation failure')
        directory.mkdir(parents=True)
        if entry['kind'] == 'native':
            actual = json.loads(native_engine_manifest.read_text())
            assert actual['runtime']['actual_kv_pool_tokens'] == 16384
            assert actual['server']['enable_hierarchical_cache'] == (entry['native']['baseline'] != 'B0')
            assert actual['runtime']['actual_host_pool_tokens'] == (0 if entry['native']['baseline'] == 'B0' else 279040)
        else:
            assert native_engine_manifest is None
        raw = directory/'fake-generation.json'
        raw.write_text(json.dumps({'report': {'turns': [{'task_id': r['task_id'], 'status': 'success'} for r in rows]}}))
        return {**entry, 'status': 'completed', 'raw_output': str(raw), 'artifact_sha256': {},
            'summary': {'measurement_kind': 'fake_process_contract', 'cohort': {'unfinished_turns': 0},
                'timing': {'failure_fraction': 0}}}
    monkeypatch.setattr(sweep, 'execute_job', job)
    monkeypatch.setattr(experiments, 'paired_latency', lambda *a, **kw: {'status': 'fake_not_measured'})
    assert sweep.execute(args, sweep.build_plan(args)) == int(failed_first_job)
    assert len(stopped) == len(set(stopped)) == len(observed) == 13
    results = [json.loads(line) for line in (args.output/'matrix-results/results.jsonl').read_text().splitlines()]
    assert sum(r['status'] == 'failed' for r in results) == int(failed_first_job)
    for row in results:
        record = json.loads((args.output/'servers'/row['job_id']/'server.json').read_text())
        assert record['status'] == ('failed' if row['status'] == 'failed' else 'completed')
        assert record['server_returncode_after_cleanup'] == 0
        if row['status'] == 'completed':
            assert row['artifact_sha256']['server_record']
    quality = json.loads((args.output/'quality-evidence.template.json').read_text())
    assert quality['frozen_protocol'] is None and all(r['scores'] is None for r in quality['methods'].values())
    assert not any(args.output.glob('*frozen*'))
