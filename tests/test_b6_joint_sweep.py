"""Owned-server and file contracts: fake processes, no GPU/model/network work."""
import asyncio

import pytest

from bench import b6_joint_sweep as sweep
from bench import experiments


def test_removed_automatic_matrix_cannot_start_from_only_a_model_path(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, 'execute', lambda *a: pytest.fail('legacy execution'))
    with pytest.raises(SystemExit) as error:
        sweep.main(['--model-path','historical-model','--output',str(tmp_path/'unused'),'--execute'])
    assert error.value.code == 2 and not (tmp_path/'unused').exists()


@pytest.mark.parametrize('verified', [None, {'status': 'running', 'server_profile': 'other', 'profile': {}}])
def test_generic_executor_rejects_owned_profile_before_generation_or_output(tmp_path, monkeypatch, verified):
    from bench import run_sessions as cli
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
