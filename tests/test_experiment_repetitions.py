"""Explicit system runs and job-bound grading, using synthetic files only."""
import asyncio
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from bench import experiments
from controller.templates import token_hash
from test_b6_quality_freeze import setup_joint
from test_experiments import spec as small_spec
from test_quality_acceptance import edit_scores, freeze_protocol


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value))


def setup_blocks(tmp_path, *, reference_candidates=False, protocols=('P2',), concurrency=(1,)):
    base = tmp_path/'base'
    base.mkdir()
    spec, _, quality = setup_joint(base)
    spec['tuning'].pop('quality_evidence')
    spec['tuning']['equal_budget'] = 2
    spec['protocols'], spec['concurrency'], spec['repetitions'] = list(protocols), list(concurrency), [1, 2]
    spec['reference_method'] = 'reference'
    spec['methods'][0].update(tune=True, candidates=[{'planner': {'fixed_token_threshold': 1000}},
                                                  {'planner': {'fixed_token_threshold': 4000}}])
    if reference_candidates:
        spec['methods'][1].update(tune=True, candidates=[{}, {'planner': {'fixed_token_threshold': 3000}}])
    identity = experiments.generation_provenance(spec)
    results, groups = [], {}
    for number, job in enumerate(experiments.matrix_jobs(spec)):
        path = tmp_path/job['job_id']
        path.mkdir()
        source = quality['methods'][job['method']]
        payload = read(source['run'])
        payload['experiment_protocol'] = {'repetition': job['repetition']}
        audits = [json.loads(line) for line in Path(source['audit']).read_text().splitlines()]
        scores = [json.loads(line) for line in Path(source['scores']).read_text().splitlines()]
        for turn, audit, score in zip(payload['report']['turns'], audits, scores):
            audit['protocol'] = job['protocol']
            turn['online_turn']['protocol'] = job['protocol']
            audit['input_ids'].append(number+100)
            turn['result']['input_ids'] = audit['input_ids']
            turn['result']['input_hash'] = score['input_hash'] = token_hash(audit['input_ids'])
            turn['assistant_text'] = audit['answer'] = score['answer'] = job['job_id']+audit['answer']
        run, audit_path, score_path = [path/name for name in ('run.json', 'audit.jsonl', 'scores.jsonl')]
        save(run, payload)
        audit_path.write_text(''.join(json.dumps(r)+'\n' for r in audits))
        score_path.write_text(''.join(json.dumps(r)+'\n' for r in scores))
        declared = {'run': str(run), 'audit': str(audit_path), 'scores': str(score_path), 'generator': source['generator']}
        key = (job['protocol'], job['concurrency'], job['seed'], job['repetition'])
        groups.setdefault(key, {})[job['job_id']] = declared
        results.append({**job, 'status': 'completed', 'experiment_provenance': identity,
            'raw_output': str(run), 'audit_output': str(audit_path),
            'artifact_sha256': {'raw_output': sha256(run.read_bytes()).hexdigest(),
                                'audit_output': sha256(audit_path.read_bytes()).hexdigest()},
            'summary': {'measurement_kind': 'synthetic_file_contract', 'cohort': {'unfinished_turns': 0},
                'timing': {'failure_fraction': 0, 'turn_seconds': {'mean': job['candidate_index']+1}}}})
    blocks = []
    for index, (key, methods) in enumerate(groups.items()):
        path = tmp_path/f'block-{index}'
        path.mkdir()
        value = {**quality, 'methods': methods,
                 'reference': next(name for name in methods if name.startswith('reference-'))}
        freeze_protocol(value, path)
        protocol = read(value['frozen_protocol'])
        protocol['reference'] = value['reference']
        save(value['frozen_protocol'], protocol)
        save(path/'quality.json', value)
        blocks.append({**dict(zip(experiments.BLOCK_FIELDS, key)), 'quality_config': str(path/'quality.json')})
    evidence = {'schema_version': 2, 'blocks': blocks}
    path = tmp_path/'evidence.json'
    save(path, evidence)
    spec['tuning']['quality_evidence'] = str(path)
    return spec, results, evidence


def test_unconfigured_ids_unchanged_and_repetitions_do_not_change_sampling_seed(tmp_path):
    spec = small_spec(tmp_path)
    original = experiments.matrix_jobs(spec)
    assert all('repetition' not in row and not row['job_id'].endswith('-r1') for row in original)
    spec['repetitions'] = [1, 2]
    repeated = experiments.matrix_jobs(spec)
    assert [r['job_id'] for r in repeated] == [r['job_id']+f'-r{rep}' for rep in (1, 2) for r in original]
    assert {r['seed'] for r in repeated} == {42}
    assert [r['repetition'] for r in repeated] == [1]*3+[2]*3
    repeated[0]['overrides']['planner']['fixed_token_threshold'] = 99
    assert repeated[3]['overrides']['planner']['fixed_token_threshold'] == 1000
    spec.pop('repetitions')
    assert experiments.matrix_jobs(spec) == original


@pytest.mark.parametrize('value', [[], 2, [0], [-1], [True], [1, 1], [1.5], ['1']])
def test_invalid_system_repetition_ids_rejected(tmp_path, value):
    spec = small_spec(tmp_path)
    spec['repetitions'] = value
    with pytest.raises(ValueError, match='repetitions'):
        experiments.matrix_jobs(spec)


def test_latency_pairs_use_same_repeat_and_keep_failed_reference(tmp_path, monkeypatch):
    spec = small_spec(tmp_path)
    spec['repetitions'], spec['reference_method'] = [1, 2], 'full'
    matched = []
    async def execute(value, job, directory):
        if job['method'] == 'full' and job['repetition'] == 2:
            raise RuntimeError('failed repeat retained')
        return {**job, 'status': 'completed', 'summary': {'repetition': job['repetition']}}
    def pair(left, right, **kwargs):
        matched.append((left['repetition'], right['repetition']))
        return {'status': 'synthetic_contract'}
    monkeypatch.setattr(experiments, 'paired_latency', pair)
    output = tmp_path/'matrix'
    results = asyncio.run(experiments.run_matrix(spec, output, executor=execute))
    assert len(results) == 6 and results[-1]['status'] == 'failed'
    assert matched == [(1, 1), (1, 1)]
    pairs = read(output/'comparisons.json')
    assert len(pairs) == 2 and all(row['reference'].endswith('-r1') for row in pairs)


def test_execute_records_repetition_in_actual_raw_without_reseeding(tmp_path, monkeypatch):
    from controller import cli
    spec = small_spec(tmp_path)
    spec['repetitions'] = [1, 2]
    seeds = []
    async def generate(args, config):
        seeds.append(args.seed)
        (Path(config.state_dir)/'benchmark-fixture.json').write_text('{"engine_kind":"mock"}')
    monkeypatch.setattr(cli, 'benchmark', generate)
    monkeypatch.setattr(experiments, 'summarize_run', lambda value: {'measurement_kind': 'simulation'})
    monkeypatch.setattr(experiments, 'markdown_report', lambda value: '')
    jobs = experiments.matrix_jobs(spec)
    for job in (jobs[0], jobs[3]):
        result = asyncio.run(experiments.execute_job(spec, job, tmp_path/job['job_id']))
        assert read(result['raw_output'])['experiment_protocol']['repetition'] == job['repetition']
    assert seeds == [42, 42]


def test_blocks_freeze_every_candidate_repeat_and_all_offline_inputs(tmp_path):
    spec, results, evidence = setup_blocks(tmp_path)
    before = experiments.generation_provenance(spec)
    frozen = experiments.select_development(spec, results)
    assert frozen['winners']['candidate']['candidate_index'] == 0
    report = frozen['quality_evidence_report']
    assert report['candidate_eligibility']['candidate'] == {'0': True, '1': True}
    assert len(report['blocks']) == 2 and 'no pooled repetition CI' in report['aggregation']
    assert all(len(block['job_ids']) == 3 and block['evidence_complete'] for block in report['blocks'])
    expected_files = {spec['tuning']['quality_evidence']}
    for block in evidence['blocks']:
        value = read(block['quality_config'])
        expected_files.update([block['quality_config'], value['frozen_protocol']])
        expected_files.update(value for method in value['methods'].values() for key, value in method.items()
                              if key in {'run', 'audit', 'scores'})
    assert expected_files <= frozen['provenance']['external_inputs'].keys()
    assert all(frozen['provenance']['external_inputs'][p] == sha256(Path(p).read_bytes()).hexdigest() for p in expected_files)
    # One bad second repeat disqualifies candidate 0, never averaged away by repeat 1.
    value = read(evidence['blocks'][1]['quality_config'])
    candidate0 = next(name for name in value['methods'] if name.startswith('candidate-c0-'))
    edit_scores(value, candidate0, lambda rows: [r.update(scores={k: 0 for k in r['scores']}) for r in rows])
    assert experiments.generation_provenance(spec) == before
    updated = experiments.select_development(spec, results)
    assert updated['winners']['candidate']['candidate_index'] == 1
    assert updated['quality_evidence_report']['candidate_eligibility']['candidate'] == {'0': False, '1': True}
    assert updated['provenance'] != frozen['provenance']


def test_multiple_protocol_and_concurrency_blocks_do_not_pool_tasks(tmp_path):
    spec, results, _ = setup_blocks(tmp_path, protocols=('P1', 'P2'), concurrency=(1, 4))
    _, report = experiments.quality_gate_from_evidence(spec, results)
    assert len(report['blocks']) == 8
    assert sum(len(block['job_ids']) for block in report['blocks']) == len(results) == 24
    assert all(len(block['report']['cohort']) == 3 for block in report['blocks'])


@pytest.mark.parametrize('kind', ['missing_block', 'duplicate_block', 'extra_block', 'missing_job', 'duplicate_job',
                                  'wrong_block_identity', 'wrong_methods', 'other_repeat_reference'])
def test_exact_block_and_job_coverage_cannot_use_intersections_or_wrong_repeat(tmp_path, kind):
    spec, results, evidence = setup_blocks(tmp_path)
    if kind == 'missing_block':
        evidence['blocks'].pop()
    elif kind == 'duplicate_block':
        evidence['blocks'].append(deepcopy(evidence['blocks'][0]))
    elif kind == 'extra_block':
        evidence['blocks'].append({**evidence['blocks'][0], 'repetition': 3})
    elif kind == 'missing_job':
        results.pop()
    elif kind == 'duplicate_job':
        results.append(deepcopy(results[0]))
    elif kind == 'wrong_block_identity':
        results[0]['repetition'] = 2
    else:
        block = evidence['blocks'][0]
        value = read(block['quality_config'])
        if kind == 'wrong_methods':
            value['methods'].pop(next(name for name in value['methods'] if name.startswith('candidate-c0')))
        else:
            value['reference'] = read(evidence['blocks'][1]['quality_config'])['reference']
        save(block['quality_config'], value)
    save(spec['tuning']['quality_evidence'], evidence)
    with pytest.raises(ValueError):
        experiments.quality_gate_from_evidence(spec, results)


def test_reference_multiple_candidates_is_ambiguous_not_first_or_fastest(tmp_path):
    spec, results, _ = setup_blocks(tmp_path, reference_candidates=True)
    with pytest.raises(ValueError, match='exactly one actual reference'):
        experiments.quality_gate_from_evidence(spec, results)


@pytest.mark.parametrize('kind', ['failed_job', 'unfinished', 'missing_score'])
def test_failed_or_ungraded_repeat_retains_all_blocks_and_blocks_freeze(tmp_path, kind):
    spec, results, evidence = setup_blocks(tmp_path)
    if kind == 'failed_job':
        results[3] = {k: v for k, v in results[3].items() if k != 'summary'}
        results[3].update(status='failed', error='injected failure')
    elif kind == 'unfinished':
        results[3]['summary']['cohort']['unfinished_turns'] = 1
    else:
        value = read(evidence['blocks'][1]['quality_config'])
        name = next(name for name in value['methods'] if name.startswith('candidate-c0'))
        edit_scores(value, name, lambda rows: rows[0]['scores'].update(citations=None))
    with pytest.raises(experiments.QualityNotReady) as error:
        experiments.select_development(spec, results)
    report = error.value.report
    assert len(report['blocks']) == 2 and sum(len(b['job_ids']) for b in report['blocks']) == 6
    assert not report['matrix_evidence_gate']['ready_to_freeze']
    if kind != 'missing_score':
        assert report['blocks'][1]['generation_incomplete_jobs'] == [results[3]['job_id']]


def test_cross_block_evaluator_change_rejected_even_if_each_local_protocol_matches(tmp_path):
    spec, results, evidence = setup_blocks(tmp_path)
    value = read(evidence['blocks'][1]['quality_config'])
    for name in value['methods']:
        edit_scores(value, name, lambda rows: [r.update(rubric_version='different-rubric') for r in rows])
    protocol = read(value['frozen_protocol'])
    protocol['evaluator']['rubric_version'] = 'different-rubric'
    save(value['frozen_protocol'], protocol)
    with pytest.raises(ValueError, match='share the complete evaluator protocol'):
        experiments.quality_gate_from_evidence(spec, results)


@pytest.mark.parametrize('kind', ['copied_run', 'wrong_raw_repetition', 'missing_task', 'changed_bytes', 'score_swap'])
def test_per_job_artifact_bindings_remain_enforced_inside_blocks(tmp_path, kind):
    spec, results, evidence = setup_blocks(tmp_path)
    value = read(evidence['blocks'][0]['quality_config'])
    job = results[0]
    declared = value['methods'][job['job_id']]
    if kind == 'copied_run':
        copy = tmp_path/'copy.json'
        copy.write_bytes(Path(declared['run']).read_bytes())
        declared['run'] = str(copy)
        save(evidence['blocks'][0]['quality_config'], value)
    elif kind == 'score_swap':
        other = read(evidence['blocks'][1]['quality_config'])
        declared['scores'] = next(m['scores'] for name, m in other['methods'].items() if name.startswith('candidate-c0'))
        save(evidence['blocks'][0]['quality_config'], value)
    else:
        path = Path(declared['run'])
        payload = read(path)
        if kind == 'wrong_raw_repetition':
            payload['experiment_protocol']['repetition'] = 2
        elif kind == 'missing_task':
            payload['report']['turns'].pop()
        else:
            payload['changed'] = True
        save(path, payload)
        if kind != 'changed_bytes':
            job['artifact_sha256']['raw_output'] = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        experiments.quality_gate_from_evidence(spec, results)


def test_frozen_selection_keeps_repeat_ids_and_changed_plan_cannot_reuse_it(tmp_path, monkeypatch):
    spec, results, _ = setup_blocks(tmp_path)
    frozen = experiments.select_development(spec, results)
    jobs = experiments.matrix_jobs(spec, split='test', frozen=frozen)
    assert {(j['method'], j['repetition']) for j in jobs} == {
        (name, repetition) for name in ('candidate', 'reference') for repetition in (1, 2)}
    assert all(j.get('method_aliases') == ['B6'] for j in jobs if j['method'] == 'candidate')
    assert all(j['candidate_index'] == 0 and j['split'] == 'test' and j['seed'] == 42 for j in jobs)
    changed = deepcopy(spec)
    changed['repetitions'] = [1, 2, 3]
    async def forbidden(*a):
        pytest.fail('changed repetition plan reached execution')
    with pytest.raises(ValueError, match='frozen development selection'):
        asyncio.run(experiments.run_matrix(changed, tmp_path/'not-created', stage='test', frozen=frozen, executor=forbidden))
    assert not (tmp_path/'not-created').exists()


def test_new_implementation_does_not_relabel_old_generation_as_same_source(tmp_path, monkeypatch):
    spec, results, _ = setup_blocks(tmp_path)
    implementation = experiments.implementation_manifest()
    implementation['files']['bench/experiments.py'] = 'older-implementation-digest'
    for row in results:
        row['experiment_provenance']['implementation'] = implementation
    with pytest.raises(ValueError, match='different or unrecorded'):
        experiments.select_development(spec, results)
