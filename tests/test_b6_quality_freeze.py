"""Synthetic file contracts only: no ratings, hardware runs or quality claims."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import pytest

from bench import experiments
from test_quality_acceptance import edit_scores, freeze_protocol, setup_methods


def setup_joint(tmp_path):
    quality = setup_methods(tmp_path)
    for method in quality['methods'].values():
        path = Path(method['run'])
        payload = json.loads(path.read_text())
        for turn in payload['report']['turns']:
            turn['online_turn']['protocol'] = 'P2'
        path.write_text(json.dumps(payload))
        audit = Path(method['audit'])
        values = [json.loads(line) for line in audit.read_text().splitlines()]
        for row in values:
            row['protocol'] = 'P2'
        audit.write_text(''.join(json.dumps(row)+'\n' for row in values))
    freeze_protocol(quality, tmp_path)
    data = tmp_path/'input.jsonl'
    data.write_text(''.join(json.dumps({'conversation_id': sid, 'task_id': f'{sid}:1', 'turn': 1,
        'Collection': 'synthetic', 'input': [{'speaker': 'user', 'text': 'Question?'}],
        'contexts': [{'document_id': str(i), 'text': 'fixture'} for i in range(5)]})+'\n'
        for sid in ('s0', 's1', 's2', 'other')))
    spec = {'schema_version': 1, 'input': str(data), 'domain': 'all', 'split_manifest': quality['split_manifest'],
        'protocols': ['P2'], 'concurrency': [1], 'seeds': [42],
        'methods': [{'id': 'candidate', 'b6_eligible': True}, {'id': 'reference', 'b6_eligible': False}],
        'tuning': {'equal_budget': 1, 'require_quality_evidence': True,
                   'quality_constraints': {'reference': 'reference', 'epsilon': .01}}}
    identity = experiments.generation_provenance(spec)
    results = [{**job, 'status': 'completed', 'experiment_provenance': identity,
        'raw_output': quality['methods'][job['method']]['run'],
        'audit_output': quality['methods'][job['method']]['audit'],
        'summary': {'measurement_kind': 'hardware_fixture_contract', 'timing': {'failure_fraction': 0,
            'turn_seconds': {'mean': 1}}, 'cohort': {'unfinished_turns': 0}}}
        for job in experiments.matrix_jobs(spec)]
    for result in results:
        result['artifact_sha256'] = {key: sha256(Path(result[key]).read_bytes()).hexdigest()
                                    for key in ('raw_output', 'audit_output')}
    path = tmp_path/'quality.json'
    path.write_text(json.dumps(quality))
    spec['tuning']['quality_evidence'] = str(path)
    return spec, results, quality


def test_real_files_gate_and_final_provenance_bind_offline_sources(tmp_path):
    spec, results, quality = setup_joint(tmp_path)
    assert experiments.generation_provenance(spec) == results[0]['experiment_provenance']
    frozen = experiments.select_development(spec, results)
    assert frozen['b6'] == 'candidate' and frozen['quality_gate_applied']
    assert frozen['quality_evidence_report']['scope'] == 'dev'
    assert not frozen['quality_noninferiority_established']
    bound = frozen['provenance']['external_inputs']
    for path in (spec['tuning']['quality_evidence'], quality['frozen_protocol'],
                 quality['methods']['candidate']['scores'], quality['methods']['reference']['audit']):
        assert bound[path] == sha256(Path(path).read_bytes()).hexdigest()


def test_quality_constraint_change_is_still_generation_identity_change(tmp_path):
    spec, results, _ = setup_joint(tmp_path)
    spec['tuning']['quality_constraints']['epsilon'] = .05
    with pytest.raises(ValueError, match='different or unrecorded'):
        experiments.select_development(spec, results)


@pytest.mark.parametrize('kind', ['booleans', 'missing_evidence', 'wrong_reference', 'no_protocol', 'heldout', 'missing_method'])
def test_required_evidence_cannot_be_replaced_by_eligibility_flags_or_other_scope(tmp_path, kind):
    spec, results, quality = setup_joint(tmp_path)
    if kind == 'booleans':
        spec['tuning']['quality_eligible'] = {'candidate': {'0': True}}
    elif kind == 'missing_evidence':
        del spec['tuning']['quality_evidence']
    else:
        if kind == 'wrong_reference':
            quality['reference'] = 'candidate'
        elif kind == 'no_protocol':
            quality.pop('frozen_protocol')
        elif kind == 'heldout':
            quality['scope'] = 'heldout'
        else:
            del quality['methods']['candidate']
        Path(spec['tuning']['quality_evidence']).write_text(json.dumps(quality))
    with pytest.raises(ValueError):
        experiments.select_development(spec, results)


@pytest.mark.parametrize('field', ['run', 'audit'])
def test_quality_must_reference_actual_generated_path_not_a_relabelled_copy(tmp_path, field):
    spec, results, quality = setup_joint(tmp_path)
    original = Path(quality['methods']['candidate'][field])
    copy = tmp_path/'copy.jsonl'
    copy.write_bytes(original.read_bytes())
    quality['methods']['candidate'][field] = str(copy)
    Path(spec['tuning']['quality_evidence']).write_text(json.dumps(quality))
    with pytest.raises(ValueError, match='actual generated artifact'):
        experiments.select_development(spec, results)


def test_changed_completed_generation_bytes_cannot_be_graded_as_original(tmp_path):
    spec, results, quality = setup_joint(tmp_path)
    run = Path(quality['methods']['candidate']['run'])
    run.write_text(run.read_text()+'\n')
    with pytest.raises(ValueError, match='bytes differ'):
        experiments.select_development(spec, results)


@pytest.mark.parametrize('kind', ['missing_dimension', 'noninferiority_failed', 'protocol_mismatch'])
def test_incomplete_or_unqualified_scores_produce_readiness_not_b6(tmp_path, kind):
    spec, results, quality = setup_joint(tmp_path)
    if kind == 'missing_dimension':
        edit_scores(quality, 'candidate', lambda values: values[0]['scores'].update(citations=None))
    elif kind == 'noninferiority_failed':
        edit_scores(quality, 'candidate', lambda values: [r.update(scores={k: 0 for k in r['scores']}) for r in values])
    else:
        edit_scores(quality, 'candidate', lambda values: [r.update(reviewer_id='different-reviewer') for r in values])
    with pytest.raises(experiments.QualityNotReady) as error:
        experiments.select_development(spec, results)
    assert error.value.report['comparisons']['candidate']['dev_quality_eligible'] is False


def test_related_family_scores_remain_diagnostic_even_if_relative_scores_pass(tmp_path):
    spec, results, quality = setup_joint(tmp_path)
    judge = {'provider': 'fixture', 'model': 'different-model', 'revision': 'different-weights',
             'prompt': 'fixture-rubric', 'temperature': 0, 'max_tokens': 128, 'base_family': 'same-family'}
    for name, method in quality['methods'].items():
        method['generator']['base_family'] = 'same-family'
        edit_scores(quality, name, lambda values: [r.update(source='offline_judge', judge=judge,
                                                           reference_targets_supplied=False) for r in values])
    protocol = json.loads(Path(quality['frozen_protocol']).read_text())
    protocol['methods'] = {name: {k: v for k, v in value.items() if k != 'scores'} for name, value in quality['methods'].items()}
    protocol['evaluator'] = {'kind': 'offline_judge', 'judge': judge, 'reference_targets_supplied': False}
    Path(quality['frozen_protocol']).write_text(json.dumps(protocol))
    Path(spec['tuning']['quality_evidence']).write_text(json.dumps(quality))
    with pytest.raises(experiments.QualityNotReady) as error:
        experiments.select_development(spec, results)
    report = error.value.report
    assert report['comparisons']['candidate']['all_dimensions_noninferior_on_evaluator']
    assert report['matrix_evidence_gate']['unaccepted_evaluator_methods'] == ['candidate', 'reference']
    assert not report['matrix_evidence_gate']['ready_to_freeze']


def test_cli_saves_incomplete_readiness_without_creating_frozen_file(tmp_path):
    import yaml
    spec, results, quality = setup_joint(tmp_path)
    edit_scores(quality, 'candidate', lambda values: values[0]['scores'].update(citations=None))
    config, recorded, output = tmp_path/'matrix.yaml', tmp_path/'results.jsonl', tmp_path/'new/frozen.json'
    config.write_text(yaml.safe_dump(spec))
    recorded.write_text(''.join(json.dumps(r)+'\n' for r in results))
    with pytest.raises(SystemExit, match='no B6 freeze'):
        experiments.main(['freeze', '--config', str(config), '--results', str(recorded), '--output', str(output)])
    assert not output.exists()
    assert json.loads(output.with_suffix('.readiness.json').read_text())['comparisons']['candidate']['dev_quality_eligible'] is False


def test_complete_expected_cohort_and_job_protocol_are_required(tmp_path):
    spec, results, quality = setup_joint(tmp_path)
    path = Path(quality['methods']['candidate']['run'])
    payload = json.loads(path.read_text())
    payload['report']['turns'].pop()
    path.write_text(json.dumps(payload))
    results[0]['artifact_sha256']['raw_output'] = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='complete declared'):
        experiments.select_development(spec, results)
    payload['report']['turns'].append(deepcopy(payload['report']['turns'][0]))
    payload['report']['turns'][-1]['task_id'] = 's2:1'
    payload['report']['turns'][0]['online_turn']['protocol'] = 'P3'
    path.write_text(json.dumps(payload))
    results[0]['artifact_sha256']['raw_output'] = sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='protocol differs'):
        experiments.select_development(spec, results)


@pytest.mark.parametrize('binding', ['run_manifest', 'server_record'])
def test_completed_sidecar_and_owned_server_record_remain_file_bound(tmp_path, binding):
    spec, results, quality = setup_joint(tmp_path)
    sidecar = tmp_path/(binding+'.json')
    sidecar.write_text('{}')
    row = results[0]
    if binding == 'run_manifest':
        row['run_manifest'] = str(sidecar)
        quality['methods']['candidate']['manifest'] = str(sidecar)
    else:
        row['server_profile'] = 'fixture_profile'
        row['server_execution'] = {'record': str(sidecar)}
        quality['methods']['candidate']['server_record'] = str(sidecar)
    row['artifact_sha256'][binding] = sha256(sidecar.read_bytes()).hexdigest()
    freeze_protocol(quality, tmp_path)
    Path(spec['tuning']['quality_evidence']).write_text(json.dumps(quality))
    assert experiments.quality_gate_from_evidence(spec, results)[0]['candidate']['0']
    sidecar.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='bytes differ|server record is not bound'):
        experiments.quality_gate_from_evidence(spec, results)
