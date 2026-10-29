"""Synthetic file contracts only: no ratings, hardware runs or quality claims."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from bench import experiments
from test_quality_acceptance import edit_scores, freeze_protocol, setup_methods

CANDIDATE = 'candidate-c0-P2-n1-s42'
REFERENCE = 'reference-c0-P2-n1-s42'


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
        'summary': {'metric_schema_version': 2, 'measurement_kind': 'hardware_fixture_contract', 'timing': {'failure_fraction': 0,
            'turn_seconds': {'mean': 1}}, 'cohort': {'unfinished_turns': 0}}}
        for job in experiments.matrix_jobs(spec)]
    quality['methods'] = {row['job_id']: quality['methods'][row['method']] for row in results}
    quality['reference'] = REFERENCE
    freeze_protocol(quality, tmp_path)
    path = tmp_path/'quality.json'
    path.write_text(json.dumps(quality))
    evidence = tmp_path/'evidence.json'
    evidence.write_text(json.dumps({'schema_version': 2, 'blocks': [{
        'protocol': 'P2', 'concurrency': 1, 'seed': 42, 'repetition': 1, 'quality_config': str(path)}]}))
    spec['tuning']['quality_evidence'] = str(evidence)
    return spec, results, quality


def test_real_files_gate_and_final_provenance_bind_offline_sources(tmp_path):
    spec, results, quality = setup_joint(tmp_path)
    assert experiments.generation_provenance(spec) == results[0]['experiment_provenance']
    frozen = experiments.select_development(spec, results)
    assert frozen['b6'] == 'candidate' and frozen['quality_gate_applied']
    assert frozen['quality_evidence_report']['scope'] == 'dev'
    assert not frozen['quality_noninferiority_established']


def test_quality_comparison_uses_the_declared_margin(tmp_path):
    spec, results, _ = setup_joint(tmp_path)
    spec['tuning']['quality_constraints']['epsilon'] = .05
    with pytest.raises(ValueError, match='quality evidence'):
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
            quality['reference'] = CANDIDATE
        elif kind == 'no_protocol':
            quality.pop('frozen_protocol')
        elif kind == 'heldout':
            quality['scope'] = 'heldout'
        else:
            del quality['methods'][CANDIDATE]
        (tmp_path/'quality.json').write_text(json.dumps(quality))
    with pytest.raises(ValueError):
        experiments.select_development(spec, results)


@pytest.mark.parametrize('field', ['run', 'audit'])
def test_quality_must_reference_actual_generated_path_not_a_relabelled_copy(tmp_path, field):
    spec, results, quality = setup_joint(tmp_path)
    original = Path(quality['methods'][CANDIDATE][field])
    copy = tmp_path/'copy.jsonl'
    copy.write_bytes(original.read_bytes())
    quality['methods'][CANDIDATE][field] = str(copy)
    (tmp_path/'quality.json').write_text(json.dumps(quality))
    with pytest.raises(ValueError, match='actual generated artifact'):
        experiments.select_development(spec, results)


@pytest.mark.parametrize('kind', ['missing_dimension', 'noninferiority_failed', 'protocol_mismatch'])
def test_incomplete_or_unqualified_scores_produce_readiness_not_b6(tmp_path, kind):
    spec, results, quality = setup_joint(tmp_path)
    if kind == 'missing_dimension':
        edit_scores(quality, CANDIDATE, lambda values: values[0]['scores'].update(citations=None))
    elif kind == 'noninferiority_failed':
        edit_scores(quality, CANDIDATE, lambda values: [r.update(scores={k: 0 for k in r['scores']}) for r in values])
    else:
        edit_scores(quality, CANDIDATE, lambda values: [r.update(reviewer_id='different-reviewer') for r in values])
    with pytest.raises(experiments.QualityNotReady) as error:
        experiments.select_development(spec, results)
    assert error.value.report['blocks'][0]['report']['comparisons'][CANDIDATE]['dev_quality_eligible'] is False


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
    (tmp_path/'quality.json').write_text(json.dumps(quality))
    with pytest.raises(experiments.QualityNotReady) as error:
        experiments.select_development(spec, results)
    report = error.value.report
    assert report['blocks'][0]['report']['comparisons'][CANDIDATE]['all_dimensions_noninferior_on_evaluator']
    assert report['matrix_evidence_gate']['unaccepted_evaluator_jobs'] == [CANDIDATE, REFERENCE]
    assert not report['matrix_evidence_gate']['ready_to_freeze']


def test_cli_saves_incomplete_readiness_without_creating_frozen_file(tmp_path):
    import yaml
    spec, results, quality = setup_joint(tmp_path)
    edit_scores(quality, CANDIDATE, lambda values: values[0]['scores'].update(citations=None))
    config, recorded, output = tmp_path/'matrix.yaml', tmp_path/'results.jsonl', tmp_path/'new/frozen.json'
    config.write_text(yaml.safe_dump(spec))
    recorded.write_text(''.join(json.dumps(r)+'\n' for r in results))
    with pytest.raises(SystemExit, match='no B6 freeze'):
        experiments.main(['freeze', '--config', str(config), '--results', str(recorded), '--output', str(output)])
    assert not output.exists()
    assert json.loads(output.with_suffix('.readiness.json').read_text())['blocks'][0]['report']['comparisons'][CANDIDATE]['dev_quality_eligible'] is False


def test_complete_expected_cohort_and_job_protocol_are_required(tmp_path):
    spec, results, quality = setup_joint(tmp_path)
    path = Path(quality['methods'][CANDIDATE]['run'])
    payload = json.loads(path.read_text())
    payload['report']['turns'].pop()
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='complete declared'):
        experiments.select_development(spec, results)
    payload['report']['turns'].append(deepcopy(payload['report']['turns'][0]))
    payload['report']['turns'][-1]['task_id'] = 's2:1'
    payload['report']['turns'][0]['online_turn']['protocol'] = 'P3'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='protocol differs'):
        experiments.select_development(spec, results)
