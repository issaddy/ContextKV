from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from adapters.mtrag import OnlineContext
from adapters.retriever_factory import HybridRetrievalConfig
from bench.source_topic_eval import evaluate_prepared, main, prepare_audit_turns, prepare_cohort, summarize_results
from controller.schema import EvidenceBlock, EvidenceKey


def block(name):
    return EvidenceBlock(EvidenceKey('fixture', name, name, 'v1', name), f'fixture://{name}',
                         f'{name} title', f'{name} complete original body')


def audits():
    a, b, hidden, current = (block(name) for name in ('earlier', 'second', 'hidden-archive', 'current-only'))
    output, history = [], []
    for turn, source in enumerate((a, b, current), 1):
        value = {'task_id': f's::{turn}', 'conversation_id': 's', 'turn': turn, 'protocol': 'P3',
            'query': f'Question {turn}', 'actual_history': deepcopy(history), 'answer': f'ACTUAL answer {turn}',
            'visible_evidence_manifest': [{'key': asdict(source.key), 'reference_id': f'E_{turn}'}],
            'audit_evidence': [asdict(source), asdict(hidden)],
            'targets': [{'text': 'OFFICIAL GOLD MUST NOT ENTER'}], 'rewrite': 'OFFICIAL REWRITE MUST NOT ENTER'}
        output.append(value)
        history.extend([{'speaker': 'user', 'text': value['query']}, {'speaker': 'assistant', 'text': value['answer']}])
    return output


def test_prior_sources_are_strictly_earlier_visible_and_snapshots_do_not_mutate():
    records = audits()
    before = deepcopy(records)
    records[1]['visible_evidence_manifest'].append({'key': asdict(block('missing').key), 'reference_id': 'E_f'})
    # A body first available in the current turn cannot repair its own prior context.
    records[2]['audit_evidence'].append(asdict(block('missing')))
    records[2]['visible_evidence_manifest'].append({'key': asdict(block('missing').key), 'reference_id': 'E_f'})
    prepared = prepare_cohort(list(reversed(records)), {'dev': ['s'], 'test': ['heldout']})
    assert [p.query.turn for p in prepared] == [1, 2, 3]
    assert not prepared[0].prior_context.sources
    assert [s.block.key.document_id for s in prepared[1].prior_context.sources] == ['earlier']
    assert [s.block.key.document_id for s in prepared[2].prior_context.sources] == ['earlier', 'second']
    assert [k.document_id for k in prepared[2].prior_context.missing_keys] == ['missing']
    assert prepared[1].prior_context.sources[0].visible_turns == (1,)
    assert prepared[2].prior_context.snapshot_version is None
    assert prepared[2].query.generated_history[-1]['text'] == 'ACTUAL answer 2'
    assert 'OFFICIAL' not in json.dumps([asdict(item) for item in prepared])
    assert records[0] == before[0]
    assert all('current-only' not in json.dumps(asdict(item)) for item in prepared)


def test_missing_prior_audits_are_reported_without_borrowing_current_sources():
    prepared = prepare_cohort([audits()[1]], {'dev': ['s']})
    assert prepared[0].missing_prior_turns == (1,)
    assert not prepared[0].prior_context.sources
    assert len(prepared[0].query.generated_history) == 2


def test_visible_inventory_scope_does_not_claim_required_equivalence_for_extra_material():
    records = audits()
    for row in records:
        shown = row['audit_evidence'][0]
        row['retrieved_evidence'] = [{'document_id': shown['key']['passage_id'], 'text': shown['text']}]
    exact = prepare_cohort(records, {'dev': ['s']})
    assert exact[-1].prior_context.source_inventory_scope == 'prior_audit_visible_equals_retrieved'
    extra = block('shown-but-not-current-required')
    records[0]['audit_evidence'].append(asdict(extra))
    records[0]['visible_evidence_manifest'].append({'key': asdict(extra.key), 'reference_id': 'E_extra'})
    broader = prepare_cohort(records, {'dev': ['s']})
    assert broader[1].prior_context.source_inventory_scope == 'prior_audit_visible'
    assert extra.key in [s.block.key for s in broader[1].prior_context.sources]
    assert len(broader[1].prior_context.sources) > len(exact[1].prior_context.sources)


def test_selector_only_export_has_same_past_context_and_no_current_answer():
    records = audits()
    cohort = prepare_cohort(records, {'dev': ['s']})
    inputs = prepare_audit_turns(records, {'dev': ['s']})
    for item, (turn, context) in zip(cohort, inputs, strict=True):
        assert context == item.prior_context and turn.protocol == 'P3' and turn.contexts == ()
        assert turn.user_questions == item.query.user_questions
        assert f'ACTUAL answer {turn.turn}' not in json.dumps(asdict(turn))


@pytest.mark.parametrize('kind', ['test', 'duplicate', 'p1', 'history', 'conflicting_source'])
def test_invalid_fixed_input_boundaries_fail_before_any_retrieval(kind):
    records = audits()
    if kind == 'test':
        records[0]['conversation_id'] = 'heldout'
    elif kind == 'duplicate':
        records.append(deepcopy(records[0]))
    elif kind == 'p1':
        records[0]['protocol'] = 'P1'
    elif kind == 'history':
        records[1]['actual_history'][1]['text'] = 'different run answer'
    else:
        records[1]['visible_evidence_manifest'] = records[0]['visible_evidence_manifest']
        records[1]['audit_evidence'] = deepcopy(records[0]['audit_evidence'])
        records[1]['audit_evidence'][0]['text'] = 'changed body under the same key'
    with pytest.raises(ValueError):
        prepare_cohort(records, {'dev': ['s'], 'test': ['heldout']})


class Resolver:
    def __init__(self, events):
        self.events, self.inputs = events, []

    def resolve(self, turn, prior):
        self.events.append(('resolve', turn.turn))
        self.inputs.append((turn, prior))
        assert turn.contexts == ()
        assert all(t < turn.turn for source in prior.sources for t in source.visible_turns)
        return {'text': turn.query, 'status': 'input_over_budget' if turn.turn == 2 else 'unchanged',
                'reason': 'selector_input_over_budget' if turn.turn == 2 else 'fixture', 'model_called': False}


def retriever(events):
    return SimpleNamespace(query_mode='source_topic', query_fusion='current_topic_rrf', stage='hybrid_rerank',
        rerank_query_mode='resolved',
        resolver=Resolver(events), sparse=object(), dense=object(), encoder=object(), reranker=object(),
        candidate_k=50, rerank_k=50, rrf_k=60, manifest={'fixture': True}, close=lambda: events.append(('close',)))


def search_fixture(events, *, fail_turn=None):
    def search(current, topic, sparse, dense, encoder, reranker, **budgets):
        turn = int(current.rsplit(' ', 1)[-1])
        events.append(('search', turn))
        assert current == topic and budgets == {'candidate_k': 50, 'rerank_k': 50, 'rrf_k': 60,
                                                'rerank_query': topic}
        if turn == fail_turn:
            raise RuntimeError('retrieval fixture failure')
        contexts = tuple(OnlineContext(str(n), f'original passage {n}', score=float(50-n)) for n in range(50))
        return {stage: {'contexts': contexts, 'elapsed_ms': 1.0, 'query_fusion_trace': {'fixture': True},
                        'rerank_candidate_ids': list(map(str, range(50)))} for stage in ('hybrid', 'hybrid_rerank')}
    return search


def test_shared_search_keeps_full_pool_and_selector_fallback_is_not_retrieval_failure(monkeypatch):
    import bench.source_topic_eval as module
    events, prepared = [], prepare_cohort(audits(), {'dev': ['s']})
    model = retriever(events)
    monkeypatch.setattr(module, 'search_current_topic', search_fixture(events, fail_turn=3))
    results = list(evaluate_prepared(prepared, model))
    assert len(results) == 6
    assert [r['status'] for r in results] == ['complete', 'complete', 'complete', 'complete', 'failed', 'failed']
    assert results[2]['query_view_trace']['status'] == 'input_over_budget'
    assert len(results[0]['candidate_ids']) == len(results[0]['candidate_scores']) == 50
    assert len(results[0]['contexts']) == 5
    assert results[-1]['elapsed_ms'] > 0 and results[-1]['candidate_ids'] == []
    assert 'OFFICIAL' not in json.dumps([asdict(turn) for turn, _ in model.resolver.inputs])
    for turn, prior in model.resolver.inputs:
        assert f'ACTUAL answer {turn.turn}' not in json.dumps(asdict(turn))
        assert 'current-only' not in json.dumps(asdict(prior))
    summary = summarize_results(results, {f's::{t}': {'0': 1} for t in (1, 2, 3)}, candidate_k=50)
    assert summary['coverage']['retrieval_status'] == {'complete': 2, 'failed': 1}
    assert summary['coverage']['selector_status'] == {'unchanged': 2, 'input_over_budget': 1}
    for value in summary['all_expected_with_failed_empty_rankings']['methods'].values():
        assert value['top5']['requested_tasks'] == 3 and value['top5']['mean_recall'] == pytest.approx(2/3)
    for value in summary['completed_only_diagnostic']['methods'].values():
        assert value['top5']['requested_tasks'] == 2 and value['top5']['mean_recall'] == 1


def test_source_offline_entry_honors_explicit_current_reranking(monkeypatch):
    import bench.source_topic_eval as module
    model = retriever([])
    model.rerank_query_mode = 'current'
    model.resolver.resolve = lambda turn, prior: {'text': turn.query+' topic expansion', 'status': 'resolved'}
    calls = []

    def search(current, topic, *args, **kwargs):
        calls.append((current, topic, kwargs['rerank_query']))
        return {stage: {'contexts': (), 'elapsed_ms': 0, 'query_fusion_trace': {},
                        'rerank_candidate_ids': []} for stage in ('hybrid', 'hybrid_rerank')}

    monkeypatch.setattr(module, 'search_current_topic', search)
    results = list(evaluate_prepared(prepare_cohort(audits(), {'dev': ['s']}), model))
    assert len(calls) == 3 and len(results) == 6
    assert all(scoring == current and topic != current for current, topic, scoring in calls)
    assert all(r['rerank_query_mode'] == 'current' for r in results)


@pytest.mark.parametrize('execute', [False, True])
def test_cli_plans_without_model_and_reads_qrels_only_after_all_attempts(tmp_path, monkeypatch, execute):
    import bench.source_topic_eval as module
    events = []
    audit, split, config, qrels = (tmp_path/name for name in ('audit.jsonl', 'split.json', 'config.json', 'qrels.tsv'))
    audit.write_text(''.join(json.dumps(row)+'\n' for row in audits()))
    split.write_text(json.dumps({'dev': ['s'], 'test': ['heldout']}))
    config.write_text('{}')
    qrels.write_text('query-id\tcorpus-id\tscore\ns::1\t0\t1\n')
    configuration = HybridRetrievalConfig('sparse', 'dense', 'encoder', query_mode='source_topic',
                                         query_fusion='current_topic_rrf', reranker_path='reranker')
    monkeypatch.setattr(module, 'load_retriever_config', lambda path: configuration)
    def create(protocol, **kwargs):
        events.append(('create',))
        assert protocol == 'P3' and kwargs == {'config_path': str(config)}
        return retriever(events)
    monkeypatch.setattr(module, 'create_retriever', create)
    monkeypatch.setattr(module, 'search_current_topic', search_fixture(events, fail_turn=2))
    def labels(path):
        assert events[-1] == ('search', 3)
        events.append(('qrels',))
        return {f's::{t}': {'0': 1} for t in (1, 2, 3)}
    monkeypatch.setattr(module, 'load_qrels', labels)
    output = tmp_path/'output'
    args = ['--audit', str(audit), '--split-manifest', str(split), '--retriever-config', str(config),
            '--output', str(output), '--qrels', str(qrels)] + (['--execute'] if execute else [])
    code = main(args)
    if not execute:
        assert code == 0 and events == [] and not output.exists()
    else:
        assert code == 1 and events[-2:] == [('qrels',), ('close',)]
        manifest = json.loads((output/'manifest.json').read_text())
        assert manifest['status'] == 'completed_with_failures' and manifest['attempted_queries'] == 3
        assert manifest['unattempted_task_ids'] == [] and manifest['qrels_read_after_all_attempts']
        results = [json.loads(line) for line in (output/'retrieval.jsonl').read_text().splitlines()]
        assert len(results) == 6 and (output/'summary.json').exists()
