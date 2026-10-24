"""Development-only retrieval evaluation from frozen actual-history audits.

Default CLI behavior is a plan. --execute loads the configured local retriever
and may call its explicitly configured selector endpoint. It generates no answer.
Qrels are opened only after every query attempt, including failed attempts.

Example (plan only; append --execute for real selector/retrieval calls):
  python -m bench.source_topic_eval --audit saved-audit.jsonl \
    --split-manifest configs/mtrag_split_manifest.json \
    --retriever-config source-topic.json --qrels dev.tsv --output runs/source-topic-dev

Prepare local model/index snapshots and the configured selector service before
execution. This module does not download weights or start/stop a model service.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

from adapters.hybrid_retrieval import rerank_query_text, search_current_topic
from adapters.mtrag import CLOUD_COLLECTION, OnlineMessage, OnlineTurn
from adapters.query_resolution import PriorContext, PriorSource, SOURCE_MODES
from adapters.retriever_factory import create_retriever, load_retriever_config
from bench.quality import load_qrels
from bench.retrieval_eval import RetrievalQuery, file_source, prepare_queries, summarize
from controller.schema import EvidenceBlock, EvidenceKey
from controller.serialization import decode


@dataclass(frozen=True)
class PreparedTopicQuery:
    query: RetrievalQuery
    prior_context: PriorContext
    missing_prior_turns: tuple[int, ...] = ()


def prepare_cohort(audits, split_manifest, *, collection=CLOUD_COLLECTION):
    """Freeze only earlier visible bodies; current/future sources cannot repair them.

    Actual history is already saved by the generation run. Newly retrieved
    passages in this evaluation are never fed into a subsequent query's context.
    """
    dev_sessions = set(split_manifest['dev'])
    if any(r['conversation_id'] not in dev_sessions for r in audits):
        raise ValueError('source-topic evaluation accepts declared development conversations only')
    positions = [(r['conversation_id'], int(r['turn'])) for r in audits]
    if len(positions) != len(set(positions)):
        raise ValueError('duplicate conversation/turn in audit cohort')
    # This minimal row has no official assistant, answer, rewrite or contexts.
    user_rows = [{'task_id': r['task_id'], 'conversation_id': r['conversation_id'], 'turn': r['turn'],
        'Collection': r.get('collection', r.get('Collection', collection)),
        'input': [{'speaker': 'user', 'text': m['text']} for m in r['actual_history'] if m['speaker'] == 'user']
                 + [{'speaker': 'user', 'text': r['query']}]}
        for r in audits]
    queries = prepare_queries(user_rows, split_manifest, split='dev', collection=collection, history_audits=audits)
    by_task = {r['task_id']: r for r in audits}
    prepared, state, equivalent = [], {}, {}
    for query in queries:
        blocks, visible, missing, previous = state.setdefault(query.conversation_id, ({}, {}, set(), {}))
        for number, earlier in previous.items():
            history = query.generated_history
            if (history[2*(number-1)]['text'] != earlier['query']
                    or history[2*(number-1)+1]['text'] != earlier.get('answer')):
                raise ValueError('actual_history does not match the prior completed audit answer')
        scope = ('prior_audit_visible_equals_retrieved' if equivalent.get(query.conversation_id, True)
                 else 'prior_audit_visible')
        context = PriorContext(tuple(PriorSource(block, tuple(visible[key])) for key, block in blocks.items()),
                               tuple(sorted(missing)), snapshot_version=None, source_inventory_scope=scope)
        prepared.append(PreparedTopicQuery(query, context,
            tuple(number for number in range(1, query.turn) if number not in previous)))
        # Import this turn only after its own frozen input has been constructed.
        # Never fill a missing earlier body from this turn's audit for this query.
        current = by_task[query.task_id]
        body_map = {decode(EvidenceKey, block['key']): block for block in current['audit_evidence']}
        # All currently retrieved passages are required by the shared policy.
        # Equality establishes the same conservative set for this common B0/RAG
        # shape; generic APPEND audits can contain extra previously visible blocks.
        visible_bodies = [(key.passage_id, body_map[key]['text'])
            for reference in current['visible_evidence_manifest']
            if (key := decode(EvidenceKey, reference['key'])) in body_map]
        retrieved = current.get('retrieved_evidence')
        same = (retrieved is not None and len(visible_bodies) == len(current['visible_evidence_manifest'])
                and sorted(visible_bodies) == sorted((r['document_id'], r['text']) for r in retrieved))
        equivalent[query.conversation_id] = equivalent.get(query.conversation_id, True) and same
        for reference in current['visible_evidence_manifest']:
            key = decode(EvidenceKey, reference['key'])
            if key not in body_map:
                missing.add(key)
                continue
            block = decode(EvidenceBlock, body_map[key])
            if key in blocks and (blocks[key].text != block.text or blocks[key].title != block.title):
                raise ValueError('the same evidence key has conflicting original body or title')
            blocks.setdefault(key, block)
            if query.turn not in visible.setdefault(key, []):
                visible[key].append(query.turn)
            missing.discard(key)
        previous[query.turn] = {'query': current['query'], 'answer': current.get('answer')}
    return tuple(prepared)


def _online_turn(query, collection):
    return OnlineTurn(query.conversation_id, query.task_id, query.turn, collection, query.user_questions[-1],
        tuple(OnlineMessage(**m) for m in query.generated_history), (), 'P3')


def prepare_audit_turns(audits, split_manifest, *, collection=CLOUD_COLLECTION):
    """Pure selector-only input export; no models, qrels or current answer fields.

    Use prepare_cohort instead when the caller also needs missing-prior-turn
    coverage. Both interfaces freeze exactly the same past-only source inventory.
    """
    return tuple((_online_turn(item.query, collection), item.prior_context)
                 for item in prepare_cohort(audits, split_manifest, collection=collection))


def evaluate_prepared(prepared, retriever, *, collection=CLOUD_COLLECTION):
    """Reuse online resolver and full-ranking search; no qrels/answer input."""
    if retriever.query_mode not in SOURCE_MODES or retriever.resolver is None:
        raise ValueError('source-topic evaluation requires source_topic or assistant_topic_unverified')
    if retriever.query_fusion != 'current_topic_rrf' or retriever.stage not in {'hybrid', 'hybrid_rerank'}:
        raise ValueError('source-topic evaluation requires current_topic_rrf with a hybrid stage')
    expected_stages = ('hybrid', 'hybrid_rerank') if retriever.stage == 'hybrid_rerank' else ('hybrid',)
    for item in prepared:
        started = time.perf_counter()
        query, prior = item.query, item.prior_context
        turn = _online_turn(query, collection)
        base = {'task_id': query.task_id, 'conversation_id': query.conversation_id, 'turn': query.turn,
            'query_mode': retriever.query_mode, 'query_fusion': retriever.query_fusion,
            'rerank_query_mode': retriever.rerank_query_mode,
            'current_query': turn.query, 'actual_history': [asdict(m) for m in turn.history],
            'history_protocol': 'frozen_actual_history_with_strictly_prior_visible_sources_v1',
            'source_inventory_scope': prior.source_inventory_scope,
            'prior_source_inventory': [{'key': asdict(s.block.key), 'visible_turns': list(s.visible_turns)} for s in prior.sources],
            'missing_source_keys': [asdict(k) for k in prior.missing_keys],
            'missing_prior_audit_turns': list(item.missing_prior_turns), 'snapshot_version': prior.snapshot_version}
        view = None
        try:
            view = retriever.resolver.resolve(turn, prior)
            resolution_ms = (time.perf_counter()-started)*1000
            stages = search_current_topic(turn.query, view['text'], retriever.sparse, retriever.dense,
                retriever.encoder, retriever.reranker if retriever.stage == 'hybrid_rerank' else None,
                candidate_k=retriever.candidate_k, rerank_k=retriever.rerank_k, rrf_k=retriever.rrf_k,
                rerank_query=rerank_query_text(turn, view['text'], retriever.rerank_query_mode))
            for stage in expected_stages:
                values = stages[stage]
                yield {**base, 'stage': stage,
                    'method': f'{retriever.query_mode}+{retriever.query_fusion}/{stage}',
                    'status': 'complete', 'query_text': view['text'], 'query_view_trace': view,
                    'query_view_ms': resolution_ms, 'query_fusion_trace': values['query_fusion_trace'],
                    'contexts': [asdict(c) for c in values['contexts'][:5]],
                    'candidate_ids': [c.document_id for c in values['contexts']],
                    'candidate_scores': [c.score for c in values['contexts']],
                    'rerank_candidate_ids': values.get('rerank_candidate_ids'),
                    'elapsed_ms': resolution_ms+values['elapsed_ms']}
        except Exception as exc:
            elapsed = (time.perf_counter()-started)*1000
            for stage in expected_stages:
                yield {**base, 'stage': stage,
                    'method': f'{retriever.query_mode}+{retriever.query_fusion}/{stage}',
                    'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                    'query_text': view['text'] if view else None, 'query_view_trace': view,
                    'query_fusion_trace': None, 'contexts': [], 'candidate_ids': [], 'candidate_scores': [],
                    'rerank_candidate_ids': None, 'elapsed_ms': elapsed}


def summarize_results(results, qrels, *, candidate_k):
    """Separate complete-only diagnostics from all expected failure denominators."""
    complete = [r for r in results if r['status'] == 'complete']
    attempts = {r['task_id']: r for r in results}
    return {'schema_version': 1, 'all_expected_with_failed_empty_rankings': summarize(results, qrels, candidate_k=candidate_k),
        'completed_only_diagnostic': summarize(complete, qrels, candidate_k=candidate_k),
        'coverage': {'expected_queries': len(attempts),
            'retrieval_status': dict(Counter(r['status'] for r in attempts.values())),
            'selector_status': dict(Counter((r.get('query_view_trace') or {}).get('status', 'no_trace') for r in attempts.values())),
            'failed_task_ids': [task for task, r in attempts.items() if r['status'] == 'failed'],
            'missing_prior_audit_tasks': [task for task, r in attempts.items() if r['missing_prior_audit_turns']],
            'missing_prior_body_tasks': [task for task, r in attempts.items() if r['missing_source_keys']]},
        'interpretation': 'Selector fallback still uses the current query and can complete retrieval; '
            'retrieval failures remain empty rankings only in the all-expected denominator. '
            'Frozen source-run history is not a new answer-generation closed loop.',
        'answer_quality_evaluated': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--split-manifest', required=True)
    parser.add_argument('--retriever-config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--qrels', required=True)
    parser.add_argument('--collection', default=CLOUD_COLLECTION)
    parser.add_argument('--execute', action='store_true', help='load local retrieval models and call the configured selector; no answer generation')
    args = parser.parse_args(argv)
    config = load_retriever_config(args.retriever_config)
    if config.query_mode not in SOURCE_MODES:
        parser.error('retriever config must explicitly choose source_topic or assistant_topic_unverified')
    audits = [json.loads(line) for line in Path(args.audit).read_text().splitlines() if line.strip()]
    split = json.loads(Path(args.split_manifest).read_text())
    prepared = prepare_cohort(audits, split, collection=args.collection)
    manifest = {'schema_version': 1, 'status': 'planned', 'split': 'dev',
        'config': vars(args), 'resolved_retriever_config': asdict(config),
        'sources': {name: file_source(path) for name, path in (
            ('audit', args.audit), ('split_manifest', args.split_manifest), ('retriever_config', args.retriever_config))},
        'task_ids': [p.query.task_id for p in prepared], 'expected_queries': len(prepared),
        'history_protocol': 'frozen_actual_history_with_strictly_prior_visible_sources_v1',
        'source_inventory_scope': 'prior_audit_visible; per-task trace states whether all prior inventories equal retrieved/required bodies',
        'collection_note': 'audit collection field when present; otherwise the caller-selected audit cohort collection',
        'model_input_boundary': 'current user query, actual prior QA and earlier visible original sources only',
        'qrels_read_after_all_attempts': False, 'answer_quality_evaluated': False,
        'pipeline': 'TopicResolver.resolve plus the shared search_current_topic; not HybridRetriever.retrieve_turn top5 wrapper',
        'timing': 'resolution plus stage wall time; initialization separate; no generation or independent latency benchmark'}
    if not args.execute:
        print(json.dumps({'mode': 'plan', 'model_calls_sent': 0, **manifest}, indent=2))
        return 0
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output/'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    retriever, results = None, []
    started = time.perf_counter()
    try:
        retriever = create_retriever('P3', config_path=args.retriever_config)
        manifest.update(status='running', initialization_seconds=time.perf_counter()-started,
                        retriever=retriever.manifest)
        with (output/'retrieval.jsonl').open('w') as stream:
            for row in evaluate_prepared(prepared, retriever, collection=args.collection):
                results.append(row)
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                stream.flush()
        # Qrels and their file bytes are deliberately unopened until here.
        summary = summarize_results(results, load_qrels(args.qrels), candidate_k=config.candidate_k)
        manifest['sources']['qrels'] = file_source(args.qrels)
        manifest['qrels_read_after_all_attempts'] = True
        (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
        manifest['status'] = 'complete' if not summary['coverage']['failed_task_ids'] else 'completed_with_failures'
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        attempted = {r['task_id'] for r in results}
        manifest['attempted_queries'] = len(attempted)
        manifest['unattempted_task_ids'] = [p.query.task_id for p in prepared if p.query.task_id not in attempted]
        if retriever is not None:
            manifest['retriever'] = retriever.manifest
            retriever.close()
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'status': manifest['status'], 'coverage': summary['coverage']}, indent=2))
    return 0 if manifest['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
