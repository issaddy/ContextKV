"""No encoder, learning, or model execution: supplied graph logits only."""
import numpy as np
from adapters.history_state import TaskPrediction
from controller.context_schema import Condition, Need, TaskState, TextRef
from training.task_relations import candidates, targets, apply_edges, graph_items


def test_owned_edges_history_scope_and_acyclic_decoding():
    a, b, x, y = [TextRef(1, 'user', i, i+1) for i in range(4)]
    old = Condition('old', value_spans=(x,))
    new = Condition('new', value_spans=(y,))
    previous = Need('previous', (x,))
    state = TaskState('history', '', conditions=(old,), needs=(previous,))
    needs = (Need('a', (a,), subject_refs=(x,), condition_ids=('old',),
                  predecessor_need_ids=('previous',), scope='all_supplied_documents'),
             Need('b', (b,), subject_refs=(y,), condition_ids=('new', 'old')))
    encoded = {'prior_tasks': (state,), 'prior_conditions': (old,)}
    items = candidates(encoded, needs, (new,), (x, y), relation="followup", resume_task_id="history")
    gold = targets(items)
    scores = {name: np.array(values)*4-2 for name, values in gold.items() if name != 'scope'}
    scores['scope'] = np.array([[-2, 2], [2, -2]])
    scores['need_predecessor'][0, 1] = 3
    scores['need_predecessor'][1, 0] = 1
    decoded = apply_edges(items, scores)
    assert decoded[0].subject_refs == (x,)
    assert decoded[1].subject_refs == (y,)
    assert decoded[0].condition_ids == ('old',)
    assert decoded[1].condition_ids == ('new', 'old')
    assert decoded[0].scope == 'all_supplied_documents'
    assert decoded[0].predecessor_need_ids == ('b', 'previous')
    assert decoded[1].predecessor_need_ids == ()
    empty = candidates(encoded, (), (), (), relation="standalone", resume_task_id="")
    assert apply_edges(empty, {'need_predecessor': np.empty((0, 1))}) == ()
    prediction = TaskPrediction('followup', conditions=(new,), needs=needs)
    renamed = TaskPrediction('followup', conditions=(Condition('renamed', value_spans=(y,)),),
        needs=(needs[0], Need('other', (b,), subject_refs=(y,), condition_ids=('renamed', 'old'))))
    assert graph_items(prediction, (old,), (state,)) == graph_items(renamed, (old,), (state,))


def test_full_decode_uses_group_edges_and_multilabel_catalogue_updates():
    from types import SimpleNamespace
    from training.task_model import encode_task, task_targets, decode_prediction, SPAN_KINDS, UPDATES, OPERATIONS
    from controller.context_schema import RetrievalState, VersionIntent
    from controller.schema import TurnRecord
    class Characters:
        cls_token_id, sep_token_id = 1000, 1001
        def __call__(self, text, **kwargs):
            result = {'input_ids': list(range(len(text)))}
            if kwargs.get('return_offsets_mapping'):
                result['offset_mapping'] = [(i, i+1) for i in range(len(text))]
            return result
    prior_ref = TextRef(1, 'user', 0, 1)
    old = (Condition('old1', value_spans=(prior_ref,)), Condition('old2', value_spans=(prior_ref,)))
    state = TaskState('prior', 'z', conditions=old, needs=(Need('priorneed', (prior_ref,)),), current_turn_id=1)
    memory = RetrievalState(1, (TurnRecord(1, 'z', 'actual'),), (state,))
    turn = SimpleNamespace(turn=2, query='abcdef')
    catalogue = {'manual': [{'version': 'v1', 'title': 'old'}, {'version': 'v2', 'title': 'new'}]}
    encoded = encode_task(Characters(), turn, (), memory, catalogue=catalogue, max_input_tokens=3000)
    a, b, key, value, predicate = [TextRef(2, 'user', i, i+1) for i in range(5)]
    condition = Condition('new', key, (value,), predicate_span=predicate,
                          target_refs=('old1', 'old2'), supersedes=('old2',), update='REPLACE')
    expected = TaskPrediction('followup', resume_task_id='prior', conditions=(condition,),
        needs=(Need('newneed', (a, b), condition_ids=('new',), scope='all_supplied_documents'),),
        version_intent=VersionIntent('compare', document_families=('manual',),
            version_ids=('v1', 'v2'), resolved_version_ids=('v2',), source_turns=(1,)))
    gold = task_targets(encoded, expected)
    length = len(encoded['input_ids'])
    output = {}
    for name, size in [('span', len(SPAN_KINDS)*3), ('history', 2), ('operation', len(OPERATIONS)),
                       ('update', len(UPDATES)), ('condition_relation', 6)]:
        output[name] = np.full((1, length, size), -4.)
    for i, labels in enumerate(gold['span']):
        for channel, label in enumerate(labels):
            if label >= 0:
                output['span'][0, i, channel*3+label] = 4
    for name in ('history', 'operation', 'update', 'condition_relation'):
        for i, label in enumerate(gold[name]):
            output[name][0, i, max(0, label)] = 4
    for name in ('relation', 'version', 'temporal_relation', 'resume'):
        values = np.full((1, 8), -4.)
        values[0, gold[name]] = 4
        output[name] = values
    for name in ('condition_target', 'condition_supersedes'):
        output[name] = np.array(gold[name], dtype=float)[None] * 8 - 4
    for name in ('document_families', 'version_ids', 'resolved_version_ids', 'source_turns'):
        output[name] = np.array(gold[name])*8-4
    output['group_scores'] = lambda name, refs: np.ones((len(refs), len(refs)))
    def graph_scores(items):
        values = {name: np.full((len(items['needs']), len(items[kind])), -4.)
                  for name, kind in [('need_subject', 'subjects'), ('need_condition', 'conditions'),
                                     ('need_predecessor', 'predecessors')]}
        values['need_condition'][:, 0] = 4
        values['scope'] = np.array([[-4, 4]])
        return values
    output['graph_scores'] = graph_scores
    decoded = decode_prediction(encoded, output, turn)
    assert decoded.needs[0].anchor_spans == (a, b)
    assert len(decoded.needs) == 1
    assert decoded.conditions[0].key_span == key
    assert decoded.conditions[0].predicate_span == predicate
    assert decoded.conditions[0].value_spans == (value,)
    assert decoded.conditions[0].target_refs == ('old1', 'old2')
    assert decoded.conditions[0].supersedes == ('old2',)
    assert decoded.version_intent == expected.version_intent
    assert decoded.needs[0].scope == 'all_supplied_documents'
    assert graph_items(decoded, old, (state,)) == graph_items(expected, old, (state,))
    # Required graph state is retained before optional history windows.
    from adapters.history_state import HistoryWindow
    required = len(encoded['input_ids'])
    with_optional = encode_task(Characters(), turn,
        (HistoryWindow(TextRef(1, 'assistant', 0, 6), 'actual', 6),), memory,
        catalogue=catalogue, max_input_tokens=required)
    assert len(with_optional['input_ids']) == required
