"""Shared span candidates and independent Need graph edges for the BGE task head."""
from dataclasses import replace
from adapters.history_state import TASK_MAX_REFS, task_graph_history

SCOPES = ("selected_sources", "all_supplied_documents")
FORMAT = "bge_owned_need_graph_v3"


def ordered_refs(refs):
    return tuple(sorted(refs, key=lambda ref: (ref.turn_id, ref.speaker, ref.start, ref.end)))


def positions(encoded, refs):
    return [i for i, token in enumerate(encoded['refs']) if token is not None and any(
        token.turn_id == ref.turn_id and token.speaker == ref.speaker
        and token.start < ref.end and ref.start < token.end for ref in refs)]


def candidates(encoded, needs, conditions, subjects, *, relation, resume_task_id):
    prior_conditions, prior_needs = task_graph_history(encoded['prior_tasks'], relation, resume_task_id)
    return {'needs': tuple(needs), 'subjects': tuple(dict.fromkeys(subjects)),
            'conditions': tuple(conditions) + tuple(c for c in prior_conditions
                if c.condition_id not in {v.condition_id for v in conditions}),
            'predecessors': tuple(needs) + prior_needs}


def targets(items):
    needs = items['needs']
    condition_ids = {c.condition_id for c in items['conditions']}
    predecessor_ids = {n.need_id for n in items['predecessors']}
    for need in needs:
        if not set(need.condition_ids) <= condition_ids or not set(need.predecessor_need_ids) <= predecessor_ids:
            raise ValueError('Need supervision names an unavailable graph candidate')
    return {'need_subject': [[int(ref in n.subject_refs) for ref in items['subjects']] for n in needs],
            'need_condition': [[int(c.condition_id in n.condition_ids) for c in items['conditions']] for n in needs],
            'need_predecessor': [[int(p.need_id in n.predecessor_need_ids) for p in items['predecessors']] for n in needs],
            'scope': [SCOPES.index(n.scope) for n in needs]}


def apply_edges(items, scores):
    needs = items['needs']
    edges = {n.need_id: set() for n in items['predecessors']}
    current = {n.need_id for n in needs}
    for n in items['predecessors']:
        if n.need_id not in current:
            edges[n.need_id].update(n.predecessor_need_ids)
    def reaches(start, goal, seen):
        if start == goal:
            return True
        if start in seen:
            return False
        seen.add(start)
        return any(reaches(p, goal, seen) for p in edges.get(start, ()))
    proposed = sorted(((float(scores['need_predecessor'][i, j]), n.need_id, p.need_id)
        for i, n in enumerate(needs) for j, p in enumerate(items['predecessors'])), reverse=True)
    for score, need, predecessor in proposed:
        if len(edges[need]) >= TASK_MAX_REFS:
            continue
        if score > 0 and not reaches(predecessor, need, set()):
            edges[need].add(predecessor)
    def selected(name, i, values):
        ranked = sorted(range(len(values)), key=lambda j: float(scores[name][i, j]), reverse=True)
        chosen = {j for j in ranked[:TASK_MAX_REFS] if scores[name][i, j] > 0}
        return tuple(value for j, value in enumerate(values) if j in chosen)
    return tuple(replace(n,
        subject_refs=selected('need_subject', i, items['subjects']),
        condition_ids=tuple(c.condition_id for c in selected('need_condition', i, items['conditions']) if c.update != 'REMOVE'),
        predecessor_need_ids=tuple(p.need_id for p in items['predecessors'] if p.need_id in edges[n.need_id]),
        scope=SCOPES[int(scores['scope'][i].argmax())]) for i, n in enumerate(needs))


def graph_items(prediction, prior_conditions=(), prior_tasks=()):
    """IDs are local; evaluation compares actual anchors and condition semantics."""
    conditions = {c.condition_id: c for c in (*prior_conditions, *prediction.conditions)}
    needs = {n.need_id: n for state in prior_tasks for n in state.needs}
    needs.update({n.need_id: n for n in prediction.needs})
    def condition_key(identity):
        c = conditions[identity]
        return (c.relation_type, c.key_span, ordered_refs(c.value_spans), c.predicate_span, c.update, tuple(sorted(c.target_refs)), tuple(sorted(c.supersedes)))
    return {
        'need_subject': {(ordered_refs(n.anchor_spans), s) for n in prediction.needs for s in n.subject_refs},
        'need_condition': {(ordered_refs(n.anchor_spans), condition_key(c)) for n in prediction.needs for c in n.condition_ids},
        'need_predecessor': {(ordered_refs(n.anchor_spans), ordered_refs(needs[p].anchor_spans)) for n in prediction.needs for p in n.predecessor_need_ids},
        'scope': {(ordered_refs(n.anchor_spans), n.scope) for n in prediction.needs},
        'need_group': {ordered_refs(n.anchor_spans) for n in prediction.needs},
        'condition_structure': {condition_key(c.condition_id) for c in prediction.conditions}}



def grouped_indices(scores, count, *, max_size=None, compatible=None):
    """Decode symmetric same-object edges into disjoint span groups."""
    groups = [{i} for i in range(count)]
    edges = sorted(((float(scores[i, j] + scores[j, i]), i, j)
                    for i in range(count) for j in range(i)), reverse=True)
    for score, i, j in edges:
        if score <= 0:
            break
        left = next(g for g in groups if i in g)
        right = next(g for g in groups if j in g)
        merged = left | right
        if (left is not right and (max_size is None or len(merged) <= max_size)
                and (compatible is None or compatible(merged))):
            left.update(right)
            groups.remove(right)
    return tuple(tuple(sorted(group)) for group in groups)


def grouping_targets(objects):
    spans = tuple(dict.fromkeys(ref for refs in objects for ref in refs))
    labels = [[int(any(left in refs and right in refs for refs in objects))
               for right in spans] for left in spans]
    return spans, labels
