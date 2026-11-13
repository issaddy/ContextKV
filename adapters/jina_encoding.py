"""Jina passage packing, marker projection and cross-batch relevance."""


def pack_passages(passages, sizes, *, overhead, budget):
    batches, current, used = [], [], overhead
    for passage, size in zip(passages, sizes):
        if current and used + size > budget:
            batches.append(tuple(current))
            current, used = [], overhead
        current.append(passage)
        used += size
    if current:
        batches.append(tuple(current))
    return batches


def project_markers(model, states, document_positions, query_position):
    import torch.nn.functional as F

    documents = model.projector(states[document_positions]).float()
    query = model.projector(states[query_position]).float()
    scores = F.cosine_similarity(documents, query[None, :], dim=-1)
    return F.normalize(documents, dim=-1), scores, query


def merge_rankings(batches):
    """Weight each batch query by its strongest local relevance, as in Jina rerank."""
    import torch

    documents = torch.cat([documents for documents, _, _ in batches])
    query_sum = torch.stack([query * ((1 + scores) / 2).max()
                             for _, scores, query in batches]).sum(0)
    scores = torch.nn.functional.cosine_similarity(documents, query_sum[None, :], dim=-1)
    return documents, scores, query_sum
