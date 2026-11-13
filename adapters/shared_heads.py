"""Task heads over the shared query and evidence backbones."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

OPERATIONS = ("lookup", "enumerate", "procedure", "compare", "count", "sum", "mean", "min", "max", "group", "infer")
SCOPES = ("selected_sources", "all_supplied_documents")
SPAN_KINDS = ("need", "subject", "condition", "value", "time", "version")
CONDITION_TYPES = ("PREDICATE", "TIME", "VERSION", "PLATFORM", "LOCATION", "SUBJECT")
UPDATES = ("ADD", "REPLACE", "REMOVE", "COMPARE")
VERSION_MODES = ("current", "explicit", "as_of", "compare")
TEMPORAL_RELATIONS = ("at", "before", "after")
EDGE_KINDS = ("subject", "condition", "predecessor", "value")
FIELD_ROLES = ("source", "measure", "entity", "period", "group", "filter", "unit", "version")
FILTER_OPERATORS = ("eq", "ne", "lt", "le", "gt", "ge", "in", "contains")


class ScaledBilinear(nn.Bilinear):
    def forward(self, left, right):
        left = F.layer_norm(left, (self.in1_features,))
        right = F.layer_norm(right, (self.in2_features,))
        return super().forward(left, right) * self.in1_features ** -.5


class QueryHeads(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.context = nn.Linear(hidden_size, hidden_size, bias=False)
        self.span = nn.Linear(hidden_size, len(SPAN_KINDS) * 3)
        self.history = nn.Linear(hidden_size, 1)
        self.operation = nn.Linear(hidden_size, len(OPERATIONS))
        self.scope = nn.Linear(hidden_size, len(SCOPES))
        self.coverage = nn.Linear(hidden_size, len(SCOPES))
        self.join = nn.Linear(hidden_size, 2)
        self.distinct = nn.Linear(hidden_size, 2)
        self.condition_type = nn.Linear(hidden_size, len(CONDITION_TYPES))
        self.update = nn.Linear(hidden_size, len(UPDATES))
        self.filter_operator = nn.Linear(hidden_size, len(FILTER_OPERATORS))
        self.version_mode = nn.Linear(hidden_size, len(VERSION_MODES))
        self.temporal_relation = nn.Linear(hidden_size, len(TEMPORAL_RELATIONS))
        self.version = ScaledBilinear(hidden_size, hidden_size, 1)
        self.family = ScaledBilinear(hidden_size, hidden_size, 1)
        self.edge = ScaledBilinear(hidden_size, hidden_size, len(EDGE_KINDS))
        self.need_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.need_projection.weight)

    def contextualize(self, token_states, pooled):
        tokens = F.layer_norm(token_states, (token_states.shape[-1],))
        current = self.context(F.layer_norm(pooled, (pooled.shape[-1],)))
        current = F.layer_norm(current, (current.shape[-1],))
        return F.layer_norm(tokens + current + tokens * current, (tokens.shape[-1],))

    def forward(self, token_states, pooled):
        token_states = self.contextualize(token_states, pooled)
        pooled = F.layer_norm(pooled, (pooled.shape[-1],))
        return {
            "features": token_states,
            "span": self.span(token_states).view(*token_states.shape[:-1], len(SPAN_KINDS), 3),
            "history": self.history(token_states).squeeze(-1),
            "version_mode": self.version_mode(pooled),
            "temporal_relation": self.temporal_relation(pooled),
            "coverage_scope": self.coverage(pooled),
        }

    def edges(self, left, right):
        a = left[:, None, :].expand(-1, right.shape[0], -1)
        b = right[None, :, :].expand(left.shape[0], -1, -1)
        return self.edge(a, b)


class EvidenceHeads(nn.Module):
    def __init__(self, hidden_size=1024, query_size=1024):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(query_size), nn.Linear(query_size, hidden_size, bias=False))
        self.field = nn.Linear(hidden_size, len(FIELD_ROLES) * 3)
        self.binding = ScaledBilinear(hidden_size, hidden_size, len(FIELD_ROLES))
        self.join = ScaledBilinear(hidden_size, hidden_size, 1)
        self.set_utility = nn.Sequential(nn.Linear(12, 16), nn.GELU(), nn.Linear(16, 1))

    def field_logits(self, states, need_vectors):
        queries = F.layer_norm(self.query(need_vectors), (states.shape[-1],))
        tokens = F.layer_norm(states, (states.shape[-1],))
        a, b = queries[:, None, :], tokens[None, :, :]
        # The product gives each Need its own token ordering in each BIO/role logit.
        return self.field(b + a + b*a).view(len(queries), len(states), len(FIELD_ROLES), 3)

    @staticmethod
    def features(embeddings, relevance, selected, support, completed):
        """Size-independent information state shared by selection and stopping."""
        selected = selected.bool()
        mask = selected.to(embeddings.dtype)
        count = mask.sum(-1, keepdim=True)
        maximum = relevance[None].expand_as(mask).masked_fill(~selected, -torch.inf).amax(-1, keepdim=True)
        minimum = relevance[None].expand_as(mask).masked_fill(~selected, torch.inf).amin(-1, keepdim=True)
        maximum = torch.where(count > 0, maximum, 0.)
        minimum = torch.where(count > 0, minimum, 0.)
        total = mask @ relevance[:, None]
        similarity = (embeddings @ embeddings.T).clamp(0, 1)
        pairs = mask[:, :, None]*mask[:, None, :]*(1-torch.eye(len(relevance), device=mask.device))
        redundancy = (pairs*similarity).sum((1, 2))[:, None]/(count*(count-1)).clamp_min(1)
        coverage = (similarity[None]*mask[:, :, None]).amax(1)
        weights = relevance.clamp_min(0)
        covered = (coverage*weights).sum(-1, keepdim=True)/weights.sum().clamp_min(1e-6)
        if support.shape[1]:
            need_coverage = (mask[:, :, None]*support[None]).amax(1)
            need_coverage = torch.maximum(need_coverage, completed[None])
            need_mean = need_coverage.mean(-1, keepdim=True)
            need_min = need_coverage.amin(-1, keepdim=True)
            known = completed.mean().expand_as(count)
        else:
            need_mean = need_min = known = torch.zeros_like(count)
        return torch.cat((maximum, minimum, total/count.clamp_min(1),
            total/(1+total.abs()), count/(count+1), redundancy, covered, need_mean, need_min, known,
            count.new_full(count.shape, support.shape[1]/(support.shape[1]+1)),
            (count > 0).to(mask.dtype)), -1)

    def utility(self, embeddings, relevance, selected, support, completed):
        """Predict additional answer quality beyond the already known facts."""
        features = self.features(embeddings, relevance, selected, support, completed)
        values = self.set_utility(features).sigmoid()
        empty_features = torch.zeros_like(features)
        empty_features[:, 7:10] = features[:, 9:10]
        if support.shape[1]:
            empty_features[:, 8] = completed.min()
        empty_features[:, 10] = features[:, 10]
        empty = self.set_utility(empty_features).sigmoid()
        return (values - empty).squeeze(-1)

    def loss(self, embeddings, relevance, masks, qualities, support, completed):
        predictions = self.utility(embeddings, relevance, masks, support, completed)
        empty = (masks == 0).all(dim=-1)
        gains = qualities-qualities[empty].mean()
        loss = F.mse_loss(predictions, gains)
        return loss + F.kl_div(predictions.log_softmax(-1), gains.softmax(-1), reduction="sum")
